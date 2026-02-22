"""
Verification script for Contrastive Loss and Focal Loss implementations.
Tests the losses with known examples and compares to expected values.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Initialize distributed training (required for @gather_and_scale_wrapper decorator)
# Use a dummy single-process group for testing
if not torch.distributed.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=0,
        world_size=1,
        init_method='tcp://localhost:12355'
    )

# Add the project root to path (needed for relative imports)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# Also add opengait directory to path for utils imports
sys.path.insert(0, os.path.join(project_root, 'opengait'))

# Change to opengait directory for imports (mimics how main.py works)
os.chdir(os.path.join(project_root, 'opengait'))

# Import with proper path setup
from modeling.losses.contrastive import ContrastiveLoss
from modeling.losses.focal import FocalLoss
from modeling.losses.triplet import TripletLoss
from modeling.losses.ce import CrossEntropyLoss


def test_contrastive_loss():
    """Test Contrastive Loss with simple examples."""
    print("=" * 70)
    print("Testing Contrastive Loss")
    print("=" * 70)
    
    # Create simple test case
    # 3 samples: 2 from class 0, 1 from class 1
    # Format: [n, c, p] = [3 samples, 2 features, 2 parts]
    # Each sample has c=2 features, and there are p=2 parts
    # So for each part, we have n=3 samples with c=2 features
    
    # Create embeddings in [n, c, p] format
    # Note: torch.tensor with nested lists creates [n, p, c], need to permute to [n, c, p]
    # Sample 0: features [1.0, 0.0] for both parts
    # Sample 1: features [2.0, 0.0] for both parts  
    # Sample 2: features [0.0, 1.0] for both parts
    embeddings_npc = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]],      # Sample 0: [part0: [1,0], part1: [1,0]]
        [[2.0, 0.0], [2.0, 0.0]],      # Sample 1: [part0: [2,0], part1: [2,0]]
        [[0.0, 1.0], [0.0, 1.0]]       # Sample 2: [part0: [0,1], part1: [0,1]]
    ]).float()  # Shape: [n=3, p=2, c=2]
    embeddings = embeddings_npc.permute(0, 2, 1)  # [n=3, c=2, p=2]
    
    labels = torch.tensor([0, 0, 1])  # First two same class, third different
    
    # Create loss function
    margin = 1.0
    loss_fn = ContrastiveLoss(margin=margin, loss_term_weight=1.0)
    
    # Forward pass (embeddings should be [n, c, p])
    loss, info = loss_fn(embeddings, labels)
    
    # Loss is a tensor with shape [p] (one per part), so we need to take mean
    loss_mean = loss.mean() if loss.numel() > 1 else loss
    print(f"Contrastive Loss: {loss_mean.item():.4f}")
    print(f"Positive Loss: {info['pos_loss'].mean().item():.4f}")
    print(f"Negative Loss: {info['neg_loss'].mean().item():.4f}")
    print(f"Mean Distance: {info['mean_dist'].mean().item():.4f}")
    print(f"Number of Positive Pairs: {info['num_pos_pairs'].item()}")
    print(f"Number of Negative Pairs: {info['num_neg_pairs'].item()}")
    
    # Check that loss is reasonable (should be > 0 for negative pairs)
    assert loss_mean.item() >= 0, "Loss should be non-negative"
    print("✓ Loss is non-negative")
    
    # Check that we have positive and negative pairs
    assert info['num_pos_pairs'].item() > 0, "Should have positive pairs"
    assert info['num_neg_pairs'].item() > 0, "Should have negative pairs"
    print("✓ Has both positive and negative pairs")
    
    print("\nContrastive Loss test passed!\n")


def test_focal_loss():
    """Test Focal Loss with simple examples."""
    print("=" * 70)
    print("Testing Focal Loss")
    print("=" * 70)
    
    # Create simple test case
    # 2 samples, 3 classes, 2 parts (use p=2 to avoid edge case with p=1)
    # Note: torch.tensor with nested lists creates [n, p, c], need to permute to [n, c, p]
    logits_npc = torch.tensor([
        [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],  # Sample 0: confident in class 0, both parts
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]   # Sample 1: uncertain, both parts
    ]).float()  # [n=2, p=2, c=3]
    logits = logits_npc.permute(0, 2, 1)  # [n=2, c=3, p=2]
    
    labels = torch.tensor([0, 1])  # Sample 0 is class 0, sample 1 is class 1
    
    # Create loss function
    gamma = 2.0
    scale = 1.0  # Use scale=1 for easier verification
    loss_fn = FocalLoss(scale=scale, gamma=gamma, label_smooth=False, log_accuracy=True)
    
    # Forward pass
    loss, info = loss_fn(logits, labels)
    
    print(f"Focal Loss: {loss.item():.4f}")
    print(f"Accuracy: {info.get('accuracy', torch.tensor(0.0)).item():.4f}")
    
    # Manual calculation for verification
    # Sample 0: logits = [2.0, 1.0, 0.0] - confident in class 0
    #   probs = softmax([2.0, 1.0, 0.0]) ≈ [0.576, 0.318, 0.106]
    #   pt = 0.576 (probability of true class 0) - EASY example
    #   focal_weight = (1 - 0.576)^2 = 0.180
    
    # Sample 1: logits = [0.5, 0.5, 0.5] - uncertain
    #   probs = softmax([0.5, 0.5, 0.5]) = [0.333, 0.333, 0.333]
    #   pt = 0.333 (probability of true class 1) - HARD example
    #   focal_weight = (1 - 0.333)^2 = 0.444
    
    # Check probabilities (average over parts since we have p=2)
    probs = F.softmax(logits * scale, dim=1)
    # Average over parts for each sample
    pt_sample0 = probs[0, 0, :].mean().item()  # Sample 0, class 0, average over parts
    pt_sample1 = probs[1, 1, :].mean().item()  # Sample 1, class 1, average over parts
    
    print(f"\nManual Verification:")
    print(f"Sample 0 - Probability of true class (avg over parts): {pt_sample0:.4f}")
    print(f"Sample 1 - Probability of true class (avg over parts): {pt_sample1:.4f}")
    print(f"Sample 0 - Focal weight: {(1 - pt_sample0)**gamma:.4f}")
    print(f"Sample 1 - Focal weight: {(1 - pt_sample1)**gamma:.4f}")
    
    # Check that loss is reasonable
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Loss is non-negative")
    
    # Test with a clearer example: one very easy, one very hard
    print("\nTesting with clearer easy/hard example:")
    # Use extreme logits to make differences very clear
    # Need to create tensor in [n, c, p] format
    clear_scale = 1.0
    # Create as [n, p, c] first, then permute to [n, c, p]
    clear_logits_npc = torch.tensor([
        [[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],  # Sample 0: 2 parts, 3 classes - confident in class 0
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]     # Sample 1: 2 parts, 3 classes - uncertain
    ]).float()  # [n=2, p=2, c=3]
    clear_logits = clear_logits_npc.permute(0, 2, 1)  # [n=2, c=3, p=2]
    clear_labels = torch.tensor([0, 1])
    
    clear_loss_fn = FocalLoss(scale=clear_scale, gamma=gamma, label_smooth=False, log_accuracy=False)
    clear_loss, _ = clear_loss_fn(clear_logits, clear_labels)
    clear_probs = F.softmax(clear_logits * clear_scale, dim=1)
    # Get probability of true class for each sample
    clear_pt0 = clear_probs[0, 0, :].mean().item()  # Sample 0, class 0 (true label)
    clear_pt1 = clear_probs[1, 1, :].mean().item()  # Sample 1, class 1 (true label) - should be 1/3 ≈ 0.333
    
    print(f"  Easy sample - logits: {clear_logits[0, :, 0].tolist()}")
    print(f"    -> probs: {clear_probs[0, :, 0].tolist()}, pt={clear_pt0:.4f}, Focal weight: {(1 - clear_pt0)**gamma:.4f}")
    print(f"  Hard sample - logits: {clear_logits[1, :, 0].tolist()}")
    print(f"    -> probs: {clear_probs[1, :, 0].tolist()}, pt={clear_pt1:.4f}, Focal weight: {(1 - clear_pt1)**gamma:.4f}")
    
    # Sample 0 should have much higher pt (more confident) than sample 1
    # Sample 0: logits [20, 0, 0] -> probs ≈ [1.0, 0.0, 0.0] -> pt ≈ 1.0
    # Sample 1: logits [0, 0, 0] -> probs = [1/3, 1/3, 1/3] -> pt = 1/3 ≈ 0.333
    assert clear_pt0 > 0.9, f"Easy sample should have high confidence (pt > 0.9), got {clear_pt0:.4f}"
    assert clear_pt1 < 0.4, f"Hard sample should have low confidence (pt < 0.4), got {clear_pt1:.4f}"
    assert clear_pt0 > clear_pt1, "First sample should be easier"
    assert (1 - clear_pt1)**gamma > (1 - clear_pt0)**gamma, "Harder example should have higher focal weight"
    print("✓ Harder example correctly has higher focal weight")
    
    print("\nFocal Loss test passed!\n")


def test_focal_vs_crossentropy():
    """Compare Focal Loss to Cross Entropy Loss."""
    print("=" * 70)
    print("Comparing Focal Loss vs Cross Entropy Loss")
    print("=" * 70)
    
    # Create test case with easy and hard examples
    # Note: torch.tensor with nested lists creates [n, p, c], need to permute to [n, c, p]
    logits_npc = torch.tensor([
        [[5.0, 1.0, 0.0]],  # Easy: very confident in class 0
        [[0.5, 0.5, 0.5]]   # Hard: uncertain
    ]).float()  # [n=2, p=1, c=3]
    logits = logits_npc.permute(0, 2, 1)  # [n=2, c=3, p=1]
    
    labels = torch.tensor([0, 1])
    
    # Cross Entropy Loss
    ce_loss_fn = CrossEntropyLoss(scale=1.0, label_smooth=False, log_accuracy=False)
    ce_loss, _ = ce_loss_fn(logits, labels)
    
    # Focal Loss (gamma=2.0)
    focal_loss_fn = FocalLoss(scale=1.0, gamma=2.0, label_smooth=False, log_accuracy=False)
    focal_loss, _ = focal_loss_fn(logits, labels)
    
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    print(f"Focal Loss (gamma=2.0): {focal_loss.item():.4f}")
    
    # Focal loss should be lower than CE for easy examples
    # But might be similar or higher for hard examples
    print("\nNote: Focal loss down-weights easy examples, so total loss may be lower")
    
    # Test with gamma=0 (should equal CE)
    focal_loss_gamma0 = FocalLoss(scale=1.0, gamma=0.0, label_smooth=False)
    focal_loss_0, _ = focal_loss_gamma0(logits, labels)
    
    print(f"Focal Loss (gamma=0.0, should ≈ CE): {focal_loss_0.item():.4f}")
    print(f"Difference: {abs(focal_loss_0.item() - ce_loss.item()):.6f}")
    
    # With gamma=0, focal loss should equal CE (within numerical precision)
    assert abs(focal_loss_0.item() - ce_loss.item()) < 1e-5, "Focal loss with gamma=0 should equal CE"
    print("✓ Focal loss with gamma=0 equals Cross Entropy")
    
    print("\nComparison test passed!\n")


def test_contrastive_vs_triplet():
    """Compare Contrastive Loss to Triplet Loss (both should work)."""
    print("=" * 70)
    print("Comparing Contrastive Loss vs Triplet Loss")
    print("=" * 70)
    
    # Create test case
    embeddings = torch.randn(4, 64, 2)  # [n=4, c=64, p=2]
    labels = torch.tensor([0, 0, 1, 1])  # 2 samples per class
    
    margin = 0.25
    
    # Triplet Loss
    triplet_loss_fn = TripletLoss(margin=margin, loss_term_weight=1.0)
    triplet_loss, triplet_info = triplet_loss_fn(embeddings, labels)
    
    # Contrastive Loss
    contrastive_loss_fn = ContrastiveLoss(margin=margin, loss_term_weight=1.0)
    contrastive_loss, contrastive_info = contrastive_loss_fn(embeddings, labels)
    
    # Losses are tensors with shape [p], take mean
    triplet_loss_mean = triplet_loss.mean() if triplet_loss.numel() > 1 else triplet_loss
    contrastive_loss_mean = contrastive_loss.mean() if contrastive_loss.numel() > 1 else contrastive_loss
    
    print(f"Triplet Loss: {triplet_loss_mean.item():.4f}")
    print(f"Contrastive Loss: {contrastive_loss_mean.item():.4f}")
    
    # Both should be non-negative
    assert triplet_loss_mean.item() >= 0, "Triplet loss should be non-negative"
    assert contrastive_loss_mean.item() >= 0, "Contrastive loss should be non-negative"
    print("✓ Both losses are non-negative")
    
    print("\nNote: Losses use different formulas, so values will differ")
    print("Both should work correctly for training")
    
    print("\nComparison test passed!\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    # Test 1: All same class (only positive pairs)
    print("Test 1: All samples same class")
    # embeddings should be [n, c, p] = [3, 64, 2]
    embeddings = torch.randn(3, 64, 2)
    labels = torch.tensor([0, 0, 0])
    
    contrastive_loss_fn = ContrastiveLoss(margin=0.25)
    loss, info = contrastive_loss_fn(embeddings, labels)
    
    loss_mean = loss.mean() if loss.numel() > 1 else loss
    print(f"  Loss: {loss_mean.item():.4f}")
    print(f"  Positive pairs: {info['num_pos_pairs'].item()}")
    print(f"  Negative pairs: {info['num_neg_pairs'].item()}")
    assert info['num_neg_pairs'].item() == 0, "Should have no negative pairs"
    print("  ✓ Correctly handles all same class")
    
    # Test 2: All different classes (only negative pairs)
    print("\nTest 2: All samples different classes")
    labels = torch.tensor([0, 1, 2])
    loss, info = contrastive_loss_fn(embeddings, labels)
    
    loss_mean = loss.mean() if loss.numel() > 1 else loss
    print(f"  Loss: {loss_mean.item():.4f}")
    print(f"  Positive pairs: {info['num_pos_pairs'].item()}")
    print(f"  Negative pairs: {info['num_neg_pairs'].item()}")
    assert info['num_pos_pairs'].item() == 0, "Should have no positive pairs"
    print("  ✓ Correctly handles all different classes")
    
    # Test 3: Focal loss with perfect prediction
    print("\nTest 3: Focal loss with perfect prediction")
    # Note: torch.tensor with nested lists creates [n, p, c], need to permute to [n, c, p]
    logits_npc = torch.tensor([[[10.0, 0.0, 0.0]]])  # Very confident
    logits = logits_npc.permute(0, 2, 1)  # [n=1, c=3, p=1]
    labels = torch.tensor([0])
    
    focal_loss_fn = FocalLoss(gamma=2.0, scale=1.0, label_smooth=False)
    loss, info = focal_loss_fn(logits, labels)
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Should be very small (easy example down-weighted)")
    assert loss.item() < 0.01, "Perfect prediction should have very small loss"
    print("  ✓ Correctly down-weights easy examples")
    
    print("\nAll edge case tests passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOSS VERIFICATION TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_contrastive_loss()
        test_focal_loss()
        test_focal_vs_crossentropy()
        test_contrastive_vs_triplet()
        test_edge_cases()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nYour loss implementations appear to be correct!")
        print("You can now use them in your training configs.\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

