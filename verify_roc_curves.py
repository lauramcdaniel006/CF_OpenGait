#!/usr/bin/env python3
"""
Verify ROC curves are correct by checking:
1. Curves start at (0,0) and end at (1,1)
2. AUC values are between 0.5 and 1.0
3. Curves are above diagonal (for good models)
4. Data is valid
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import glob

def verify_roc_curve(prob_file):
    """Verify a ROC curve file."""
    print("="*70)
    print(f"Verifying: {prob_file}")
    print("="*70)
    
    data = np.load(prob_file)
    probs = data['probs']
    true_ids = data['true_ids']
    class_names = data['class_names']
    
    print(f"\nData Info:")
    print(f"  Total samples: {len(true_ids)}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Class distribution: {np.bincount(true_ids)}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probabilities sum to 1.0: {np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)}")
    
    print(f"\nROC Curve Verification:")
    print("-"*70)
    
    all_valid = True
    
    for i in range(len(class_names)):
        y_binary = (true_ids == i).astype(int)
        y_score = probs[:, i]
        
        # Check if we have both positive and negative samples
        n_positive = np.sum(y_binary == 1)
        n_negative = np.sum(y_binary == 0)
        
        if n_positive == 0:
            print(f"\n{class_names[i]}: ⚠️  WARNING - No positive samples (all predictions are negative)")
            print(f"  Cannot compute ROC curve for this class.")
            all_valid = False
            continue
        
        if n_negative == 0:
            print(f"\n{class_names[i]}: ⚠️  WARNING - No negative samples (all predictions are positive)")
            print(f"  Cannot compute ROC curve for this class.")
            all_valid = False
            continue
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Verify curve properties
        starts_at_origin = (fpr[0] == 0.0 and tpr[0] == 0.0)
        ends_at_one = (fpr[-1] == 1.0 and tpr[-1] == 1.0)
        auc_valid = (0.0 <= roc_auc <= 1.0)
        above_random = (roc_auc > 0.5)
        
        print(f"\n{class_names[i]}:")
        print(f"  ✓ Starts at (0,0): {starts_at_origin}")
        print(f"  ✓ Ends at (1,1): {ends_at_one}")
        print(f"  ✓ AUC is valid (0-1): {auc_valid} (AUC = {roc_auc:.4f})")
        print(f"  ✓ Above random (>0.5): {above_random}")
        print(f"  ✓ Number of points: {len(fpr)}")
        print(f"  ✓ FPR range: [{fpr.min():.4f}, {fpr.max():.4f}]")
        print(f"  ✓ TPR range: [{tpr.min():.4f}, {tpr.max():.4f}]")
        
        if not (starts_at_origin and ends_at_one and auc_valid):
            print(f"  ❌ ERROR: ROC curve is invalid!")
            all_valid = False
        elif not above_random:
            print(f"  ⚠️  WARNING: Model performs worse than random for this class")
        
        # Check if curve is monotonic (should be non-decreasing)
        tpr_diff = np.diff(tpr)
        is_monotonic = np.all(tpr_diff >= -1e-10)  # Allow small numerical errors
        print(f"  ✓ TPR is monotonic (non-decreasing): {is_monotonic}")
        
        if not is_monotonic:
            print(f"  ⚠️  WARNING: TPR is not strictly monotonic (unusual but possible)")
    
    print("\n" + "="*70)
    if all_valid:
        print("✓ All ROC curves are valid!")
    else:
        print("⚠️  Some issues found - see warnings above")
    print("="*70)
    
    return all_valid

if __name__ == '__main__':
    # Find all probability files
    prob_dir = 'output/roc_probabilities'
    
    if not os.path.exists(prob_dir):
        print(f"Error: Directory {prob_dir} does not exist")
        exit(1)
    
    prob_files = glob.glob(os.path.join(prob_dir, '*.npz'))
    
    if not prob_files:
        print(f"No probability files found in {prob_dir}")
        exit(1)
    
    print(f"Found {len(prob_files)} probability file(s)\n")
    
    for prob_file in sorted(prob_files):
        verify_roc_curve(prob_file)
        print()

