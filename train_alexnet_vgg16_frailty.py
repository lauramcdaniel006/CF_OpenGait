#!/usr/bin/env python3
"""
Train AlexNet and VGG16 for frailty classification using pre-trained ImageNet weights
Uses reshaped 64x64 GaitSet embeddings as input
"""

import os
import sys

# Fix library path issues (similar to batch extraction script)
# Set LD_LIBRARY_PATH to include conda environment lib directory
if 'LD_LIBRARY_PATH' not in os.environ:
    # Try to find conda environment lib directory
    python_executable = sys.executable
    conda_env_lib = os.path.join(os.path.dirname(os.path.dirname(python_executable)), 'lib')
    if os.path.exists(conda_env_lib):
        os.environ['LD_LIBRARY_PATH'] = conda_env_lib
    else:
        # Fallback: try common conda paths
        for path in ['/cis/home/lmcdan11/r38_conda_envs/myGait38/lib',
                     '/cis/home/lmcdan11/r38/miniconda3/envs/myGait38/lib']:
            if os.path.exists(path):
                current_ld = os.environ.get('LD_LIBRARY_PATH', '')
                os.environ['LD_LIBRARY_PATH'] = f"{path}:{current_ld}" if current_ld else path
                break

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
# Try to import roc_auc_score, but handle gracefully if it fails
try:
    from sklearn.metrics import roc_auc_score
    HAS_ROC_AUC = True
except ImportError:
    HAS_ROC_AUC = False
    print("Warning: roc_auc_score not available, ROC AUC metrics will be skipped")
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


class GaitEmbeddingDataset(Dataset):
    """Dataset for reshaped 64x64 GaitSet embeddings"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # Embeddings should be (64, 64) - add channel dimension: (1, 64, 64)
        emb = self.embeddings[idx]
        if len(emb.shape) == 2:
            emb = emb[np.newaxis, :, :]  # Add channel dimension
        elif len(emb.shape) == 3 and emb.shape[0] == 1:
            emb = emb[0]  # Remove batch dimension if present
            emb = emb[np.newaxis, :, :]
        
        return torch.FloatTensor(emb), self.labels[idx]


def load_embeddings_and_labels(embeddings_dir):
    """Load all reshaped 64x64 embeddings and extract labels from directory structure"""
    embeddings_dir = Path(embeddings_dir)
    embeddings = []
    labels = []
    participant_ids = []
    
    label_map = {'Frail': 0, 'Prefrail': 1, 'Nonfrail': 2}
    
    # Scan directory structure: participant_id/frailty_label/embeddings_64x64.npy
    for participant_dir in sorted(embeddings_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        
        participant_id = participant_dir.name
        
        for frailty_dir in sorted(participant_dir.iterdir()):
            if not frailty_dir.is_dir():
                continue
            
            frailty_label = frailty_dir.name
            if frailty_label not in label_map:
                continue
            
            # Try both possible filenames
            embeddings_file = frailty_dir / 'embeddings_64x64.npy'
            if not embeddings_file.exists():
                embeddings_file = frailty_dir / 'embeddings.npy'
                if not embeddings_file.exists():
                    continue
            
            # Load embedding
            emb = np.load(embeddings_file)
            
            # Handle different shapes
            if len(emb.shape) == 3:
                # (1, 64, 64) -> (64, 64)
                emb = emb[0]
            elif len(emb.shape) == 2:
                # (64, 64) - already correct
                pass
            else:
                print(f"Warning: Unexpected shape {emb.shape} for {embeddings_file}, skipping")
                continue
            
            embeddings.append(emb)
            labels.append(label_map[frailty_label])
            participant_ids.append(participant_id)
    
    return np.array(embeddings), np.array(labels), participant_ids


def create_alexnet_model(num_classes=3, pretrained=True, freeze_backbone=False):
    """
    Create AlexNet model with pre-trained ImageNet weights
    Modified for 1-channel input and 3-class output
    
    Args:
        num_classes: Number of output classes (default: 3 for Frail/Prefrail/Nonfrail)
        pretrained: Use pre-trained ImageNet weights
        freeze_backbone: Freeze convolutional layers (only train classifier)
    
    Returns:
        Modified AlexNet model
    """
    # Load pre-trained AlexNet
    if pretrained:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        print("✓ Loaded pre-trained AlexNet with ImageNet weights")
    else:
        model = models.alexnet(weights=None)
        print("✓ Loaded AlexNet with random weights")
    
    # Modify first layer for 1-channel input (grayscale)
    # Original: Conv2d(3, 64, ...) for RGB
    # Modified: Conv2d(1, 64, ...) for grayscale
    original_conv1 = model.features[0]
    model.features[0] = nn.Conv2d(
        1, 64, 
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding
    )
    
    # Initialize first layer weights (average RGB channels or use small random init)
    if pretrained:
        # Average the 3 RGB channels to 1 grayscale channel
        with torch.no_grad():
            model.features[0].weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            model.features[0].bias.data = original_conv1.bias.data
    
    # Modify last layer for 3 classes
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("✓ Frozen convolutional layers (backbone)")
        print("✓ Only classifier layers will be trained")
    else:
        print("✓ All layers will be trained (fine-tuning)")
    
    return model


def create_vgg16_model(num_classes=3, pretrained=True, freeze_backbone=False):
    """
    Create VGG16 model with pre-trained ImageNet weights
    Modified for 1-channel input and 3-class output
    
    Args:
        num_classes: Number of output classes (default: 3 for Frail/Prefrail/Nonfrail)
        pretrained: Use pre-trained ImageNet weights
        freeze_backbone: Freeze convolutional layers (only train classifier)
    
    Returns:
        Modified VGG16 model
    """
    # Load pre-trained VGG16
    if pretrained:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        print("✓ Loaded pre-trained VGG16 with ImageNet weights")
    else:
        model = models.vgg16(weights=None)
        print("✓ Loaded VGG16 with random weights")
    
    # Modify first layer for 1-channel input (grayscale)
    # Original: Conv2d(3, 64, ...) for RGB
    # Modified: Conv2d(1, 64, ...) for grayscale
    original_conv1 = model.features[0]
    model.features[0] = nn.Conv2d(
        1, 64,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding
    )
    
    # Initialize first layer weights (average RGB channels)
    if pretrained:
        # Average the 3 RGB channels to 1 grayscale channel
        with torch.no_grad():
            model.features[0].weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            model.features[0].bias.data = original_conv1.bias.data
    
    # Modify last layer for 3 classes
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("✓ Frozen convolutional layers (backbone)")
        print("✓ Only classifier layers will be trained")
    else:
        print("✓ All layers will be trained (fine-tuning)")
    
    return model


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', lr=0.001):
    """Train the classifier"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc*100:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc*100:.2f}%')
            print()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✓ Loaded best model (Val Acc: {best_val_acc*100:.2f}%)")
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and compute all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm(test_loader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (macro and micro)
    if HAS_ROC_AUC:
        try:
            auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
            auc_micro = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            auc_macro = 0.0
            auc_micro = 0.0
    else:
        auc_macro = 0.0
        auc_micro = 0.0
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = ['Frail', 'Prefrail', 'Nonfrail']
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (macro): {precision_macro*100:.2f}%")
    print(f"Recall (macro): {recall_macro*100:.2f}%")
    print(f"F1 (macro): {f1_macro*100:.2f}%")
    print(f"ROC AUC (macro): {auc_macro:.4f}")
    print(f"ROC AUC (micro): {auc_micro:.4f}")
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name}: Precision={precision_per_class[i]*100:.2f}%, "
              f"Recall={recall_per_class[i]*100:.2f}%, F1={f1_per_class[i]*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*70)
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'auc_macro': float(auc_macro),
        'auc_micro': float(auc_micro),
        'confusion_matrix': cm.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train AlexNet/VGG16 for frailty classification using pre-trained ImageNet weights'
    )
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing reshaped 64x64 embeddings')
    parser.add_argument('--model', type=str, choices=['alexnet', 'vgg16'], required=True,
                        help='Model to train (alexnet or vgg16)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size (of train)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze convolutional layers (only train classifier)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pre-trained weights (train from scratch)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results JSON (default: {model}_results.json)')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("="*70)
    print(f"Training {args.model.upper()} for Frailty Classification")
    print("="*70)
    
    # Load embeddings and labels
    print("\nLoading embeddings and labels...")
    embeddings, labels, participant_ids = load_embeddings_and_labels(args.embeddings_dir)
    print(f"Loaded {len(embeddings)} samples")
    print(f"Class distribution: {np.bincount(labels)} (Frail/Prefrail/Nonfrail)")
    print(f"Embedding shape: {embeddings[0].shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=42, stratify=y_train
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    pretrained = not args.no_pretrained
    
    if args.model == 'alexnet':
        model = create_alexnet_model(
            num_classes=3, 
            pretrained=pretrained,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'vgg16':
        model = create_vgg16_model(
            num_classes=3,
            pretrained=pretrained,
            freeze_backbone=args.freeze_backbone
        )
    
    # Create datasets and loaders
    train_dataset = GaitEmbeddingDataset(X_train, y_train)
    val_dataset = GaitEmbeddingDataset(X_val, y_val)
    test_dataset = GaitEmbeddingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train model
    print(f"\nTraining {args.model.upper()}...")
    model = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        device=args.device,
        lr=args.lr
    )
    
    # Evaluate model
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=args.device)
    
    # Save results
    output_file = args.output or f"{args.model}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Save model if requested
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': args.model,
            'num_classes': 3,
            'results': results
        }, args.save_model)
        print(f"✓ Model saved to {args.save_model}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

