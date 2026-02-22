#!/usr/bin/env python3
"""
Simple classifier for GaitSet embeddings (Liu et al. approach - simplified version)
Uses MLP instead of CAE + AlexNet/VGG16 for simplicity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
import json

class GaitEmbeddingDataset(Dataset):
    """Dataset for GaitSet embeddings"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.embeddings[idx]), self.labels[idx]


class SimpleMLPClassifier(nn.Module):
    """
    Simple MLP classifier for GaitSet embeddings
    Alternative to CAE + AlexNet/VGG16 approach
    """
    def __init__(self, input_dim=256*62, num_classes=3, hidden_dims=[512, 256, 128]):
        super(SimpleMLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten embeddings: (batch, 256, 62) -> (batch, 256*62)
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)
        return self.classifier(x)


class AlexNetClassifier(nn.Module):
    """
    Simplified AlexNet for 64x64 input
    Adapted from standard AlexNet architecture
    """
    def __init__(self, num_classes=3, input_channels=1):
        super(AlexNetClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: 64x64 -> 16x16
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 16x16 -> 8x8
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3-5: 8x8 -> 4x4
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),  # Adjusted for 64x64 input
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16Classifier(nn.Module):
    """
    Simplified VGG16 for 64x64 input
    Adapted from standard VGG16 architecture
    """
    def __init__(self, num_classes=3, input_channels=1):
        super(VGG16Classifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_embeddings_and_labels(embeddings_dir):
    """Load all embeddings and extract labels from directory structure"""
    embeddings_dir = Path(embeddings_dir)
    embeddings = []
    labels = []
    participant_ids = []
    
    label_map = {'Frail': 0, 'Prefrail': 1, 'Nonfrail': 2}
    
    # Scan directory structure: participant_id/frailty_label/embeddings.npy
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
            
            embeddings_file = frailty_dir / 'embeddings.npy'
            if not embeddings_file.exists():
                continue
            
            # Load embedding
            emb = np.load(embeddings_file)
            # Flatten: (1, 256, 62) -> (256*62,)
            emb_flat = emb.flatten()
            
            embeddings.append(emb_flat)
            labels.append(label_map[frailty_label])
            participant_ids.append(participant_id)
    
    return np.array(embeddings), np.array(labels), participant_ids


def reshape_for_cnn(embeddings, target_size=(64, 64)):
    """
    Reshape embeddings to 2D format for CNN input
    Simple approach: interpolate or pad to target size
    """
    # embeddings shape: (N, 256*62) = (N, 15872)
    # Reshape to (N, 256, 62) first
    N = embeddings.shape[0]
    emb_2d = embeddings.reshape(N, 256, 62)
    
    # Convert to torch tensor for interpolation
    emb_tensor = torch.FloatTensor(emb_2d).unsqueeze(1)  # (N, 1, 256, 62)
    
    # Interpolate to target size
    emb_resized = torch.nn.functional.interpolate(
        emb_tensor, size=target_size, mode='bilinear', align_corners=False
    )
    
    return emb_resized.squeeze(1).numpy()  # (N, 64, 64)


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train the classifier"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and compute all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
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
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
        auc_micro = roc_auc_score(all_labels, all_probs, average='micro', multi_class='ovr')
    except:
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
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Classify GaitSet embeddings for frailty')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing embeddings (participant_id/frailty_label/embeddings.npy)')
    parser.add_argument('--model', type=str, choices=['mlp', 'alexnet', 'vgg16'], default='mlp',
                        help='Classifier model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size (of train)')
    parser.add_argument('--output', type=str, default='classifier_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("Loading embeddings and labels...")
    embeddings, labels, participant_ids = load_embeddings_and_labels(args.embeddings_dir)
    print(f"Loaded {len(embeddings)} samples")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    if args.model == 'mlp':
        model = SimpleMLPClassifier(input_dim=embeddings.shape[1], num_classes=3)
        train_dataset = GaitEmbeddingDataset(X_train, y_train)
        val_dataset = GaitEmbeddingDataset(X_val, y_val)
        test_dataset = GaitEmbeddingDataset(X_test, y_test)
    elif args.model == 'alexnet':
        # Reshape to 2D for CNN
        X_train_2d = reshape_for_cnn(X_train)
        X_val_2d = reshape_for_cnn(X_val)
        X_test_2d = reshape_for_cnn(X_test)
        
        model = AlexNetClassifier(num_classes=3, input_channels=1)
        train_dataset = GaitEmbeddingDataset(X_train_2d, y_train)
        val_dataset = GaitEmbeddingDataset(X_val_2d, y_val)
        test_dataset = GaitEmbeddingDataset(X_test_2d, y_test)
    elif args.model == 'vgg16':
        # Reshape to 2D for CNN
        X_train_2d = reshape_for_cnn(X_train)
        X_val_2d = reshape_for_cnn(X_val)
        X_test_2d = reshape_for_cnn(X_test)
        
        model = VGG16Classifier(num_classes=3, input_channels=1)
        train_dataset = GaitEmbeddingDataset(X_train_2d, y_train)
        val_dataset = GaitEmbeddingDataset(X_val_2d, y_val)
        test_dataset = GaitEmbeddingDataset(X_test_2d, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nTraining {args.model} model...")
    model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=args.device)
    
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device=args.device)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

