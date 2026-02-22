#!/usr/bin/env python3
"""
Calculate class weights for Frailty dataset
Classes: Frail (0), Prefrail (1), Nonfrail (2)
"""

import pandas as pd
import numpy as np
from collections import Counter

# Path to your frailty label file
frailty_label_file = '/cis/home/lmcdan11/OpenGait_silhall/opengait/frailty_label.csv'

# Load the CSV file
df = pd.read_csv(frailty_label_file)

# Assuming the frailty status is in a column (adjust column name as needed)
# Common column names: 'frailty', 'status', 'label', 'class', etc.
# Check what columns exist
print("Columns in CSV:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Try to find the frailty status column
status_col = None
for col in ['frailty', 'status', 'label', 'class', 'Frailty', 'Status', 'Label', 'Class']:
    if col in df.columns:
        status_col = col
        break

if status_col is None:
    # If not found, use the last column or second column (adjust as needed)
    status_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]
    print(f"\nUsing column: {status_col}")

# Count classes
class_counts = Counter(df[status_col])
print(f"\n{'='*60}")
print("CLASS DISTRIBUTION:")
print(f"{'='*60}")
for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
    count = class_counts.get(class_name, 0)
    print(f"{class_name:12s}: {count:5d} samples")

total_samples = len(df)
num_classes = 3

# Map class names to indices
class_map = {'Frail': 0, 'Prefrail': 1, 'Nonfrail': 2}
counts = [class_counts.get('Frail', 0), class_counts.get('Prefrail', 0), class_counts.get('Nonfrail', 0)]

print(f"\n{'='*60}")
print("CLASS WEIGHT CALCULATIONS:")
print(f"{'='*60}")

# Method 1: Inverse Frequency (Simple)
print("\n1. INVERSE FREQUENCY WEIGHTING:")
print("   Formula: weight_i = total_samples / (num_classes * count_i)")
weights_inv = [total_samples / (num_classes * count) if count > 0 else 0 for count in counts]
print(f"   Frail:    {weights_inv[0]:.4f}")
print(f"   Prefrail: {weights_inv[1]:.4f}")
print(f"   Nonfrail: {weights_inv[2]:.4f}")
print(f"\n   YAML format:")
print(f"   class_weights: [{weights_inv[0]:.4f}, {weights_inv[1]:.4f}, {weights_inv[2]:.4f}]")
print(f"   class_weights: [{weights_inv[0]:.2f}, {weights_inv[1]:.2f}, {weights_inv[2]:.2f}]")

# Method 2: Balanced (sklearn style)
print("\n2. BALANCED WEIGHTING (sklearn style):")
print("   Formula: weight_i = total_samples / (num_classes * count_i), then normalize")
weights_balanced = [total_samples / (num_classes * count) if count > 0 else 0 for count in counts]
mean_weight = np.mean(weights_balanced)
weights_balanced_norm = [w / mean_weight for w in weights_balanced]
print(f"   Frail:    {weights_balanced_norm[0]:.4f}")
print(f"   Prefrail: {weights_balanced_norm[1]:.4f}")
print(f"   Nonfrail: {weights_balanced_norm[2]:.4f}")
print(f"\n   YAML format:")
print(f"   class_weights: [{weights_balanced_norm[0]:.4f}, {weights_balanced_norm[1]:.4f}, {weights_balanced_norm[2]:.4f}]")
print(f"   class_weights: [{weights_balanced_norm[0]:.2f}, {weights_balanced_norm[1]:.2f}, {weights_balanced_norm[2]:.2f}]")

# Method 3: Inverse Square Root
print("\n3. INVERSE SQUARE ROOT WEIGHTING:")
print("   Formula: weight_i = sqrt(total_samples / count_i)")
weights_sqrt = [np.sqrt(total_samples / count) if count > 0 else 0 for count in counts]
# Normalize to have mean = 1
mean_sqrt = np.mean(weights_sqrt)
weights_sqrt_norm = [w / mean_sqrt for w in weights_sqrt]
print(f"   Frail:    {weights_sqrt_norm[0]:.4f}")
print(f"   Prefrail: {weights_sqrt_norm[1]:.4f}")
print(f"   Nonfrail: {weights_sqrt_norm[2]:.4f}")
print(f"\n   YAML format:")
print(f"   class_weights: [{weights_sqrt_norm[0]:.4f}, {weights_sqrt_norm[1]:.4f}, {weights_sqrt_norm[2]:.4f}]")
print(f"   class_weights: [{weights_sqrt_norm[0]:.2f}, {weights_sqrt_norm[1]:.2f}, {weights_sqrt_norm[2]:.2f}]")

# Method 4: Manual (based on your previous config)
print("\n4. MANUAL WEIGHTING (from your swin_baseline.yaml):")
print("   class_weights: [1.0, 1.6, 1.0]  # Prefrail gets 1.6x weight")
print("   (This suggests Prefrail is harder to classify or more important)")

# Recommendations
print(f"\n{'='*60}")
print("RECOMMENDATIONS:")
print(f"{'='*60}")
print("1. If classes are imbalanced, use Method 1 (Inverse Frequency)")
print("2. If Prefrail is harder to classify, try: [1.0, 1.5-2.0, 1.0]")
print("3. Start with balanced weights, then adjust based on per-class performance")
print("4. Check per-class recall in your evaluation - if one class has low recall,")
print("   increase its weight")

print(f"\n{'='*60}")
print("SUGGESTED TESTING ORDER:")
print(f"{'='*60}")
print("1. No class weights (baseline)")
print("2. Inverse frequency weights")
print("3. Manual weights: [1.0, 1.5, 1.0] (if Prefrail is hard)")
print("4. Manual weights: [1.0, 1.6, 1.0] (your previous config)")
print("5. Manual weights: [1.0, 2.0, 1.0] (if Prefrail still struggles)")

