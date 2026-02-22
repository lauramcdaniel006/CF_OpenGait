#!/usr/bin/env python3
"""
Count Frail, Prefrail, and Nonfrail samples in the dataset
Shows total counts and train/test split counts
"""

import pandas as pd
import json
from collections import Counter

# Paths to your files
frailty_label_file = '/cis/home/lmcdan11/OpenGait_silhall/opengait/frailty_label.csv'
dataset_partition_file = '/cis/home/lmcdan11/OpenGait_silhall/opengait/FrailtySet.json'

print("="*70)
print("FRAILTY DATASET CLASS DISTRIBUTION")
print("="*70)

# Load frailty labels
print("\n1. Loading frailty labels...")
# Try reading with different separators
try:
    df_labels = pd.read_csv(frailty_label_file)
except:
    df_labels = pd.read_csv(frailty_label_file, sep=',')

print(f"   Columns: {df_labels.columns.tolist()}")
print(f"   Total rows: {len(df_labels)}")
print(f"\n   First few rows:")
print(df_labels.head())

# Check if columns are combined (like 'subject_id,frailty_score')
if len(df_labels.columns) == 1 and ',' in str(df_labels.columns[0]):
    # Split the combined column
    col_name = df_labels.columns[0]
    df_labels = df_labels[col_name].str.split(',', expand=True)
    df_labels.columns = ['subject_id', 'frailty_score']
    print(f"\n   Fixed columns: {df_labels.columns.tolist()}")
    print(f"\n   First few rows after fix:")
    print(df_labels.head())

# Find the frailty status column
status_col = None
for col in ['frailty_score', 'frailty', 'status', 'label', 'class', 'Frailty', 'Status', 'Label', 'Class', 'frailty_status']:
    if col in df_labels.columns:
        status_col = col
        break

if status_col is None:
    # Use last column if not found
    status_col = df_labels.columns[-1] if len(df_labels.columns) > 1 else df_labels.columns[0]

print(f"\n   Using status column: '{status_col}'")

# Map numeric scores to class names
# Based on your code: 0 = Frail, 1 = Prefrail, 2 = Nonfrail
# But the CSV shows 1, 2 - need to check mapping
unique_scores = sorted(df_labels[status_col].unique())
print(f"   Unique frailty scores: {unique_scores}")

# Create mapping - adjust based on your actual mapping
# Common mappings:
# Option 1: 0=Frail, 1=Prefrail, 2=Nonfrail
# Option 2: 1=Frail, 2=Prefrail, 3=Nonfrail
# Option 3: 1=Frail, 2=Nonfrail (binary)

def map_score_to_class(score):
    """Map numeric score to class name
    Based on code: 0=Frail, 1=Prefrail, 2=Nonfrail
    But CSV might be 1-indexed: 1=Frail, 2=Prefrail, 3=Nonfrail
    Or: 1=Prefrail, 2=Nonfrail (if Frail is 0 or missing)
    """
    score = int(score) if pd.notna(score) else -1
    
    # Try 0-indexed mapping first (matches code)
    if score == 0:
        return 'Frail'
    elif score == 1:
        return 'Prefrail'
    elif score == 2:
        return 'Nonfrail'
    # Try 1-indexed mapping (if CSV uses 1,2,3)
    elif score == 3:
        return 'Nonfrail'
    else:
        return f'Unknown_{score}'

# Apply mapping
df_labels['class_name'] = df_labels[status_col].apply(map_score_to_class)
print(f"\n   Class distribution in CSV:")
print(df_labels['class_name'].value_counts().sort_index())

# Load dataset partition (train/test split)
print("\n2. Loading dataset partition...")
with open(dataset_partition_file, 'r') as f:
    partition = json.load(f)

print(f"   Partition keys: {partition.keys()}")

# Get train and test IDs - handle both lowercase and uppercase keys
train_ids = set(partition.get('train', partition.get('TRAIN_SET', [])))
test_ids = set(partition.get('test', partition.get('TEST_SET', [])))
val_ids = set(partition.get('val', partition.get('VAL_SET', []))) if 'val' in partition or 'VAL_SET' in partition else set()

print(f"   Train samples: {len(train_ids)}")
print(f"   Test samples: {len(test_ids)}")
if val_ids:
    print(f"   Val samples: {len(val_ids)}")

# Find the ID column in labels (to match with partition)
id_col = None
for col in ['subject_id', 'id', 'ID', 'sample_id', 'sample_ID', 'subject_ID', 'index']:
    if col in df_labels.columns:
        id_col = col
        break

if id_col is None:
    # Use first column as ID
    id_col = df_labels.columns[0]

print(f"   Using ID column: '{id_col}'")

# Create a mapping from ID to frailty status (using class_name we created)
id_to_status = {}
for idx, row in df_labels.iterrows():
    sample_id = str(row[id_col]).strip()
    status = row['class_name']  # Use the mapped class name
    id_to_status[sample_id] = status

print(f"\n   Total IDs in label file: {len(id_to_status)}")
print(f"   Sample IDs: {list(id_to_status.keys())[:5]}...")

# Count total distribution
print("\n" + "="*70)
print("TOTAL CLASS DISTRIBUTION (All Data):")
print("="*70)
total_counts = Counter(id_to_status.values())
for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
    count = total_counts.get(class_name, 0)
    pct = (count / len(id_to_status) * 100) if len(id_to_status) > 0 else 0
    print(f"  {class_name:12s}: {count:5d} samples ({pct:5.2f}%)")

total_all = sum(total_counts.values())
print(f"  {'Total':12s}: {total_all:5d} samples")

# Count train distribution
print("\n" + "="*70)
print("TRAIN SET CLASS DISTRIBUTION:")
print("="*70)
# Convert partition IDs to strings and match
train_statuses = []
for sid in train_ids:
    sid_str = str(sid).strip()
    if sid_str in id_to_status:
        train_statuses.append(id_to_status[sid_str])
    # Also try without leading zeros or with different formats
    elif sid_str.lstrip('0') in id_to_status:
        train_statuses.append(id_to_status[sid_str.lstrip('0')])

train_counts = Counter(train_statuses)
for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
    count = train_counts.get(class_name, 0)
    pct = (count / len(train_statuses) * 100) if len(train_statuses) > 0 else 0
    print(f"  {class_name:12s}: {count:5d} samples ({pct:5.2f}%)")

total_train = sum(train_counts.values())
print(f"  {'Total':12s}: {total_train:5d} samples")

# Count test distribution
print("\n" + "="*70)
print("TEST SET CLASS DISTRIBUTION:")
print("="*70)
test_statuses = []
for sid in test_ids:
    sid_str = str(sid).strip()
    if sid_str in id_to_status:
        test_statuses.append(id_to_status[sid_str])
    elif sid_str.lstrip('0') in id_to_status:
        test_statuses.append(id_to_status[sid_str.lstrip('0')])

test_counts = Counter(test_statuses)
for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
    count = test_counts.get(class_name, 0)
    pct = (count / len(test_statuses) * 100) if len(test_statuses) > 0 else 0
    print(f"  {class_name:12s}: {count:5d} samples ({pct:5.2f}%)")

total_test = sum(test_counts.values())
print(f"  {'Total':12s}: {total_test:5d} samples")

# Count val distribution if exists
if val_ids:
    print("\n" + "="*70)
    print("VAL SET CLASS DISTRIBUTION:")
    print("="*70)
    val_statuses = []
    for sid in val_ids:
        sid_str = str(sid).strip()
        if sid_str in id_to_status:
            val_statuses.append(id_to_status[sid_str])
        elif sid_str.lstrip('0') in id_to_status:
            val_statuses.append(id_to_status[sid_str.lstrip('0')])
    
    val_counts = Counter(val_statuses)
    for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
        count = val_counts.get(class_name, 0)
        pct = (count / len(val_statuses) * 100) if len(val_statuses) > 0 else 0
        print(f"  {class_name:12s}: {count:5d} samples ({pct:5.2f}%)")
    
    total_val = sum(val_counts.values())
    print(f"  {'Total':12s}: {total_val:5d} samples")

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE:")
print("="*70)
print(f"{'Class':<12} {'Total':>8} {'Train':>8} {'Test':>8}", end="")
if val_ids:
    print(f" {'Val':>8}")
else:
    print()
print("-" * 70)

for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
    total_c = total_counts.get(class_name, 0)
    train_c = train_counts.get(class_name, 0)
    test_c = test_counts.get(class_name, 0)
    print(f"{class_name:<12} {total_c:>8} {train_c:>8} {test_c:>8}", end="")
    if val_ids:
        val_c = val_counts.get(class_name, 0)
        print(f" {val_c:>8}")
    else:
        print()

# Calculate class weights based on train set
print("\n" + "="*70)
print("DIFFERENT TYPES OF CLASS WEIGHTS (based on TRAIN set):")
print("="*70)

train_total = sum([train_counts.get(c, 0) for c in ['Frail', 'Prefrail', 'Nonfrail']])
num_classes = 3

if train_total > 0:
    frail_count = train_counts.get('Frail', 1)
    prefrail_count = train_counts.get('Prefrail', 1)
    nonfrail_count = train_counts.get('Nonfrail', 1)
    
    import math
    
    # 1. UNIFORM (No weighting - baseline)
    weights_uniform = [1.0, 1.0, 1.0]
    print("\n1. UNIFORM WEIGHTING (No weighting - baseline):")
    print("   Formula: weight_i = 1.0 for all classes")
    print(f"   class_weights: {weights_uniform}")
    print("   Use when: Classes are balanced")
    
    # 2. INVERSE FREQUENCY (Aggressive - high weight for minority)
    weights_inv = [
        train_total / (num_classes * frail_count),
        train_total / (num_classes * prefrail_count),
        train_total / (num_classes * nonfrail_count)
    ]
    print("\n2. INVERSE FREQUENCY WEIGHTING (Aggressive):")
    print("   Formula: weight_i = total_samples / (num_classes × count_i)")
    print(f"   class_weights: [{weights_inv[0]:.4f}, {weights_inv[1]:.4f}, {weights_inv[2]:.4f}]")
    print(f"   class_weights: [{weights_inv[0]:.2f}, {weights_inv[1]:.2f}, {weights_inv[2]:.2f}]")
    print("   Use when: Strong emphasis on minority classes needed")
    
    # 2b. BALANCED (Normalized inverse frequency - sklearn style)
    mean_weight = sum(weights_inv) / num_classes
    weights_balanced = [w / mean_weight for w in weights_inv]
    print("\n2b. BALANCED WEIGHTING (Normalized inverse frequency - sklearn style):")
    print("   Formula: weight_i = [total_samples / (num_classes × count_i)] / mean_weight")
    print(f"   class_weights: [{weights_balanced[0]:.4f}, {weights_balanced[1]:.4f}, {weights_balanced[2]:.4f}]")
    print(f"   class_weights: [{weights_balanced[0]:.2f}, {weights_balanced[1]:.2f}, {weights_balanced[2]:.2f}]")
    print("   Use when: Standard sklearn approach (normalized for interpretability)")
    
    # 3. INVERSE SQUARE ROOT (Moderate - less aggressive)
    weights_sqrt = [
        math.sqrt(train_total / frail_count),
        math.sqrt(train_total / prefrail_count),
        math.sqrt(train_total / nonfrail_count)
    ]
    # Normalize to mean = 1
    mean_sqrt = sum(weights_sqrt) / num_classes
    weights_sqrt_norm = [w / mean_sqrt for w in weights_sqrt]
    print("\n3. INVERSE SQUARE ROOT WEIGHTING (Moderate):")
    print("   Formula: weight_i = sqrt(total_samples / count_i), normalized")
    print(f"   class_weights: [{weights_sqrt_norm[0]:.4f}, {weights_sqrt_norm[1]:.4f}, {weights_sqrt_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_sqrt_norm[0]:.2f}, {weights_sqrt_norm[1]:.2f}, {weights_sqrt_norm[2]:.2f}]")
    print("   Use when: Moderate emphasis on minority classes")
    
    # 4. LOGARITHMIC WEIGHTING (Gentle - smooth weighting)
    weights_log = [
        math.log(train_total / frail_count + 1),
        math.log(train_total / prefrail_count + 1),
        math.log(train_total / nonfrail_count + 1)
    ]
    # Normalize to mean = 1
    mean_log = sum(weights_log) / num_classes
    weights_log_norm = [w / mean_log for w in weights_log]
    print("\n4. LOGARITHMIC WEIGHTING (Gentle):")
    print("   Formula: weight_i = log(total_samples / count_i + 1), normalized")
    print(f"   class_weights: [{weights_log_norm[0]:.4f}, {weights_log_norm[1]:.4f}, {weights_log_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_log_norm[0]:.2f}, {weights_log_norm[1]:.2f}, {weights_log_norm[2]:.2f}]")
    print("   Use when: Gentle emphasis on minority classes")
    
    # 5. EFFECTIVE NUMBER WEIGHTING (Sophisticated - accounts for overlap)
    beta = 0.9999
    def effective_number(count):
        return (1 - beta) / (1 - beta ** count) if count > 0 else 0
    
    en_frail = effective_number(frail_count)
    en_prefrail = effective_number(prefrail_count)
    en_nonfrail = effective_number(nonfrail_count)
    en_total = en_frail + en_prefrail + en_nonfrail
    
    weights_eff = [
        en_total / (num_classes * en_frail) if en_frail > 0 else 1.0,
        en_total / (num_classes * en_prefrail) if en_prefrail > 0 else 1.0,
        en_total / (num_classes * en_nonfrail) if en_nonfrail > 0 else 1.0
    ]
    # Normalize
    mean_eff = sum(weights_eff) / num_classes
    weights_eff_norm = [w / mean_eff for w in weights_eff]
    print("\n5. EFFECTIVE NUMBER WEIGHTING (Sophisticated - accounts for overlap):")
    print("   Formula: effective_number_i = (1 - β) / (1 - β^count_i), where β=0.9999")
    print("            weight_i = effective_total / (num_classes × effective_number_i)")
    print(f"   class_weights: [{weights_eff_norm[0]:.4f}, {weights_eff_norm[1]:.4f}, {weights_eff_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_eff_norm[0]:.2f}, {weights_eff_norm[1]:.2f}, {weights_eff_norm[2]:.2f}]")
    print("   Use when: Classes have significant overlap or complex imbalance")
    
    # 5b. SMOOTHED EFFECTIVE NUMBER WEIGHTING (Adds smoothing to prevent extremes)
    # Smoothing: Add a small constant to prevent division by zero and extreme values
    smoothing_factor = 1.0  # Add 1 to each count for smoothing
    def smoothed_effective_number(count):
        smoothed_count = count + smoothing_factor
        return (1 - beta) / (1 - beta ** smoothed_count) if smoothed_count > 0 else 0
    
    sen_frail = smoothed_effective_number(frail_count)
    sen_prefrail = smoothed_effective_number(prefrail_count)
    sen_nonfrail = smoothed_effective_number(nonfrail_count)
    sen_total = sen_frail + sen_prefrail + sen_nonfrail
    
    weights_smoothed_eff = [
        sen_total / (num_classes * sen_frail) if sen_frail > 0 else 1.0,
        sen_total / (num_classes * sen_prefrail) if sen_prefrail > 0 else 1.0,
        sen_total / (num_classes * sen_nonfrail) if sen_nonfrail > 0 else 1.0
    ]
    # Normalize
    mean_smoothed_eff = sum(weights_smoothed_eff) / num_classes
    weights_smoothed_eff_norm = [w / mean_smoothed_eff for w in weights_smoothed_eff]
    print("\n5b. SMOOTHED EFFECTIVE NUMBER WEIGHTING (Smoothed variant):")
    print("   Formula: effective_number_i = (1 - β) / (1 - β^(count_i + smoothing)), where smoothing=1.0")
    print("            weight_i = effective_total / (num_classes × effective_number_i)")
    print(f"   class_weights: [{weights_smoothed_eff_norm[0]:.4f}, {weights_smoothed_eff_norm[1]:.4f}, {weights_smoothed_eff_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_smoothed_eff_norm[0]:.2f}, {weights_smoothed_eff_norm[1]:.2f}, {weights_smoothed_eff_norm[2]:.2f}]")
    print("   Use when: Want effective number approach but with smoothing to prevent extreme values")
    
    # 6. POWER-BASED WEIGHTING (Customizable aggressiveness)
    power = 0.5  # Less than 1 = less aggressive, > 1 = more aggressive
    weights_power = [
        (train_total / frail_count) ** power,
        (train_total / prefrail_count) ** power,
        (train_total / nonfrail_count) ** power
    ]
    # Normalize
    mean_power = sum(weights_power) / num_classes
    weights_power_norm = [w / mean_power for w in weights_power]
    print("\n6. POWER-BASED WEIGHTING (Customizable, power=0.5):")
    print("   Formula: weight_i = (total_samples / count_i)^power, normalized")
    print(f"   class_weights: [{weights_power_norm[0]:.4f}, {weights_power_norm[1]:.4f}, {weights_power_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_power_norm[0]:.2f}, {weights_power_norm[1]:.2f}, {weights_power_norm[2]:.2f}]")
    print("   Use when: You want to control aggressiveness (power < 1 = gentle, > 1 = aggressive)")
    
    # 7. MANUAL/DOMAIN-SPECIFIC (Your previous config)
    weights_manual = [1.0, 1.6, 1.0]
    print("\n7. MANUAL WEIGHTING (Domain-specific - your previous config):")
    print("   Formula: Based on domain knowledge or observed difficulty")
    print(f"   class_weights: {weights_manual}")
    print("   Use when: You know which class is harder/more important (e.g., Prefrail)")
    
    # 8. INVERSE PROPORTION (Simple ratio-based)
    max_count = max(frail_count, prefrail_count, nonfrail_count)
    weights_prop = [
        max_count / frail_count if frail_count > 0 else 1.0,
        max_count / prefrail_count if prefrail_count > 0 else 1.0,
        max_count / nonfrail_count if nonfrail_count > 0 else 1.0
    ]
    # Normalize
    mean_prop = sum(weights_prop) / num_classes
    weights_prop_norm = [w / mean_prop for w in weights_prop]
    print("\n8. INVERSE PROPORTION WEIGHTING (Ratio-based):")
    print("   Formula: weight_i = max_count / count_i, normalized")
    print(f"   class_weights: [{weights_prop_norm[0]:.4f}, {weights_prop_norm[1]:.4f}, {weights_prop_norm[2]:.4f}]")
    print(f"   class_weights: [{weights_prop_norm[0]:.2f}, {weights_prop_norm[1]:.2f}, {weights_prop_norm[2]:.2f}]")
    print("   Use when: Simple ratio-based weighting needed")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON (All normalized to similar scale):")
    print("="*70)
    print(f"{'Method':<30} {'Frail':>8} {'Prefrail':>10} {'Nonfrail':>10}")
    print("-" * 70)
    print(f"{'1. Uniform':<30} {weights_uniform[0]:>8.2f} {weights_uniform[1]:>10.2f} {weights_uniform[2]:>10.2f}")
    print(f"{'2. Inverse Frequency':<30} {weights_inv[0]:>8.2f} {weights_inv[1]:>10.2f} {weights_inv[2]:>10.2f}")
    print(f"{'2b. Balanced (normalized)':<30} {weights_balanced[0]:>8.2f} {weights_balanced[1]:>10.2f} {weights_balanced[2]:>10.2f}")
    print(f"{'3. Inverse Square Root':<30} {weights_sqrt_norm[0]:>8.2f} {weights_sqrt_norm[1]:>10.2f} {weights_sqrt_norm[2]:>10.2f}")
    print(f"{'4. Logarithmic':<30} {weights_log_norm[0]:>8.2f} {weights_log_norm[1]:>10.2f} {weights_log_norm[2]:>10.2f}")
    print(f"{'5. Effective Number':<30} {weights_eff_norm[0]:>8.2f} {weights_eff_norm[1]:>10.2f} {weights_eff_norm[2]:>10.2f}")
    print(f"{'5b. Smoothed Effective':<30} {weights_smoothed_eff_norm[0]:>8.2f} {weights_smoothed_eff_norm[1]:>10.2f} {weights_smoothed_eff_norm[2]:>10.2f}")
    print(f"{'6. Power-based (0.5)':<30} {weights_power_norm[0]:>8.2f} {weights_power_norm[1]:>10.2f} {weights_power_norm[2]:>10.2f}")
    print(f"{'7. Manual':<30} {weights_manual[0]:>8.2f} {weights_manual[1]:>10.2f} {weights_manual[2]:>10.2f}")
    print(f"{'8. Inverse Proportion':<30} {weights_prop_norm[0]:>8.2f} {weights_prop_norm[1]:>10.2f} {weights_prop_norm[2]:>10.2f}")
    
    print("\n" + "="*70)
    print("RECOMMENDED TESTING ORDER:")
    print("="*70)
    print("1. Uniform [1.0, 1.0, 1.0] - Baseline")
    print("2. Manual [1.0, 1.6, 1.0] - Your previous config")
    print("3. Balanced (normalized) - Standard sklearn approach")
    print("4. Inverse Square Root - Moderate weighting")
    print("5. Logarithmic - Gentle weighting")
    print("6. Inverse Frequency - Aggressive weighting")
    print("7. Effective Number - Sophisticated approach")
    print("8. Smoothed Effective Number - Effective number with smoothing")

print("\n" + "="*70)

