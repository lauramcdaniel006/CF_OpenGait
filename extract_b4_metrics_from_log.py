#!/usr/bin/env python3
"""
Extract all metrics for B4 from the log file.
The log has per-class metrics, we need to calculate macro-averaged metrics.
"""

import re
import numpy as np

log_file = 'output/REDO_Frailty_ccpg_pt4a_deepgaitv2_B4_unfrozen_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4a_deepgaitv2_B4_unfrozen_with_weights/logs/2026-01-14-10-52-43.txt'

with open(log_file, 'r') as f:
    content = f.read()

# Split by evaluation sections
eval_sections = re.split(r'EVALUATION RESULTS', content)

results = []

for i, section in enumerate(eval_sections[1:], 1):  # Skip first empty split
    # Find iteration number before this section
    before_idx = content.find(section) - 1000  # Look back 1000 chars
    before_text = content[max(0, before_idx):content.find(section)]
    
    iter_match = re.search(r'Iteration\s+0*(\d+)', before_text)
    if not iter_match:
        continue
    
    iter_num = int(iter_match.group(1))
    
    # Extract overall accuracy
    acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
    if not acc_match:
        continue
    
    accuracy = float(acc_match.group(1))
    
    # Extract per-class metrics
    frail_recall = re.search(r'Frail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
    frail_prec = re.search(r'Frail Precision:\s*(\d+\.?\d*)%', section)
    
    prefrail_recall = re.search(r'Prefrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
    prefrail_prec = re.search(r'Prefrail Precision:\s*(\d+\.?\d*)%', section)
    
    nonfrail_recall = re.search(r'Nonfrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
    nonfrail_prec = re.search(r'Nonfrail Precision:\s*(\d+\.?\d*)%', section)
    
    if not all([frail_recall, frail_prec, prefrail_recall, prefrail_prec, nonfrail_recall, nonfrail_prec]):
        continue
    
    # Convert to decimals
    frail_recall_val = float(frail_recall.group(1)) / 100.0
    frail_prec_val = float(frail_prec.group(1)) / 100.0
    frail_f1 = 2 * (frail_prec_val * frail_recall_val) / (frail_prec_val + frail_recall_val) if (frail_prec_val + frail_recall_val) > 0 else 0.0
    
    prefrail_recall_val = float(prefrail_recall.group(1)) / 100.0
    prefrail_prec_val = float(prefrail_prec.group(1)) / 100.0
    prefrail_f1 = 2 * (prefrail_prec_val * prefrail_recall_val) / (prefrail_prec_val + prefrail_recall_val) if (prefrail_prec_val + prefrail_recall_val) > 0 else 0.0
    
    nonfrail_recall_val = float(nonfrail_recall.group(1)) / 100.0
    nonfrail_prec_val = float(nonfrail_prec.group(1)) / 100.0
    nonfrail_f1 = 2 * (nonfrail_prec_val * nonfrail_recall_val) / (nonfrail_prec_val + nonfrail_recall_val) if (nonfrail_prec_val + nonfrail_recall_val) > 0 else 0.0
    
    # Calculate macro-averaged metrics
    macro_precision = (frail_prec_val + prefrail_prec_val + nonfrail_prec_val) / 3.0
    macro_recall = (frail_recall_val + prefrail_recall_val + nonfrail_recall_val) / 3.0
    macro_f1 = (frail_f1 + prefrail_f1 + nonfrail_f1) / 3.0
    
    results.append({
        'iteration': iter_num,
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    })

# Sort by iteration
results.sort(key=lambda x: x['iteration'])

print("=" * 80)
print("B4: UNFROZEN WITH WEIGHTS - ALL METRICS")
print("=" * 80)
print(f"{'Iteration':<12} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 80)

for r in results:
    print(f"{r['iteration']:<12} {r['accuracy']:>10.2f}% {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")

if results:
    # Find best accuracy (if tie, pick best F1)
    max_acc = max(r['accuracy'] for r in results)
    best = max([r for r in results if r['accuracy'] == max_acc], key=lambda x: x['f1'])
    
    accuracies = [r['accuracy'] for r in results]
    mean_acc = sum(accuracies) / len(accuracies)
    min_acc = min(accuracies)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Best Overall Accuracy: {best['accuracy']:.2f}% (iter {best['iteration']})")
    print(f"  Corresponding F1: {best['f1']:.4f}")
    print(f"  Corresponding Precision: {best['precision']:.4f}")
    print(f"  Corresponding Recall: {best['recall']:.4f}")
    print(f"\nMean Accuracy: {mean_acc:.2f}%")
    print(f"Lowest Accuracy: {min_acc:.2f}%")
    print(f"Total Evaluations: {len(results)}")
