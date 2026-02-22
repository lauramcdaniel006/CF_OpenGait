#!/usr/bin/env python3
"""
Find iterations where Run 1 and Run 2 have identical results.
"""

import os
import sys
import re
from view_two_runs_results import RUN1_PATH, RUN2_PATH, extract_from_log_file, get_latest_log_file

def find_identical_results():
    """Find iterations where both runs have identical metrics."""
    
    # Extract results
    log_file1 = get_latest_log_file(RUN1_PATH)
    log_file2 = get_latest_log_file(RUN2_PATH)
    
    if not log_file1 or not log_file2:
        print("Error: Could not find log files")
        return
    
    run1_results = extract_from_log_file(log_file1)
    run2_results = extract_from_log_file(log_file2)
    
    # Create lookup dictionaries
    run1_dict = {r['iteration']: r for r in run1_results}
    run2_dict = {r['iteration']: r for r in run2_results}
    
    # Find common iterations
    common_iters = sorted(set(run1_dict.keys()) & set(run2_dict.keys()))
    
    print("="*120)
    print("SEARCHING FOR IDENTICAL RESULTS BETWEEN RUN 1 AND RUN 2")
    print("="*120)
    print(f"\nTotal common iterations: {len(common_iters)}")
    print(f"Iterations: {common_iters}\n")
    
    identical_iters = []
    nearly_identical_iters = []
    
    for iter_num in common_iters:
        r1 = run1_dict[iter_num]
        r2 = run2_dict[iter_num]
        
        # Check if all metrics are identical
        metrics_match = True
        differences = []
        
        # Check accuracy (with small tolerance for floating point)
        if abs(r1.get('accuracy', 0) - r2.get('accuracy', 0)) > 0.0001:
            metrics_match = False
            differences.append(f"Accuracy: {r1.get('accuracy', 0):.4f} vs {r2.get('accuracy', 0):.4f}")
        
        # Check F1
        if 'macro_f1' in r1 and 'macro_f1' in r2:
            if abs(r1['macro_f1'] - r2['macro_f1']) > 0.0001:
                metrics_match = False
                differences.append(f"F1: {r1['macro_f1']:.4f} vs {r2['macro_f1']:.4f}")
        
        # Check precision
        if 'macro_precision' in r1 and 'macro_precision' in r2:
            if abs(r1['macro_precision'] - r2['macro_precision']) > 0.0001:
                metrics_match = False
                differences.append(f"Precision: {r1['macro_precision']:.4f} vs {r2['macro_precision']:.4f}")
        
        # Check recall
        if 'macro_recall' in r1 and 'macro_recall' in r2:
            if abs(r1['macro_recall'] - r2['macro_recall']) > 0.0001:
                metrics_match = False
                differences.append(f"Recall: {r1['macro_recall']:.4f} vs {r2['macro_recall']:.4f}")
        
        # Check AUC
        if r1.get('auc_macro') and r2.get('auc_macro'):
            if abs(r1['auc_macro'] - r2['auc_macro']) > 0.0001:
                metrics_match = False
                differences.append(f"AUC: {r1['auc_macro']:.4f} vs {r2['auc_macro']:.4f}")
        
        # Check confusion matrix
        cm_match = True
        if 'confusion_matrix' in r1 and 'confusion_matrix' in r2:
            if r1['confusion_matrix'] != r2['confusion_matrix']:
                metrics_match = False
                cm_match = False
                differences.append("Confusion Matrix: Different")
        
        if metrics_match:
            identical_iters.append(iter_num)
        elif len(differences) <= 1:  # Nearly identical (only one small difference)
            nearly_identical_iters.append((iter_num, differences))
    
    # Print results
    print("="*120)
    print("IDENTICAL RESULTS (All metrics match exactly)")
    print("="*120)
    
    if identical_iters:
        print(f"\n✅ Found {len(identical_iters)} iteration(s) with IDENTICAL results:\n")
        for iter_num in identical_iters:
            r1 = run1_dict[iter_num]
            print(f"Iteration {iter_num}:")
            print(f"  Accuracy: {r1.get('accuracy', 0)*100:.2f}%")
            if 'macro_f1' in r1:
                print(f"  F1: {r1['macro_f1']:.4f}")
                print(f"  Precision: {r1['macro_precision']:.4f}")
                print(f"  Recall: {r1['macro_recall']:.4f}")
            if r1.get('auc_macro'):
                print(f"  AUC: {r1['auc_macro']:.4f}")
            if 'confusion_matrix' in r1:
                print(f"  Confusion Matrix: {r1['confusion_matrix']}")
            print()
    else:
        print("\n❌ No iterations with IDENTICAL results found.\n")
    
    print("="*120)
    print("NEARLY IDENTICAL RESULTS (Only minor differences)")
    print("="*120)
    
    if nearly_identical_iters:
        print(f"\n⚠️  Found {len(nearly_identical_iters)} iteration(s) with NEARLY identical results:\n")
        for iter_num, differences in nearly_identical_iters:
            r1 = run1_dict[iter_num]
            r2 = run2_dict[iter_num]
            print(f"Iteration {iter_num}:")
            print(f"  Run 1 Accuracy: {r1.get('accuracy', 0)*100:.2f}%")
            print(f"  Run 2 Accuracy: {r2.get('accuracy', 0)*100:.2f}%")
            print(f"  Differences: {', '.join(differences)}")
            print()
    else:
        print("\n⚠️  No iterations with nearly identical results found.\n")
    
    # Summary statistics
    print("="*120)
    print("SUMMARY")
    print("="*120)
    
    if identical_iters:
        print(f"\n✅ {len(identical_iters)}/{len(common_iters)} iterations have IDENTICAL results")
        print(f"   Iterations: {identical_iters}")
    else:
        print(f"\n❌ 0/{len(common_iters)} iterations have IDENTICAL results")
    
    # Calculate accuracy differences
    acc_diffs = []
    for iter_num in common_iters:
        r1 = run1_dict[iter_num]
        r2 = run2_dict[iter_num]
        diff = abs(r1.get('accuracy', 0) - r2.get('accuracy', 0)) * 100
        acc_diffs.append((iter_num, diff))
    
    acc_diffs.sort(key=lambda x: x[1])  # Sort by difference
    
    print(f"\nSmallest accuracy differences:")
    for iter_num, diff in acc_diffs[:5]:
        r1 = run1_dict[iter_num]
        r2 = run2_dict[iter_num]
        print(f"  Iter {iter_num}: {diff:.2f}% difference ({r1.get('accuracy', 0)*100:.2f}% vs {r2.get('accuracy', 0)*100:.2f}%)")
    
    print("="*120)

if __name__ == '__main__':
    find_identical_results()
