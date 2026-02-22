#!/usr/bin/env python3
"""
Compare accuracies from first 1500 iterations of:
- swin_part1_pretrained_unfrozen
- swin_part4a_B3_unfrozen_cnn
"""

import re
import os

def extract_accuracies(log_file, max_iter=1500):
    """Extract softmax_accuracy for iterations <= max_iter."""
    accuracies = {}
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return accuracies
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Match: Iteration 00100, Cost 59.85s, triplet_loss=0.3810, softmax_loss=0.1234, softmax_accuracy=0.5678
                match = re.search(r'Iteration\s+(\d+).*?softmax_accuracy=([\d.]+)', line)
                if match:
                    iter_num = int(match.group(1))
                    if iter_num <= max_iter:
                        acc = float(match.group(2))
                        accuracies[iter_num] = acc
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return accuracies

# Paths to log files
log1 = "output/REDO_Frailty_ccpg_pt1_pretrained(UF)/SwinGait/REDO_Frailty_ccpg_pt1_pretrained(UF)/logs/2026-01-29-02-32-31.txt"
log2 = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/logs/2026-01-28-19-03-19.txt"

print("=" * 100)
print("Comparing accuracies for first 1500 iterations")
print("=" * 100)
print("\nConfig 1: swin_part1_pretrained_unfrozen")
print("Config 2: swin_part4a_B3_unfrozen_cnn")
print()

# Extract accuracies
acc1 = extract_accuracies(log1, max_iter=1500)
acc2 = extract_accuracies(log2, max_iter=1500)

print(f"Found {len(acc1)} iterations in config 1")
print(f"Found {len(acc2)} iterations in config 2")
print()

# Find common iterations
common_iters = set(acc1.keys()) & set(acc2.keys())
print(f"Common iterations: {len(common_iters)}")
print()

if len(common_iters) == 0:
    print("No common iterations found!")
else:
    # Compare accuracies
    matches = 0
    mismatches = 0
    max_diff = 0
    max_diff_iter = None
    
    for iter_num in sorted(common_iters):
        a1 = acc1[iter_num]
        a2 = acc2[iter_num]
        diff = abs(a1 - a2)
        
        if diff < 1e-6:  # Consider identical if difference < 1e-6
            matches += 1
        else:
            mismatches += 1
            if diff > max_diff:
                max_diff = diff
                max_diff_iter = iter_num
    
    print(f"✓ Matching accuracies: {matches}")
    print(f"✗ Mismatched accuracies: {mismatches}")
    print()
    
    if mismatches > 0:
        print(f"Maximum difference: {max_diff:.6f} at iteration {max_diff_iter}")
        print(f"  Config 1: {acc1[max_diff_iter]:.6f}")
        print(f"  Config 2: {acc2[max_diff_iter]:.6f}")
        print()
    else:
        print("✅ All accuracies match perfectly!")
    
    # Show all comparisons up to 1500
    print("\n" + "=" * 100)
    print("All iterations up to 1500:")
    print("=" * 100)
    print(f"{'Iter':<8} {'Pretrained UF Acc':<20} {'B3 Unfrozen Acc':<20} {'Match':<8} {'Diff':<12}")
    print("-" * 100)
    for iter_num in sorted(common_iters):
        a1 = acc1[iter_num]
        a2 = acc2[iter_num]
        diff = abs(a1 - a2)
        match = "✓" if diff < 1e-6 else "✗"
        print(f"{iter_num:<8} {a1:<20.6f} {a2:<20.6f} {match:<8} {diff:<12.6f}")
