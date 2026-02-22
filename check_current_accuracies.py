#!/usr/bin/env python3
"""
Check current accuracies for:
- swin_part1_pretrained_unfrozen
- swin_part4a_B3_unfrozen_cnn
"""

import re
import os
import glob

def find_latest_log(exp_dir):
    """Find the most recent log file."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)

def extract_all_accuracies(log_file):
    """Extract all softmax_accuracy values."""
    accuracies = {}
    
    if not log_file or not os.path.exists(log_file):
        return accuracies
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r'Iteration\s+(\d+).*?softmax_accuracy=([\d.]+)', line)
                if match:
                    iter_num = int(match.group(1))
                    acc = float(match.group(2))
                    accuracies[iter_num] = acc
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return accuracies

# Find latest log files
log1_dir = "output/REDO_Frailty_ccpg_pt1_pretrained(UF)/SwinGait/REDO_Frailty_ccpg_pt1_pretrained(UF)"
log2_dir = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn"

log1 = find_latest_log(log1_dir)
log2 = find_latest_log(log2_dir)

print("=" * 100)
print("Current Accuracies Comparison")
print("=" * 100)
print(f"\nConfig 1: swin_part1_pretrained_unfrozen")
print(f"  Latest log: {os.path.basename(log1) if log1 else 'NOT FOUND'}")
print(f"\nConfig 2: swin_part4a_B3_unfrozen_cnn")
print(f"  Latest log: {os.path.basename(log2) if log2 else 'NOT FOUND'}")
print()

# Extract accuracies
acc1 = extract_all_accuracies(log1) if log1 else {}
acc2 = extract_all_accuracies(log2) if log2 else {}

print(f"Found {len(acc1)} iterations in config 1")
print(f"Found {len(acc2)} iterations in config 2")
print()

# Find common iterations
common_iters = set(acc1.keys()) & set(acc2.keys())
print(f"Common iterations: {len(common_iters)}")
print()

if len(common_iters) == 0:
    print("No common iterations found!")
    print(f"\nConfig 1 iterations: {sorted(acc1.keys())[:20]}")
    print(f"Config 2 iterations: {sorted(acc2.keys())[:20]}")
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
        
        if diff < 1e-6:
            matches += 1
        else:
            mismatches += 1
            if diff > max_diff:
                max_diff = diff
                max_diff_iter = iter_num
    
    print(f"✓ Matching accuracies: {matches}")
    print(f"✗ Mismatched accuracies: {mismatches}")
    if mismatches > 0:
        print(f"\nMaximum difference: {max_diff:.6f} at iteration {max_diff_iter}")
        print(f"  Config 1: {acc1[max_diff_iter]:.6f}")
        print(f"  Config 2: {acc2[max_diff_iter]:.6f}")
    else:
        print("\n✅ All accuracies match perfectly!")
    
    # Show all common iterations
    print("\n" + "=" * 100)
    print("All Common Iterations:")
    print("=" * 100)
    print(f"{'Iter':<8} {'Pretrained UF Acc':<20} {'B3 Unfrozen Acc':<20} {'Match':<8} {'Diff':<12}")
    print("-" * 100)
    for iter_num in sorted(common_iters):
        a1 = acc1[iter_num]
        a2 = acc2[iter_num]
        diff = abs(a1 - a2)
        match = "✓" if diff < 1e-6 else "✗"
        print(f"{iter_num:<8} {a1:<20.6f} {a2:<20.6f} {match:<8} {diff:<12.6f}")

# Show unique iterations for each config
print("\n" + "=" * 100)
print("Unique Iterations (only in one config):")
print("=" * 100)
only1 = set(acc1.keys()) - set(acc2.keys())
only2 = set(acc2.keys()) - set(acc1.keys())

if only1:
    print(f"\nOnly in Config 1 ({len(only1)} iterations):")
    for iter_num in sorted(only1)[:20]:
        print(f"  Iter {iter_num}: {acc1[iter_num]:.6f}")
    if len(only1) > 20:
        print(f"  ... and {len(only1) - 20} more")

if only2:
    print(f"\nOnly in Config 2 ({len(only2)} iterations):")
    for iter_num in sorted(only2)[:20]:
        print(f"  Iter {iter_num}: {acc2[iter_num]:.6f}")
    if len(only2) > 20:
        print(f"  ... and {len(only2) - 20} more")

if not only1 and not only2:
    print("\nNo unique iterations - both configs have the same iteration numbers logged.")
