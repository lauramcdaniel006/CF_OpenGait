#!/usr/bin/env python3
"""
Print evaluation accuracies at every 500 iteration checkpoint for both runs.
"""

import os
import re
import glob

# Paths for the two runs
RUN1_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn"
RUN2_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest"

RUN1_NAME = "swin_part4a_B3_unfrozen_cnn"
RUN2_NAME = "swin_testb3"

def find_latest_log(exp_dir):
    """Find the most recent log file."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)

def extract_evaluations(log_file):
    """Extract all evaluation results."""
    results = {}
    if not log_file or not os.path.exists(log_file):
        return results
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find all evaluation sections
        eval_sections = re.split(r'EVALUATION RESULTS', content)
        
        for section in eval_sections[1:]:
            # Find iteration number from context before this section
            prev_context = content[:content.find(section)]
            iter_match = re.search(r'Iteration\s+(\d+)', prev_context[-500:])
            if not iter_match:
                continue
            
            iter_num = int(iter_match.group(1))
            
            # Extract accuracy
            acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', section)
            if acc_match:
                accuracy = float(acc_match.group(1))
                results[iter_num] = accuracy
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return results

def main():
    print("=" * 100)
    print("EVALUATION ACCURACIES AT EVERY 500 ITERATION CHECKPOINT")
    print("=" * 100)
    print()
    
    # Get logs
    log1 = find_latest_log(RUN1_PATH)
    log2 = find_latest_log(RUN2_PATH)
    
    if not log1:
        print(f"❌ No log file found for {RUN1_NAME}")
        return
    
    if not log2:
        print(f"❌ No log file found for {RUN2_NAME}")
        return
    
    print(f"✓ Run 1 log: {os.path.basename(log1)}")
    print(f"✓ Run 2 log: {os.path.basename(log2)}")
    print()
    
    # Extract evaluations
    evals1 = extract_evaluations(log1)
    evals2 = extract_evaluations(log2)
    
    # Get all evaluation iterations
    all_iters = sorted(set(list(evals1.keys()) + list(evals2.keys())))
    
    # Filter to multiples of 500 (or closest evaluation)
    # We'll show evaluations that are at or near multiples of 500
    checkpoints = []
    for iter_num in all_iters:
        # Check if it's close to a multiple of 500 (within 100 iterations)
        nearest_500 = round(iter_num / 500) * 500
        if abs(iter_num - nearest_500) <= 100:
            checkpoints.append(iter_num)
    
    # Remove duplicates and sort
    checkpoints = sorted(set(checkpoints))
    
    if not checkpoints:
        print("⚠ No evaluations found near 500-iteration checkpoints")
        print(f"\nAll evaluation iterations found:")
        print(f"  Run 1: {sorted(evals1.keys())}")
        print(f"  Run 2: {sorted(evals2.keys())}")
        return
    
    # Print header
    print(f"{'Iteration':<12} {'Run 1 Accuracy':<20} {'Run 2 Accuracy':<20} {'Match':<10}")
    print("-" * 100)
    
    # Print results
    for iter_num in checkpoints:
        acc1 = evals1.get(iter_num, None)
        acc2 = evals2.get(iter_num, None)
        
        acc1_str = f"{acc1:.2f}%" if acc1 is not None else "N/A"
        acc2_str = f"{acc2:.2f}%" if acc2 is not None else "N/A"
        
        # Check if they match
        if acc1 is not None and acc2 is not None:
            match = abs(acc1 - acc2) < 0.01
            match_str = "✅" if match else "❌"
        else:
            match_str = "N/A"
        
        print(f"{iter_num:<12} {acc1_str:<20} {acc2_str:<20} {match_str:<10}")
    
    print()
    print("=" * 100)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Total checkpoints shown: {len(checkpoints)}")
    if evals1:
        best1_iter = max(evals1.keys(), key=lambda k: evals1[k])
        print(f"  Run 1 best: {evals1[best1_iter]:.2f}% at iteration {best1_iter}")
    if evals2:
        best2_iter = max(evals2.keys(), key=lambda k: evals2[k])
        print(f"  Run 2 best: {evals2[best2_iter]:.2f}% at iteration {best2_iter}")
    print("=" * 100)

if __name__ == '__main__':
    main()
