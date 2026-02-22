#!/usr/bin/env python3
"""
Investigate why results aren't matching between runs.
Checks training losses, batch counter behavior, and other potential issues.
"""

import os
import re
import glob

# Paths for the two runs
RUN1_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn"
RUN2_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest"

def find_latest_log(exp_dir):
    """Find the most recent log file."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)

def extract_training_losses(log_file):
    """Extract all training losses."""
    losses = []
    if not log_file or not os.path.exists(log_file):
        return losses
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Match: Iteration 00100, Cost 59.85s, triplet_loss=0.3810, ...
                match = re.search(r'Iteration\s+(\d+).*?triplet_loss=([\d.]+).*?softmax_loss=([\d.]+).*?softmax_accuracy=([\d.]+)', line)
                if match:
                    iter_num = int(match.group(1))
                    triplet_loss = float(match.group(2))
                    softmax_loss = float(match.group(3))
                    softmax_acc = float(match.group(4))
                    losses.append({
                        'iteration': iter_num,
                        'triplet_loss': triplet_loss,
                        'softmax_loss': softmax_loss,
                        'softmax_accuracy': softmax_acc
                    })
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return losses

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
            # Find iteration number
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
        print(f"Error: {e}")
    
    return results

def check_seed_initialization(log_file):
    """Check if seed initialization messages are present."""
    if not log_file or not os.path.exists(log_file):
        return []
    
    seed_info = []
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for seed-related messages
            if 'seed' in content.lower():
                # Find lines with seed info
                for line in content.split('\n'):
                    if 'seed' in line.lower() or 'random' in line.lower():
                        seed_info.append(line.strip())
    except:
        pass
    
    return seed_info[:10]  # Return first 10 matches

def main():
    print("=" * 100)
    print("REPRODUCIBILITY INVESTIGATION")
    print("=" * 100)
    print()
    
    # Get logs
    log1 = find_latest_log(RUN1_PATH)
    log2 = find_latest_log(RUN2_PATH)
    
    if not log1 or not log2:
        print("❌ Could not find log files")
        return
    
    print(f"✓ Run 1 log: {os.path.basename(log1)}")
    print(f"✓ Run 2 log: {os.path.basename(log2)}")
    print()
    
    # 1. CHECK TRAINING LOSSES
    print("=" * 100)
    print("1. TRAINING LOSSES COMPARISON (First 20 iterations)")
    print("=" * 100)
    print()
    
    losses1 = extract_training_losses(log1)
    losses2 = extract_training_losses(log2)
    
    if not losses1 or not losses2:
        print("⚠ No training losses found")
    else:
        # Compare first 20 iterations
        common_iters = sorted(set([l['iteration'] for l in losses1[:20]] + [l['iteration'] for l in losses2[:20]]))
        
        print(f"{'Iter':<8} {'Run1 Triplet':<15} {'Run2 Triplet':<15} {'Diff':<12} {'Run1 Softmax':<15} {'Run2 Softmax':<15} {'Diff':<12}")
        print("-" * 100)
        
        match_count = 0
        total_count = 0
        
        for iter_num in common_iters[:20]:
            l1 = next((l for l in losses1 if l['iteration'] == iter_num), None)
            l2 = next((l for l in losses2 if l['iteration'] == iter_num), None)
            
            if l1 and l2:
                diff_triplet = abs(l1['triplet_loss'] - l2['triplet_loss'])
                diff_softmax = abs(l1['softmax_loss'] - l2['softmax_loss'])
                
                match_triplet = diff_triplet < 0.0001
                match_softmax = diff_softmax < 0.0001
                
                if match_triplet and match_softmax:
                    match_count += 1
                total_count += 1
                
                match_str_triplet = "✅" if match_triplet else f"❌ ({diff_triplet:.6f})"
                match_str_softmax = "✅" if match_softmax else f"❌ ({diff_softmax:.6f})"
                
                print(f"{iter_num:<8} {l1['triplet_loss']:<15.6f} {l2['triplet_loss']:<15.6f} {diff_triplet:<12.6f} "
                      f"{l1['softmax_loss']:<15.6f} {l2['softmax_loss']:<15.6f} {diff_softmax:<12.6f}")
        
        print()
        print(f"Match rate: {match_count}/{total_count} ({100*match_count/total_count if total_count > 0 else 0:.1f}%)")
        
        if match_count == 0:
            print("❌ CRITICAL: Training losses don't match - data loading is non-deterministic!")
        elif match_count < total_count:
            print("⚠️  WARNING: Some training losses differ - partial non-determinism")
        else:
            print("✅ Training losses match - data loading appears deterministic")
    
    print()
    
    # 2. CHECK EVALUATION ACCURACIES AT FIRST FEW CHECKPOINTS
    print("=" * 100)
    print("2. EVALUATION ACCURACIES (First 5 checkpoints)")
    print("=" * 100)
    print()
    
    evals1 = extract_evaluations(log1)
    evals2 = extract_evaluations(log2)
    
    checkpoints = sorted(set(list(evals1.keys()) + list(evals2.keys())))[:5]
    
    print(f"{'Iter':<8} {'Run1 Acc':<15} {'Run2 Acc':<15} {'Match':<10}")
    print("-" * 100)
    
    for iter_num in checkpoints:
        acc1 = evals1.get(iter_num)
        acc2 = evals2.get(iter_num)
        
        if acc1 is not None and acc2 is not None:
            match = abs(acc1 - acc2) < 0.01
            match_str = "✅" if match else "❌"
            print(f"{iter_num:<8} {acc1:<15.2f} {acc2:<15.2f} {match_str:<10}")
    
    print()
    
    # 3. CHECK LOG FILE SIZES AND ITERATION COUNTS
    print("=" * 100)
    print("3. RUN STATISTICS")
    print("=" * 100)
    print()
    
    size1 = os.path.getsize(log1) / 1024  # KB
    size2 = os.path.getsize(log2) / 1024
    
    print(f"Run 1:")
    print(f"  Log size: {size1:.1f} KB")
    print(f"  Training iterations: {len(losses1)}")
    print(f"  Evaluations: {len(evals1)}")
    if losses1:
        print(f"  Iteration range: {losses1[0]['iteration']} - {losses1[-1]['iteration']}")
    
    print(f"\nRun 2:")
    print(f"  Log size: {size2:.1f} KB")
    print(f"  Training iterations: {len(losses2)}")
    print(f"  Evaluations: {len(evals2)}")
    if losses2:
        print(f"  Iteration range: {losses2[0]['iteration']} - {losses2[-1]['iteration']}")
    
    print()
    
    # 4. CHECK FOR SEED INITIALIZATION MESSAGES
    print("=" * 100)
    print("4. SEED INITIALIZATION CHECK")
    print("=" * 100)
    print()
    
    seed_info1 = check_seed_initialization(log1)
    seed_info2 = check_seed_initialization(log2)
    
    print(f"Run 1 seed-related messages (first 5):")
    for msg in seed_info1[:5]:
        print(f"  {msg}")
    
    print(f"\nRun 2 seed-related messages (first 5):")
    for msg in seed_info2[:5]:
        print(f"  {msg}")
    
    print()
    
    # 5. SUMMARY AND RECOMMENDATIONS
    print("=" * 100)
    print("5. DIAGNOSIS SUMMARY")
    print("=" * 100)
    print()
    
    if losses1 and losses2:
        first_iter_match = False
        if losses1[0]['iteration'] == losses2[0]['iteration']:
            l1_first = losses1[0]
            l2_first = losses2[0]
            if abs(l1_first['triplet_loss'] - l2_first['triplet_loss']) < 0.0001:
                first_iter_match = True
        
        if not first_iter_match:
            print("❌ FIRST ITERATION LOSSES DON'T MATCH")
            print("   → Batch counter may not be resetting properly")
            print("   → Or data loading is non-deterministic from the start")
        else:
            print("✅ First iteration losses match")
            print("   → Batch counter reset appears to work")
            print("   → But divergence happens later - check increment logic")
    
    print()
    print("=" * 100)

if __name__ == '__main__':
    main()
