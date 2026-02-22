#!/usr/bin/env python3
"""
Compare logs from two test configs to check if they produce identical results:
1. swin_part4a_B3_unfrozen_cnn.yaml
2. swin_testb3.yaml.yaml

Usage:
    python compare_test_configs.py
"""

import os
import sys
import re
import glob
from datetime import datetime

# Paths for the two runs based on save_name in configs
RUN1_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn"
RUN2_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnntest"

RUN1_NAME = "swin_part4a_B3_unfrozen_cnn"
RUN2_NAME = "swin_testb3"

def find_latest_log(exp_dir):
    """Find the most recent log file in an experiment directory."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    
    # Return the most recently modified file
    return max(log_files, key=os.path.getmtime)

def extract_training_losses(log_file):
    """Extract training losses from log file."""
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

def extract_evaluation_metrics(log_file):
    """Extract evaluation metrics from log file."""
    results = []
    
    if not log_file or not os.path.exists(log_file):
        return results
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find all evaluation sections
        eval_sections = re.split(r'EVALUATION RESULTS', content)
        
        for i, section in enumerate(eval_sections[1:], 1):  # Skip first empty split
            metrics = {}
            
            # Extract iteration number from previous lines
            iter_match = re.search(r'Iteration\s+(\d+)', section)
            if iter_match:
                metrics['iteration'] = int(iter_match.group(1))
            else:
                # Try to find iteration from context before this section
                prev_context = content[:content.find(section)]
                iter_match = re.search(r'Iteration\s+(\d+)', prev_context[-500:])
                if iter_match:
                    metrics['iteration'] = int(iter_match.group(1))
                else:
                    continue  # Skip if we can't find iteration
            
            # Extract confusion matrix
            cm_match = re.search(r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]', section)
            if cm_match:
                metrics['confusion_matrix'] = [
                    [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                    [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                    [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
                ]
            
            # Extract overall accuracy
            acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', section)
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1))
            
            # Extract ROC AUC
            auc_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', section)
            if auc_match:
                metrics['roc_auc_macro'] = float(auc_match.group(1))
            
            # Extract per-class metrics
            frail_match = re.search(r'Frail.*?Recall:\s*([\d.]+)%.*?Precision:\s*([\d.]+)%.*?F1=([\d.]+)', section, re.DOTALL)
            if frail_match:
                metrics['frail'] = {
                    'recall': float(frail_match.group(1)),
                    'precision': float(frail_match.group(2)),
                    'f1': float(frail_match.group(3))
                }
            
            prefrail_match = re.search(r'Prefrail.*?Recall:\s*([\d.]+)%.*?Precision:\s*([\d.]+)%.*?F1=([\d.]+)', section, re.DOTALL)
            if prefrail_match:
                metrics['prefrail'] = {
                    'recall': float(prefrail_match.group(1)),
                    'precision': float(prefrail_match.group(2)),
                    'f1': float(prefrail_match.group(3))
                }
            
            nonfrail_match = re.search(r'Nonfrail.*?Recall:\s*([\d.]+)%.*?Precision:\s*([\d.]+)%.*?F1=([\d.]+)', section, re.DOTALL)
            if nonfrail_match:
                metrics['nonfrail'] = {
                    'recall': float(nonfrail_match.group(1)),
                    'precision': float(nonfrail_match.group(2)),
                    'f1': float(nonfrail_match.group(3))
                }
            
            if metrics:
                results.append(metrics)
    
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def compare_values(val1, val2, name, tolerance=1e-6):
    """Compare two values and return if they match."""
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        diff = abs(val1 - val2)
        match = diff < tolerance
        return match, diff
    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False, None
        for i, (v1, v2) in enumerate(zip(val1, val2)):
            match, diff = compare_values(v1, v2, f"{name}[{i}]", tolerance)
            if not match:
                return False, diff
        return True, 0
    else:
        match = val1 == val2
        return match, None

def main():
    print("=" * 100)
    print("COMPARING TWO TEST CONFIGS FOR IDENTICAL RESULTS")
    print("=" * 100)
    print()
    
    # Find log files
    log1 = find_latest_log(RUN1_PATH)
    log2 = find_latest_log(RUN2_PATH)
    
    if not log1:
        print(f"❌ No log file found for {RUN1_NAME}")
        print(f"   Expected at: {RUN1_PATH}/logs/")
        return
    
    if not log2:
        print(f"❌ No log file found for {RUN2_NAME}")
        print(f"   Expected at: {RUN2_PATH}/logs/")
        return
    
    print(f"✓ Found Run 1 log: {os.path.basename(log1)}")
    print(f"✓ Found Run 2 log: {os.path.basename(log2)}")
    print()
    
    # Extract training losses
    print("=" * 100)
    print("COMPARING TRAINING LOSSES")
    print("=" * 100)
    print()
    
    losses1 = extract_training_losses(log1)
    losses2 = extract_training_losses(log2)
    
    if not losses1:
        print("⚠ No training losses found in Run 1")
    if not losses2:
        print("⚠ No training losses found in Run 2")
    
    if losses1 and losses2:
        # Compare losses at common iterations
        common_iters = set(l['iteration'] for l in losses1) & set(l['iteration'] for l in losses2)
        common_iters = sorted(common_iters)
        
        if common_iters:
            print(f"{'Iter':<8} {'Run1 Triplet':<15} {'Run2 Triplet':<15} {'Match':<8} {'Run1 Softmax':<15} {'Run2 Softmax':<15} {'Match':<8}")
            print("-" * 100)
            
            all_match = True
            for iter_num in common_iters[:20]:  # Show first 20 iterations
                l1 = next(l for l in losses1 if l['iteration'] == iter_num)
                l2 = next(l for l in losses2 if l['iteration'] == iter_num)
                
                match_triplet, diff_triplet = compare_values(l1['triplet_loss'], l2['triplet_loss'], 'triplet')
                match_softmax, diff_softmax = compare_values(l1['softmax_loss'], l2['softmax_loss'], 'softmax')
                
                match_str_triplet = "✅" if match_triplet else f"❌ ({diff_triplet:.6f})"
                match_str_softmax = "✅" if match_softmax else f"❌ ({diff_softmax:.6f})"
                
                if not match_triplet or not match_softmax:
                    all_match = False
                
                print(f"{iter_num:<8} {l1['triplet_loss']:<15.6f} {l2['triplet_loss']:<15.6f} {match_str_triplet:<8} "
                      f"{l1['softmax_loss']:<15.6f} {l2['softmax_loss']:<15.6f} {match_str_softmax:<8}")
            
            if len(common_iters) > 20:
                print(f"\n... (showing first 20 of {len(common_iters)} iterations)")
            
            print()
            if all_match:
                print("✅ ALL TRAINING LOSSES MATCH!")
            else:
                print("❌ SOME TRAINING LOSSES DIFFER!")
        else:
            print("⚠ No common iterations found for comparison")
    
    print()
    
    # Extract evaluation metrics
    print("=" * 100)
    print("COMPARING EVALUATION METRICS")
    print("=" * 100)
    print()
    
    metrics1 = extract_evaluation_metrics(log1)
    metrics2 = extract_evaluation_metrics(log2)
    
    if not metrics1:
        print("⚠ No evaluation metrics found in Run 1")
    if not metrics2:
        print("⚠ No evaluation metrics found in Run 2")
    
    if metrics1 and metrics2:
        # Compare metrics at common iterations
        common_iters = set(m['iteration'] for m in metrics1) & set(m['iteration'] for m in metrics2)
        common_iters = sorted(common_iters)
        
        if common_iters:
            print(f"{'Iter':<8} {'Metric':<20} {'Run1':<15} {'Run2':<15} {'Match':<8}")
            print("-" * 100)
            
            all_match = True
            for iter_num in common_iters:
                m1 = next(m for m in metrics1 if m['iteration'] == iter_num)
                m2 = next(m for m in metrics2 if m['iteration'] == iter_num)
                
                # Compare accuracy
                if 'accuracy' in m1 and 'accuracy' in m2:
                    match, diff = compare_values(m1['accuracy'], m2['accuracy'], 'accuracy')
                    match_str = "✅" if match else f"❌ ({diff:.4f}%)"
                    if not match:
                        all_match = False
                    print(f"{iter_num:<8} {'Accuracy':<20} {m1['accuracy']:<15.2f} {m2['accuracy']:<15.2f} {match_str:<8}")
                
                # Compare ROC AUC
                if 'roc_auc_macro' in m1 and 'roc_auc_macro' in m2:
                    match, diff = compare_values(m1['roc_auc_macro'], m2['roc_auc_macro'], 'roc_auc')
                    match_str = "✅" if match else f"❌ ({diff:.6f})"
                    if not match:
                        all_match = False
                    print(f"{iter_num:<8} {'ROC AUC (macro)':<20} {m1['roc_auc_macro']:<15.6f} {m2['roc_auc_macro']:<15.6f} {match_str:<8}")
                
                # Compare confusion matrix
                if 'confusion_matrix' in m1 and 'confusion_matrix' in m2:
                    match, _ = compare_values(m1['confusion_matrix'], m2['confusion_matrix'], 'cm')
                    match_str = "✅" if match else "❌"
                    if not match:
                        all_match = False
                    cm1_str = str(m1['confusion_matrix']).replace('\n', ' ')
                    cm2_str = str(m2['confusion_matrix']).replace('\n', ' ')
                    print(f"{iter_num:<8} {'Confusion Matrix':<20} {cm1_str:<15} {cm2_str:<15} {match_str:<8}")
                
                print()
            
            print("=" * 100)
            if all_match:
                print("✅ ALL EVALUATION METRICS MATCH! Results are IDENTICAL!")
            else:
                print("❌ SOME EVALUATION METRICS DIFFER!")
        else:
            print("⚠ No common iterations found for comparison")
    
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Run 1 ({RUN1_NAME}): {len(losses1)} training iterations, {len(metrics1)} evaluations")
    print(f"Run 2 ({RUN2_NAME}): {len(losses2)} training iterations, {len(metrics2)} evaluations")
    print()
    print("Log files:")
    print(f"  Run 1: {log1}")
    print(f"  Run 2: {log2}")
    print("=" * 100)

if __name__ == '__main__':
    main()
