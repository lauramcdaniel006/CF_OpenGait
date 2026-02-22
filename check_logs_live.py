#!/usr/bin/env python3
"""
Quick script to check the current state of logs from both test configs.
Can be run while training is still in progress.

Usage:
    python check_logs_live.py
"""

import os
import glob
import re
from datetime import datetime

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

def get_last_n_lines(file_path, n=50):
    """Get last N lines from a file."""
    if not file_path or not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def get_last_iteration(log_file):
    """Get the last training iteration number."""
    if not log_file:
        return None
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            matches = re.findall(r'Iteration\s+(\d+)', content)
            if matches:
                return int(matches[-1])
    except:
        pass
    return None

def get_last_evaluation(log_file):
    """Get the last evaluation result."""
    if not log_file:
        return None
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Find last evaluation section
            eval_sections = content.split('EVALUATION RESULTS')
            if len(eval_sections) < 2:
                return None
            
            last_section = eval_sections[-1]
            
            # Extract iteration
            iter_match = re.search(r'Iteration\s+(\d+)', content[:content.rfind('EVALUATION RESULTS')])
            iter_num = int(iter_match.group(1)) if iter_match else None
            
            # Extract accuracy
            acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', last_section)
            accuracy = float(acc_match.group(1)) if acc_match else None
            
            # Extract confusion matrix
            cm_match = re.search(r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]', last_section)
            cm = None
            if cm_match:
                cm = [
                    [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                    [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                    [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
                ]
            
            return {
                'iteration': iter_num,
                'accuracy': accuracy,
                'confusion_matrix': cm
            }
    except Exception as e:
        print(f"Error extracting evaluation: {e}")
    return None

def main():
    print("=" * 100)
    print("LIVE LOG CHECKER - Test Configs Comparison")
    print("=" * 100)
    print()
    
    log1 = find_latest_log(RUN1_PATH)
    log2 = find_latest_log(RUN2_PATH)
    
    # Check Run 1
    print("RUN 1: swin_part4a_B3_unfrozen_cnn")
    print("-" * 100)
    if log1:
        print(f"✓ Log file: {os.path.basename(log1)}")
        file_size = os.path.getsize(log1) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(log1))
        print(f"  Size: {file_size:.1f} KB")
        print(f"  Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        last_iter = get_last_iteration(log1)
        if last_iter:
            print(f"  Last iteration: {last_iter}")
        
        last_eval = get_last_evaluation(log1)
        if last_eval:
            print(f"  Last evaluation:")
            print(f"    Iteration: {last_eval['iteration']}")
            if last_eval['accuracy']:
                print(f"    Accuracy: {last_eval['accuracy']:.2f}%")
            if last_eval['confusion_matrix']:
                print(f"    Confusion Matrix: {last_eval['confusion_matrix']}")
    else:
        print("❌ No log file found")
        print(f"   Expected at: {RUN1_PATH}/logs/")
    print()
    
    # Check Run 2
    print("RUN 2: swin_testb3")
    print("-" * 100)
    if log2:
        print(f"✓ Log file: {os.path.basename(log2)}")
        file_size = os.path.getsize(log2) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(log2))
        print(f"  Size: {file_size:.1f} KB")
        print(f"  Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        last_iter = get_last_iteration(log2)
        if last_iter:
            print(f"  Last iteration: {last_iter}")
        
        last_eval = get_last_evaluation(log2)
        if last_eval:
            print(f"  Last evaluation:")
            print(f"    Iteration: {last_eval['iteration']}")
            if last_eval['accuracy']:
                print(f"    Accuracy: {last_eval['accuracy']:.2f}%")
            if last_eval['confusion_matrix']:
                print(f"    Confusion Matrix: {last_eval['confusion_matrix']}")
    else:
        print("❌ No log file found")
        print(f"   Expected at: {RUN2_PATH}/logs/")
    print()
    
    # Compare if both exist
    if log1 and log2:
        print("=" * 100)
        print("QUICK COMPARISON")
        print("=" * 100)
        
        iter1 = get_last_iteration(log1)
        iter2 = get_last_iteration(log2)
        
        if iter1 and iter2:
            print(f"Iteration progress: Run1={iter1}, Run2={iter2}")
            if iter1 == iter2:
                print("✅ Both runs at same iteration")
            else:
                print(f"⚠️  Runs at different iterations (diff: {abs(iter1 - iter2)})")
        
        eval1 = get_last_evaluation(log1)
        eval2 = get_last_evaluation(log2)
        
        if eval1 and eval2 and eval1['iteration'] == eval2['iteration']:
            print(f"\nComparing evaluation at iteration {eval1['iteration']}:")
            
            if eval1['accuracy'] and eval2['accuracy']:
                match_acc = abs(eval1['accuracy'] - eval2['accuracy']) < 0.01
                print(f"  Accuracy: Run1={eval1['accuracy']:.2f}%, Run2={eval2['accuracy']:.2f}%", end="")
                print(" ✅ MATCH" if match_acc else " ❌ DIFFER")
            
            if eval1['confusion_matrix'] and eval2['confusion_matrix']:
                match_cm = eval1['confusion_matrix'] == eval2['confusion_matrix']
                print(f"  Confusion Matrix: ", end="")
                print("✅ MATCH" if match_cm else "❌ DIFFER")
                if not match_cm:
                    print(f"    Run1: {eval1['confusion_matrix']}")
                    print(f"    Run2: {eval2['confusion_matrix']}")
        elif eval1 or eval2:
            print("\n⚠️  Cannot compare: evaluations at different iterations or missing")
    
    print()
    print("=" * 100)
    print("For detailed comparison, run: python compare_test_configs.py")
    print("=" * 100)

if __name__ == '__main__':
    main()
