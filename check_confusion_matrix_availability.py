#!/usr/bin/env python3
"""
Check if confusion matrices are available for Part 3 experiments.
Confusion matrices are calculated but may not be saved to TensorBoard.
"""

import os
import glob
import re

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

def check_tensorboard_for_confusion_matrix(summary_dir):
    """Check if confusion matrix is in TensorBoard."""
    if not TB_AVAILABLE or not os.path.isdir(summary_dir):
        return False, []
    
    try:
        event_files = glob.glob(os.path.join(summary_dir, "events.out.tfevents.*"))
        if not event_files:
            return False, []
        
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        scalar_tags = ea.Tags().get('scalars', [])
        
        # Check for confusion matrix related tags
        confusion_tags = [tag for tag in scalar_tags if 'confusion' in tag.lower() or 'matrix' in tag.lower()]
        
        return len(confusion_tags) > 0, confusion_tags
    except Exception as e:
        return False, []

def find_log_files(exp_dir):
    """Find log files that might contain confusion matrices."""
    log_dirs = [
        os.path.join(exp_dir, 'logs'),
        os.path.join(exp_dir, 'log'),
        exp_dir
    ]
    
    log_files = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            # Look for .log, .txt files
            for pattern in ['*.log', '*.txt']:
                log_files.extend(glob.glob(os.path.join(log_dir, pattern)))
    
    return log_files

def check_log_file_for_confusion_matrix(log_file):
    """Check if log file contains confusion matrix."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for confusion matrix pattern
            if re.search(r'Confusion Matrix:\s*\[\[', content):
                return True
    except:
        pass
    return False

def main():
    print("=" * 120)
    print("CHECKING CONFUSION MATRIX AVAILABILITY FOR PART 3")
    print("=" * 120)
    
    part3_experiments = [
        'REDO_Frailty_ccpg_pt3_ce_contrastive',
        'REDO_Frailty_ccpg_pt3_triplet_focal',
        'REDO_Frailty_ccpg_pt3_contrastive_focal'
    ]
    
    print("\nChecking Part 3 experiments...\n")
    
    all_available = True
    results = {}
    
    for exp_name in part3_experiments:
        exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
        
        if not os.path.exists(exp_path):
            print(f"⚠️  {exp_name}: Experiment directory not found")
            results[exp_name] = {'status': 'not_found'}
            continue
        
        summary_dir = os.path.join(exp_path, 'summary')
        log_files = find_log_files(exp_path)
        
        # Check TensorBoard
        tb_has_cm, tb_tags = check_tensorboard_for_confusion_matrix(summary_dir)
        
        # Check log files
        log_has_cm = False
        cm_log_files = []
        for log_file in log_files:
            if check_log_file_for_confusion_matrix(log_file):
                log_has_cm = True
                cm_log_files.append(log_file)
        
        results[exp_name] = {
            'status': 'found',
            'tensorboard_has_cm': tb_has_cm,
            'tensorboard_tags': tb_tags,
            'log_files_has_cm': log_has_cm,
            'cm_log_files': cm_log_files,
            'all_log_files': log_files
        }
        
        print(f"{exp_name}:")
        print(f"  TensorBoard confusion matrix: {'✓ YES' if tb_has_cm else '✗ NO'}")
        if tb_tags:
            print(f"    Tags: {tb_tags}")
        print(f"  Log files confusion matrix: {'✓ YES' if log_has_cm else '✗ NO'}")
        if cm_log_files:
            print(f"    Files with CM: {cm_log_files}")
        if log_files and not log_has_cm:
            print(f"    Log files found but no CM: {[os.path.basename(f) for f in log_files]}")
        print()
        
        if not tb_has_cm and not log_has_cm:
            all_available = False
    
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    if all_available:
        print("✓ Confusion matrices are available for all experiments!")
    else:
        print("✗ Confusion matrices are NOT directly available.")
        print("\nOptions to get confusion matrices:")
        print("1. Re-run evaluation on checkpoints (confusion matrices are printed to console)")
        print("2. Check if confusion matrices are in log files (if --log_to_file was used)")
        print("3. Confusion matrices are calculated but not saved to TensorBoard")
        print("\nThe evaluator calculates confusion matrices but only saves scalar metrics to TensorBoard.")
        print("To get confusion matrices, you need to:")
        print("  - Re-run evaluation: python opengait/main.py --cfgs <config> --phase test --iter <iteration>")
        print("  - Or parse from console output if you have it saved")
    
    return results

if __name__ == '__main__':
    main()

