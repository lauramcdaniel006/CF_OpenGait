#!/usr/bin/env python3
"""
Re-run evaluation on checkpoints to capture all metrics including specificity.
This uses the updated evaluator that saves per-class metrics to TensorBoard.
"""

import os
import glob
import re
import subprocess
from pathlib import Path

def find_checkpoints(exp_dir):
    """Find all checkpoint files, sorted by iteration."""
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    
    def get_iter(cp):
        match = re.search(r'-(\d+)\.pt$', cp)
        return int(match.group(1)) if match else 0
    
    return sorted(checkpoints, key=get_iter)

def find_config_file(exp_name):
    """Try to find the config file for this experiment."""
    # Common patterns for REDO experiments
    patterns = [
        f"configs/swingait/*{exp_name}*.yaml",
        f"configs/swingait/swin_part2*.yaml",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Default
    return "configs/swingait/swin_part2_baseline.yaml"

def main():
    print("=" * 80)
    print("Re-evaluate Checkpoints to Capture Specificity")
    print("=" * 80)
    print("\nThis will re-run evaluation on checkpoints using the updated evaluator")
    print("that saves per-class metrics (including specificity) to TensorBoard.\n")
    
    redo_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_smooth',
        'REDO_log',
        'REDO_uniform'
    ]
    
    print("Options:")
    print("1. Generate evaluation commands (you run them manually)")
    print("2. Show which checkpoints will be evaluated")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 80)
        print("EVALUATION COMMANDS")
        print("=" * 80)
        print("\nRun these commands to re-evaluate and capture all metrics:")
        print("(The updated evaluator will save specificity to TensorBoard)\n")
        
        for exp_name in redo_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            checkpoints = find_checkpoints(exp_path)
            
            if not checkpoints:
                print(f"\n# {exp_name}: No checkpoints found")
                continue
            
            config_file = find_config_file(exp_name)
            
            print(f"\n# {exp_name}")
            print(f"# Found {len(checkpoints)} checkpoints")
            print(f"# Config: {config_file}")
            print(f"# Evaluating every 5th checkpoint (or specify iterations manually)")
            print()
            
            # Evaluate every 5th checkpoint to save time
            for cp in checkpoints[::5]:
                cp_name = os.path.basename(cp)
                match = re.search(r'-(\d+)\.pt$', cp_name)
                if match:
                    iter_num = match.group(1)
                    print(f"# Iteration {iter_num}")
                    print(f"CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \\")
                    print(f"  /cis/home/lmcdan11/Documents_Swin/OpenGait/opengait/main.py \\")
                    print(f"  --cfgs {config_file} \\")
                    print(f"  --phase test \\")
                    print(f"  --iter {iter_num} \\")
                    print(f"  --log_to_file")
                    print()
        
        print("=" * 80)
        print("\nAfter running evaluations, use extract_all_checkpoint_metrics.py")
        print("to extract all metrics including specificity from TensorBoard.")
        print("=" * 80)
    
    else:
        print("\n" + "=" * 80)
        print("CHECKPOINT SUMMARY")
        print("=" * 80)
        
        for exp_name in redo_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            checkpoints = find_checkpoints(exp_path)
            
            if checkpoints:
                print(f"\n{exp_name}: {len(checkpoints)} checkpoints")
                print(f"  First: {os.path.basename(checkpoints[0])}")
                print(f"  Last: {os.path.basename(checkpoints[-1])}")
            else:
                print(f"\n{exp_name}: No checkpoints found")

if __name__ == '__main__':
    main()

