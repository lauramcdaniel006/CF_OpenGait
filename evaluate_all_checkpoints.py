#!/usr/bin/env python3
"""
Re-run evaluation on all checkpoints to extract metrics without TensorBoard.
This will evaluate each checkpoint and extract all metrics from the output.
"""

import os
import re
import subprocess
import glob
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

def extract_metrics_from_output(output_text):
    """Extract all metrics from evaluation output."""
    metrics = {}
    
    # Overall Accuracy
    acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', output_text)
    if acc_match:
        metrics['overall_accuracy'] = float(acc_match.group(1))
    
    # Per-class metrics
    class_names = ['Frail', 'Prefrail', 'Nonfrail']
    for class_name in class_names:
        # Sensitivity
        sens_match = re.search(
            rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
            output_text
        )
        if sens_match:
            metrics[f'{class_name.lower()}_sensitivity'] = float(sens_match.group(1))
        
        # Specificity
        spec_match = re.search(
            rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%', 
            output_text
        )
        if spec_match:
            metrics[f'{class_name.lower()}_specificity'] = float(spec_match.group(1))
        
        # Precision
        prec_match = re.search(
            rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
            output_text
        )
        if prec_match:
            metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1))
    
    # Confusion Matrix
    cm_match = re.search(
        r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
        output_text
    )
    if cm_match:
        metrics['confusion_matrix'] = [
            [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
            [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
            [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
        ]
    
    return metrics

def find_config_file(exp_name):
    """Try to find the config file for this experiment."""
    # Common patterns
    patterns = [
        f"configs/swingait/*{exp_name}*.yaml",
        f"configs/swingait/swin_part2*.yaml",
        f"configs/**/*{exp_name}*.yaml",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    # Default to baseline if not found
    return "configs/swingait/swin_part2_baseline.yaml"

def main():
    print("=" * 70)
    print("Evaluate All Checkpoints to Extract Metrics")
    print("=" * 70)
    print("\nNOTE: This will re-run evaluation on checkpoints.")
    print("This may take a while. You can also do this manually.\n")
    
    redo_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_smooth',
        'REDO_log',
        'REDO_uniform'
    ]
    
    print("Options:")
    print("1. Generate evaluation commands (you run them manually)")
    print("2. Run evaluations automatically (requires GPU and time)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Generate commands
        print("\n" + "=" * 70)
        print("Evaluation Commands for REDO Part 2 Experiments")
        print("=" * 70)
        
        for exp_name in redo_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            checkpoints = find_checkpoints(exp_path)
            
            if not checkpoints:
                continue
            
            # Find config
            config_file = find_config_file(exp_name)
            
            print(f"\n# {exp_name}")
            print(f"# Found {len(checkpoints)} checkpoints")
            
            # Evaluate every 5th checkpoint to save time, or all if user wants
            for cp in checkpoints[::5]:  # Every 5th checkpoint
                cp_name = os.path.basename(cp)
                match = re.search(r'-(\d+)\.pt$', cp_name)
                if match:
                    iter_num = match.group(1)
                    print(f"CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \\")
                    print(f"  /cis/home/lmcdan11/Documents_Swin/OpenGait/opengait/main.py \\")
                    print(f"  --cfgs {config_file} \\")
                    print(f"  --phase test \\")
                    print(f"  --iter {iter_num} \\")
                    print(f"  --log_to_file")
                    print(f"# Save output to: {exp_name}_eval_iter{iter_num}.txt")
                    print()
        
        print("\n" + "=" * 70)
        print("Run these commands and save output to text files.")
        print("Then use extract_all_metrics_no_tb.py to parse the results.")
        print("=" * 70)
    
    else:
        print("\nAutomatic evaluation not implemented yet.")
        print("Please use option 1 to generate commands, or evaluate manually.")

if __name__ == '__main__':
    main()

