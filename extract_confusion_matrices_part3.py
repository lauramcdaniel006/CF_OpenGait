#!/usr/bin/env python3
"""
Extract confusion matrices for Part 3 experiments by re-running evaluation on checkpoints.
Confusion matrices are calculated but not saved to TensorBoard, so we need to re-run evaluation.
"""

import os
import re
import subprocess
import glob
import csv
from pathlib import Path
from collections import defaultdict

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
    """Find the config file for this experiment."""
    config_map = {
        'REDO_Frailty_ccpg_pt3_ce_contrastive': 'configs/swingait/swin_part3_ce_contrastive.yaml',
        'REDO_Frailty_ccpg_pt3_triplet_focal': 'configs/swingait/swin_part3_triplet_focal.yaml',
        'REDO_Frailty_ccpg_pt3_contrastive_focal': 'configs/swingait/swin_part3_contrastive_focal.yaml',
    }
    
    if exp_name in config_map:
        config_path = config_map[exp_name]
        if os.path.exists(config_path):
            return config_path
    
    # Try to find by pattern
    patterns = [
        f"configs/swingait/*{exp_name.lower().replace('redo_frailty_ccpg_pt3_', '')}*.yaml",
        f"configs/swingait/swin_part3*.yaml",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def extract_confusion_matrix_from_output(output_text):
    """Extract confusion matrix from evaluation output."""
    # Pattern: Confusion Matrix: [[a b c] [d e f] [g h i]]
    cm_match = re.search(
        r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
        output_text,
        re.MULTILINE | re.DOTALL
    )
    
    if cm_match:
        cm = [
            [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
            [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
            [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
        ]
        return cm
    return None

def evaluate_checkpoint(config_file, checkpoint_path, iteration):
    """Run evaluation on a single checkpoint and extract confusion matrix."""
    print(f"  Evaluating iteration {iteration}...", end=' ', flush=True)
    
    # Build command
    cmd = [
        'python', '/cis/home/lmcdan11/Documents_Swin/OpenGait/opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',
        '--iter', str(iteration)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd='/cis/home/lmcdan11/Documents_Swin/OpenGait'
        )
        
        if result.returncode != 0:
            print(f"❌ Error (code {result.returncode})")
            return None
        
        cm = extract_confusion_matrix_from_output(result.stdout + result.stderr)
        
        if cm:
            print("✓")
            return cm
        else:
            print("⚠️  No CM found")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout")
        return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("=" * 120)
    print("EXTRACT CONFUSION MATRICES FOR PART 3")
    print("=" * 120)
    print("\nThis will re-run evaluation on Part 3 checkpoints to extract confusion matrices.")
    print("Confusion matrices are calculated but not saved to TensorBoard.\n")
    
    part3_experiments = [
        'REDO_Frailty_ccpg_pt3_ce_contrastive',
        'REDO_Frailty_ccpg_pt3_triplet_focal',
        'REDO_Frailty_ccpg_pt3_contrastive_focal'
    ]
    
    print("Options:")
    print("1. Generate evaluation commands (you run them manually)")
    print("2. Auto-run evaluation and extract confusion matrices to CSV")
    print("3. Show which checkpoints will be evaluated")
    print("4. Sample checkpoints only (every 5th iteration, faster)")
    
    choice = input("\nEnter choice (1, 2, 3, or 4): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 120)
        print("EVALUATION COMMANDS FOR PART 3")
        print("=" * 120)
        print("\nRun these commands to re-evaluate and extract confusion matrices:\n")
        
        for exp_name in part3_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            if not os.path.exists(exp_path):
                print(f"\n# {exp_name}: Experiment directory not found")
                continue
                
            checkpoints = find_checkpoints(exp_path)
            config_file = find_config_file(exp_name)
            
            if not checkpoints:
                print(f"\n# {exp_name}: No checkpoints found")
                continue
            
            if not config_file:
                print(f"\n# {exp_name}: Config file not found")
                continue
            
            print(f"\n# {exp_name}")
            print(f"# Found {len(checkpoints)} checkpoints")
            print(f"# Config: {config_file}")
            print()
            
            for cp in checkpoints:
                cp_name = os.path.basename(cp)
                match = re.search(r'-(\d+)\.pt$', cp_name)
                if match:
                    iter_num = match.group(1)
                    print(f"# Iteration {iter_num}")
                    print(f"python /cis/home/lmcdan11/Documents_Swin/OpenGait/opengait/main.py \\")
                    print(f"  --cfgs {config_file} \\")
                    print(f"  --phase test \\")
                    print(f"  --iter {iter_num}")
                    print()
        
        print("=" * 120)
        print("\nAfter running evaluations, confusion matrices will be in the console output.")
        print("You can parse them or re-run this script with option 2.")
        print("=" * 120)
    
    elif choice == "2" or choice == "4":
        sample_only = (choice == "4")
        print("\n" + "=" * 120)
        print("AUTO-RUNNING EVALUATION")
        print("=" * 120)
        if sample_only:
            print("\n⚠️  Sampling mode: Evaluating every 5th checkpoint (faster)\n")
        else:
            print("\n⚠️  This will take a while! Evaluating all checkpoints...\n")
        
        all_data = []
        
        for exp_name in part3_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            if not os.path.exists(exp_path):
                print(f"⚠️  {exp_name}: Experiment directory not found")
                continue
            
            checkpoints = find_checkpoints(exp_path)
            config_file = find_config_file(exp_name)
            
            if not checkpoints:
                print(f"⚠️  {exp_name}: No checkpoints found")
                continue
            
            if not config_file:
                print(f"⚠️  {exp_name}: Config file not found")
                continue
            
            # Sample checkpoints if requested
            if sample_only:
                checkpoints = checkpoints[::5]  # Every 5th checkpoint
            
            print(f"\n{'='*120}")
            print(f"Processing: {exp_name}")
            print(f"  Config: {config_file}")
            print(f"  Checkpoints: {len(checkpoints)} {'(sampled)' if sample_only else ''}")
            print(f"{'='*120}\n")
            
            for cp in checkpoints:
                cp_name = os.path.basename(cp)
                match = re.search(r'-(\d+)\.pt$', cp_name)
                if not match:
                    continue
                
                iteration = int(match.group(1))
                cm = evaluate_checkpoint(config_file, cp, iteration)
                
                if cm:
                    row = {
                        'Part': 'Part 3: Loss Functions',
                        'Experiment': exp_name,
                        'Iteration': iteration,
                        'CM_Frail_Frail': cm[0][0],
                        'CM_Frail_Prefrail': cm[0][1],
                        'CM_Frail_Nonfrail': cm[0][2],
                        'CM_Prefrail_Frail': cm[1][0],
                        'CM_Prefrail_Prefrail': cm[1][1],
                        'CM_Prefrail_Nonfrail': cm[1][2],
                        'CM_Nonfrail_Frail': cm[2][0],
                        'CM_Nonfrail_Prefrail': cm[2][1],
                        'CM_Nonfrail_Nonfrail': cm[2][2],
                        'CM_String': f"[[{cm[0][0]} {cm[0][1]} {cm[0][2]}] [{cm[1][0]} {cm[1][1]} {cm[1][2]}] [{cm[2][0]} {cm[2][1]} {cm[2][2]}]]"
                    }
                    all_data.append(row)
        
        # Save to CSV
        if all_data:
            output_dir = 'results_visualization'
            os.makedirs(output_dir, exist_ok=True)
            
            csv_file = os.path.join(output_dir, 'part3_confusion_matrices_all_iterations.csv')
            
            fieldnames = [
                'Part', 'Experiment', 'Iteration',
                'CM_Frail_Frail', 'CM_Frail_Prefrail', 'CM_Frail_Nonfrail',
                'CM_Prefrail_Frail', 'CM_Prefrail_Prefrail', 'CM_Prefrail_Nonfrail',
                'CM_Nonfrail_Frail', 'CM_Nonfrail_Prefrail', 'CM_Nonfrail_Nonfrail',
                'CM_String'
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)
            
            print(f"\n{'='*120}")
            print(f"✓ Saved confusion matrices to: {csv_file}")
            print(f"  Total rows: {len(all_data)}")
            print(f"{'='*120}")
        else:
            print("\n❌ No confusion matrices extracted!")
    
    else:  # choice == "3"
        print("\n" + "=" * 120)
        print("CHECKPOINT SUMMARY FOR PART 3")
        print("=" * 120)
        
        for exp_name in part3_experiments:
            exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
            if not os.path.exists(exp_path):
                print(f"\n{exp_name}: Experiment directory not found")
                continue
            
            checkpoints = find_checkpoints(exp_path)
            config_file = find_config_file(exp_name)
            
            if checkpoints:
                print(f"\n{exp_name}:")
                print(f"  Config: {config_file if config_file else 'NOT FOUND'}")
                print(f"  Checkpoints: {len(checkpoints)}")
                print(f"  First: {os.path.basename(checkpoints[0])}")
                print(f"  Last: {os.path.basename(checkpoints[-1])}")
                if len(checkpoints) > 5:
                    print(f"  Sample (every 5th): {len(checkpoints[::5])} checkpoints")
            else:
                print(f"\n{exp_name}: No checkpoints found")

if __name__ == '__main__':
    main()

