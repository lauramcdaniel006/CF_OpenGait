#!/usr/bin/env python3
"""
Re-run evaluation on Part 2 (Class Weights) checkpoints to extract per-class metrics.
This will evaluate each checkpoint and extract per-class F1, Precision, Recall for each iteration.
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
    # Part 2 experiments
    config_map = {
        'REDO_insqrt': 'configs/swingait/swin_part2_insqrt.yaml',
        'REDO_balnormal': 'configs/swingait/swin_part2_balnormal.yaml',
        'REDO_log': 'configs/swingait/swin_part2_log.yaml',
        'REDO_smooth': 'configs/swingait/swin_part2_smooth.yaml',
        'REDO_uniform': 'configs/swingait/swin_part2_uniform.yaml',
    }
    
    if exp_name in config_map:
        config_path = config_map[exp_name]
        if os.path.exists(config_path):
            return config_path
    
    # Try to find by pattern
    patterns = [
        f"configs/swingait/*{exp_name.lower().replace('redo_', '')}*.yaml",
        f"configs/swingait/swin_part2*.yaml",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def extract_per_class_metrics_from_output(output_text):
    """Extract per-class metrics from evaluation output."""
    metrics = {}
    
    # Overall Accuracy
    acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', output_text)
    if acc_match:
        metrics['overall_accuracy'] = float(acc_match.group(1))
    
    # Per-class metrics
    class_names = ['Frail', 'Prefrail', 'Nonfrail']
    for class_name in class_names:
        # Precision
        prec_match = re.search(
            rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
            output_text
        )
        if prec_match:
            metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1)) / 100.0
        
        # Sensitivity (Recall)
        sens_match = re.search(
            rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
            output_text
        )
        if sens_match:
            metrics[f'{class_name.lower()}_recall'] = float(sens_match.group(1)) / 100.0
        
        # Specificity
        spec_match = re.search(
            rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%', 
            output_text
        )
        if spec_match:
            metrics[f'{class_name.lower()}_specificity'] = float(spec_match.group(1)) / 100.0
        
        # F1 (may need to calculate from precision and recall)
        # Or try to find it directly
        f1_match = re.search(
            rf'{re.escape(class_name)}\s+F1[:\s]+(\d+\.?\d*)%?', 
            output_text
        )
        if f1_match:
            val = f1_match.group(1)
            metrics[f'{class_name.lower()}_f1'] = float(val) / 100.0 if float(val) > 1.0 else float(val)
    
    return metrics

def evaluate_checkpoint(config_file, checkpoint_path, iteration):
    """Run evaluation on a single checkpoint and extract metrics."""
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
        
        metrics = extract_per_class_metrics_from_output(result.stdout + result.stderr)
        
        if metrics:
            print("✓")
            return metrics
        else:
            print("⚠️  No metrics found")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏱️  Timeout")
        return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("=" * 120)
    print("RE-EVALUATE PART 2 CHECKPOINTS FOR PER-CLASS METRICS")
    print("=" * 120)
    print("\nThis will re-run evaluation on Part 2 (Class Weights) checkpoints")
    print("to extract per-class F1, Precision, Recall for each class.\n")
    
    # Part 2 experiments
    part2_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_log',
        'REDO_smooth',
        'REDO_uniform'
    ]
    
    print("Options:")
    print("1. Generate evaluation commands (you run them manually)")
    print("2. Auto-run evaluation and extract metrics to CSV")
    print("3. Show which checkpoints will be evaluated")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 120)
        print("EVALUATION COMMANDS FOR PART 2")
        print("=" * 120)
        print("\nRun these commands to re-evaluate and extract per-class metrics:\n")
        
        for exp_name in part2_experiments:
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
            
            # Evaluate every checkpoint (or you can modify to sample)
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
        print("\nAfter running evaluations, metrics will be in the console output.")
        print("You can then parse them or re-run this script with option 2.")
        print("=" * 120)
    
    elif choice == "2":
        print("\n" + "=" * 120)
        print("AUTO-RUNNING EVALUATION")
        print("=" * 120)
        print("\n⚠️  This will take a while! Evaluating all checkpoints...\n")
        
        all_data = []
        
        for exp_name in part2_experiments:
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
            
            print(f"\n{'='*120}")
            print(f"Processing: {exp_name}")
            print(f"  Config: {config_file}")
            print(f"  Checkpoints: {len(checkpoints)}")
            print(f"{'='*120}\n")
            
            for cp in checkpoints:
                cp_name = os.path.basename(cp)
                match = re.search(r'-(\d+)\.pt$', cp_name)
                if not match:
                    continue
                
                iteration = int(match.group(1))
                metrics = evaluate_checkpoint(config_file, cp, iteration)
                
                if metrics:
                    row = {
                        'Part': 'Part 2: Class Weights',
                        'Experiment': exp_name,
                        'Iteration': iteration,
                        'Overall_Accuracy': metrics.get('overall_accuracy', ''),
                    }
                    
                    # Add per-class metrics
                    for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
                        class_lower = class_name.lower()
                        row[f'{class_name}_Precision'] = metrics.get(f'{class_lower}_precision', '')
                        row[f'{class_name}_Recall'] = metrics.get(f'{class_lower}_recall', '')
                        row[f'{class_name}_Specificity'] = metrics.get(f'{class_lower}_specificity', '')
                        row[f'{class_name}_F1'] = metrics.get(f'{class_lower}_f1', '')
                    
                    all_data.append(row)
        
        # Save to CSV
        if all_data:
            output_dir = 'results_visualization'
            os.makedirs(output_dir, exist_ok=True)
            
            csv_file = os.path.join(output_dir, 'part2_per_class_metrics_all_iterations.csv')
            
            fieldnames = ['Part', 'Experiment', 'Iteration', 'Overall_Accuracy']
            for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
                fieldnames.extend([
                    f'{class_name}_Precision',
                    f'{class_name}_Recall',
                    f'{class_name}_Specificity',
                    f'{class_name}_F1'
                ])
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_data)
            
            print(f"\n{'='*120}")
            print(f"✓ Saved per-class metrics to: {csv_file}")
            print(f"  Total rows: {len(all_data)}")
            print(f"{'='*120}")
        else:
            print("\n❌ No metrics extracted!")
    
    else:  # choice == "3"
        print("\n" + "=" * 120)
        print("CHECKPOINT SUMMARY FOR PART 2")
        print("=" * 120)
        
        for exp_name in part2_experiments:
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
            else:
                print(f"\n{exp_name}: No checkpoints found")

if __name__ == '__main__':
    main()

