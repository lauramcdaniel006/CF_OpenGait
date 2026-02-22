#!/usr/bin/env python3
"""
Compute AUC-ROC for SwinGait M1 and DeepGaitV2 M6 models.
This script re-evaluates the best checkpoints and computes AUC-ROC metrics.
"""

import os
import sys
import subprocess
import glob
import re
import json
from pathlib import Path

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_best_checkpoint(exp_dir):
    """Find the checkpoint with the highest iteration number."""
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None, None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not checkpoints:
        return None, None
    
    def get_iter(cp):
        match = re.search(r'-(\d+)\.pt$', cp)
        return int(match.group(1)) if match else 0
    
    best_checkpoint = max(checkpoints, key=get_iter)
    best_iter = get_iter(best_checkpoint)
    
    return best_checkpoint, best_iter

def find_config_file(exp_name):
    """Find the config file for an experiment."""
    # Map experiment names to config files
    config_map = {
        'REDO_Frailty_ccpg_pt4_swingait': 'configs/swingait/swin_part4_deepgaitv2_comparison.yaml',
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights': 'configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml'
    }
    
    if exp_name in config_map:
        config_path = config_map[exp_name]
        if os.path.exists(config_path):
            return config_path
    
    # Try to find config by searching
    for root, dirs, files in os.walk('configs'):
        for file in files:
            if exp_name.lower().replace('redo_frailty_ccpg_', '') in file.lower():
                return os.path.join(root, file)
    
    return None

def extract_auc_from_output(output_text):
    """Extract AUC-ROC values from evaluation output."""
    auc_macro = None
    auc_micro = None
    
    for line in output_text.split('\n'):
        if 'ROC AUC (macro)' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                auc_macro = float(match.group(1))
        elif 'ROC AUC (micro)' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                auc_micro = float(match.group(1))
    
    return auc_macro, auc_micro

def evaluate_checkpoint(config_file, checkpoint_path, iteration, exp_name):
    """Run evaluation on a checkpoint and extract AUC-ROC."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {exp_name}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)} (iteration {iteration})")
    print(f"Config: {config_file}")
    print(f"{'='*80}")
    
    # Run evaluation using opengait/main.py
    cmd = [
        'python', 'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',
        '--iter', str(iteration)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=600  # 10 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Extract AUC-ROC from output
        auc_macro, auc_micro = extract_auc_from_output(output)
        
        # Also extract other metrics
        accuracy = None
        precision = None
        recall = None
        f1 = None
        
        for line in output.split('\n'):
            if 'Overall Accuracy' in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    accuracy = float(match.group(1))
            elif 'test_precision/' in line and 'macro' in line.lower():
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    precision = float(match.group(1))
            elif 'test_recall/' in line and 'macro' in line.lower():
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    recall = float(match.group(1))
            elif 'test_f1/' in line and 'macro' in line.lower():
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    f1 = float(match.group(1))
        
        return {
            'iteration': iteration,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_macro': auc_macro,
            'auc_micro': auc_micro,
            'output': output
        }
    except subprocess.TimeoutExpired:
        print(f"ERROR: Evaluation timed out for {exp_name}")
        return None
    except Exception as e:
        print(f"ERROR: Evaluation failed for {exp_name}: {e}")
        return None

def main():
    # Define the two models to evaluate
    models = {
        'SwinGait M1': {
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4_swingait/SwinGait/REDO_Frailty_ccpg_pt4_swingait',
            'exp_name': 'REDO_Frailty_ccpg_pt4_swingait',
            'config': 'configs/swingait/swin_part4_deepgaitv2_comparison.yaml'
        },
        'DeepGaitV2 M6': {
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights',
            'exp_name': 'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights',
            'config': 'configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml'
        }
    }
    
    results = {}
    
    for model_name, model_info in models.items():
        exp_dir = model_info['exp_dir']
        exp_name = model_info['exp_name']
        config_file = model_info['config']
        
        if not os.path.exists(exp_dir):
            print(f"WARNING: Experiment directory not found: {exp_dir}")
            continue
        
        if not os.path.exists(config_file):
            print(f"WARNING: Config file not found: {config_file}")
            continue
        
        # Find best checkpoint
        checkpoint_path, iteration = find_best_checkpoint(exp_dir)
        
        if checkpoint_path is None:
            print(f"WARNING: No checkpoint found for {model_name}")
            continue
        
        # Evaluate
        result = evaluate_checkpoint(config_file, checkpoint_path, iteration, exp_name)
        
        if result:
            results[model_name] = result
    
    # Print summary
    print("\n" + "="*80)
    print("AUC-ROC RESULTS SUMMARY")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Iteration: {result['iteration']}")
        if result['accuracy'] is not None:
            print(f"  Accuracy: {result['accuracy']:.2f}%")
        if result['precision'] is not None:
            print(f"  Precision: {result['precision']:.4f}")
        if result['recall'] is not None:
            print(f"  Recall: {result['recall']:.4f}")
        if result['f1'] is not None:
            print(f"  F1: {result['f1']:.4f}")
        if result['auc_macro'] is not None:
            print(f"  AUC-ROC (macro): {result['auc_macro']:.4f}")
        if result['auc_micro'] is not None:
            print(f"  AUC-ROC (micro): {result['auc_micro']:.4f}")
    
    # Save results to JSON
    output_file = 'auc_roc_results_part4.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()

