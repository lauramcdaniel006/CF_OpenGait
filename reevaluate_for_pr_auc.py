#!/usr/bin/env python3
"""
Re-evaluate all checkpoints every 500 iterations to compute PR-AUC curves.
This script evaluates checkpoints at 500, 1000, 1500, ... iterations and extracts PR-AUC.
Creates a graph showing PR-AUC over training iterations.
"""

import os
import sys
import subprocess
import glob
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_checkpoints(exp_dir, step=500):
    """Find checkpoints at regular intervals (every step iterations)."""
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    
    def get_iter(cp):
        match = re.search(r'-(\d+)\.pt$', cp)
        return int(match.group(1)) if match else 0
    
    # Filter to only checkpoints at step intervals (500, 1000, 1500, ...)
    filtered = []
    for cp in checkpoints:
        iter_num = get_iter(cp)
        if iter_num > 0 and iter_num % step == 0:
            filtered.append((cp, iter_num))
    
    # Sort by iteration
    filtered.sort(key=lambda x: x[1])
    return filtered

def extract_pr_auc_from_output(output_text):
    """Extract PR-AUC values from evaluation output."""
    pr_auc_macro = None
    pr_auc_micro = None
    
    # Try multiple patterns
    patterns = [
        (r'PR AUC \(macro\):\s*(\d+\.\d+)', 'PR AUC (macro):'),
        (r'PR AUC \(micro\):\s*(\d+\.\d+)', 'PR AUC (micro):'),
        (r'PR AUC.*macro.*?(\d+\.\d+)', 'PR AUC macro'),
        (r'PR AUC.*micro.*?(\d+\.\d+)', 'PR AUC micro'),
        (r'pr.*auc.*macro.*?(\d+\.\d+)', 'pr auc macro'),
        (r'pr.*auc.*micro.*?(\d+\.\d+)', 'pr auc micro'),
    ]
    
    for pattern, desc in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            if 'macro' in desc.lower():
                pr_auc_macro = float(match.group(1))
            elif 'micro' in desc.lower():
                pr_auc_micro = float(match.group(1))
    
    return pr_auc_macro, pr_auc_micro

def extract_all_metrics_from_output(output_text):
    """Extract all metrics from evaluation output."""
    metrics = {}
    
    # Accuracy
    acc_match = re.search(r'Overall Accuracy:\s*(\d+\.\d+)%', output_text)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    
    # Precision, Recall, F1 (macro-averaged)
    prec_match = re.search(r'Precision.*?(\d+\.\d+)', output_text, re.IGNORECASE)
    rec_match = re.search(r'Recall.*?(\d+\.\d+)', output_text, re.IGNORECASE)
    f1_match = re.search(r'F1.*?(\d+\.\d+)', output_text, re.IGNORECASE)
    
    if prec_match:
        metrics['precision'] = float(prec_match.group(1))
    if rec_match:
        metrics['recall'] = float(rec_match.group(1))
    if f1_match:
        metrics['f1'] = float(f1_match.group(1))
    
    # ROC AUC (using same pattern but looking for ROC AUC)
    roc_patterns = [
        (r'ROC AUC \(macro\):\s*(\d+\.\d+)', 'ROC AUC (macro):'),
        (r'ROC AUC \(micro\):\s*(\d+\.\d+)', 'ROC AUC (micro):'),
    ]
    roc_auc_macro = None
    roc_auc_micro = None
    for pattern, desc in roc_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            if 'macro' in desc.lower():
                roc_auc_macro = float(match.group(1))
            elif 'micro' in desc.lower():
                roc_auc_micro = float(match.group(1))
    
    if roc_auc_macro:
        metrics['roc_auc_macro'] = roc_auc_macro
    if roc_auc_micro:
        metrics['roc_auc_micro'] = roc_auc_micro
    
    # PR AUC
    pr_auc_macro, pr_auc_micro = extract_pr_auc_from_output(output_text)
    if pr_auc_macro:
        metrics['pr_auc_macro'] = pr_auc_macro
    if pr_auc_micro:
        metrics['pr_auc_micro'] = pr_auc_micro
    
    return metrics

def evaluate_checkpoint(config_file, iteration, exp_name, device='0,1', nproc=2):
    """Run evaluation on a checkpoint and extract PR-AUC."""
    print(f"Evaluating iteration {iteration}...")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    
    # Get the Python interpreter from the current environment
    python_cmd = sys.executable
    
    # Ensure PYTHONPATH includes OpenGait root
    opengait_root = os.path.dirname(os.path.abspath(__file__))
    opengait_dir = os.path.join(opengait_root, 'opengait')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}"
    
    # Set LD_LIBRARY_PATH for Conda environment
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        conda_lib = os.path.join(conda_env, 'lib')
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{conda_lib}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = conda_lib
    
    # Build command
    cmd = [
        python_cmd, '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        '--master_port=29500',
        'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',
        '--iter', str(iteration)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=opengait_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode != 0:
            print(f"Error evaluating iteration {iteration}:")
            print(result.stderr[:500])  # Print first 500 chars of error
            return None
        
        # Extract PR-AUC from output
        output_text = result.stdout + result.stderr
        pr_auc_macro, pr_auc_micro = extract_pr_auc_from_output(output_text)
        
        if pr_auc_macro is None:
            print(f"Warning: Could not extract PR-AUC from output for iteration {iteration}")
            # Try to extract all metrics for debugging
            all_metrics = extract_all_metrics_from_output(output_text)
            print(f"Extracted metrics: {all_metrics}")
            return None
        
        # Extract all metrics for verification
        all_metrics = extract_all_metrics_from_output(output_text)
        all_metrics['pr_auc_macro'] = pr_auc_macro
        if pr_auc_micro:
            all_metrics['pr_auc_micro'] = pr_auc_micro
        
        return all_metrics
        
    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out for iteration {iteration}")
        return None
    except Exception as e:
        print(f"Error evaluating iteration {iteration}: {e}")
        return None

def plot_pr_auc_curve(iterations, pr_auc_values, model_name, output_file):
    """Plot PR-AUC over training iterations."""
    plt.figure(figsize=(12, 8))
    
    plt.plot(iterations, pr_auc_values, 
             marker='o', 
             linewidth=2.5, 
             markersize=8,
             label=f'{model_name}',
             color='#2ca02c' if 'DeepGait' in model_name else '#1f77b4')
    
    plt.xlabel('Training Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('PR-AUC (Macro)', fontsize=14, fontweight='bold')
    plt.title(f'PR-AUC Over Training Iterations - {model_name}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add value annotations
    for i, (iter_val, auc_val) in enumerate(zip(iterations, pr_auc_values)):
        if i % 3 == 0 or i == len(iterations) - 1:  # Annotate every 3rd point and last point
            plt.annotate(f'{auc_val:.3f}', 
                        (iter_val, auc_val),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PR-AUC curve to: {output_file}")
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-evaluate checkpoints and create PR-AUC curves')
    parser.add_argument('--device', type=str, default='0,1', help='GPU devices (e.g., "0,1")')
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes')
    parser.add_argument('--swingait_config', type=str,
                       default='configs/swingait/swin_part4_deepgaitv2_comparison.yaml',
                       help='SwinGait config file')
    parser.add_argument('--deepgait_config', type=str,
                       default='configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml',
                       help='DeepGaitV2 config file')
    
    args = parser.parse_args()
    
    # Model configurations
    models = [
        {
            'name': 'SwinGait_M1',
            'config': args.swingait_config,
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4_swingait'
        },
        {
            'name': 'DeepGaitV2_M6',
            'config': args.deepgait_config,
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights'
        }
    ]
    
    all_results = {}
    
    for model in models:
        print(f"\n{'='*70}")
        print(f"Processing {model['name']}")
        print(f"{'='*70}")
        
        exp_dir = model['exp_dir']
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            continue
        
        # Find checkpoints
        checkpoints = find_checkpoints(exp_dir, step=500)
        if not checkpoints:
            print(f"No checkpoints found in {exp_dir}")
            continue
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        # Evaluate each checkpoint
        iterations = []
        pr_aucs = []
        all_metrics_list = []
        
        for checkpoint_path, iteration in checkpoints:
            metrics = evaluate_checkpoint(
                model['config'],
                iteration,
                model['name'],
                args.device,
                args.nproc
            )
            
            if metrics and 'pr_auc_macro' in metrics:
                iterations.append(iteration)
                pr_aucs.append(metrics['pr_auc_macro'])
                all_metrics_list.append(metrics)
                print(f"Iteration {iteration}: PR-AUC (macro) = {metrics['pr_auc_macro']:.4f}")
            else:
                print(f"Failed to get PR-AUC for iteration {iteration}")
        
        if not iterations:
            print(f"No valid PR-AUC values extracted for {model['name']}")
            continue
        
        # Plot PR-AUC curve
        output_file = f"pr_auc_curve_{model['name']}.png"
        plot_pr_auc_curve(iterations, pr_aucs, model['name'], output_file)
        
        all_results[model['name']] = {
            'iterations': iterations,
            'pr_aucs': pr_aucs,
            'all_metrics': all_metrics_list
        }
    
    # Create comparison plot
    if len(all_results) == 2:
        plt.figure(figsize=(12, 8))
        
        colors = {'SwinGait_M1': '#1f77b4', 'DeepGaitV2_M6': '#2ca02c'}
        
        for model_name, result in all_results.items():
            plt.plot(result['iterations'], result['pr_aucs'],
                    marker='o',
                    linewidth=2.5,
                    markersize=8,
                    label=model_name,
                    color=colors.get(model_name, '#000000'))
        
        plt.xlabel('Training Iteration', fontsize=14, fontweight='bold')
        plt.ylabel('PR-AUC (Macro)', fontsize=14, fontweight='bold')
        plt.title('PR-AUC Over Training Iterations - Comparison', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        comparison_file = 'pr_auc_comparison_part4.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to: {comparison_file}")
        plt.close()
    
    # Save results to JSON
    json_file = 'pr_auc_curves_data.json'
    json_data = {}
    for model_name, result in all_results.items():
        json_data[model_name] = {
            'iterations': result['iterations'],
            'pr_aucs': result['pr_aucs']
        }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved data to: {json_file}")
    
    # Print verification summary
    print(f"\n{'='*70}")
    print("VERIFICATION: PR-AUC Summary (Final Checkpoint)")
    print(f"{'='*70}")
    for model_name, result in all_results.items():
        if result['iterations']:
            final_iter = result['iterations'][-1]
            final_pr_auc = result['pr_aucs'][-1]
            print(f"{model_name} (iteration {final_iter}): PR-AUC (macro) = {final_pr_auc:.4f}")

if __name__ == '__main__':
    main()

