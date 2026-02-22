#!/usr/bin/env python3
"""
Re-evaluate all checkpoints every 500 iterations to compute AUC-ROC curves.
This script evaluates checkpoints at 500, 1000, 1500, ... iterations and extracts AUC-ROC.
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

def extract_auc_from_output(output_text):
    """Extract AUC-ROC values from evaluation output."""
    auc_macro = None
    auc_micro = None
    
    # Try multiple patterns
    patterns = [
        (r'ROC AUC \(macro\):\s*(\d+\.\d+)', 'ROC AUC (macro):'),
        (r'ROC AUC \(micro\):\s*(\d+\.\d+)', 'ROC AUC (micro):'),
        (r'ROC AUC.*macro.*?(\d+\.\d+)', 'ROC AUC macro'),
        (r'ROC AUC.*micro.*?(\d+\.\d+)', 'ROC AUC micro'),
        (r'test_auc_macro.*?(\d+\.\d+)', 'test_auc_macro'),
        (r'test_auc_micro.*?(\d+\.\d+)', 'test_auc_micro'),
    ]
    
    for line in output_text.split('\n'):
        # Check for macro AUC
        if 'ROC AUC (macro)' in line or 'test_auc_macro' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                auc_macro = float(match.group(1))
        # Check for micro AUC
        elif 'ROC AUC (micro)' in line or 'test_auc_micro' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                auc_micro = float(match.group(1))
    
    return auc_macro, auc_micro

def extract_all_metrics_from_output(output_text):
    """Extract all metrics (accuracy, precision, recall, F1, AUC-ROC) for verification."""
    metrics = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'auc_macro': None,
        'auc_micro': None
    }
    
    for line in output_text.split('\n'):
        # Accuracy
        if 'Overall Accuracy' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                metrics['accuracy'] = float(match.group(1)) / 100.0
        
        # Precision (macro)
        if 'test_precision/' in line.lower() and 'macro' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['precision'] = float(match.group(1))
        
        # Recall (macro)
        if 'test_recall/' in line.lower() and 'macro' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['recall'] = float(match.group(1))
        
        # F1 (macro)
        if 'test_f1/' in line.lower() and 'macro' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['f1'] = float(match.group(1))
        
        # AUC-ROC
        if 'ROC AUC (macro)' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['auc_macro'] = float(match.group(1))
        elif 'ROC AUC (micro)' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                metrics['auc_micro'] = float(match.group(1))
    
    return metrics

def evaluate_checkpoint(config_file, iteration, exp_name, device='0,1', nproc=2):
    """Run evaluation on a checkpoint and extract AUC-ROC."""
    print(f"  Iteration {iteration:5d}...", end=' ', flush=True)
    
    # Use torchrun for multi-GPU or single GPU
    # Get the Python interpreter from the current environment
    python_cmd = sys.executable  # Use the same Python that's running this script
    
    if nproc > 1:
        # Use python -m torch.distributed.launch (same as training script)
        # This matches the format used in train.sh
        cmd = [
            python_cmd, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={nproc}',
            'opengait/main.py',
            '--cfgs', config_file,
            '--phase', 'test',
            '--iter', str(iteration)
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = device
        # Ensure PYTHONPATH includes OpenGait root (where opengait package is)
        opengait_root = os.path.dirname(os.path.abspath(__file__))
        opengait_dir = os.path.join(opengait_root, 'opengait')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}"
        # Ensure LD_LIBRARY_PATH includes conda environment lib directory
        python_lib_dir = os.path.join(os.path.dirname(python_cmd), '..', 'lib')
        python_lib_dir = os.path.abspath(python_lib_dir)
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{python_lib_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = python_lib_dir
    else:
        # Single GPU
        cmd = [
            python_cmd, 'opengait/main.py',
            '--cfgs', config_file,
            '--phase', 'test',
            '--iter', str(iteration)
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = device.split(',')[0] if ',' in device else device
        # Ensure PYTHONPATH includes OpenGait root (where opengait package is)
        opengait_root = os.path.dirname(os.path.abspath(__file__))
        opengait_dir = os.path.join(opengait_root, 'opengait')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}"
        # Ensure LD_LIBRARY_PATH includes conda environment lib directory
        python_lib_dir = os.path.join(os.path.dirname(python_cmd), '..', 'lib')
        python_lib_dir = os.path.abspath(python_lib_dir)
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{python_lib_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = python_lib_dir
    
    try:
        # Get the OpenGait root directory (where this script is located)
        opengait_root = os.path.dirname(os.path.abspath(__file__))
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=opengait_root,  # Run from OpenGait root directory
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Debug: Check if evaluation ran successfully
        if result.returncode != 0:
            print(f"❌ Evaluation failed (code {result.returncode})")
            # Print more error details
            error_lines = output.split('\n')
            # Find the actual error message
            error_start = None
            for i, line in enumerate(error_lines):
                if 'Error' in line or 'Traceback' in line or 'File' in line:
                    error_start = max(0, i - 2)
                    break
            if error_start is not None:
                print(f"   Error details:")
                for line in error_lines[error_start:error_start+15]:
                    if line.strip():
                        print(f"   {line}")
            else:
                # Print last 10 lines if we can't find error
                print(f"   Last error lines:")
                for line in error_lines[-10:]:
                    if line.strip():
                        print(f"   {line}")
            return None
        
        auc_macro, auc_micro = extract_auc_from_output(output)
        all_metrics = extract_all_metrics_from_output(output)
        
        if auc_macro is not None:
            print(f"✓ AUC={auc_macro:.4f}", end='')
            if all_metrics['accuracy'] is not None:
                print(f" Acc={all_metrics['accuracy']:.4f}", end='')
            print()
            return {
                'iteration': iteration, 
                'auc_macro': auc_macro, 
                'auc_micro': auc_micro,
                'accuracy': all_metrics['accuracy'],
                'precision': all_metrics['precision'],
                'recall': all_metrics['recall'],
                'f1': all_metrics['f1']
            }
        else:
            # Debug: Check what's in the output
            if 'ROC AUC' in output or 'auc' in output.lower():
                # Try more flexible parsing
                print("⚠️  Trying alternative parsing...", end=' ')
                # Look for any line with ROC AUC and a number
                for line in output.split('\n'):
                    if 'ROC AUC' in line and 'macro' in line.lower():
                        # Try multiple patterns
                        patterns = [
                            r'ROC AUC.*?(\d+\.\d{4})',
                            r'ROC AUC.*?(\d+\.\d{3})',
                            r'ROC AUC.*?(\d+\.\d{2})',
                            r'ROC AUC.*?(\d+\.\d+)',
                            r':\s*(\d+\.\d+)',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, line)
                            if match:
                                try:
                                    auc_macro = float(match.group(1))
                                    print(f"✓ Found AUC={auc_macro:.4f}")
                                    return {
                                        'iteration': iteration,
                                        'auc_macro': auc_macro,
                                        'auc_micro': auc_micro,
                                        'accuracy': all_metrics.get('accuracy'),
                                        'precision': all_metrics.get('precision'),
                                        'recall': all_metrics.get('recall'),
                                        'f1': all_metrics.get('f1')
                                    }
                                except:
                                    continue
                print("⚠️  Could not parse AUC")
                # Print a sample of the output for debugging
                auc_lines = [l for l in output.split('\n') if 'ROC AUC' in l or 'auc' in l.lower()]
                if auc_lines:
                    print(f"   Sample lines: {auc_lines[:3]}")
            else:
                print("⚠️  No AUC found (evaluation may not have run scoliosis evaluator)")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def plot_auc_curve(iterations, auc_values, model_name, output_file):
    """Plot AUC-ROC curve."""
    if not iterations or not auc_values:
        return None
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, auc_values, marker='o', markersize=6, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('AUC-ROC (Macro)', fontsize=12)
    plt.title(f'{model_name} - AUC-ROC vs Iteration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    # Add value labels on points
    for i, (iter_val, auc_val) in enumerate(zip(iterations, auc_values)):
        if i % 2 == 0 or i == len(iterations) - 1:  # Label every other point or last
            plt.annotate(f'{auc_val:.3f}', (iter_val, auc_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot to: {output_file}")
    return plt

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-evaluate checkpoints for AUC-ROC curves')
    parser.add_argument('--device', type=str, default='0,1',
                       help='CUDA devices to use (e.g., "0,1" or "0")')
    parser.add_argument('--nproc', type=int, default=2,
                       help='Number of processes (should match number of GPUs)')
    args = parser.parse_args()
    
    print("="*80)
    print("Re-evaluating Checkpoints for AUC-ROC Curves")
    print(f"Using CUDA devices: {args.device}")
    print(f"Number of processes: {args.nproc}")
    print("="*80)
    
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
    
    all_results = {}
    
    for model_name, model_info in models.items():
        print(f"\n{model_name}:")
        print("-" * 80)
        
        exp_dir = model_info['exp_dir']
        config_file = model_info['config']
        
        if not os.path.exists(exp_dir):
            print(f"  ⚠️  Experiment directory not found: {exp_dir}")
            continue
        
        if not os.path.exists(config_file):
            print(f"  ⚠️  Config file not found: {config_file}")
            continue
        
        # Find checkpoints every 500 iterations
        checkpoints = find_checkpoints(exp_dir, step=500)
        
        if not checkpoints:
            print(f"  ⚠️  No checkpoints found")
            continue
        
        print(f"  Found {len(checkpoints)} checkpoints (every 500 iterations)")
        
        # Evaluate each checkpoint
        results = []
        for checkpoint_path, iteration in checkpoints:
            result = evaluate_checkpoint(config_file, iteration, model_info['exp_name'],
                                       device=args.device, nproc=args.nproc)
            if result:
                results.append(result)
        
        if results:
            iterations = [r['iteration'] for r in results]
            auc_values = [r['auc_macro'] for r in results]
            
            all_results[model_name] = {
                'iterations': iterations,
                'auc_macro': auc_values,
                'auc_micro': [r.get('auc_micro', None) for r in results],
                'accuracy': [r.get('accuracy', None) for r in results],
                'precision': [r.get('precision', None) for r in results],
                'recall': [r.get('recall', None) for r in results],
                'f1': [r.get('f1', None) for r in results]
            }
            
            # Plot individual curve
            output_file = f'auc_roc_curve_{model_name.replace(" ", "_")}.png'
            plot_auc_curve(iterations, auc_values, model_name, output_file)
        else:
            print(f"  ⚠️  No AUC-ROC values extracted")
    
    # Plot comparison if both models have data
    if len(all_results) == 2:
        print("\n" + "="*80)
        print("Creating Comparison Plot")
        print("="*80)
        
        model_names = list(all_results.keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, model_name in enumerate(model_names):
            data = all_results[model_name]
            ax.plot(data['iterations'], data['auc_macro'], 
                   marker='o', markersize=5, linewidth=2, label=model_name)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('AUC-ROC (Macro)', fontsize=12)
        ax.set_title('AUC-ROC Comparison: SwinGait M1 vs DeepGaitV2 M6', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        output_file = 'auc_roc_comparison_part4.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {output_file}")
    
    # Save results to JSON
    json_file = 'auc_roc_curves_data.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved data to: {json_file}")
    
    # Verification: Print summary of metrics for final checkpoint
    print("\n" + "="*80)
    print("VERIFICATION: Metrics Summary (Final Checkpoint)")
    print("="*80)
    for model_name, model_data in all_results.items():
        if model_data['iterations']:
            final_idx = -1
            final_iter = model_data['iterations'][final_idx]
            final_auc = model_data['auc_macro'][final_idx]
            print(f"\n{model_name} (Iteration {final_iter}):")
            print(f"  AUC-ROC (macro): {final_auc:.4f}")
            # Check if we have other metrics
            if 'accuracy' in model_data and model_data['accuracy']:
                print(f"  Accuracy: {model_data['accuracy'][final_idx]:.4f}")
            if 'precision' in model_data and model_data['precision']:
                print(f"  Precision: {model_data['precision'][final_idx]:.4f}")
            if 'recall' in model_data and model_data['recall']:
                print(f"  Recall: {model_data['recall'][final_idx]:.4f}")
            if 'f1' in model_data and model_data['f1']:
                print(f"  F1 Score: {model_data['f1'][final_idx]:.4f}")
            print(f"\n  ✅ These metrics should match your training evaluation results!")
            print(f"  ✅ Same weights → Same predictions → Same metrics")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    print("\n📊 SUMMARY:")
    print("  Type 1 Graph: AUC-ROC over ALL checkpoints (500, 1000, 1500, ...)")
    print("  - Each checkpoint evaluated separately")
    print("  - One AUC-ROC value per checkpoint")
    print("  - Shows training progression")
    print("="*80)

if __name__ == '__main__':
    main()

