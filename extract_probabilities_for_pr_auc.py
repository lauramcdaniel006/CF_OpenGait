#!/usr/bin/env python3
"""
Extract probabilities from model evaluation and create traditional PR (Precision-Recall) curves.
This script evaluates the final checkpoint and creates PR curves showing Precision vs Recall.
"""

import os
import sys
import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import shutil
from pathlib import Path

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def modify_evaluator_to_save_probs():
    """Temporarily modify evaluator to save probabilities."""
    evaluator_path = 'opengait/evaluation/evaluator.py'
    backup_path = 'opengait/evaluation/evaluator.py.backup'
    
    # Create backup
    if not os.path.exists(backup_path):
        shutil.copy(evaluator_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Check if already modified
    with open(evaluator_path, 'r') as f:
        content = f.read()
    
    if 'SAVE_PROBS_FOR_ROC' in content:
        print("Evaluator already modified for probability saving.")
        return True
    
    return False

def restore_evaluator():
    """Restore original evaluator from backup."""
    evaluator_path = 'opengait/evaluation/evaluator.py'
    backup_path = 'opengait/evaluation/evaluator.py.backup'
    
    if os.path.exists(backup_path):
        shutil.copy(backup_path, evaluator_path)
        print("Restored original evaluator.")
        return True
    return False

def evaluate_and_save_probs(config_file, iteration, model_name, device='0,1', nproc=2):
    """Run evaluation with probability saving enabled."""
    print(f"\nEvaluating {model_name} at iteration {iteration}...")
    
    # Set environment variable to enable probability saving
    env = os.environ.copy()
    env['SAVE_PROBS_FOR_ROC'] = '1'
    env['SAVE_PROBS_MODEL_NAME'] = model_name  # Add model name to distinguish files
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
    
    # Set LD_LIBRARY_PATH for Conda environment (needed for cv2 and other libraries)
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
    
    print(f"Running: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=opengait_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Error running evaluation:")
            print(result.stderr)
            return None
        
        # Find the saved probability file
        prob_dir = os.path.join(opengait_root, 'output', 'roc_probabilities')
        
        # Look for model-specific file first
        model_specific_file = os.path.join(prob_dir, f'probs_{model_name}_iter_{iteration}.npz')
        if os.path.exists(model_specific_file):
            print(f"Found model-specific probability file: {model_specific_file}")
            return model_specific_file
        
        # Fallback: look for any file with model name
        prob_files = [f for f in os.listdir(prob_dir) if model_name.lower() in f.lower() and f.endswith('.npz')]
        if prob_files:
            prob_file = os.path.join(prob_dir, prob_files[0])
            print(f"Found probability file: {prob_file}")
            return prob_file
        
        # Last resort: look for any recent file
        prob_files = [f for f in os.listdir(prob_dir) if f.startswith('probs_') and f.endswith('.npz')]
        if not prob_files:
            print(f"No probability files found in {prob_dir}")
            return None
        
        # Get the most recent file
        prob_files.sort(reverse=True)
        prob_file = os.path.join(prob_dir, prob_files[0])
        print(f"Found probability file (may be from different model): {prob_file}")
        return prob_file
        
    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def create_pr_curves_from_file(prob_file, model_name, output_file):
    """Load saved probabilities and create PR curves."""
    print(f"\nCreating PR curves from {prob_file}...")
    
    # Load probabilities
    data = np.load(prob_file, allow_pickle=True)
    probs = data['probs']
    true_ids = data['true_ids']
    class_names = data['class_names'] if 'class_names' in data else ['Frail', 'Prefrail', 'Nonfrail']
    
    print(f"Loaded probabilities shape: {probs.shape}")
    print(f"True labels shape: {true_ids.shape}")
    print(f"Class names: {class_names}")
    
    # Create PR curves for each class (one-vs-rest)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    pr_aucs = []
    
    for i, class_name in enumerate(class_names):
        # One-vs-rest: treat class i as positive, others as negative
        y_true_binary = (true_ids == i).astype(int)
        y_probs_binary = probs[:, i]
        
        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_probs_binary)
        
        # Compute PR-AUC
        pr_auc = average_precision_score(y_true_binary, y_probs_binary)
        pr_aucs.append(pr_auc)
        
        # Plot PR curve
        ax.plot(recall, precision, 
                label=f'{class_name} (PR-AUC = {pr_auc:.3f})',
                color=colors[i],
                linewidth=2.5,
                alpha=0.8)
    
    # Compute macro-averaged PR-AUC
    macro_pr_auc = np.mean(pr_aucs)
    
    # Baseline (random classifier) - for imbalanced data, baseline is class prevalence
    class_counts = np.bincount(true_ids)
    baseline_precision = class_counts / len(true_ids)
    
    # Plot baseline for each class
    for i, class_name in enumerate(class_names):
        baseline = baseline_precision[i]
        ax.axhline(y=baseline, 
                  color=colors[i], 
                  linestyle='--', 
                  alpha=0.5,
                  linewidth=1.5,
                  label=f'{class_name} Baseline ({baseline:.3f})')
    
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add macro PR-AUC text
    ax.text(0.02, 0.98, f'Macro PR-AUC: {macro_pr_auc:.3f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PR curves to: {output_file}")
    print(f"Macro PR-AUC: {macro_pr_auc:.4f}")
    print(f"Per-class PR-AUC: {dict(zip(class_names, pr_aucs))}")
    
    plt.close()
    
    return macro_pr_auc, pr_aucs

def find_best_checkpoint(exp_dir):
    """Find the best checkpoint (highest iteration)."""
    import glob
    
    # Try direct path first
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        # Try nested structure: exp_dir/ModelName/exp_name/checkpoints
        nested_dirs = glob.glob(os.path.join(exp_dir, '*', '*', 'checkpoints'))
        if nested_dirs:
            checkpoint_dir = nested_dirs[0]
        else:
            # Try recursive search
            checkpoints = glob.glob(os.path.join(exp_dir, '**', 'checkpoints', '*.pt'), recursive=True)
            if checkpoints:
                def get_iter(cp):
                    match = re.search(r'-(\d+)\.pt$', cp)
                    return int(match.group(1)) if match else 0
                best_checkpoint = max(checkpoints, key=get_iter)
                best_iter = get_iter(best_checkpoint)
                return best_checkpoint, best_iter
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PR-AUC curves from model evaluation')
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
    
    # Modify evaluator to save probabilities
    modify_evaluator_to_save_probs()
    
    try:
        results = {}
        
        for model in models:
            print(f"\n{'='*70}")
            print(f"Processing {model['name']}")
            print(f"{'='*70}")
            
            # Find best checkpoint
            exp_dir = model['exp_dir']
            if not os.path.exists(exp_dir):
                print(f"Experiment directory not found: {exp_dir}")
                continue
            
            best_checkpoint, best_iter = find_best_checkpoint(exp_dir)
            if best_checkpoint is None:
                print(f"No checkpoints found in {exp_dir}")
                continue
            
            print(f"Found best checkpoint: {best_checkpoint} (iteration {best_iter})")
            
            # Evaluate and save probabilities
            prob_file = evaluate_and_save_probs(
                model['config'],
                best_iter,
                model['name'],
                args.device,
                args.nproc
            )
            
            if prob_file is None:
                print(f"Failed to evaluate {model['name']}")
                continue
            
            # Create PR curves
            output_file = f"pr_curves_{model['name']}.png"
            macro_pr_auc, pr_aucs = create_pr_curves_from_file(
                prob_file,
                model['name'],
                output_file
            )
            
            results[model['name']] = {
                'macro_pr_auc': macro_pr_auc,
                'pr_aucs': pr_aucs
            }
        
        # Print summary
        print(f"\n{'='*70}")
        print("PR-AUC SUMMARY")
        print(f"{'='*70}")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Macro PR-AUC: {result['macro_pr_auc']:.4f}")
            print(f"  Per-class PR-AUC: {dict(zip(['Frail', 'Prefrail', 'Nonfrail'], result['pr_aucs']))}")
        
    finally:
        # Restore original evaluator
        restore_evaluator()

if __name__ == '__main__':
    main()

