#!/usr/bin/env python3
"""
Extract probabilities from model evaluation and create traditional ROC curves.
This script temporarily modifies the evaluator to save probabilities and labels.
"""

import os
import sys
import numpy as np
import pickle
import subprocess
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shutil

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
    
    # Read current evaluator
    with open(evaluator_path, 'r') as f:
        content = f.read()
    
    # Check if already modified
    if 'SAVE_PROBS_FOR_ROC' in content:
        print("Evaluator already modified for probability saving.")
        return True
    
    # Find the location to add probability saving
    # Look for where probs are computed
    if 'probs = exp_logits' in content:
        # Add code to save probabilities
        save_code = '''
    # SAVE_PROBS_FOR_ROC: Save probabilities for ROC curve creation
    if os.environ.get('SAVE_PROBS_FOR_ROC', '0') == '1':
        save_dir = os.path.join('output', 'roc_probabilities')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'probs_iter_{model.iteration if hasattr(model, "iteration") else "final"}.npz')
        np.savez(save_file, 
                probs=probs, 
                true_ids=true_ids, 
                pred_ids=pred_ids,
                class_names=class_names)
        msg_mgr.log_info(f"Saved probabilities to: {save_file}")
'''
        
        # Insert after probs calculation
        content = content.replace(
            'probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # [n_samples, n_classes]',
            'probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # [n_samples, n_classes]' + save_code
        )
        
        # Write modified evaluator
        with open(evaluator_path, 'w') as f:
            f.write(content)
        
        print("Modified evaluator to save probabilities.")
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
    env['CUDA_VISIBLE_DEVICES'] = device
    # Get the Python interpreter from the current environment (needed for LD_LIBRARY_PATH)
    python_cmd = sys.executable
    
    # Ensure PYTHONPATH includes OpenGait root (where opengait package is)
    opengait_root = os.path.dirname(os.path.abspath(__file__))
    opengait_dir = os.path.join(opengait_root, 'opengait')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = f"{opengait_dir}:{opengait_root}"
    # Ensure LD_LIBRARY_PATH includes conda environment lib directory (prepend for priority)
    python_lib_dir = os.path.join(os.path.dirname(python_cmd), '..', 'lib')
    python_lib_dir = os.path.abspath(python_lib_dir)
    if 'LD_LIBRARY_PATH' in env:
        # Prepend to prioritize conda environment libraries
        env['LD_LIBRARY_PATH'] = f"{python_lib_dir}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = python_lib_dir
    
    # Use python -m torch.distributed.launch (same as training script)
    cmd = [
        python_cmd, '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',
        '--iter', str(iteration)
    ]
    
    try:
        # Get the OpenGait root directory (where this script is located)
        opengait_root = os.path.dirname(os.path.abspath(__file__))
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=opengait_root,  # Run from OpenGait root directory
            timeout=300
        )
        
        # Find saved probability file
        save_dir = os.path.join('output', 'roc_probabilities')
        prob_files = [f for f in os.listdir(save_dir) if f.endswith('.npz')]
        
        if prob_files:
            # Get most recent
            prob_file = max([os.path.join(save_dir, f) for f in prob_files], 
                          key=os.path.getmtime)
            return prob_file
        else:
            print("  ⚠️  No probability file found")
            return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def create_roc_curves_from_file(prob_file, model_name, output_file):
    """Load probabilities and create ROC curves."""
    data = np.load(prob_file)
    probs = data['probs']
    true_ids = data['true_ids']
    class_names = data['class_names']
    
    n_classes = len(class_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        y_binary = (true_ids == i).astype(int)
        y_score = probs[:, i]
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
               label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
           label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved ROC curves to: {output_file}")
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract probabilities and create ROC curves')
    parser.add_argument('--device', type=str, default='0,1',
                       help='CUDA devices to use (e.g., "0,1" or "0")')
    parser.add_argument('--nproc', type=int, default=2,
                       help='Number of processes (should match number of GPUs)')
    args = parser.parse_args()
    
    print("="*80)
    print("Creating Traditional ROC Curves (Type 2)")
    print(f"Using CUDA devices: {args.device}")
    print(f"Number of processes: {args.nproc}")
    print("="*80)
    
    models = {
        'SwinGait M1': {
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4_swingait/SwinGait/REDO_Frailty_ccpg_pt4_swingait',
            'config': 'configs/swingait/swin_part4_deepgaitv2_comparison.yaml'
        },
        'DeepGaitV2 M6': {
            'exp_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights',
            'config': 'configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml'
        }
    }
    
    # Modify evaluator to save probabilities
    print("\n1. Modifying evaluator to save probabilities...")
    if not modify_evaluator_to_save_probs():
        print("  ⚠️  Could not modify evaluator")
        return
    
    # Find best checkpoints and evaluate
    print("\n2. Evaluating models at final checkpoint...")
    for model_name, model_info in models.items():
        exp_dir = model_info['exp_dir']
        config_file = model_info['config']
        
        if not os.path.exists(exp_dir) or not os.path.exists(config_file):
            print(f"  ⚠️  Missing files for {model_name}")
            continue
        
        # Find best checkpoint
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        if not checkpoints:
            print(f"  ⚠️  No checkpoints for {model_name}")
            continue
        
        # Get highest iteration
        def get_iter(cp):
            match = re.search(r'-(\d+)\.pt$', cp)
            return int(match.group(1)) if match else 0
        
        best_checkpoint = max(checkpoints, key=get_iter)
        best_iter = get_iter(best_checkpoint)
        
        # Evaluate and save probabilities
        prob_file = evaluate_and_save_probs(config_file, best_iter, model_name,
                                           device=args.device, nproc=args.nproc)
        
        if prob_file:
            # Create ROC curves
            output_file = f'roc_curves_{model_name.replace(" ", "_")}.png'
            create_roc_curves_from_file(prob_file, model_name, output_file)
    
    # Restore original evaluator
    print("\n3. Restoring original evaluator...")
    restore_evaluator()
    
    print("\n" + "="*80)
    print("Done! Traditional ROC curves created.")
    print("="*80)

if __name__ == '__main__':
    main()

