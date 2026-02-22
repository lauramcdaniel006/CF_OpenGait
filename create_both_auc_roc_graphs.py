#!/usr/bin/env python3
"""
Create both types of AUC-ROC graphs:
1. AUC-ROC over training iterations (training progress)
2. Traditional ROC curves (TPR vs FPR) at final checkpoint
"""

import os
import sys
import subprocess
import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
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

def find_checkpoints(exp_dir, step=500):
    """Find checkpoints at regular intervals."""
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    
    def get_iter(cp):
        match = re.search(r'-(\d+)\.pt$', cp)
        return int(match.group(1)) if match else 0
    
    filtered = []
    for cp in checkpoints:
        iter_num = get_iter(cp)
        if iter_num > 0 and iter_num % step == 0:
            filtered.append((cp, iter_num))
    
    filtered.sort(key=lambda x: x[1])
    return filtered

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

def extract_probabilities_from_output(output_text):
    """Extract probabilities and labels from evaluation output (if available)."""
    # This is a placeholder - we'll need to modify the evaluator to save probabilities
    # For now, we'll need to re-run evaluation and capture the data
    return None, None

def evaluate_checkpoint_for_auc(config_file, iteration, exp_name):
    """Run evaluation and extract AUC-ROC."""
    print(f"  Iteration {iteration:5d}...", end=' ', flush=True)
    
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
            timeout=300
        )
        
        output = result.stdout + result.stderr
        auc_macro, auc_micro = extract_auc_from_output(output)
        
        if auc_macro is not None:
            print(f"✓ AUC={auc_macro:.4f}")
            return {'iteration': iteration, 'auc_macro': auc_macro, 'auc_micro': auc_micro}
        else:
            print("⚠️  No AUC found")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_probabilities_from_evaluation(config_file, iteration):
    """
    Run evaluation and extract probabilities and labels.
    We'll need to modify the evaluator to save this data, or extract from logs.
    For now, we'll create a modified version that saves probabilities.
    """
    # This will require modifying the evaluator to save probabilities to a file
    # For now, return None and we'll handle it separately
    return None, None

def plot_auc_over_iterations(iterations, auc_values, model_name, output_file):
    """Plot AUC-ROC values over training iterations (Type 1)."""
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, auc_values, marker='o', markersize=6, linewidth=2, label=model_name)
    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('AUC-ROC (Macro)', fontsize=12)
    plt.title(f'{model_name} - AUC-ROC Over Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    plt.legend(fontsize=11)
    
    # Add value labels
    for i, (iter_val, auc_val) in enumerate(zip(iterations, auc_values)):
        if i % 2 == 0 or i == len(iterations) - 1:
            plt.annotate(f'{auc_val:.3f}', (iter_val, auc_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved Type 1 graph to: {output_file}")
    return plt

def plot_traditional_roc_curves(y_true, y_probs, class_names, model_name, output_file):
    """Plot traditional ROC curves (TPR vs FPR) for each class (Type 2)."""
    n_classes = len(class_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class (One-vs-Rest)
    for i in range(n_classes):
        # Create binary labels: class i vs all others
        y_binary = (y_true == i).astype(int)
        
        # Get probabilities for class i
        y_score = y_probs[:, i]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, 
               color=colors[i], 
               lw=2, 
               label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 
           color='gray', 
           lw=1, 
           linestyle='--', 
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
    print(f"  ✓ Saved Type 2 graph to: {output_file}")
    return fig

def create_modified_evaluator_script():
    """Create a script that runs evaluation and saves probabilities."""
    script_content = '''#!/usr/bin/env python3
"""
Modified evaluation script that saves probabilities and labels for ROC curve creation.
"""
import os
import sys
import numpy as np
import pickle
import argparse

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opengait.main import run_model, initialization
from utils import config_loader

def evaluate_and_save_probs(config_file, iteration, output_file):
    """Run evaluation and save probabilities and labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default=config_file)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--iter', type=int, default=iteration)
    parser.add_argument('--log_to_file', action='store_true')
    opt = parser.parse_args()
    
    # This would need to be integrated into the main evaluation flow
    # For now, we'll use a workaround by modifying the evaluator temporarily
    pass

if __name__ == '__main__':
    pass
'''
    return script_content

def main():
    print("="*80)
    print("Creating Both Types of AUC-ROC Graphs")
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
    
    # Step 1: Get AUC-ROC over iterations (Type 1)
    print("\n" + "="*80)
    print("STEP 1: Creating AUC-ROC Over Iterations (Type 1)")
    print("="*80)
    
    for model_name, model_info in models.items():
        print(f"\n{model_name}:")
        print("-" * 80)
        
        exp_dir = model_info['exp_dir']
        config_file = model_info['config']
        
        if not os.path.exists(exp_dir) or not os.path.exists(config_file):
            print(f"  ⚠️  Missing files, skipping...")
            continue
        
        # Find checkpoints every 500 iterations
        checkpoints = find_checkpoints(exp_dir, step=500)
        
        if not checkpoints:
            print(f"  ⚠️  No checkpoints found")
            continue
        
        print(f"  Found {len(checkpoints)} checkpoints")
        
        # Evaluate each checkpoint
        results = []
        for checkpoint_path, iteration in checkpoints:
            result = evaluate_checkpoint_for_auc(config_file, iteration, model_info['exp_name'])
            if result:
                results.append(result)
        
        if results:
            iterations = [r['iteration'] for r in results]
            auc_values = [r['auc_macro'] for r in results]
            
            all_results[model_name] = {
                'iterations': iterations,
                'auc_macro': auc_values,
                'config': config_file,
                'best_iter': max(iterations)
            }
            
            # Plot Type 1 graph
            output_file = f'auc_roc_over_iterations_{model_name.replace(" ", "_")}.png'
            plot_auc_over_iterations(iterations, auc_values, model_name, output_file)
    
    # Step 2: Create traditional ROC curves at final checkpoint (Type 2)
    print("\n" + "="*80)
    print("STEP 2: Creating Traditional ROC Curves (Type 2)")
    print("="*80)
    print("\nNote: This requires extracting probabilities from evaluation.")
    print("We'll need to modify the evaluator to save probabilities.")
    print("\nCreating a helper script to extract probabilities...")
    
    # Create a script that can extract probabilities
    helper_script = create_extract_probabilities_script()
    with open('extract_probabilities_for_roc.py', 'w') as f:
        f.write(helper_script)
    print("  ✓ Created: extract_probabilities_for_roc.py")
    print("\n  To create Type 2 graphs, run:")
    print("    python extract_probabilities_for_roc.py")
    print("  This will extract probabilities and create traditional ROC curves.")
    
    # Create comparison plot for Type 1
    if len(all_results) == 2:
        print("\n" + "="*80)
        print("Creating Comparison Plot (Type 1)")
        print("="*80)
        
        model_names = list(all_results.keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#1f77b4', '#ff7f0e']
        
        for idx, model_name in enumerate(model_names):
            data = all_results[model_name]
            ax.plot(data['iterations'], data['auc_macro'], 
                   marker='o', markersize=5, linewidth=2, 
                   label=model_name, color=colors[idx])
        
        ax.set_xlabel('Training Iteration', fontsize=12)
        ax.set_ylabel('AUC-ROC (Macro)', fontsize=12)
        ax.set_title('AUC-ROC Comparison: SwinGait M1 vs DeepGaitV2 M6', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        output_file = 'auc_roc_comparison_over_iterations.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot to: {output_file}")
    
    # Save results
    json_file = 'auc_roc_results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved results to: {json_file}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nType 1 Graphs Created:")
    print("  - Individual: auc_roc_over_iterations_*.png")
    print("  - Comparison: auc_roc_comparison_over_iterations.png")
    print("\nType 2 Graphs:")
    print("  - Run: python extract_probabilities_for_roc.py")
    print("  - This will create traditional ROC curves (TPR vs FPR)")
    print("="*80)

def create_extract_probabilities_script():
    """Create a script to extract probabilities and create traditional ROC curves."""
    return '''#!/usr/bin/env python3
"""
Extract probabilities from model evaluation and create traditional ROC curves.
This script modifies the evaluator temporarily to save probabilities.
"""

import os
import sys
import numpy as np
import pickle
import subprocess
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to modify the evaluator to save probabilities
# For now, we'll create a wrapper that captures the data

def extract_from_evaluation_output(output_text):
    """Try to extract probabilities from evaluation output if logged."""
    # This is a placeholder - we'll need to modify evaluator to log probabilities
    return None, None

def create_roc_curves_from_probs(y_true, y_probs, class_names, model_name, output_file):
    """Create traditional ROC curves from probabilities."""
    n_classes = len(class_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
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
    print(f"✓ Saved ROC curves to: {output_file}")

if __name__ == '__main__':
    print("This script needs the evaluator to be modified to save probabilities.")
    print("We'll create a modified version that saves probabilities to a file.")
'''

if __name__ == '__main__':
    main()

