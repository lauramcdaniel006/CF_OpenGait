#!/usr/bin/env python3
"""
Create PR-AUC curves from existing probability files (if available) or guide user to run evaluation.
This script can work with probability files saved from ROC evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path
import glob

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
    print(f"Number of samples: {len(true_ids)}")
    
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
                alpha=0.8,
                marker='o' if len(recall) <= 20 else None,  # Show markers for small datasets
                markersize=6 if len(recall) <= 20 else 0)
    
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
    ax.set_title(f'Precision-Recall Curves - {model_name}\n(Test Set: {len(true_ids)} samples)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add macro PR-AUC text
    ax.text(0.02, 0.98, f'Macro PR-AUC: {macro_pr_auc:.3f}\nTest Samples: {len(true_ids)}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add note about small dataset if applicable
    if len(true_ids) < 50:
        ax.text(0.98, 0.02, 'Note: Curves appear stepped due to small test set',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                style='italic',
                alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PR curves to: {output_file}")
    print(f"Macro PR-AUC: {macro_pr_auc:.4f}")
    print(f"Per-class PR-AUC: {dict(zip(class_names, pr_aucs))}")
    
    plt.close()
    
    return macro_pr_auc, pr_aucs

def find_probability_files():
    """Find existing probability files."""
    prob_dir = 'output/roc_probabilities'
    if not os.path.exists(prob_dir):
        return []
    
    prob_files = glob.glob(os.path.join(prob_dir, '*.npz'))
    return sorted(prob_files)

def main():
    print("="*70)
    print("PR-AUC Curve Generator")
    print("="*70)
    
    # Check for existing probability files
    prob_files = find_probability_files()
    
    if not prob_files:
        print("\nNo existing probability files found.")
        print("\nTo generate PR-AUC curves, you need to:")
        print("1. Run evaluation with probability saving enabled")
        print("2. Use the script: extract_probabilities_for_pr_auc.py")
        print("\nOr run evaluation manually:")
        print("  python opengait/main.py --cfgs <config> --phase test --iter <iteration>")
        print("  (with SAVE_PROBS_FOR_ROC=1 environment variable)")
        return
    
    print(f"\nFound {len(prob_files)} probability file(s):")
    for pf in prob_files:
        print(f"  - {pf}")
    
    # Try to identify which model each file belongs to
    # For now, create PR curves for all found files
    results = {}
    
    for prob_file in prob_files:
        # Try to infer model name from filename or directory
        filename = os.path.basename(prob_file)
        
        # Check if it's from a specific model evaluation
        if 'swingait' in prob_file.lower() or 'swin' in filename.lower():
            model_name = 'SwinGait_M1'
        elif 'deepgait' in prob_file.lower() or 'deep' in filename.lower():
            model_name = 'DeepGaitV2_M6'
        else:
            # Use iteration number or generic name
            model_name = f"Model_{filename.replace('.npz', '').replace('probs_iter_', '')}"
        
        print(f"\n{'='*70}")
        print(f"Processing {model_name}")
        print(f"{'='*70}")
        
        try:
            output_file = f"pr_curves_{model_name}.png"
            macro_pr_auc, pr_aucs = create_pr_curves_from_file(
                prob_file,
                model_name,
                output_file
            )
            
            results[model_name] = {
                'macro_pr_auc': macro_pr_auc,
                'pr_aucs': pr_aucs,
                'prob_file': prob_file
            }
        except Exception as e:
            print(f"Error processing {prob_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if results:
        print(f"\n{'='*70}")
        print("PR-AUC SUMMARY")
        print(f"{'='*70}")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Macro PR-AUC: {result['macro_pr_auc']:.4f}")
            print(f"  Per-class PR-AUC: {dict(zip(['Frail', 'Prefrail', 'Nonfrail'], result['pr_aucs']))}")
            print(f"  Source: {result['prob_file']}")
    else:
        print("\nNo PR curves were generated. Please run evaluation first.")

if __name__ == '__main__':
    main()

