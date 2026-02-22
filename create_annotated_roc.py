#!/usr/bin/env python3
"""
Create an annotated ROC curve plot that clearly shows where ROC and AUC are.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def create_annotated_roc(prob_file, model_name, output_file):
    """Create ROC curve with clear annotations showing ROC and AUC."""
    data = np.load(prob_file)
    probs = data['probs']
    true_ids = data['true_ids']
    class_names = data['class_names']
    
    n_classes = len(class_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot ROC curves
    for i in range(n_classes):
        y_binary = (true_ids == i).astype(int)
        y_score = probs[:, i]
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve (THIS IS THE ROC)
        line = ax.plot(fpr, tpr, color=colors[i], lw=3, 
                      label=f'{class_names[i]} (AUC = {roc_auc:.3f})',
                      marker='o', markersize=6)
        
        # Fill area under curve to visualize AUC
        ax.fill_between(fpr, 0, tpr, alpha=0.2, color=colors[i])
        
        # Add annotation pointing to the curve
        mid_idx = len(fpr) // 2
        if mid_idx < len(fpr):
            ax.annotate(f'ROC Curve\n(AUC={roc_auc:.3f})', 
                       xy=(fpr[mid_idx], tpr[mid_idx]),
                       xytext=(fpr[mid_idx] + 0.15, tpr[mid_idx] + 0.15),
                       fontsize=10, color=colors[i],
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=2),
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
           label='Random Classifier (AUC = 0.500)')
    
    # Add text box explaining ROC and AUC
    textstr = ('ROC = The colored lines (TPR vs FPR)\n'
               'AUC = Area Under Curve (shown in legend)')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - ROC Curves (Annotated)', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved annotated ROC curve to: {output_file}")
    return fig

if __name__ == '__main__':
    # Find probability files
    prob_dir = 'output/roc_probabilities'
    
    if not os.path.exists(prob_dir):
        print(f"Error: {prob_dir} does not exist")
        exit(1)
    
    # Check for DeepGaitV2 M6
    prob_file = os.path.join(prob_dir, 'probs_iter_final.npz')
    
    if os.path.exists(prob_file):
        # Try to determine model from context (you may need to adjust this)
        # For now, create annotated version
        create_annotated_roc(prob_file, 'DeepGaitV2 M6', 
                            'roc_curves_annotated_DeepGaitV2_M6.png')
        print("\n" + "="*70)
        print("ANNOTATED PLOT CREATED!")
        print("="*70)
        print("The new plot shows:")
        print("  • ROC curves with filled areas (showing AUC visually)")
        print("  • Arrows pointing to each ROC curve")
        print("  • Text box explaining what ROC and AUC are")
        print("  • Legend with AUC values clearly labeled")
        print("="*70)
    else:
        print(f"Error: {prob_file} not found")

