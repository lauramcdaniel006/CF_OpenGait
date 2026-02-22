#!/usr/bin/env python3
"""
Create ROC curves for frailty classification models.
This script can create:
1. Traditional ROC curves (TPR vs FPR) for each class
2. AUC-ROC over training iterations (from evaluation data)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import sys

def plot_traditional_roc_curve(y_true, y_probs, class_names, output_file='roc_curves.png'):
    """
    Create traditional ROC curves (TPR vs FPR) for each class.
    
    Args:
        y_true: True labels (0, 1, 2)
        y_probs: Probability matrix [n_samples, n_classes]
        class_names: List of class names
        output_file: Output filename
    """
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
    ax.set_title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved traditional ROC curves to: {output_file}")
    return fig

def plot_auc_over_iterations(iterations, auc_values, model_name, output_file):
    """
    Plot AUC-ROC values over training iterations.
    This is what your existing scripts create.
    """
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
    print(f"✓ Saved AUC over iterations to: {output_file}")
    return plt

def explain_roc_curve_creation():
    """
    Explain how ROC curves are created step-by-step.
    """
    print("="*80)
    print("HOW ROC CURVES ARE CREATED")
    print("="*80)
    print("\n1. TRADITIONAL ROC CURVE (TPR vs FPR):")
    print("   - For each class (Frail, Prefrail, Nonfrail):")
    print("     a. Treat that class as 'positive', others as 'negative'")
    print("     b. Test many thresholds (0.0, 0.1, 0.2, ..., 1.0)")
    print("     c. For each threshold:")
    print("        - Calculate True Positive Rate (TPR)")
    print("        - Calculate False Positive Rate (FPR)")
    print("        - Plot point (FPR, TPR)")
    print("     d. Connect points to form curve")
    print("     e. Calculate area under curve = AUC")
    print("\n2. AUC-ROC OVER ITERATIONS (Training Progress):")
    print("   - Evaluate model at checkpoints (500, 1000, 1500, ...)")
    print("   - Calculate AUC-ROC at each checkpoint")
    print("   - Plot: Iteration (x-axis) vs AUC-ROC (y-axis)")
    print("   - Shows how model improves during training")
    print("\n" + "="*80)

if __name__ == '__main__':
    explain_roc_curve_creation()
    
    print("\nTo create ROC curves, you need:")
    print("1. True labels (y_true)")
    print("2. Probability predictions (y_probs)")
    print("\nThese come from model evaluation.")
    print("\nYour scripts (reevaluate_for_auc_roc.py) create Type 2 graphs:")
    print("  - AUC-ROC values over training iterations")
    print("  - Shows training progress")

