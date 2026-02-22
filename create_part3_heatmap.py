#!/usr/bin/env python3
"""
Create a heatmap for Part 3 (Loss Functions) showing the impact of different loss combinations.
2x2 matrix: Rows = Metric Learning Loss (Triplet vs Contrastive), Columns = Classification Loss (CE vs Focal)
"""

import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_part3_data(csv_file):
    """Load Part 3 experiment data from CSV, including Part 1 p+CNN (triplet + CE)."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Include Part 3 experiments
            if 'Part 3' in row['Part']:
                exp_name = row['Experiment']
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['Accuracy (%)']),
                    'f1': float(row['F1']) if row['F1'] else None,
                    'precision': float(row['Precision']) if row['Precision'] else None,
                    'recall': float(row['Recall']) if row['Recall'] else None
                })
            # Also include Part 1 p+CNN (pretrained + CNN) which uses triplet + CE
            elif 'Part 1' in row['Part'] and 'p+CNN' in row['Experiment'] and 'Tintro' not in row['Experiment']:
                exp_name = row['Experiment']
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['Accuracy (%)']),
                    'f1': float(row['F1']) if row['F1'] else None,
                    'precision': float(row['Precision']) if row['Precision'] else None,
                    'recall': float(row['Recall']) if row['Recall'] else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
    return experiments

def get_best_metrics(experiments):
    """Extract best accuracy and F1 for each experiment."""
    best_metrics = {}
    
    for exp_name, data in experiments.items():
        if not data:
            continue
        
        best_acc = max(data, key=lambda x: x['accuracy'])
        best_f1_data = max([d for d in data if d['f1'] is not None], 
                          key=lambda x: x['f1'], default=None)
        
        best_metrics[exp_name] = {
            'best_accuracy': best_acc['accuracy'],
            'best_acc_iter': best_acc['iteration'],
            'best_f1': best_f1_data['f1'] if best_f1_data else None,
            'best_f1_iter': best_f1_data['iteration'] if best_f1_data else None
        }
    
    return best_metrics

def map_experiment_to_matrix(exp_name):
    """Map experiment name to matrix position (row, col)."""
    exp_lower = exp_name.lower()
    
    # Determine metric learning loss (row)
    if 'triplet' in exp_lower or 'p+cnn' in exp_lower:
        row = 0  # Triplet
    elif 'contrastive' in exp_lower:
        row = 1  # Contrastive
    else:
        return None, None
    
    # Determine classification loss (column)
    if 'focal' in exp_lower:
        col = 1  # Focal
    elif 'ce' in exp_lower or 'p+cnn' in exp_lower:
        col = 0  # CE (CrossEntropy)
    else:
        return None, None
    
    return row, col

def create_heatmap(best_metrics, metric='accuracy', output_dir='results_visualization'):
    """Create a 2x2 heatmap showing loss function combinations."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize 2x2 matrix
    matrix = np.full((2, 2), np.nan)
    matrix_labels = np.empty((2, 2), dtype=object)
    matrix_iters = np.empty((2, 2), dtype=object)
    
    # Fill matrix
    for exp_name, metrics in best_metrics.items():
        row, col = map_experiment_to_matrix(exp_name)
        if row is None or col is None:
            continue
        
        if metric == 'accuracy':
            value = metrics['best_accuracy']
            iter_val = metrics['best_acc_iter']
        else:  # f1
            value = metrics['best_f1']
            iter_val = metrics['best_f1_iter']
        
        if value is not None:
            matrix[row, col] = value
            matrix_labels[row, col] = f"{value:.2f}%"
            matrix_iters[row, col] = f"(iter {iter_val})"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    # Use a colormap that works well for accuracy (higher is better)
    cmap = 'YlOrRd' if metric == 'accuracy' else 'YlGnBu'
    
    sns.heatmap(matrix, 
                annot=False,  # We'll add custom annotations
                fmt='.2f',
                cmap=cmap,
                vmin=0,
                vmax=100 if metric == 'accuracy' else 1.0,
                cbar_kws={'label': f'Best {metric.capitalize()} (%)' if metric == 'accuracy' else f'Best {metric.upper()}'},
                ax=ax,
                linewidths=2,
                linecolor='black',
                square=True)
    
    # Add custom annotations with values and iterations
    for i in range(2):
        for j in range(2):
            if not np.isnan(matrix[i, j]):
                # Main value
                ax.text(j + 0.5, i + 0.35, matrix_labels[i, j],
                       ha='center', va='center', fontsize=16, fontweight='bold', color='black')
                # Iteration info
                if matrix_iters[i, j]:
                    ax.text(j + 0.5, i + 0.65, matrix_iters[i, j],
                           ha='center', va='center', fontsize=10, color='black', style='italic')
            else:
                ax.text(j + 0.5, i + 0.5, 'N/A',
                       ha='center', va='center', fontsize=14, color='gray')
    
    # Set labels
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['CrossEntropy (CE)', 'Focal Loss'], fontsize=12, fontweight='bold')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Triplet Loss', 'Contrastive Loss'], fontsize=12, fontweight='bold', rotation=0)
    
    # Set title
    metric_label = 'Accuracy' if metric == 'accuracy' else 'F1 Score'
    ax.set_title(f'Part 3: Loss Functions - Best {metric_label} Comparison\n2×2 Matrix: Metric Learning Loss × Classification Loss',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid lines for clarity
    ax.axhline(y=1, color='black', linewidth=2)
    ax.axvline(x=1, color='black', linewidth=2)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, f'part3_heatmap_{metric}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def create_combined_heatmap(best_metrics, output_dir='results_visualization'):
    """Create a combined heatmap showing both accuracy and F1 side by side."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize matrices
    acc_matrix = np.full((2, 2), np.nan)
    f1_matrix = np.full((2, 2), np.nan)
    acc_labels = np.empty((2, 2), dtype=object)
    f1_labels = np.empty((2, 2), dtype=object)
    acc_iters = np.empty((2, 2), dtype=object)
    f1_iters = np.empty((2, 2), dtype=object)
    
    # Fill matrices
    for exp_name, metrics in best_metrics.items():
        row, col = map_experiment_to_matrix(exp_name)
        if row is None or col is None:
            continue
        
        if metrics['best_accuracy'] is not None:
            acc_matrix[row, col] = metrics['best_accuracy']
            acc_labels[row, col] = f"{metrics['best_accuracy']:.2f}%"
            acc_iters[row, col] = f"(iter {metrics['best_acc_iter']})"
        
        if metrics['best_f1'] is not None:
            f1_matrix[row, col] = metrics['best_f1'] * 100  # Convert to percentage for consistency
            f1_labels[row, col] = f"{metrics['best_f1']:.3f}"
            f1_iters[row, col] = f"(iter {metrics['best_f1_iter']})"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Heatmap 1: Accuracy
    sns.heatmap(acc_matrix,
                annot=False,
                fmt='.2f',
                cmap='YlOrRd',
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Best Accuracy (%)'},
                ax=ax1,
                linewidths=2,
                linecolor='black',
                square=True)
    
    # Add annotations for accuracy
    for i in range(2):
        for j in range(2):
            if not np.isnan(acc_matrix[i, j]):
                ax1.text(j + 0.5, i + 0.35, acc_labels[i, j],
                        ha='center', va='center', fontsize=16, fontweight='bold', color='black')
                if acc_iters[i, j]:
                    ax1.text(j + 0.5, i + 0.65, acc_iters[i, j],
                            ha='center', va='center', fontsize=10, color='black', style='italic')
            else:
                ax1.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=14, color='gray')
    
    ax1.set_xticks([0.5, 1.5])
    ax1.set_xticklabels(['CrossEntropy (CE)', 'Focal Loss'], fontsize=11, fontweight='bold')
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(['Triplet Loss', 'Contrastive Loss'], fontsize=11, fontweight='bold', rotation=0)
    ax1.set_title('Best Accuracy (%)', fontsize=13, fontweight='bold', pad=15)
    ax1.axhline(y=1, color='black', linewidth=2)
    ax1.axvline(x=1, color='black', linewidth=2)
    
    # Heatmap 2: F1 Score
    sns.heatmap(f1_matrix,
                annot=False,
                fmt='.2f',
                cmap='YlGnBu',
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Best F1 Score (%)'},
                ax=ax2,
                linewidths=2,
                linecolor='black',
                square=True)
    
    # Add annotations for F1
    for i in range(2):
        for j in range(2):
            if not np.isnan(f1_matrix[i, j]):
                ax2.text(j + 0.5, i + 0.35, f1_labels[i, j],
                        ha='center', va='center', fontsize=16, fontweight='bold', color='black')
                if f1_iters[i, j]:
                    ax2.text(j + 0.5, i + 0.65, f1_iters[i, j],
                            ha='center', va='center', fontsize=10, color='black', style='italic')
            else:
                ax2.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center', fontsize=14, color='gray')
    
    ax2.set_xticks([0.5, 1.5])
    ax2.set_xticklabels(['CrossEntropy (CE)', 'Focal Loss'], fontsize=11, fontweight='bold')
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(['Triplet Loss', 'Contrastive Loss'], fontsize=11, fontweight='bold', rotation=0)
    ax2.set_title('Best F1 Score', fontsize=13, fontweight='bold', pad=15)
    ax2.axhline(y=1, color='black', linewidth=2)
    ax2.axvline(x=1, color='black', linewidth=2)
    
    # Overall title
    fig.suptitle('Part 3: Loss Functions - Best Performance Comparison\n2×2 Matrix: Metric Learning Loss × Classification Loss',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'part3_heatmap_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    print("=" * 120)
    print("CREATING PART 3 LOSS FUNCTIONS HEATMAP")
    print("=" * 120)
    
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print(f"\nLoading data from: {csv_file}")
    experiments = load_part3_data(csv_file)
    
    if not experiments:
        print("❌ No Part 3 experiments found!")
        return
    
    print(f"\nFound {len(experiments)} experiment(s):")
    for exp_name in sorted(experiments.keys()):
        print(f"  - {exp_name}")
    
    print("\nExtracting best metrics...")
    best_metrics = get_best_metrics(experiments)
    
    print("\nBest metrics per experiment:")
    for exp_name, metrics in sorted(best_metrics.items()):
        print(f"  {exp_name}:")
        print(f"    Best Accuracy: {metrics['best_accuracy']:.2f}% (iter {metrics['best_acc_iter']})")
        if metrics['best_f1']:
            print(f"    Best F1: {metrics['best_f1']:.3f} (iter {metrics['best_f1_iter']})")
    
    print("\nCreating heatmaps...")
    
    # Create individual heatmaps
    create_heatmap(best_metrics, metric='accuracy')
    create_heatmap(best_metrics, metric='f1')
    
    # Create combined heatmap
    create_combined_heatmap(best_metrics)
    
    print("\n" + "=" * 120)
    print("✓ Heatmaps created successfully!")
    print("=" * 120)
    print("\nGenerated files:")
    print("  - part3_heatmap_accuracy.png (Accuracy heatmap)")
    print("  - part3_heatmap_f1.png (F1 Score heatmap)")
    print("  - part3_heatmap_combined.png (Both metrics side by side)")

if __name__ == '__main__':
    main()

