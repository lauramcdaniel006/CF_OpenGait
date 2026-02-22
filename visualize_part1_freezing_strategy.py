#!/usr/bin/env python3
"""
Visualize Part 1 (Freezing Strategy) experiment results.
Creates graphs showing metrics over iterations and comparison across experiments.
"""

import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_part1_data(csv_file):
    """Load Part 1 experiment data from CSV."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Part 1' in row['Part']:
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

def plot_metrics_over_iterations(experiments, output_dir='results_visualization'):
    """Plot all metrics (Accuracy, F1, Precision, Recall) over iterations for each experiment."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Part 1: Freezing Strategy - Metrics Over Iterations', fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy (%)', axes[0, 0], lambda x: x['accuracy']),
        ('f1', 'F1 Score', axes[0, 1], lambda x: x['f1']),
        ('precision', 'Precision', axes[1, 0], lambda x: x['precision']),
        ('recall', 'Recall', axes[1, 1], lambda x: x['recall'])
    ]
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for metric_name, metric_label, ax, get_value in metrics:
        for i, (exp_name, data) in enumerate(sorted(experiments.items())):
            iterations = [d['iteration'] for d in data]
            values = [get_value(d) for d in data]
            
            # Filter out None values
            valid_data = [(it, val) for it, val in zip(iterations, values) if val is not None]
            if valid_data:
                iters, vals = zip(*valid_data)
                ax.plot(iters, vals, marker='o', label=exp_name, color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} Over Iterations', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part1_metrics_over_iterations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_accuracy_comparison(experiments, output_dir='results_visualization'):
    """Plot accuracy comparison - most important metric."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        iterations = [d['iteration'] for d in data]
        accuracies = [d['accuracy'] for d in data]
        ax.plot(iterations, accuracies, marker='o', label=exp_name, 
                color=colors[i], linewidth=2.5, markersize=5, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Part 1: Freezing Strategy - Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part1_accuracy_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_final_metrics_comparison(experiments, output_dir='results_visualization'):
    """Bar chart comparing final metrics across experiments."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    exp_names = sorted(experiments.keys())
    final_acc = [experiments[exp][-1]['accuracy'] for exp in exp_names]
    final_f1 = [experiments[exp][-1]['f1'] for exp in exp_names if experiments[exp][-1]['f1'] is not None]
    exp_names_f1 = [exp for exp in exp_names if experiments[exp][-1]['f1'] is not None]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy bar chart
    bars1 = ax1.bar(range(len(exp_names)), final_acc, color=plt.cm.viridis(np.linspace(0, 1, len(exp_names))))
    ax1.set_xlabel('Experiment', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, final_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # F1 bar chart
    if final_f1:
        bars2 = ax2.bar(range(len(exp_names_f1)), final_f1, color=plt.cm.plasma(np.linspace(0, 1, len(exp_names_f1))))
        ax2.set_xlabel('Experiment', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Final F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Final F1 Score Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(exp_names_f1)))
        ax2.set_xticklabels(exp_names_f1, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars2, final_f1)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Part 1: Freezing Strategy - Final Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part1_final_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_best_metrics_comparison(experiments, output_dir='results_visualization'):
    """Bar chart comparing best (max) metrics across experiments."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    exp_names = sorted(experiments.keys())
    best_acc = [max(d['accuracy'] for d in experiments[exp]) for exp in exp_names]
    best_f1 = []
    exp_names_f1 = []
    for exp in exp_names:
        f1_values = [d['f1'] for d in experiments[exp] if d['f1'] is not None]
        if f1_values:
            best_f1.append(max(f1_values))
            exp_names_f1.append(exp)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best Accuracy
    bars1 = ax1.bar(range(len(exp_names)), best_acc, color=plt.cm.viridis(np.linspace(0, 1, len(exp_names))))
    ax1.set_xlabel('Experiment', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Best Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Best Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    for i, (bar, val) in enumerate(zip(bars1, best_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Best F1
    if best_f1:
        bars2 = ax2.bar(range(len(exp_names_f1)), best_f1, color=plt.cm.plasma(np.linspace(0, 1, len(exp_names_f1))))
        ax2.set_xlabel('Experiment', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Best F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Best F1 Score Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(exp_names_f1)))
        ax2.set_xticklabels(exp_names_f1, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        for i, (bar, val) in enumerate(zip(bars2, best_f1)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Part 1: Freezing Strategy - Best Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part1_best_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("   Run: python view_all_redo_parts1-4.py first")
        return
    
    print("=" * 80)
    print("Part 1: Freezing Strategy - Visualization")
    print("=" * 80)
    
    experiments = load_part1_data(csv_file)
    
    if not experiments:
        print("\n❌ No Part 1 data found in CSV!")
        print("   Make sure you've run the extraction script with TensorBoard available.")
        return
    
    print(f"\nFound {len(experiments)} Part 1 experiment(s):")
    for exp_name in sorted(experiments.keys()):
        print(f"  - {exp_name} ({len(experiments[exp_name])} evaluations)")
    
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    
    # Generate all plots
    plot_metrics_over_iterations(experiments)
    plot_accuracy_comparison(experiments)
    plot_final_metrics_comparison(experiments)
    plot_best_metrics_comparison(experiments)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations generated!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. part1_metrics_over_iterations.png - All 4 metrics (Accuracy, F1, Precision, Recall)")
    print("  2. part1_accuracy_comparison.png - Accuracy over iterations (most important)")
    print("  3. part1_final_metrics_comparison.png - Final Accuracy and F1 comparison")
    print("  4. part1_best_metrics_comparison.png - Best Accuracy and F1 comparison")
    print("\nRecommendation for paper:")
    print("  - Use 'part1_accuracy_comparison.png' as main figure (most important metric)")
    print("  - Use 'part1_final_metrics_comparison.png' or 'part1_best_metrics_comparison.png' for table/summary")
    print("=" * 80)

if __name__ == '__main__':
    main()

