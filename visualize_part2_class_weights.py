#!/usr/bin/env python3
"""
Visualize Part 2 (Class Weights) experiment results.
Creates graphs showing metrics over iterations and comparison across different class weighting strategies.
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

def load_part2_data(csv_file):
    """Load Part 2 experiment data from CSV."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Part 2' in row['Part']:
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

def get_clean_experiment_name(exp_name):
    """Get a cleaner name for display."""
    name_map = {
        'REDO_insqrt': 'Inverse Sqrt',
        'REDO_balnormal': 'Balanced Normal',
        'REDO_log': 'Log',
        'REDO_smooth': 'Smooth',
        'REDO_uniform': 'Uniform'
    }
    return name_map.get(exp_name, exp_name.replace('REDO_', ''))

def get_class_weights(exp_name):
    """Get class weights for an experiment (train-only split weights)."""
    weights_map = {
        'REDO_insqrt': [1.130, 0.935, 0.935],
        'REDO_balnormal': [1.267, 0.867, 0.867],
        'REDO_log': [1.138, 0.931, 0.931],
        'REDO_smooth': [0.778, 1.111, 1.111],
        'REDO_uniform': [1.000, 1.000, 1.000]
    }
    return weights_map.get(exp_name, [1.0, 1.0, 1.0])

def get_legend_label(exp_name):
    """Get legend label with class weights."""
    clean_name = get_clean_experiment_name(exp_name)
    weights = get_class_weights(exp_name)
    # Format: "Name [Frail, Prefrail, Nonfrail]"
    weights_str = f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
    return f"{clean_name} {weights_str}"

def plot_accuracy_comparison(experiments, output_dir='results_visualization'):
    """Plot accuracy comparison - most important metric for Part 2."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font to Times New Roman with fallbacks
    # Try Times New Roman first, fall back to available serif fonts
    import matplotlib.font_manager as fm
    
    # List of preferred serif fonts (in order of preference)
    preferred_fonts = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']
    
    # Find available font
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_to_use = None
    
    for font in preferred_fonts:
        if font in available_fonts or font == 'serif':
            font_to_use = font
            break
    
    if font_to_use:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_to_use] + [f for f in preferred_fonts if f != font_to_use]
        plt.rcParams['mathtext.fontset'] = 'stix'  # For math text, use STIX (similar to Times)
        print(f"Using font: {font_to_use}")
    else:
        # Fallback to default serif
        plt.rcParams['font.family'] = 'serif'
        print("Using default serif font (Times New Roman not available)")
    
    # Create figure with better styling
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Professional color palette (colorblind-friendly, distinct colors)
    # Using tab10 colormap which is designed for categorical data
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Marker styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Track best accuracy for annotation
    best_overall = {'value': 0, 'exp': None, 'iter': None, 'idx': None}
    
    for i, (exp_name, data) in enumerate(sorted(experiments.items())):
        iterations = [d['iteration'] for d in data]
        accuracies = [d['accuracy'] for d in data]
        clean_name = get_clean_experiment_name(exp_name)
        legend_label = get_legend_label(exp_name)
        
        # Find best accuracy for this experiment
        best_idx = np.argmax(accuracies)
        best_acc = accuracies[best_idx]
        best_iter = iterations[best_idx]
        
        # Track overall best
        if best_acc > best_overall['value']:
            best_overall['value'] = best_acc
            best_overall['exp'] = clean_name
            best_overall['iter'] = best_iter
            best_overall['idx'] = i
        
        # Plot line with better styling
        line = ax.plot(iterations, accuracies, 
                      marker=markers[i % len(markers)],
                      label=legend_label, 
                      color=colors[i], 
                      linewidth=2.8,
                      markersize=6,
                      markevery=1,
                      alpha=0.85,
                      linestyle=linestyles[i % len(linestyles)],
                      markerfacecolor='white',
                      markeredgewidth=1.5,
                      markeredgecolor=colors[i])
        
        # Annotate best point for each experiment
        ax.annotate(f'{best_acc:.1f}%',
                   xy=(best_iter, best_acc),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   fontweight='bold',
                   color=colors[i],
                   family='serif',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=colors[i], alpha=0.8, linewidth=1.5),
                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))
    
    # Styling improvements with Times New Roman
    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold', labelpad=10, family='serif')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10, family='serif')
    ax.set_title('Part 2: Class Weights - Accuracy Comparison Over Training', 
                fontsize=15, fontweight='bold', pad=15, family='serif')
    
    # Better grid styling
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # Better legend with Times New Roman font
    legend = ax.legend(fontsize=10, loc='lower right', framealpha=0.95, 
                      fancybox=True, shadow=True, frameon=True,
                      edgecolor='gray', facecolor='white',
                      prop={'family': 'serif', 'size': 10})
    legend.get_frame().set_linewidth(1.2)
    
    # Y-axis improvements with Times New Roman
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 101, 10))
    ax.tick_params(axis='both', which='major', labelsize=11)
    # Set tick labels to serif font
    for label in ax.get_xticklabels():
        label.set_family('serif')
    for label in ax.get_yticklabels():
        label.set_family('serif')
    
    # X-axis improvements
    all_iterations = [[d['iteration'] for d in data] for data in experiments.values()]
    if all_iterations:
        max_iter = max(max(iter_list) for iter_list in all_iterations)
        ax.set_xlim([0, max_iter * 1.02])
        # Set nice x-axis ticks
        if max_iter <= 15000:
            ax.set_xticks(range(0, int(max_iter) + 1, 2000))
        else:
            ax.set_xticks(range(0, int(max_iter) + 1, 5000))
    
    # Add subtle background color for better contrast
    ax.set_facecolor('#fafafa')
    
    # Add a subtle horizontal line at best accuracy
    if best_overall['value'] > 0:
        ax.axhline(y=best_overall['value'], color='red', linestyle=':', 
                  linewidth=1.5, alpha=0.4, zorder=0)
        ax.text(ax.get_xlim()[1] * 0.98, best_overall['value'] + 1,
               f'Best: {best_overall["value"]:.1f}% ({best_overall["exp"]})',
               fontsize=9, ha='right', va='bottom', family='Times New Roman',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                        alpha=0.7, edgecolor='red', linewidth=1))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part2_accuracy_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_metrics_over_iterations(experiments, output_dir='results_visualization'):
    """Plot all metrics (Accuracy, F1, Precision, Recall) over iterations."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Part 2: Class Weights - Metrics Over Iterations', fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy (%)', axes[0, 0], lambda x: x['accuracy']),
        ('f1', 'F1 Score', axes[0, 1], lambda x: x['f1']),
        ('precision', 'Precision', axes[1, 0], lambda x: x['precision']),
        ('recall', 'Recall', axes[1, 1], lambda x: x['recall'])
    ]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))
    
    for metric_name, metric_label, ax, get_value in metrics:
        for i, (exp_name, data) in enumerate(sorted(experiments.items())):
            iterations = [d['iteration'] for d in data]
            values = [get_value(d) for d in data]
            
            valid_data = [(it, val) for it, val in zip(iterations, values) if val is not None]
            if valid_data:
                iters, vals = zip(*valid_data)
                clean_name = get_clean_experiment_name(exp_name)
                ax.plot(iters, vals, marker='o', label=clean_name, color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} Over Iterations', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part2_metrics_over_iterations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_final_metrics_comparison(experiments, output_dir='results_visualization'):
    """Bar chart comparing final metrics across class weight strategies."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    exp_names = sorted(experiments.keys())
    clean_names = [get_clean_experiment_name(exp) for exp in exp_names]
    final_acc = [experiments[exp][-1]['accuracy'] for exp in exp_names]
    final_f1 = [experiments[exp][-1]['f1'] for exp in exp_names if experiments[exp][-1]['f1'] is not None]
    exp_names_f1 = [get_clean_experiment_name(exp) for exp in exp_names if experiments[exp][-1]['f1'] is not None]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy bar chart
    bars1 = ax1.bar(range(len(clean_names)), final_acc, color=plt.cm.viridis(np.linspace(0, 1, len(clean_names))))
    ax1.set_xlabel('Class Weight Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Final Accuracy by Class Weight Strategy', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(clean_names)))
    ax1.set_xticklabels(clean_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    for i, (bar, val) in enumerate(zip(bars1, final_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # F1 bar chart
    if final_f1:
        bars2 = ax2.bar(range(len(exp_names_f1)), final_f1, color=plt.cm.plasma(np.linspace(0, 1, len(exp_names_f1))))
        ax2.set_xlabel('Class Weight Strategy', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Final F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Final F1 Score by Class Weight Strategy', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(exp_names_f1)))
        ax2.set_xticklabels(exp_names_f1, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        for i, (bar, val) in enumerate(zip(bars2, final_f1)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Part 2: Class Weights - Final Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part2_final_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_best_metrics_comparison(experiments, output_dir='results_visualization'):
    """Bar chart comparing best (max) metrics across class weight strategies."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    exp_names = sorted(experiments.keys())
    clean_names = [get_clean_experiment_name(exp) for exp in exp_names]
    best_acc = [max(d['accuracy'] for d in experiments[exp]) for exp in exp_names]
    best_f1 = []
    exp_names_f1 = []
    for exp in exp_names:
        f1_values = [d['f1'] for d in experiments[exp] if d['f1'] is not None]
        if f1_values:
            best_f1.append(max(f1_values))
            exp_names_f1.append(get_clean_experiment_name(exp))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best Accuracy
    bars1 = ax1.bar(range(len(clean_names)), best_acc, color=plt.cm.viridis(np.linspace(0, 1, len(clean_names))))
    ax1.set_xlabel('Class Weight Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Best Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Best Accuracy by Class Weight Strategy', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(clean_names)))
    ax1.set_xticklabels(clean_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    for i, (bar, val) in enumerate(zip(bars1, best_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Best F1
    if best_f1:
        bars2 = ax2.bar(range(len(exp_names_f1)), best_f1, color=plt.cm.plasma(np.linspace(0, 1, len(exp_names_f1))))
        ax2.set_xlabel('Class Weight Strategy', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Best F1 Score', fontsize=11, fontweight='bold')
        ax2.set_title('Best F1 Score by Class Weight Strategy', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(exp_names_f1)))
        ax2.set_xticklabels(exp_names_f1, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        for i, (bar, val) in enumerate(zip(bars2, best_f1)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Part 2: Class Weights - Best Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part2_best_metrics_comparison.png')
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
    print("Part 2: Class Weights - Visualization")
    print("=" * 80)
    
    experiments = load_part2_data(csv_file)
    
    if not experiments:
        print("\n❌ No Part 2 data found in CSV!")
        return
    
    print(f"\nFound {len(experiments)} Part 2 experiment(s):")
    for exp_name in sorted(experiments.keys()):
        clean_name = get_clean_experiment_name(exp_name)
        print(f"  - {clean_name} ({exp_name}) - {len(experiments[exp_name])} evaluations")
    
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    
    # Generate all plots
    plot_accuracy_comparison(experiments)
    plot_metrics_over_iterations(experiments)
    plot_final_metrics_comparison(experiments)
    plot_best_metrics_comparison(experiments)
    
    print("\n" + "=" * 80)
    print("✓ All visualizations generated!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. part2_accuracy_comparison.png - Accuracy over iterations (MAIN FIGURE)")
    print("  2. part2_final_metrics_comparison.png - Final Accuracy and F1 comparison")
    print("  3. part2_best_metrics_comparison.png - Best Accuracy and F1 comparison")
    print("  4. part2_metrics_over_iterations.png - All 4 metrics (optional)")
    print("\nRecommendation for paper:")
    print("  ✓ Use 'part2_accuracy_comparison.png' as main figure")
    print("  ✓ Use 'part2_final_metrics_comparison.png' or 'part2_best_metrics_comparison.png' for summary")
    print("  ✓ Shows which class weighting strategy works best for your imbalanced dataset")
    print("=" * 80)

if __name__ == '__main__':
    main()

