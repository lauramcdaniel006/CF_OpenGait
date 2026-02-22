#!/usr/bin/env python3
"""
Visualize Part 3 (Loss Functions) - Best Accuracy and Corresponding F1 Score.
Shows both metrics on the same graph with a legend.
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
                    'f1': float(row['F1']) if row['F1'] and row['F1'].strip() else None,
                    'precision': float(row['Precision']) if row['Precision'] and row['Precision'].strip() else None,
                    'recall': float(row['Recall']) if row['Recall'] and row['Recall'].strip() else None
                })
            # Also include Part 1 p+CNN (pretrained + CNN) which uses triplet + CE
            elif 'Part 1' in row['Part'] and 'p+CNN' in row['Experiment'] and 'Tintro' not in row['Experiment']:
                exp_name = row['Experiment']
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['Accuracy (%)']),
                    'f1': float(row['F1']) if row['F1'] and row['F1'].strip() else None,
                    'precision': float(row['Precision']) if row['Precision'] and row['Precision'].strip() else None,
                    'recall': float(row['Recall']) if row['Recall'] and row['Recall'].strip() else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
    return experiments

def get_clean_experiment_name(exp_name):
    """Get a cleaner name for display."""
    name_map = {
        'REDO_Frailty_ccpg_pt1_p+CNN': 'Triplet + CE',
        'REDO_Frailty_ccpg_pt3_ce_contrastive': 'CE + Contrastive',
        'REDO_Frailty_ccpg_pt3_triplet_focal': 'Triplet + Focal',
        'REDO_Frailty_ccpg_pt3_contrastive_focal': 'Contrastive + Focal',
        'Frailty_part3_baseline_tripfocal': 'Triplet + CE',
    }
    return name_map.get(exp_name, exp_name.replace('REDO_', '').replace('Frailty_ccpg_pt3_', ''))

def plot_best_accuracy_and_f1(experiments, output_dir='results_visualization'):
    """Plot best accuracy and corresponding F1 score for each loss combination."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font to Times New Roman with fallbacks
    import matplotlib.font_manager as fm
    preferred_fonts = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif', 'serif']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_to_use = None
    
    for font in preferred_fonts:
        if font in available_fonts or font == 'serif':
            font_to_use = font
            break
    
    if font_to_use:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_to_use] + [f for f in preferred_fonts if f != font_to_use]
        plt.rcParams['mathtext.fontset'] = 'stix'
        print(f"Using font: {font_to_use}")
    else:
        plt.rcParams['font.family'] = 'serif'
        print("Using default serif font")
    
    # Calculate best metrics for each experiment
    summary_data = []
    
    for exp_name in sorted(experiments.keys()):
        data = experiments[exp_name]
        if not data:
            continue
        
        # Find best accuracy and corresponding F1
        best_idx = max(range(len(data)), key=lambda i: data[i]['accuracy'])
        best_acc = data[best_idx]['accuracy']
        best_f1 = data[best_idx]['f1']
        clean_name = get_clean_experiment_name(exp_name)
        
        if best_f1 is not None:
            summary_data.append({
                'name': clean_name,
                'accuracy': best_acc,
                'f1': best_f1
            })
    
    # Sort by accuracy (descending)
    summary_data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    # Prepare data
    exp_names = [d['name'] for d in summary_data]
    accuracies = [d['accuracy'] for d in summary_data]
    f1_scores = [d['f1'] * 100 for d in summary_data]  # Convert to percentage for better comparison
    
    x = np.arange(len(exp_names))
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Best Accuracy (%)', 
                    color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1 Score (%)', 
                    color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # Styling
    ax1.set_xlabel('Loss Combination', fontsize=13, fontweight='bold', family='serif')
    ax1.set_ylabel('Score (%)', fontsize=13, fontweight='bold', family='serif')
    ax1.set_title('Part 3: Loss Functions - Best Accuracy and Corresponding F1 Score', 
                  fontsize=14, fontweight='bold', pad=15, family='serif')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=15, ha='right', family='serif')
    ax1.set_ylim([0, 80])
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray', axis='y')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('#fafafa')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, family='serif', fontweight='bold')
    
    # Legend
    legend = ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                       fancybox=True, shadow=True, frameon=True,
                       edgecolor='gray', facecolor='white',
                       prop={'family': 'serif', 'size': 11})
    legend.get_frame().set_linewidth(1.2)
    
    # Set tick labels to serif font
    for label in ax1.get_xticklabels():
        label.set_family('serif')
    for label in ax1.get_yticklabels():
        label.set_family('serif')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'part3_best_accuracy_f1_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        return
    
    experiments = load_part3_data(csv_file)
    
    if not experiments:
        print("❌ No Part 3 experiments found!")
        return
    
    print(f"Found {len(experiments)} Part 3 experiments:")
    for exp_name in sorted(experiments.keys()):
        print(f"  - {exp_name}")
    
    plot_best_accuracy_and_f1(experiments)

if __name__ == '__main__':
    main()

