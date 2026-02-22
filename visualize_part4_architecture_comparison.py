#!/usr/bin/env python3
"""
Visualize Part 4 (Architecture Comparison) - Best Accuracy and Corresponding F1 Score.
Compares SwinGait vs DeepGaitV2 with frozen and unfrozen CNN layers.
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

def load_part4_data(csv_file):
    """Load Part 4 experiment data from CSV."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Include Part 4 experiments
            if 'Part 4' in row['Part']:
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
        'REDO_Frailty_ccpg_pt4_swingait': 'SwinGait (Frozen CNN)',
        'REDO_Frailty_ccpg_pt4_deepgaitv2': 'DeepGaitV2 (Frozen CNN)',
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn': 'SwinGait (Unfrozen CNN)',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn': 'DeepGaitV2 (Unfrozen CNN)',
    }
    return name_map.get(exp_name, exp_name.replace('REDO_', '').replace('Frailty_ccpg_pt4_', ''))

def plot_best_accuracy_and_f1(experiments, output_dir='results_visualization'):
    """Plot best accuracy and corresponding F1 score for each architecture."""
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
    exp_data = {}
    
    # Map experiments to their data
    exp_map = {
        'swingait_frozen': 'REDO_Frailty_ccpg_pt4_swingait',
        'deepgaitv2_frozen': 'REDO_Frailty_ccpg_pt4_deepgaitv2',
        'swingait_unfrozen': 'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn',
        'deepgaitv2_unfrozen': 'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn'
    }
    
    for key, exp_name in exp_map.items():
        if exp_name not in experiments:
            continue
        
        data = experiments[exp_name]
        if not data:
            continue
        
        # Find best accuracy and corresponding F1
        best_idx = max(range(len(data)), key=lambda i: data[i]['accuracy'])
        best_acc = data[best_idx]['accuracy']
        best_f1 = data[best_idx]['f1']
        
        if best_f1 is not None:
            exp_data[key] = {
                'accuracy': best_acc,
                'f1': best_f1 * 100  # Convert to percentage
            }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    # Architecture names for x-axis
    architectures = ['SwinGait', 'DeepGaitV2']
    x = np.arange(len(architectures))
    width = 0.35  # Width of bars
    
    # Prepare data: Group by architecture, show frozen/unfrozen as grouped bars
    # For Accuracy
    swingait_acc = [exp_data.get('swingait_frozen', {}).get('accuracy', 0),
                    exp_data.get('swingait_unfrozen', {}).get('accuracy', 0)]
    deepgaitv2_acc = [exp_data.get('deepgaitv2_frozen', {}).get('accuracy', 0),
                      exp_data.get('deepgaitv2_unfrozen', {}).get('accuracy', 0)]
    
    # For F1
    swingait_f1 = [exp_data.get('swingait_frozen', {}).get('f1', 0),
                   exp_data.get('swingait_unfrozen', {}).get('f1', 0)]
    deepgaitv2_f1 = [exp_data.get('deepgaitv2_frozen', {}).get('f1', 0),
                     exp_data.get('deepgaitv2_unfrozen', {}).get('f1', 0)]
    
    # Plot Accuracy
    bars1 = ax1.bar(x - width/2, [swingait_acc[0], deepgaitv2_acc[0]], width, 
                    label='Frozen CNN', color='#1f77b4', alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, [swingait_acc[1], deepgaitv2_acc[1]], width, 
                    label='Unfrozen CNN', color='#2ca02c', alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    
    # Plot F1
    bars3 = ax2.bar(x - width/2, [swingait_f1[0], deepgaitv2_f1[0]], width, 
                    label='Frozen CNN', color='#ff7f0e', alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, [swingait_f1[1], deepgaitv2_f1[1]], width, 
                    label='Unfrozen CNN', color='#d62728', alpha=0.85, 
                    edgecolor='black', linewidth=1.2)
    
    # Styling for Accuracy subplot
    ax1.set_xlabel('Architecture', fontsize=13, fontweight='bold', family='serif')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', family='serif')
    ax1.set_title('Best Accuracy by Architecture', fontsize=14, fontweight='bold', pad=15, family='serif')
    ax1.set_xticks(x)
    ax1.set_xticklabels(architectures, family='serif')
    ax1.set_ylim([0, 80])
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray', axis='y')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('#fafafa')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, 
               fancybox=True, shadow=True, prop={'family': 'serif'})
    
    # Styling for F1 subplot
    ax2.set_xlabel('Architecture', fontsize=13, fontweight='bold', family='serif')
    ax2.set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold', family='serif')
    ax2.set_title('Best F1 Score by Architecture', fontsize=14, fontweight='bold', pad=15, family='serif')
    ax2.set_xticks(x)
    ax2.set_xticklabels(architectures, family='serif')
    ax2.set_ylim([0, 80])
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray', axis='y')
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#fafafa')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95, 
               fancybox=True, shadow=True, prop={'family': 'serif'})
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9, family='serif', fontweight='bold')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9, family='serif', fontweight='bold')
    
    # Set tick labels to serif font
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_family('serif')
        for label in ax.get_yticklabels():
            label.set_family('serif')
    
    plt.suptitle('Part 4: Architecture Comparison - SwinGait vs DeepGaitV2', 
                 fontsize=16, fontweight='bold', y=1.02, family='serif')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_file = os.path.join(output_dir, 'part4_best_accuracy_f1_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        return
    
    experiments = load_part4_data(csv_file)
    
    if not experiments:
        print("❌ No Part 4 experiments found!")
        return
    
    print(f"Found {len(experiments)} Part 4 experiments:")
    for exp_name in sorted(experiments.keys()):
        print(f"  - {exp_name}")
    
    plot_best_accuracy_and_f1(experiments)

if __name__ == '__main__':
    main()

