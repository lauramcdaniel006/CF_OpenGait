#!/usr/bin/env python3
"""
Update Part 4 visualization to include class weights experiments.
"""

import csv
import os
import json
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def load_existing_results():
    """Load existing Part 4 baseline results."""
    # Baseline results (from previous analysis)
    baseline = {
        'SwinGait (Frozen CNN)': {'accuracy': 73.33, 'f1': 73.89},
        'DeepGaitV2 (Frozen CNN)': {'accuracy': 60.00, 'f1': 60.33},
        'SwinGait (Unfrozen CNN)': {'accuracy': 73.33, 'f1': 72.78},
        'DeepGaitV2 (Unfrozen CNN)': {'accuracy': 66.67, 'f1': 60.32}
    }
    return baseline

def load_with_weights_results():
    """Load class weights experiment results."""
    csv_file = 'results_visualization/part4_with_weights_results.csv'
    results = {}
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['Experiment']
                acc = float(row['Accuracy (%)'])
                f1 = float(row['F1 Score (%)']) if row['F1 Score (%)'] else None
                results[name] = {
                    'accuracy': acc,
                    'f1': f1,
                    'iteration': int(row['Best Iteration'])
                }
    
    return results

def plot_comparison(baseline, with_weights, output_dir='results_visualization'):
    """Create updated visualization with class weights experiments."""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font
    import matplotlib.font_manager as fm
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Prepare data
    architectures = ['SwinGait', 'DeepGaitV2']
    x = np.arange(len(architectures))
    width = 0.2  # Width of bars
    
    # Extract data
    swingait_frozen_baseline = baseline['SwinGait (Frozen CNN)']['accuracy']
    swingait_frozen_weights = with_weights.get('SwinGait (Frozen CNN) + Class Weights', {}).get('accuracy', 0)
    
    deepgaitv2_frozen_baseline = baseline['DeepGaitV2 (Frozen CNN)']['accuracy']
    deepgaitv2_frozen_weights = with_weights.get('DeepGaitV2 (Frozen CNN) + Class Weights', {}).get('accuracy', 0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    # Accuracy comparison
    acc_baseline = [swingait_frozen_baseline, deepgaitv2_frozen_baseline]
    acc_weights = [swingait_frozen_weights, deepgaitv2_frozen_weights]
    
    bars1 = ax1.bar(x - width/2, acc_baseline, width, 
                    label='Baseline (No Weights)', color='#1f77b4', alpha=0.85,
                    edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, acc_weights, width,
                    label='With Class Weights', color='#ff7f0e', alpha=0.85,
                    edgecolor='black', linewidth=1.2)
    
    # F1 comparison (if available)
    swingait_f1_baseline = baseline['SwinGait (Frozen CNN)']['f1']
    swingait_f1_weights = with_weights.get('SwinGait (Frozen CNN) + Class Weights', {}).get('f1', 0)
    
    deepgaitv2_f1_baseline = baseline['DeepGaitV2 (Frozen CNN)']['f1']
    deepgaitv2_f1_weights = with_weights.get('DeepGaitV2 (Frozen CNN) + Class Weights', {}).get('f1', 0)
    
    f1_baseline = [swingait_f1_baseline, deepgaitv2_f1_baseline]
    f1_weights = [swingait_f1_weights if swingait_f1_weights else 0, 
                  deepgaitv2_f1_weights if deepgaitv2_f1_weights else 0]
    
    bars3 = ax2.bar(x - width/2, f1_baseline, width,
                    label='Baseline (No Weights)', color='#1f77b4', alpha=0.85,
                    edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, f1_weights, width,
                    label='With Class Weights', color='#ff7f0e', alpha=0.85,
                    edgecolor='black', linewidth=1.2)
    
    # Styling
    for ax, metric_name in [(ax1, 'Accuracy'), (ax2, 'F1 Score')]:
        ax.set_xlabel('Architecture', fontsize=13, fontweight='bold', family='serif')
        ax.set_ylabel(f'{metric_name} (%)', fontsize=13, fontweight='bold', family='serif')
        ax.set_title(f'Best {metric_name} by Architecture', fontsize=14, fontweight='bold', pad=15, family='serif')
        ax.set_xticks(x)
        ax.set_xticklabels(architectures, family='serif')
        ax.set_ylim([0, 80])
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray', axis='y')
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
                  fancybox=True, shadow=True, prop={'family': 'serif'})
        
        # Add value labels
        for bars in [bars1, bars2] if ax == ax1 else [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9, family='serif', fontweight='bold')
    
    plt.suptitle('Part 4: Effect of Class Weights on Architecture Performance', 
                 fontsize=16, fontweight='bold', y=1.02, family='serif')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_file = os.path.join(output_dir, 'part4_class_weights_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_file}")
    plt.close()

def print_summary(baseline, with_weights):
    """Print comprehensive summary and recommendations."""
    print("\n" + "="*80)
    print("PART 4 CLASS WEIGHTS EXPERIMENTS - RESULTS SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("BASELINE RESULTS (No Class Weights):")
    print("-"*80)
    for name, metrics in baseline.items():
        print(f"  {name}:")
        print(f"    Accuracy: {metrics['accuracy']:.2f}%")
        print(f"    F1 Score: {metrics['f1']:.2f}%")
    
    print("\n" + "-"*80)
    print("RESULTS WITH CLASS WEIGHTS:")
    print("-"*80)
    for name, metrics in with_weights.items():
        print(f"  {name}:")
        print(f"    Accuracy: {metrics['accuracy']:.2f}%")
        if metrics.get('f1'):
            print(f"    F1 Score: {metrics['f1']:.2f}%")
        print(f"    Best Iteration: {metrics.get('iteration', 'N/A')}")
    
    print("\n" + "-"*80)
    print("COMPARISON:")
    print("-"*80)
    
    swingait_baseline = baseline['SwinGait (Frozen CNN)']['accuracy']
    swingait_weights = with_weights.get('SwinGait (Frozen CNN) + Class Weights', {}).get('accuracy', 0)
    swingait_change = swingait_weights - swingait_baseline
    
    deepgaitv2_baseline = baseline['DeepGaitV2 (Frozen CNN)']['accuracy']
    deepgaitv2_weights = with_weights.get('DeepGaitV2 (Frozen CNN) + Class Weights', {}).get('accuracy', 0)
    deepgaitv2_change = deepgaitv2_weights - deepgaitv2_baseline
    
    print(f"\n  SwinGait (Frozen CNN):")
    print(f"    Baseline: {swingait_baseline:.2f}%")
    print(f"    With Weights: {swingait_weights:.2f}%")
    print(f"    Change: {swingait_change:+.2f}%")
    
    print(f"\n  DeepGaitV2 (Frozen CNN):")
    print(f"    Baseline: {deepgaitv2_baseline:.2f}%")
    print(f"    With Weights: {deepgaitv2_weights:.2f}%")
    print(f"    Change: {deepgaitv2_change:+.2f}%")
    
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    if swingait_change == 0:
        print("  ✓ SwinGait: No change with class weights (maintains 73.33%)")
    elif swingait_change > 0:
        print(f"  ✓ SwinGait: Improved by {swingait_change:.2f}% with class weights")
    else:
        print(f"  ✗ SwinGait: Decreased by {abs(swingait_change):.2f}% with class weights")
    
    if deepgaitv2_change > 0:
        print(f"  ✓ DeepGaitV2: Improved by {deepgaitv2_change:.2f}% with class weights")
        print(f"    Significant improvement! (60.00% → {deepgaitv2_weights:.2f}%)")
    elif deepgaitv2_change == 0:
        print("  - DeepGaitV2: No change with class weights")
    else:
        print(f"  ✗ DeepGaitV2: Decreased by {abs(deepgaitv2_change):.2f}% with class weights")
    
    # Calculate gap
    gap_baseline = swingait_baseline - deepgaitv2_baseline
    gap_weights = swingait_weights - deepgaitv2_weights
    
    print(f"\n  Performance Gap:")
    print(f"    Baseline: {gap_baseline:.2f} percentage points")
    print(f"    With Weights: {gap_weights:.2f} percentage points")
    print(f"    Gap Change: {gap_weights - gap_baseline:+.2f} percentage points")
    
    print("\n" + "-"*80)
    print("RECOMMENDATIONS:")
    print("-"*80)
    
    if deepgaitv2_change >= 5:
        print("  ✓ DeepGaitV2 shows significant improvement with class weights")
        print("    → Class weights help DeepGaitV2, but it still underperforms SwinGait")
        print("    → Architecture difference remains the main factor")
    elif deepgaitv2_change > 0:
        print("  ✓ DeepGaitV2 shows improvement with class weights")
        print("    → Class weights help, but improvement is modest")
    else:
        print("  ✗ DeepGaitV2 did not improve with class weights")
        print("    → Class weights alone may not be sufficient")
    
    if swingait_change == 0:
        print("  → SwinGait already performs well without class weights")
        print("    → Transformer architecture may inherently handle imbalance better")
    
    if gap_weights < gap_baseline:
        print(f"\n  ✓ Gap narrowed by {gap_baseline - gap_weights:.2f} percentage points")
        print("    → Class weights help reduce architecture performance gap")
    elif gap_weights > gap_baseline:
        print(f"\n  ✗ Gap widened by {gap_weights - gap_baseline:.2f} percentage points")
        print("    → SwinGait benefits more from class weights")
    
    print("\n" + "-"*80)
    print("NEXT STEPS - ADDITIONAL EXPERIMENTS:")
    print("-"*80)
    
    if deepgaitv2_weights < 70:
        print("  RECOMMENDED: Test Focal Loss on DeepGaitV2")
        print("    - DeepGaitV2 still underperforms (< 70%)")
        print("    - Focal Loss focuses on hard examples")
        print("    - Might help DeepGaitV2 learn better from minority class")
        print("    - Experiment: DeepGaitV2 (Frozen) + Triplet + Focal + Class Weights")
    else:
        print("  OPTIONAL: Test Focal Loss on DeepGaitV2")
        print("    - DeepGaitV2 is performing reasonably well")
        print("    - Could test if Focal Loss provides additional improvement")
    
    if swingait_change == 0:
        print("\n  OPTIONAL: Test Focal Loss on SwinGait")
        print("    - SwinGait didn't benefit from class weights")
        print("    - Could test if Focal Loss helps (though unlikely needed)")
    
    print("\n" + "="*80)

def main():
    baseline = load_existing_results()
    with_weights = load_with_weights_results()
    
    if not with_weights:
        print("❌ No class weights results found!")
        print("   Run extract_part4_with_weights_from_logs.py first")
        return
    
    print_summary(baseline, with_weights)
    
    if MATPLOTLIB_AVAILABLE:
        plot_comparison(baseline, with_weights)
    else:
        print("\n⚠ matplotlib not available - skipping visualization")

if __name__ == '__main__':
    main()

