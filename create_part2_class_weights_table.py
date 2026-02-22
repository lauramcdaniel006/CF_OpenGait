#!/usr/bin/env python3
"""
Create a comprehensive table for Part 2 (Class Weights) that shows:
1. Class distribution (imbalance)
2. Class weights used by each strategy
3. Performance metrics (how well each strategy handles imbalance)
"""

import csv
import os
import math
from collections import defaultdict

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
                    'f1': float(row['F1']) if row['F1'] else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
    return experiments

def get_class_distribution():
    """Get class distribution from frailty label file."""
    frailty_file = '/cis/home/lmcdan11/OpenGait_silhall/opengait/frailty_label.csv'
    
    class_counts = {'Frail': 0, 'Prefrail': 0, 'Nonfrail': 0}
    
    try:
        with open(frailty_file, 'r') as f:
            content = f.read()
            # Try to parse CSV
            lines = content.strip().split('\n')
            if len(lines) > 1:
                # Skip header
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            score = int(parts[1].strip())
                            if score == 0:
                                class_counts['Nonfrail'] += 1
                            elif score == 1:
                                class_counts['Prefrail'] += 1
                            elif score == 2:
                                class_counts['Frail'] += 1
                        except:
                            pass
    except Exception as e:
        # Use known values from previous analysis
        class_counts = {'Frail': 25, 'Prefrail': 24, 'Nonfrail': 17}
    
    total = sum(class_counts.values())
    if total == 0:
        # Fallback to known values
        class_counts = {'Frail': 25, 'Prefrail': 24, 'Nonfrail': 17}
        total = 66
    
    return class_counts, total

def calculate_class_weights(class_counts, total):
    """Calculate class weights for each strategy."""
    num_classes = 3
    frail_count = class_counts['Frail']
    prefrail_count = class_counts['Prefrail']
    nonfrail_count = class_counts['Nonfrail']
    
    weights = {}
    
    # 1. UNIFORM
    weights['uniform'] = [1.0, 1.0, 1.0]
    
    # 2. INVERSE SQUARE ROOT (insqrt)
    weights_sqrt = [
        math.sqrt(total / frail_count) if frail_count > 0 else 1.0,
        math.sqrt(total / prefrail_count) if prefrail_count > 0 else 1.0,
        math.sqrt(total / nonfrail_count) if nonfrail_count > 0 else 1.0
    ]
    mean_sqrt = sum(weights_sqrt) / num_classes
    weights['insqrt'] = [w / mean_sqrt for w in weights_sqrt]
    
    # 3. BALANCED NORMAL (balnormal) - sklearn style
    weights_inv = [
        total / (num_classes * frail_count) if frail_count > 0 else 1.0,
        total / (num_classes * prefrail_count) if prefrail_count > 0 else 1.0,
        total / (num_classes * nonfrail_count) if nonfrail_count > 0 else 1.0
    ]
    mean_weight = sum(weights_inv) / num_classes
    weights['balnormal'] = [w / mean_weight for w in weights_inv]
    
    # 4. LOGARITHMIC (log)
    weights_log = [
        math.log(total / frail_count + 1) if frail_count > 0 else 1.0,
        math.log(total / prefrail_count + 1) if prefrail_count > 0 else 1.0,
        math.log(total / nonfrail_count + 1) if nonfrail_count > 0 else 1.0
    ]
    mean_log = sum(weights_log) / num_classes
    weights['log'] = [w / mean_log for w in weights_log]
    
    # 5. SMOOTH (smoothed effective number)
    beta = 0.9999
    smoothing_factor = 1.0
    
    def smoothed_effective_number(count):
        smoothed_count = count + smoothing_factor
        return (1 - beta) / (1 - beta ** smoothed_count) if smoothed_count > 0 else 0
    
    sen_frail = smoothed_effective_number(frail_count)
    sen_prefrail = smoothed_effective_number(prefrail_count)
    sen_nonfrail = smoothed_effective_number(nonfrail_count)
    sen_total = sen_frail + sen_prefrail + sen_nonfrail
    
    weights_smoothed_eff = [
        sen_total / (num_classes * sen_frail) if sen_frail > 0 else 1.0,
        sen_total / (num_classes * sen_prefrail) if sen_prefrail > 0 else 1.0,
        sen_total / (num_classes * sen_nonfrail) if sen_nonfrail > 0 else 1.0
    ]
    mean_smoothed_eff = sum(weights_smoothed_eff) / num_classes
    weights['smooth'] = [w / mean_smoothed_eff for w in weights_smoothed_eff]
    
    return weights

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

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    # Get class distribution
    class_counts, total = get_class_distribution()
    
    # Calculate class weights
    class_weights = calculate_class_weights(class_counts, total)
    
    # Load performance data
    experiments = load_part2_data(csv_file)
    
    if not experiments:
        print("❌ No Part 2 data found!")
        return
    
    print("=" * 120)
    print("PART 2: CLASS WEIGHTS - COMPREHENSIVE TABLE")
    print("=" * 120)
    
    # Show class distribution
    print(f"\nCLASS DISTRIBUTION (Training Set):")
    print(f"  Frail:    {class_counts['Frail']} subjects ({class_counts['Frail']/total*100:.1f}%)")
    print(f"  Prefrail: {class_counts['Prefrail']} subjects ({class_counts['Prefrail']/total*100:.1f}%)")
    print(f"  Nonfrail: {class_counts['Nonfrail']} subjects ({class_counts['Nonfrail']/total*100:.1f}%)")
    print(f"  Total:    {total} subjects")
    print(f"\n  Note: Nonfrail is the minority class (needs higher weight)")
    
    # Calculate statistics for each experiment
    summary_data = []
    
    for exp_name in sorted(experiments.keys()):
        data = experiments[exp_name]
        
        accuracies = [d['accuracy'] for d in data]
        f1_scores = [d['f1'] for d in data if d['f1'] is not None]
        
        best_acc = max(accuracies)
        lowest_acc = min(accuracies)
        mean_acc = sum(accuracies) / len(accuracies)
        best_acc_row = max(data, key=lambda x: x['accuracy'])
        best_acc_iter = best_acc_row['iteration']
        f1_at_best_acc = best_acc_row['f1'] if best_acc_row['f1'] is not None else None
        
        best_f1 = max(f1_scores) if f1_scores else None
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
        
        clean_name = get_clean_experiment_name(exp_name)
        
        # Get class weights for this strategy
        strategy_key = exp_name.replace('REDO_', '').lower()
        weights = class_weights.get(strategy_key, [1.0, 1.0, 1.0])
        
        summary_data.append({
            'Experiment': clean_name,
            'Original Name': exp_name,
            'Frail Weight': weights[0],
            'Prefrail Weight': weights[1],
            'Nonfrail Weight': weights[2],
            'Best Accuracy (%)': best_acc,
            'F1 at Best Acc': f1_at_best_acc,
            'Best Acc Iteration': best_acc_iter,
            'Lowest Accuracy (%)': lowest_acc,
            'Mean Accuracy (%)': mean_acc,
            'Best F1': best_f1,
            'Mean F1': mean_f1,
            'Num Evaluations': len(data)
        })
    
    # Print comprehensive table
    print(f"\n{'='*120}")
    print("COMPREHENSIVE TABLE: Class Weights + Performance")
    print(f"{'='*120}")
    print(f"\n{'Strategy':<18} {'Frail W':<8} {'Prefrail W':<10} {'Nonfrail W':<11} {'Best Acc':<10} {'F1@Best':<10} {'Mean Acc':<10} {'Mean F1':<10}")
    print("-" * 120)
    
    for row in summary_data:
        f1_at_best_str = f"{row['F1 at Best Acc']:.4f}" if row['F1 at Best Acc'] is not None else "N/A"
        mean_f1_str = f"{row['Mean F1']:.4f}" if row['Mean F1'] is not None else "N/A"
        
        print(f"{row['Experiment']:<18} {row['Frail Weight']:<8.3f} {row['Prefrail Weight']:<10.3f} "
              f"{row['Nonfrail Weight']:<11.3f} {row['Best Accuracy (%)']:<10.2f} {f1_at_best_str:<10} "
              f"{row['Mean Accuracy (%)']:<10.2f} {mean_f1_str:<10}")
    
    # Save to CSV
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_output = os.path.join(output_dir, 'part2_class_weights_summary_table.csv')
    
    with open(csv_output, 'w', newline='') as f:
        fieldnames = ['Experiment', 'Original Name', 'Frail Weight', 'Prefrail Weight', 'Nonfrail Weight',
                     'Best Accuracy (%)', 'F1 at Best Acc', 'Best Acc Iteration',
                     'Lowest Accuracy (%)', 'Mean Accuracy (%)', 'Best F1', 'Mean F1', 'Num Evaluations']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in summary_data:
            csv_row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(csv_row)
    
    print(f"\n{'='*120}")
    print(f"✓ Saved summary table to: {csv_output}")
    print(f"{'='*120}")
    
    # Create LaTeX table
    latex_file = os.path.join(output_dir, 'part2_class_weights_summary_table_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Part 2: Class Weights - Performance vs. Class Distribution}\n")
        f.write("\\label{tab:part2_class_weights}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("Strategy & \\multicolumn{3}{c}{Class Weights} & Best Acc (\\%) & F1@BestAcc & Mean Acc (\\%) & Mean F1 \\\\\n")
        f.write(" & Frail & Prefrail & Nonfrail & & & & \\\\\n")
        f.write("\\midrule\n")
        
        for row in summary_data:
            clean_name = row['Experiment'].replace('&', '\\&')
            f1_at_best_str = f"{row['F1 at Best Acc']:.4f}" if row['F1 at Best Acc'] is not None else "N/A"
            mean_f1_str = f"{row['Mean F1']:.4f}" if row['Mean F1'] is not None else "N/A"
            
            f.write(f"{clean_name} & {row['Frail Weight']:.3f} & {row['Prefrail Weight']:.3f} & "
                   f"{row['Nonfrail Weight']:.3f} & {row['Best Accuracy (%)']:.2f} & {f1_at_best_str} & "
                   f"{row['Mean Accuracy (%)']:.2f} & {mean_f1_str} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to: {latex_file}")
    print(f"{'='*120}")
    
    # Analysis
    print(f"\n{'='*120}")
    print("ANALYSIS:")
    print(f"{'='*120}")
    
    # Find best performing strategy
    best_mean_acc = max(summary_data, key=lambda x: x['Mean Accuracy (%)'])
    print(f"\n✓ Best Mean Accuracy: {best_mean_acc['Experiment']} ({best_mean_acc['Mean Accuracy (%)']:.2f}%)")
    print(f"  - Nonfrail weight: {best_mean_acc['Nonfrail Weight']:.3f} (minority class)")
    
    # Compare to uniform (baseline)
    uniform_row = next((r for r in summary_data if 'Uniform' in r['Experiment']), None)
    if uniform_row:
        print(f"\n✓ Baseline (Uniform, no weighting): {uniform_row['Mean Accuracy (%)']:.2f}%")
        improvement = best_mean_acc['Mean Accuracy (%)'] - uniform_row['Mean Accuracy (%)']
        print(f"  - Improvement with weighting: +{improvement:.2f}%")
    
    print(f"\n{'='*120}")

if __name__ == '__main__':
    main()

