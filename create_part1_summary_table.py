#!/usr/bin/env python3
"""
Create a summary table for Part 1 (Freezing Strategy) experiments.
Shows best accuracy, lowest accuracy, mean accuracy, best F1, and mean F1 for each experiment.
"""

import csv
import os
from collections import defaultdict

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
                    'f1': float(row['F1']) if row['F1'] else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
    return experiments

def get_clean_experiment_name(exp_name):
    """Get a cleaner name for display."""
    # Remove prefix
    name = exp_name.replace('REDO_Frailty_ccpg_pt1_', '')
    # Make it more readable
    name_map = {
        'p+CNN': 'Pretrained + CNN',
        'p+CNN+Tintro': 'Pretrained + CNN + Tintro',
        'p+CNN+Tintro+T1': 'Pretrained + CNN + Tintro + T1',
        'p+CNN+Tintro+T1+T2': 'Pretrained + CNN + Tintro + T1 + T2',
        'pretrained(UF)': 'Pretrained (Unfrozen)'
    }
    return name_map.get(name, name)

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    experiments = load_part1_data(csv_file)
    
    if not experiments:
        print("❌ No Part 1 data found!")
        return
    
    print("=" * 100)
    print("PART 1: FREEZING STRATEGY - SUMMARY TABLE")
    print("=" * 100)
    
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
        
        # Get F1 score at the iteration where best accuracy occurred
        f1_at_best_acc = best_acc_row['f1'] if best_acc_row['f1'] is not None else None
        
        best_f1 = max(f1_scores) if f1_scores else None
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
        
        clean_name = get_clean_experiment_name(exp_name)
        
        # Determine frozen/unfrozen status based on experiment name
        exp_lower = exp_name.lower()
        cnn_frozen = '✓' if 'pretrained(uf)' not in exp_lower else '✗'
        has_t1 = 'tintro+t1' in exp_lower or 'tintro+t1+t2' in exp_lower
        has_t2 = 'tintro+t1+t2' in exp_lower
        t1_frozen = '✓' if has_t1 else '—'
        t2_frozen = '✓' if has_t2 else '—'
        
        summary_data.append({
            'Experiment': clean_name,
            'Original Name': exp_name,
            'Best Accuracy (%)': best_acc,
            'F1 at Best Acc': f1_at_best_acc,
            'Best Acc Iteration': best_acc_iter,
            'Lowest Accuracy (%)': lowest_acc,
            'Mean Accuracy (%)': mean_acc,
            'Best F1': best_f1,
            'Mean F1': mean_f1,
            'CNN Frozen': cnn_frozen,
            'T1 Frozen': t1_frozen,
            'T2 Frozen': t2_frozen,
            'Num Evaluations': len(data)
        })
    
    # Print table
    print(f"\n{'Experiment':<35} {'Best Acc (%)':<12} {'F1@BestAcc':<12} {'Lowest Acc (%)':<14} {'Mean Acc (%)':<13} {'CNN':<6} {'T1':<6} {'T2':<6}")
    print("-" * 110)
    
    for row in summary_data:
        f1_at_best_str = f"{row['F1 at Best Acc']:.4f}" if row['F1 at Best Acc'] is not None else "N/A"
        
        print(f"{row['Experiment']:<35} {row['Best Accuracy (%)']:<12.2f} {f1_at_best_str:<12} "
              f"{row['Lowest Accuracy (%)']:<14.2f} {row['Mean Accuracy (%)']:<13.2f} "
              f"{row['CNN Frozen']:<6} {row['T1 Frozen']:<6} {row['T2 Frozen']:<6}")
    
    # Save to CSV
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_output = os.path.join(output_dir, 'part1_summary_table.csv')
    
    with open(csv_output, 'w', newline='') as f:
        fieldnames = ['Experiment', 'Original Name', 'Best Accuracy (%)', 'F1 at Best Acc', 'Best Acc Iteration', 
                     'Lowest Accuracy (%)', 'Mean Accuracy (%)', 'Best F1', 'Mean F1', 
                     'CNN Frozen', 'T1 Frozen', 'T2 Frozen', 'Num Evaluations']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in summary_data:
            # Convert None to empty string for CSV
            csv_row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(csv_row)
    
    print(f"\n{'='*100}")
    print(f"✓ Saved summary table to: {csv_output}")
    print(f"{'='*100}")
    
    # Also create a LaTeX table format (useful for papers)
    latex_file = os.path.join(output_dir, 'part1_summary_table_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Part 1: Freezing Strategy - Performance Summary}\n")
        f.write("\\label{tab:part1_summary}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("Experiment & Best Acc (\\%) & F1@BestAcc & Lowest Acc (\\%) & Mean Acc (\\%) & CNN & T1 & T2 \\\\\n")
        f.write("\\midrule\n")
        
        for row in summary_data:
            clean_name = row['Experiment'].replace('&', '\\&')
            f1_at_best_str = f"{row['F1 at Best Acc']:.4f}" if row['F1 at Best Acc'] is not None else "N/A"
            # Convert checkmarks to LaTeX
            cnn_latex = "$\\checkmark$" if row['CNN Frozen'] == '✓' else "$\\times$"
            t1_latex = "$\\checkmark$" if row['T1 Frozen'] == '✓' else ("$\\times$" if row['T1 Frozen'] == '✗' else "—")
            t2_latex = "$\\checkmark$" if row['T2 Frozen'] == '✓' else ("$\\times$" if row['T2 Frozen'] == '✗' else "—")
            
            f.write(f"{clean_name} & {row['Best Accuracy (%)']:.2f} & {f1_at_best_str} & "
                   f"{row['Lowest Accuracy (%)']:.2f} & {row['Mean Accuracy (%)']:.2f} & "
                   f"{cnn_latex} & {t1_latex} & {t2_latex} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to: {latex_file}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()

