#!/usr/bin/env python3
"""
Create a summary table for Part 3 (Loss Functions) experiments.
Shows best accuracy, F1, precision, and recall for each loss combination.
"""

import csv
import os
from collections import defaultdict

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
        'REDO_Frailty_ccpg_pt1_p+CNN': 'Triplet + CE (Baseline)',
        'REDO_Frailty_ccpg_pt3_ce_contrastive': 'CE + Contrastive',
        'REDO_Frailty_ccpg_pt3_triplet_focal': 'Triplet + Focal',
        'REDO_Frailty_ccpg_pt3_contrastive_focal': 'Contrastive + Focal',
        'Frailty_part3_baseline_tripfocal': 'Triplet + CE (Baseline)',
    }
    return name_map.get(exp_name, exp_name.replace('REDO_', '').replace('Frailty_ccpg_pt3_', ''))

def create_summary_table(experiments, output_dir='results_visualization'):
    """Create a summary table with best metrics for each experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate best metrics for each experiment
    summary_data = []
    
    for exp_name in sorted(experiments.keys()):
        data = experiments[exp_name]
        if not data:
            continue
        
        # Find best accuracy and corresponding metrics
        best_idx = max(range(len(data)), key=lambda i: data[i]['accuracy'])
        best_acc = data[best_idx]['accuracy']
        best_f1 = data[best_idx]['f1']
        best_precision = data[best_idx]['precision']
        best_recall = data[best_idx]['recall']
        best_iter = data[best_idx]['iteration']
        
        clean_name = get_clean_experiment_name(exp_name)
        
        summary_data.append({
            'experiment': clean_name,
            'best_accuracy': best_acc,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_iteration': best_iter
        })
    
    # Sort by best accuracy (descending)
    summary_data.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    # Create CSV
    csv_file = os.path.join(output_dir, 'part3_summary_table.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'best_accuracy', 'best_f1', 'best_precision', 'best_recall', 'best_iteration'])
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    
    print(f"✓ Saved CSV: {csv_file}")
    
    # Create LaTeX table
    latex_file = os.path.join(output_dir, 'part3_summary_table_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Part 3: Loss Functions - Best Performance Metrics}\n")
        f.write("\\label{tab:part3_loss_functions}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Loss Combination & Best Acc. (\\%) & F1 & Precision & Recall \\\\\n")
        f.write("\\midrule\n")
        
        for row in summary_data:
            exp = row['experiment'].replace('&', '\\&')
            acc = f"{row['best_accuracy']:.2f}"
            f1 = f"{row['best_f1']:.3f}" if row['best_f1'] is not None else "N/A"
            prec = f"{row['best_precision']:.3f}" if row['best_precision'] is not None else "N/A"
            rec = f"{row['best_recall']:.3f}" if row['best_recall'] is not None else "N/A"
            
            # Bold the best accuracy row
            if row == summary_data[0]:
                f.write(f"\\textbf{{{exp}}} & \\textbf{{{acc}}} & \\textbf{{{f1}}} & \\textbf{{{prec}}} & \\textbf{{{rec}}} \\\\\n")
            else:
                f.write(f"{exp} & {acc} & {f1} & {prec} & {rec} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX: {latex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("Part 3: Loss Functions - Summary Table")
    print("="*80)
    print(f"{'Loss Combination':<30} {'Best Acc (%)':<15} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*80)
    
    for row in summary_data:
        exp = row['experiment']
        acc = f"{row['best_accuracy']:.2f}"
        f1 = f"{row['best_f1']:.3f}" if row['best_f1'] is not None else "N/A"
        prec = f"{row['best_precision']:.3f}" if row['best_precision'] is not None else "N/A"
        rec = f"{row['best_recall']:.3f}" if row['best_recall'] is not None else "N/A"
        print(f"{exp:<30} {acc:<15} {f1:<10} {prec:<10} {rec:<10}")
    
    print("="*80)
    print(f"\nBest performing: {summary_data[0]['experiment']} with {summary_data[0]['best_accuracy']:.2f}% accuracy")
    print(f"  - F1: {summary_data[0]['best_f1']:.3f}" if summary_data[0]['best_f1'] else "  - F1: N/A")
    print(f"  - Precision: {summary_data[0]['best_precision']:.3f}" if summary_data[0]['best_precision'] else "  - Precision: N/A")
    print(f"  - Recall: {summary_data[0]['best_recall']:.3f}" if summary_data[0]['best_recall'] else "  - Recall: N/A")

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
    
    create_summary_table(experiments)

if __name__ == '__main__':
    main()


