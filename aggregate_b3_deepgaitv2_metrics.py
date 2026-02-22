#!/usr/bin/env python3
"""
Aggregate metrics for DeepGaitV2 B3 unfrozen model across k-folds.
Extracts metrics at specific iterations for each fold and computes mean ± std.
"""

import os
import re
import glob
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Model configuration: {fold: iteration}
MODEL_CONFIG = {
    1: 6000,
    2: 500,
    3: 4000,
    4: 5500,
    5: 1500
}

# Model directory pattern
MODEL_DIR_PATTERN = 'REDO_Frailty_ccpg_pt4a_deepgaitv2_B3_unfrozen_fold'

def extract_metrics_from_log(log_file, target_iteration):
    """Extract metrics from log file at specific iteration (or closest)."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Split by evaluation sections
        eval_sections = re.split(r'EVALUATION RESULTS', content)
        
        # Find all evaluations with their iteration numbers
        evaluations = []
        for section in eval_sections[1:]:  # Skip first empty part
            # Look backwards for iteration number
            section_start = content.find(section)
            before_text = content[max(0, section_start - 2000):section_start]
            
            # Try to find iteration number before this section
            iter_matches = list(re.finditer(r'Iteration\s+0*(\d+)', before_text))
            if not iter_matches:
                continue
            
            # Get the iteration number just before this evaluation
            last_iter_match = iter_matches[-1]
            iter_num = int(last_iter_match.group(1))
            evaluations.append((iter_num, section))
        
        if not evaluations:
            return None
        
        # Find the evaluation closest to target_iteration
        best_iter = None
        best_section = None
        min_diff = float('inf')
        
        for iter_num, section in evaluations:
            diff = abs(iter_num - target_iteration)
            if diff < min_diff:
                min_diff = diff
                best_iter = iter_num
                best_section = section
        
        # Only use if within reasonable range (within 500 iterations)
        if min_diff > 500:
            return None
        
        section = best_section
        
        # Extract metrics from this section
        metrics = {}
        
        # Overall Accuracy
        acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
        if acc_match:
            metrics['overall_accuracy'] = float(acc_match.group(1))
        
        # Macro-averaged metrics
        prec_match = re.search(r'Precision \(macro\):\s*(\d+\.?\d*)%', section)
        if prec_match:
            metrics['precision_macro'] = float(prec_match.group(1))
        
        recall_match = re.search(r'Recall \(macro\):\s*(\d+\.?\d*)%', section)
        if recall_match:
            metrics['recall_macro'] = float(recall_match.group(1))
        
        f1_match = re.search(r'F1 Score \(macro\):\s*(\d+\.?\d*)%', section)
        if f1_match:
            metrics['f1_macro'] = float(f1_match.group(1))
        
        # Cohen's Kappa
        kappa_match = re.search(r"Cohen's Kappa \(linear weighted\):\s*([\d.]+)", section)
        if kappa_match:
            metrics['cohen_kappa'] = float(kappa_match.group(1))
        
        # ROC AUC
        auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', section)
        if auc_macro_match:
            metrics['auc_macro'] = float(auc_macro_match.group(1))
        
        auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', section)
        if auc_micro_match:
            metrics['auc_micro'] = float(auc_micro_match.group(1))
        
        # Per-class metrics
        class_names = ['Frail', 'Prefrail', 'Nonfrail']
        for class_name in class_names:
            # Sensitivity (Recall)
            sens_match = re.search(
                rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%',
                section
            )
            if sens_match:
                metrics[f'{class_name.lower()}_sensitivity'] = float(sens_match.group(1))
            
            # Specificity
            spec_match = re.search(
                rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%',
                section
            )
            if spec_match:
                metrics[f'{class_name.lower()}_specificity'] = float(spec_match.group(1))
            
            # Precision
            prec_match = re.search(
                rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%',
                section
            )
            if prec_match:
                metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1))
        
        # Confusion Matrix
        cm_match = re.search(
            r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
            section
        )
        if cm_match:
            metrics['confusion_matrix'] = [
                [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
            ]
        
        if metrics:
            metrics['iteration'] = best_iter
            return metrics
        
        return None
        
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None

def find_log_file(fold_num):
    """Find the log file for a specific fold."""
    exp_dir = f"output/{MODEL_DIR_PATTERN}{fold_num}"
    
    # Try to find DeepGaitV2 subdirectory
    deepgait_dir = os.path.join(exp_dir, 'DeepGaitV2')
    if os.path.exists(deepgait_dir):
        # Find the model subdirectory
        model_dirs = [d for d in os.listdir(deepgait_dir) if os.path.isdir(os.path.join(deepgait_dir, d))]
        if model_dirs:
            exp_full_dir = os.path.join(deepgait_dir, model_dirs[0])
        else:
            exp_full_dir = deepgait_dir
    else:
        exp_full_dir = exp_dir
    
    logs_dir = os.path.join(exp_full_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    
    # Return the most recent log file
    return max(log_files, key=os.path.getmtime)

def aggregate_metrics():
    """Aggregate metrics across all folds."""
    all_metrics = defaultdict(list)
    
    print(f"\n{'='*80}")
    print("Processing DeepGaitV2 B3 Unfrozen")
    print(f"{'='*80}")
    
    for fold_num, target_iter in MODEL_CONFIG.items():
        log_file = find_log_file(fold_num)
        
        if not log_file:
            print(f"  ✗ Fold {fold_num}: Log file not found")
            continue
        
        print(f"  Fold {fold_num}: Looking for iteration {target_iter} in {os.path.basename(log_file)}")
        
        metrics = extract_metrics_from_log(log_file, target_iter)
        
        if not metrics:
            print(f"    ✗ No metrics found near iteration {target_iter}")
            continue
        
        actual_iter = metrics['iteration']
        if actual_iter != target_iter:
            print(f"    ✓ Found metrics at iteration {actual_iter} (target was {target_iter}, diff: {abs(actual_iter - target_iter)})")
        else:
            print(f"    ✓ Found metrics at iteration {actual_iter}")
        
        # Store all metrics
        for key, value in metrics.items():
            if key != 'iteration' and key != 'confusion_matrix':
                all_metrics[key].append(value)
    
    # Compute statistics
    aggregated = {}
    for key, values in all_metrics.items():
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_values'] = values
    
    return aggregated

def print_focused_summary(results):
    """Print focused summary with only key metrics."""
    print(f"\n{'='*80}")
    print("FOCUSED METRICS SUMMARY - DeepGaitV2 B3 Unfrozen")
    print("(Cohen's Kappa, Accuracy, ROC AUC Macro, ROC AUC Micro)")
    print(f"{'='*80}\n")
    
    # Create a table format
    print(f"{'Model':<30} {'Accuracy (%)':<20} {'Cohen Kappa':<20} {'ROC AUC Macro':<20} {'ROC AUC Micro':<20}")
    print("-" * 110)
    
    # Accuracy
    acc_mean = results.get('overall_accuracy_mean', 0)
    acc_std = results.get('overall_accuracy_std', 0)
    acc_str = f"{acc_mean:.2f} ± {acc_std:.2f}"
    
    # Cohen's Kappa
    kappa_mean = results.get('cohen_kappa_mean', 0)
    kappa_std = results.get('cohen_kappa_std', 0)
    kappa_str = f"{kappa_mean:.4f} ± {kappa_std:.4f}"
    
    # ROC AUC Macro
    auc_macro_mean = results.get('auc_macro_mean', 0)
    auc_macro_std = results.get('auc_macro_std', 0)
    auc_macro_str = f"{auc_macro_mean:.4f} ± {auc_macro_std:.4f}"
    
    # ROC AUC Micro
    auc_micro_mean = results.get('auc_micro_mean', 0)
    auc_micro_std = results.get('auc_micro_std', 0)
    auc_micro_str = f"{auc_micro_mean:.4f} ± {auc_micro_std:.4f}"
    
    print(f"{'DeepGaitV2 B3 Unfrozen':<30} {acc_str:<20} {kappa_str:<20} {auc_macro_str:<20} {auc_micro_str:<20}")
    
    print("\n" + "=" * 110)
    print("CSV FORMAT (for easy copy-paste):")
    print("Model,Accuracy_mean,Accuracy_std,Cohen_Kappa_mean,Cohen_Kappa_std,ROC_AUC_Macro_mean,ROC_AUC_Macro_std,ROC_AUC_Micro_mean,ROC_AUC_Micro_std")
    
    row = [
        'DeepGaitV2_B3_Unfrozen',
        f"{results.get('overall_accuracy_mean', 0):.4f}",
        f"{results.get('overall_accuracy_std', 0):.4f}",
        f"{results.get('cohen_kappa_mean', 0):.4f}",
        f"{results.get('cohen_kappa_std', 0):.4f}",
        f"{results.get('auc_macro_mean', 0):.4f}",
        f"{results.get('auc_macro_std', 0):.4f}",
        f"{results.get('auc_micro_mean', 0):.4f}",
        f"{results.get('auc_micro_std', 0):.4f}"
    ]
    print(','.join(row))
    print()

def print_summary(results):
    """Print a formatted summary of aggregated results."""
    print(f"\n{'-'*80}")
    
    # Overall metrics
    print("  Overall Metrics:")
    overall_metrics = [
        ('overall_accuracy', 'overall accuracy'),
        ('precision_macro', 'precision macro'),
        ('recall_macro', 'recall macro'),
        ('f1_macro', 'f1 macro'),
        ('cohen_kappa', 'cohen kappa'),
        ('auc_macro', 'auc macro'),
        ('auc_micro', 'auc micro')
    ]
    
    for metric_key, metric_display in overall_metrics:
        mean_key = f'{metric_key}_mean'
        std_key = f'{metric_key}_std'
        if mean_key in results and std_key in results:
            mean_val = results[mean_key]
            std_val = results[std_key]
            if 'kappa' in metric_key or 'auc' in metric_key:
                print(f"    {metric_display:25s}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"    {metric_display:25s}: {mean_val:.2f}/% ± {std_val:.2f}/%")
    
    # Per-class metrics
    print("\n  Per-Class Metrics:")
    per_class_metrics = ['sensitivity', 'specificity', 'precision']
    classes = ['frail', 'prefrail', 'nonfrail']
    
    for class_name in classes:
        print(f"    {class_name.capitalize()}:")
        for metric in per_class_metrics:
            full_metric = f'{class_name}_{metric}'
            mean_key = f'{full_metric}_mean'
            std_key = f'{full_metric}_std'
            if mean_key in results and std_key in results:
                mean_val = results[mean_key]
                std_val = results[std_key]
                print(f"      {metric:15s}: {mean_val:.2f}/% ± {std_val:.2f}/%")
    
    print()

def main():
    aggregated = aggregate_metrics()
    
    if not aggregated:
        print("\n✗ No metrics found. Please check the log files.")
        return
    
    # Save to JSON
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'deepgaitv2_b3_unfrozen_aggregated_metrics.json')
    with open(json_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n✓ Saved detailed results to: {json_file}")
    
    # Print focused summary first (key metrics only)
    print_focused_summary(aggregated)
    
    # Print full summary
    print_summary(aggregated)

if __name__ == '__main__':
    main()
