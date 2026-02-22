#!/usr/bin/env python3
"""
Aggregate results from best checkpoints for all SwinGait Part 1 models.
Reads metrics from training logs and computes mean/std across folds.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

def find_metrics_at_checkpoint(log_file, checkpoint_iter):
    """Find metrics at a specific checkpoint iteration from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {log_file}: {e}")
        return None
    
    # Look for iteration followed by "Running validation..." or "Running test..." or "Running evaluation..."
    # Then find metrics within the next 50 lines
    pattern = rf'Iteration\s+0*{checkpoint_iter}[^\d].*?Running (?:validation|test|evaluation)\.\.\..*?Overall Accuracy:\s*([\d.]+)%.*?Precision \(macro\):\s*([\d.]+)%.*?Recall \(macro\):\s*([\d.]+)%.*?F1 Score \(macro\):\s*([\d.]+)%.*?ROC AUC \(macro\):\s*([\d.]+).*?ROC AUC \(micro\):\s*([\d.]+)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return {
            'iteration': checkpoint_iter,
            'accuracy': float(match.group(1)),
            'precision': float(match.group(2)),
            'recall': float(match.group(3)),
            'f1': float(match.group(4)),
            'auc_macro': float(match.group(5)),
            'auc_micro': float(match.group(6)),
        }
    
    return None

def aggregate_results(all_results):
    """Aggregate results across all folds"""
    if not all_results:
        return None
    
    # Collect all metrics (only numeric metrics, skip fold, checkpoint_iter, note, model_name)
    skip_keys = {'fold', 'checkpoint_iter', 'note', 'model_name'}
    metrics_dict = {}
    for result in all_results:
        for key, value in result.items():
            if key in skip_keys:
                continue
            if key not in metrics_dict:
                metrics_dict[key] = []
            # Only include non-None numeric values
            if value is not None and isinstance(value, (int, float)):
                metrics_dict[key].append(value)
    
    # Compute mean and std for numeric metrics
    summary = {}
    for key, values in metrics_dict.items():
        if values:  # Only compute if we have values
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_values'] = values
    
    return summary

def process_model(model_name, model_dir_prefix, checkpoints_dict, output_base_dir='output'):
    """Process a single model and aggregate results across folds"""
    print(f"\n{'='*70}")
    print(f"Processing: {model_name}")
    print(f"{'='*70}")
    print(f"Checkpoint iterations:")
    for fold_num, checkpoint_iter in sorted(checkpoints_dict.items()):
        print(f"  Fold {fold_num}: {checkpoint_iter}")
    print()
    
    all_results = []
    
    # Process each fold
    for fold_num in range(1, 6):
        checkpoint_iter = checkpoints_dict.get(fold_num)
        if checkpoint_iter is None:
            print(f"  ⚠️  No checkpoint specified for fold {fold_num}, skipping...")
            continue
        
        print(f"Processing fold {fold_num} (checkpoint {checkpoint_iter})...")
        
        # Find the log file for this fold
        fold_dir = os.path.join(output_base_dir, f"{model_dir_prefix}_fold{fold_num}")
        
        if not os.path.exists(fold_dir):
            print(f"  ⚠️  Directory not found: {fold_dir}")
            continue
        
        # Find SwinGait subdirectory
        swingait_dir = os.path.join(fold_dir, 'SwinGait')
        if not os.path.exists(swingait_dir):
            print(f"  ⚠️  SwinGait directory not found: {swingait_dir}")
            continue
        
        # Find the experiment directory (should match the save_name)
        exp_dirs = [d for d in os.listdir(swingait_dir) if os.path.isdir(os.path.join(swingait_dir, d))]
        if not exp_dirs:
            print(f"  ⚠️  No experiment directory found in {swingait_dir}")
            continue
        
        exp_dir = os.path.join(swingait_dir, exp_dirs[0])
        log_dir = os.path.join(exp_dir, 'logs')
        
        if not os.path.exists(log_dir):
            print(f"  ⚠️  Log directory not found: {log_dir}")
            continue
        
        # Find log files
        log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.txt')])
        if not log_files:
            print(f"  ⚠️  No log files found in {log_dir}")
            continue
        
        # Try each log file
        checkpoint_metrics = None
        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            checkpoint_metrics = find_metrics_at_checkpoint(log_path, checkpoint_iter)
            if checkpoint_metrics:
                break
        
        if checkpoint_metrics is not None:
            print(f"  ✓ Found metrics at iteration {checkpoint_iter}")
            
            # Convert metrics to 0-1 range for consistency
            metrics = {
                'model_name': model_name,
                'fold': fold_num,
                'checkpoint_iter': checkpoint_iter,
                'accuracy': checkpoint_metrics['accuracy'] / 100.0,  # Convert to 0-1 range
                'precision': checkpoint_metrics.get('precision', None) / 100.0 if checkpoint_metrics.get('precision') is not None else None,
                'recall': checkpoint_metrics.get('recall', None) / 100.0 if checkpoint_metrics.get('recall') is not None else None,
                'f1': checkpoint_metrics.get('f1', None) / 100.0 if checkpoint_metrics.get('f1') is not None else None,
                'auc_macro': checkpoint_metrics.get('auc_macro', None),
                'auc_micro': checkpoint_metrics.get('auc_micro', None),
                'note': 'Best checkpoint'
            }
            all_results.append(metrics)
        else:
            print(f"  ⚠️  Could not find metrics at checkpoint {checkpoint_iter} for fold {fold_num}")
    
    # Aggregate results
    if all_results:
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS BY FOLD - {model_name}")
        print(f"{'='*70}")
        
        # Print table with separators
        col_widths = [6, 12, 12, 12, 12, 12, 12, 12]
        header = f"| {'Fold':<{col_widths[0]}} | {'Checkpoint':<{col_widths[1]}} | {'Accuracy':<{col_widths[2]}} | {'Precision':<{col_widths[3]}} | {'Recall':<{col_widths[4]}} | {'F1':<{col_widths[5]}} | {'AUC Macro':<{col_widths[6]}} | {'AUC Micro':<{col_widths[7]}} |"
        separator = "+" + "-" * (col_widths[0] + 2) + "+" + "-" * (col_widths[1] + 2) + "+" + "-" * (col_widths[2] + 2) + "+" + "-" * (col_widths[3] + 2) + "+" + "-" * (col_widths[4] + 2) + "+" + "-" * (col_widths[5] + 2) + "+" + "-" * (col_widths[6] + 2) + "+" + "-" * (col_widths[7] + 2) + "+"
        
        print(f"\n{separator}")
        print(header)
        print(separator)
        
        for result in sorted(all_results, key=lambda x: x['fold']):
            fold_num = result['fold']
            checkpoint_iter = result['checkpoint_iter']
            accuracy = result.get('accuracy', None)
            precision = result.get('precision', None)
            recall = result.get('recall', None)
            f1 = result.get('f1', None)
            auc_macro = result.get('auc_macro', None)
            auc_micro = result.get('auc_micro', None)
            
            acc_str = f"{accuracy*100:.2f}%" if accuracy is not None else "N/A"
            prec_str = f"{precision*100:.2f}%" if precision is not None else "N/A"
            rec_str = f"{recall*100:.2f}%" if recall is not None else "N/A"
            f1_str = f"{f1*100:.2f}%" if f1 is not None else "N/A"
            auc_macro_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
            auc_micro_str = f"{auc_micro:.4f}" if auc_micro is not None else "N/A"
            
            row = f"| {fold_num:<{col_widths[0]}} | {checkpoint_iter:<{col_widths[1]}} | {acc_str:<{col_widths[2]}} | {prec_str:<{col_widths[3]}} | {rec_str:<{col_widths[4]}} | {f1_str:<{col_widths[5]}} | {auc_macro_str:<{col_widths[6]}} | {auc_micro_str:<{col_widths[7]}} |"
            print(row)
            print(separator)
        
        print(f"\n{'='*70}")
        print(f"AGGREGATED RESULTS - {model_name}")
        print(f"{'='*70}")
        
        summary = aggregate_results(all_results)
        
        # Print summary
        print("\nMean ± Std across folds:")
        metric_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_macro', 'auc_micro']
        for key in metric_order:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in summary:
                mean_val = summary[mean_key]
                std_val = summary[std_key]
                # Format percentage metrics (0-1 range) as percentages
                if key in ['accuracy', 'precision', 'recall', 'f1']:
                    print(f"  {key:15s}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")
                else:
                    print(f"  {key:15s}: {mean_val:.4f} ± {std_val:.4f}")
        
        return {
            'model_name': model_name,
            'summary': summary,
            'individual_results': all_results
        }
    else:
        print(f"\n❌ ERROR: No results collected for {model_name}!")
        return None

def main():
    # Define models and their best checkpoints
    models_config = {
        'SwinGait (UF)': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_pretrained(UF)',
            'checkpoints': {
                1: 1500,
                2: 500,
                3: 2000,
                4: 500,
                5: 9000,
            }
        },
        'p+CNN': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_p+CNN',
            'checkpoints': {
                1: 1500,
                2: 500,
                3: 5000,
                4: 500,
                5: 8000,
            }
        },
        'p+CNN+Tintro': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro',
            'checkpoints': {
                1: 1500,
                2: 500,
                3: 2000,
                4: 500,
                5: 8000,
            }
        },
        'T1': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1',
            'checkpoints': {
                1: 1000,
                2: 1000,
                3: 7000,
                4: 7000,
                5: 9500,
            }
        },
        'T2': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2',
            'checkpoints': {
                1: 9500,
                2: 8500,
                3: 1500,
                4: 6500,
                5: 4000,
            }
        },
    }
    
    print(f"\n{'='*70}")
    print("AGGREGATING SWINGAIT PART 1 RESULTS")
    print(f"{'='*70}")
    print(f"Total models: {len(models_config)}")
    print(f"{'='*70}\n")
    
    all_model_results = []
    
    # Process each model
    for model_name, config in models_config.items():
        result = process_model(
            model_name,
            config['dir_prefix'],
            config['checkpoints']
        )
        if result:
            all_model_results.append(result)
    
    # Create summary across all models
    if all_model_results:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL MODELS")
        print(f"{'='*70}")
        
        # Print comparison table with separators
        col_widths = [20, 25, 25, 25, 25, 25, 25]
        header = f"| {'Model':<{col_widths[0]}} | {'Accuracy':<{col_widths[1]}} | {'Precision':<{col_widths[2]}} | {'Recall':<{col_widths[3]}} | {'F1':<{col_widths[4]}} | {'AUC Macro':<{col_widths[5]}} | {'AUC Micro':<{col_widths[6]}} |"
        separator = "+" + "-" * (col_widths[0] + 2) + "+" + "-" * (col_widths[1] + 2) + "+" + "-" * (col_widths[2] + 2) + "+" + "-" * (col_widths[3] + 2) + "+" + "-" * (col_widths[4] + 2) + "+" + "-" * (col_widths[5] + 2) + "+" + "-" * (col_widths[6] + 2) + "+"
        
        print(f"\n{separator}")
        print(header)
        print(separator)
        
        for model_result in all_model_results:
            model_name = model_result['model_name']
            summary = model_result['summary']
            
            acc_str = f"{summary.get('accuracy_mean', 0)*100:.2f}% ± {summary.get('accuracy_std', 0)*100:.2f}%" if 'accuracy_mean' in summary else "N/A"
            prec_str = f"{summary.get('precision_mean', 0)*100:.2f}% ± {summary.get('precision_std', 0)*100:.2f}%" if 'precision_mean' in summary else "N/A"
            rec_str = f"{summary.get('recall_mean', 0)*100:.2f}% ± {summary.get('recall_std', 0)*100:.2f}%" if 'recall_mean' in summary else "N/A"
            f1_str = f"{summary.get('f1_mean', 0)*100:.2f}% ± {summary.get('f1_std', 0)*100:.2f}%" if 'f1_mean' in summary else "N/A"
            auc_macro_str = f"{summary.get('auc_macro_mean', 0):.4f} ± {summary.get('auc_macro_std', 0):.4f}" if 'auc_macro_mean' in summary else "N/A"
            auc_micro_str = f"{summary.get('auc_micro_mean', 0):.4f} ± {summary.get('auc_micro_std', 0):.4f}" if 'auc_micro_mean' in summary else "N/A"
            
            row = f"| {model_name:<{col_widths[0]}} | {acc_str:<{col_widths[1]}} | {prec_str:<{col_widths[2]}} | {rec_str:<{col_widths[3]}} | {f1_str:<{col_widths[4]}} | {auc_macro_str:<{col_widths[5]}} | {auc_micro_str:<{col_widths[6]}} |"
            print(row)
            print(separator)
        
        # Save all results
        output_dir = 'kfold_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual model summaries
        for model_result in all_model_results:
            model_name = model_result['model_name']
            safe_name = model_name.replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
            
            summary_file = os.path.join(output_dir, f'{safe_name}_aggregated.json')
            with open(summary_file, 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'summary': model_result['summary'],
                    'individual_results': model_result['individual_results']
                }, f, indent=2)
            print(f"\n✓ Saved {model_name} results to: {summary_file}")
        
        # Save combined summary
        combined_summary = {
            'models': all_model_results
        }
        combined_file = os.path.join(output_dir, 'swingait_part1_all_models_aggregated.json')
        with open(combined_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"\n✓ Saved combined summary to: {combined_file}")
        
        # Create CSV for easy viewing
        all_individual_results = []
        for model_result in all_model_results:
            all_individual_results.extend(model_result['individual_results'])
        
        df = pd.DataFrame(all_individual_results)
        csv_file = os.path.join(output_dir, 'swingait_part1_all_models_results.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV to: {csv_file}")
    
    print(f"\n{'='*70}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
