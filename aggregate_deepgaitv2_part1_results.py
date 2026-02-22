#!/usr/bin/env python3
"""
Aggregate results from best checkpoints for all DeepGaitV2 Part 1 models.
Reads metrics from training logs and computes mean/std across folds.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

def find_metrics_at_checkpoint_in_log(log_file, fold_num, checkpoint_iter):
    """Find metrics at a specific checkpoint iteration from log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Warning: Could not read {log_file}: {e}")
        return None
    
    # Find the iteration line and then look for evaluation results
    found_iteration = False
    evaluation_start = None
    
    for i, line in enumerate(lines):
        # Look for the iteration line
        if f'Iteration {checkpoint_iter:05d}' in line or f'Iteration {checkpoint_iter:04d}' in line or f'Iteration {checkpoint_iter}' in line:
            found_iteration = True
            # Look ahead for evaluation results (within next 50 lines)
            for j in range(i, min(i + 50, len(lines))):
                if 'Running evaluation on TEST_SET' in lines[j] or 'EVALUATION RESULTS' in lines[j]:
                    evaluation_start = j
                    break
            if evaluation_start:
                break
    
    if not found_iteration or evaluation_start is None:
        return None
    
    # Extract metrics from the evaluation section
    metrics = {}
    for i in range(evaluation_start, min(evaluation_start + 30, len(lines))):
        line = lines[i]
        
        # Overall Accuracy
        if 'Overall Accuracy:' in line:
            match = re.search(r'Overall Accuracy:\s*([\d.]+)%', line)
            if match:
                metrics['accuracy'] = float(match.group(1)) / 100.0
        
        # Precision (macro)
        if 'Precision (macro):' in line:
            match = re.search(r'Precision \(macro\):\s*([\d.]+)%', line)
            if match:
                metrics['precision'] = float(match.group(1)) / 100.0
        
        # Recall (macro)
        if 'Recall (macro):' in line:
            match = re.search(r'Recall \(macro\):\s*([\d.]+)%', line)
            if match:
                metrics['recall'] = float(match.group(1)) / 100.0
        
        # F1 Score (macro)
        if 'F1 Score (macro):' in line:
            match = re.search(r'F1 Score \(macro\):\s*([\d.]+)%', line)
            if match:
                metrics['f1'] = float(match.group(1)) / 100.0
        
        # ROC AUC (macro)
        if 'ROC AUC (macro):' in line:
            match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', line)
            if match:
                metrics['auc_macro'] = float(match.group(1))
        
        # ROC AUC (micro)
        if 'ROC AUC (micro):' in line:
            match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', line)
            if match:
                metrics['auc_micro'] = float(match.group(1))
    
    if len(metrics) >= 4:  # At least accuracy, precision, recall, f1
        metrics['fold'] = fold_num
        metrics['checkpoint_iter'] = checkpoint_iter
        return metrics
    
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
    """Process a single model across all folds"""
    print(f"\n{'='*70}")
    print(f"Processing: {model_name}")
    print(f"{'='*70}")
    
    print("Checkpoint iterations:")
    for fold_num, checkpoint_iter in sorted(checkpoints_dict.items()):
        print(f"  Fold {fold_num}: {checkpoint_iter}")
    
    all_results = []
    
    for fold_num in sorted(checkpoints_dict.keys()):
        checkpoint_iter = checkpoints_dict[fold_num]
        print(f"\nProcessing fold {fold_num} (checkpoint {checkpoint_iter})...")
        
        # Find the log file
        fold_dir = os.path.join(output_base_dir, f"{model_dir_prefix}_fold{fold_num}")
        model_subdir = os.path.join(fold_dir, "DeepGaitV2", f"{model_dir_prefix}_fold{fold_num}")
        logs_dir = os.path.join(model_subdir, "logs")
        
        if not os.path.exists(logs_dir):
            print(f"  ✗ Logs directory not found: {logs_dir}")
            continue
        
        # Find the most recent log file
        log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.txt')])
        if not log_files:
            print(f"  ✗ No log files found in {logs_dir}")
            continue
        
        log_file = os.path.join(logs_dir, log_files[-1])
        
        # Extract metrics
        result = find_metrics_at_checkpoint_in_log(log_file, fold_num, checkpoint_iter)
        
        if result:
            print(f"  ✓ Found metrics at iteration {checkpoint_iter}")
            all_results.append(result)
        else:
            print(f"  ✗ Could not find metrics at iteration {checkpoint_iter}")
    
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
        'Heavy Frozen': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen',
            'checkpoints': {
                1: 3000,
                2: 10000,
                3: 2000,
                4: 10000,
                5: 3000,
            }
        },
        'First Two Frozen': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen',
            'checkpoints': {
                1: 10000,
                2: 4500,
                3: 500,
                4: 4500,
                5: 8000,
            }
        },
        'First Layer Frozen': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen',
            'checkpoints': {
                1: 8500,
                2: 2500,
                3: 4500,
                4: 500,
                5: 8500,
            }
        },
        'Early Frozen': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen',
            'checkpoints': {
                1: 500,
                2: 5500,
                3: 1500,
                4: 10000,
                5: 6000,
            }
        },
        'All Trainable': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable',
            'checkpoints': {
                1: 1000,
                2: 6000,
                3: 2000,
                4: 500,
                5: 5000,
            }
        },
        'All Frozen': {
            'dir_prefix': 'REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen',
            'checkpoints': {
                1: 500,
                2: 9000,
                3: 8000,
                4: 3500,
                5: 7500,
            }
        },
    }
    
    print("="*70)
    print("AGGREGATING DEEPGaitV2 PART 1 RESULTS")
    print("="*70)
    print(f"Total models: {len(models_config)}")
    print("="*70)
    
    all_model_results = []
    
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
            
            summary_file = os.path.join(output_dir, f'deepgaitv2_part1_{safe_name}_aggregated.json')
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
        combined_file = os.path.join(output_dir, 'deepgaitv2_part1_all_models_aggregated.json')
        with open(combined_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"\n✓ Saved combined summary to: {combined_file}")
        
        # Create CSV for easy viewing
        all_individual_results = []
        for model_result in all_model_results:
            for result in model_result['individual_results']:
                result['model_name'] = model_result['model_name']
            all_individual_results.extend(model_result['individual_results'])
        
        df = pd.DataFrame(all_individual_results)
        csv_file = os.path.join(output_dir, 'deepgaitv2_part1_all_models_results.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV to: {csv_file}")
    
    print(f"\n{'='*70}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
