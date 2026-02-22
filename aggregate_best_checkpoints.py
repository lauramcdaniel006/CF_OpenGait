#!/usr/bin/env python3
"""
Simple script to aggregate results from best checkpoints across all folds.
Reads metrics from training logs and computes mean/std.
"""

import os
import sys
import argparse
import json
import glob
import re
import yaml
import numpy as np
import pandas as pd

def find_metrics_at_checkpoint_in_log(log_file, fold_num, checkpoint_iter):
    """Find metrics at a specific checkpoint iteration for a specific fold from log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Warning: Could not read {log_file}: {e}")
        return None
    
    # Find the section for this fold
    fold_start_idx = None
    fold_end_idx = None
    
    for i, line in enumerate(lines):
        # Look for fold marker (e.g., "PROCESSING FOLD 1/5" or "FOLD 1: Training")
        if f'PROCESSING FOLD {fold_num}/' in line or f'FOLD {fold_num}:' in line:
            fold_start_idx = i
            # Look for next fold marker or end of file
            for j in range(i + 1, len(lines)):
                if f'PROCESSING FOLD {fold_num + 1}/' in lines[j] or f'FOLD {fold_num + 1}:' in lines[j]:
                    fold_end_idx = j
                    break
            if fold_end_idx is None:
                fold_end_idx = len(lines)
            break
    
    if fold_start_idx is None:
        return None
    
    # Search within this fold's section
    search_lines = lines[fold_start_idx:fold_end_idx]
    
    i = 0
    while i < len(search_lines):
        # Look for iteration line
        iter_match = re.search(r'Iteration\s+(\d+)', search_lines[i])
        if iter_match:
            iteration = int(iter_match.group(1))
            
            # Check if this is the checkpoint we're looking for
            if iteration == checkpoint_iter:
                # Look forward for "Running validation..." or "Running test..."
                found_validation = False
                for j in range(i, min(i + 10, len(search_lines))):
                    if 'Running validation...' in search_lines[j] or 'Running test...' in search_lines[j]:
                        found_validation = True
                        # Now look forward for metrics (within next 50 lines)
                        for k in range(j, min(j + 50, len(search_lines))):
                            line = search_lines[k]
                            
                            # Find accuracy
                            acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', line)
                            if acc_match:
                                accuracy = float(acc_match.group(1))
                                
                                # Get section for other metrics
                                section_start = max(0, k - 5)
                                section_end = min(len(search_lines), k + 25)
                                section_text = ''.join(search_lines[section_start:section_end])
                                
                                auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', section_text)
                                auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', section_text)
                                f1_match = re.search(r'F1 Score \(macro\):\s*([\d.]+)%', section_text)
                                precision_match = re.search(r'Precision \(macro\):\s*([\d.]+)%', section_text)
                                recall_match = re.search(r'Recall \(macro\):\s*([\d.]+)%', section_text)
                                
                                metric_dict = {
                                    'iteration': iteration,
                                    'accuracy': accuracy,
                                    'auc_macro': float(auc_macro_match.group(1)) if auc_macro_match else None,
                                    'auc_micro': float(auc_micro_match.group(1)) if auc_micro_match else None,
                                    'f1': float(f1_match.group(1)) if f1_match else None,
                                    'precision': float(precision_match.group(1)) if precision_match else None,
                                    'recall': float(recall_match.group(1)) if recall_match else None,
                                }
                                return metric_dict
                        break
                if not found_validation:
                    i += 1
                    continue
        i += 1
    
    return None

def aggregate_results(all_results):
    """Aggregate results across all folds"""
    if not all_results:
        return None
    
    # Collect all metrics (only numeric metrics, skip fold, checkpoint_iter, note)
    skip_keys = {'fold', 'checkpoint_iter', 'note'}
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

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from best checkpoints across all folds'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/swingait/swin_part1_p_CNN.yaml',
        help='Base config file (default: configs/swingait/swin_part1_p_CNN.yaml)'
    )
    parser.add_argument(
        '--k', 
        type=int, 
        default=5,
        help='Number of folds (default: 5)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='kfold_results',
        help='Output directory where fold configs are stored (default: kfold_results)'
    )
    parser.add_argument(
        '--checkpoints', 
        type=str, 
        default=None,
        help='Comma-separated checkpoint iterations for each fold (e.g., "4500,9500,7000,2000,8000"). If not provided, uses handpicked values.'
    )
    
    args = parser.parse_args()
    
    # Define handpicked checkpoints
    HANDPICKED_CHECKPOINTS = {
        1: 4500,
        2: 9500,
        3: 7000,
        4: 2000,
        5: 8000,
    }
    
    # Parse checkpoint iterations
    if args.checkpoints:
        checkpoint_list = [int(x.strip()) for x in args.checkpoints.split(',')]
        if len(checkpoint_list) != args.k:
            print(f"❌ Error: Expected {args.k} checkpoint values, got {len(checkpoint_list)}")
            sys.exit(1)
        checkpoint_dict = {i+1: checkpoint_list[i] for i in range(len(checkpoint_list))}
    else:
        checkpoint_dict = HANDPICKED_CHECKPOINTS
    
    print(f"\n{'='*70}")
    print("AGGREGATING HANDPICKED CHECKPOINT RESULTS")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Folds: {args.k}")
    print(f"Checkpoint iterations:")
    for fold_num, checkpoint_iter in sorted(checkpoint_dict.items()):
        print(f"  Fold {fold_num}: {checkpoint_iter}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    all_results = []
    
    # Process each fold
    for fold_num in range(1, args.k + 1):
        checkpoint_iter = checkpoint_dict.get(fold_num)
        if checkpoint_iter is None:
            print(f"  ⚠️  No checkpoint specified for fold {fold_num}, skipping...")
            continue
        
        print(f"Processing fold {fold_num} (checkpoint {checkpoint_iter})...")
        
        # Find fold config file
        fold_config = os.path.join(args.output_dir, f'config_fold{fold_num}.yaml')
        
        if not os.path.exists(fold_config):
            print(f"  ⚠️  Config file not found: {fold_config}")
            continue
        
        # Load fold config to get experiment directory
        with open(fold_config, 'r') as f:
            fold_config_data = yaml.safe_load(f)
        
        save_name = fold_config_data['trainer_cfg']['save_name']
        dataset_name = fold_config_data['data_cfg']['dataset_name']
        exp_dir = os.path.join('output', dataset_name, 'SwinGait', save_name)
        
        # Try to find metrics from main kfold log file first (find most recent)
        kfold_logs = glob.glob(os.path.join(args.output_dir, 'kfold_run_*.log'))
        checkpoint_metrics = None
        
        if kfold_logs:
            # Use most recent log file
            kfold_log = max(kfold_logs, key=os.path.getmtime)
            checkpoint_metrics = find_metrics_at_checkpoint_in_log(kfold_log, fold_num, checkpoint_iter)
        
        # Fallback to individual log files if main log doesn't work
        if checkpoint_metrics is None:
            log_dir = os.path.join(exp_dir, 'logs')
            if os.path.exists(log_dir):
                log_files = glob.glob(os.path.join(log_dir, '*.txt'))
                # Try to find in individual log files (simpler search without fold context)
                for log_file in sorted(log_files):
                    # Simple search - just look for iteration and metrics
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                        
                        # Look for iteration followed by validation and metrics
                        pattern = rf'Iteration\s+0*{checkpoint_iter}[^\d].*?Running (?:validation|test)\.\.\..*?Overall Accuracy:\s*([\d.]+)%.*?Precision \(macro\):\s*([\d.]+)%.*?Recall \(macro\):\s*([\d.]+)%.*?F1 Score \(macro\):\s*([\d.]+)%.*?ROC AUC \(macro\):\s*([\d.]+).*?ROC AUC \(micro\):\s*([\d.]+)'
                        match = re.search(pattern, content, re.DOTALL)
                        if match:
                            checkpoint_metrics = {
                                'iteration': checkpoint_iter,
                                'accuracy': float(match.group(1)),
                                'precision': float(match.group(2)),
                                'recall': float(match.group(3)),
                                'f1': float(match.group(4)),
                                'auc_macro': float(match.group(5)),
                                'auc_micro': float(match.group(6)),
                            }
                            break
                    except Exception as e:
                        continue
        
        if checkpoint_metrics is not None:
            print(f"  ✓ Found metrics at iteration {checkpoint_iter}")
            
            # Convert metrics to 0-1 range for consistency
            metrics = {
                'fold': fold_num,
                'checkpoint_iter': checkpoint_iter,
                'accuracy': checkpoint_metrics['accuracy'] / 100.0,  # Convert to 0-1 range
                'precision': checkpoint_metrics.get('precision', None) / 100.0 if checkpoint_metrics.get('precision') is not None else None,
                'recall': checkpoint_metrics.get('recall', None) / 100.0 if checkpoint_metrics.get('recall') is not None else None,
                'f1': checkpoint_metrics.get('f1', None) / 100.0 if checkpoint_metrics.get('f1') is not None else None,
                'auc_macro': checkpoint_metrics.get('auc_macro', None),
                'auc_micro': checkpoint_metrics.get('auc_micro', None),
                'note': 'Handpicked checkpoint'
            }
            all_results.append(metrics)
        else:
            print(f"  ⚠️  Could not find metrics at checkpoint {checkpoint_iter} for fold {fold_num}")
    
    # Aggregate results
    if all_results:
        print(f"\n{'='*70}")
        print("DETAILED RESULTS BY FOLD")
        print(f"{'='*70}")
        
        # Print table
        print(f"\n{'Fold':<6} {'Checkpoint':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC Macro':<12} {'AUC Micro':<12}")
        print("-" * 100)
        
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
            
            print(f"{fold_num:<6} {checkpoint_iter:<12} {acc_str:<12} {prec_str:<12} {rec_str:<12} {f1_str:<12} {auc_macro_str:<12} {auc_micro_str:<12}")
        
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS ACROSS ALL FOLDS")
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
        
        # Save summary
        summary_file = os.path.join(args.output_dir, 'kfold_summary_best_checkpoints.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_file}")
        
        # Save individual results
        results_file = os.path.join(args.output_dir, 'kfold_results_best_checkpoints.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Individual results saved to: {results_file}")
        
        # Create CSV for easy viewing
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(args.output_dir, 'kfold_results_best_checkpoints.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ Results CSV saved to: {csv_file}")
    else:
        print("\n❌ ERROR: No results collected!")
        sys.exit(1)
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
