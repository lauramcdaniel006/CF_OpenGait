#!/usr/bin/env python3
"""
Aggregate metrics from best checkpoints across folds for three models:
- Tintro: REDO_Frailty_ccpg_pt1_p+CNN+Tintro
- T1: REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1
- T2: REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2
"""

import os
import sys
import json
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def find_metrics_at_checkpoint(log_file, checkpoint_iter):
    """Find metrics at a specific checkpoint iteration from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {log_file}: {e}")
        return None
    
    # Look for iteration followed by evaluation results
    # Pattern: Iteration XXXX ... EVALUATION RESULTS ... metrics
    pattern = rf'Iteration\s+0*{checkpoint_iter}[^\d].*?EVALUATION RESULTS.*?Overall Accuracy:\s*([\d.]+)%.*?Precision \(macro\):\s*([\d.]+)%.*?Recall \(macro\):\s*([\d.]+)%.*?F1 Score \(macro\):\s*([\d.]+)%.*?ROC AUC \(macro\):\s*([\d.]+).*?ROC AUC \(micro\):\s*([\d.]+)'
    
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
    
    # Collect all metrics (only numeric metrics, skip fold, checkpoint_iter, note)
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

def main():
    # Model configurations
    models_config = {
        'Tintro': {
            'base_name': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro',
            'checkpoints': {
                1: 6500,
                2: 7000,
                3: 10000,
                4: 3000,
                5: 2500
            }
        },
        'T1': {
            'base_name': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1',
            'checkpoints': {
                1: 500,
                2: 5000,
                3: 10000,
                4: 5500,
                5: 1500
            }
        },
        'T2': {
            'base_name': 'REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2',
            'checkpoints': {
                1: 3000,
                2: 2000,
                3: 2500,
                4: 10000,
                5: 1000
            }
        },
        'UF': {
            'base_name': 'REDO_Frailty_ccpg_pt1_pretrained(UF)',
            'checkpoints': {
                1: 7500,
                2: 8000,
                3: 5500,
                4: 3500,
                5: 2000
            }
        },
        'p+CNN': {
            'base_name': 'REDO_Frailty_ccpg_pt1_p+CNN',
            'checkpoints': {
                1: 4500,
                2: 4500,
                3: 7000,
                4: 2000,
                5: 8000
            }
        }
    }
    
    output_dir = 'kfold_results'
    output_base = 'output'
    
    print(f"\n{'='*80}")
    print("AGGREGATING METRICS FROM BEST CHECKPOINTS")
    print(f"{'='*80}")
    print(f"Models: {len(models_config)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_model_results = {}
    all_detailed_results = []
    
    # Process each model
    for model_short_name, model_info in models_config.items():
        base_name = model_info['base_name']
        fold_checkpoints = model_info['checkpoints']
        
        print(f"\n{'='*80}")
        print(f"Processing model: {model_short_name} ({base_name})")
        print(f"{'='*80}")
        
        model_results = []
        
        # Process each fold
        for fold_num, checkpoint_iter in sorted(fold_checkpoints.items()):
            # Construct full model name with fold
            model_name = f"{base_name}_fold{fold_num}"
            
            # Find experiment directory
            exp_dir = os.path.join(output_base, model_name, 'SwinGait', model_name)
            
            if not os.path.exists(exp_dir):
                print(f"  ⚠️  Experiment directory not found: {exp_dir}")
                continue
            
            log_dir = os.path.join(exp_dir, 'logs')
            if not os.path.exists(log_dir):
                print(f"  ⚠️  Log directory not found: {log_dir}")
                continue
            
            # Get all log files (sorted by modification time, most recent first)
            log_files = sorted(glob.glob(os.path.join(log_dir, '*.txt')), key=os.path.getmtime, reverse=True)
            if not log_files:
                print(f"  ⚠️  No log files found in {log_dir}")
                continue
            
            print(f"\n  Fold {fold_num} (checkpoint {checkpoint_iter})...")
            
            # Try to find metrics in log files (start with most recent)
            metrics = None
            for log_file in log_files:
                metrics = find_metrics_at_checkpoint(log_file, checkpoint_iter)
                if metrics:
                    print(f"    Log: {os.path.basename(log_file)}")
                    break
            
            if not metrics:
                print(f"    Log: {os.path.basename(log_files[0])} (checked {len(log_files)} log files)")
            
            if metrics:
                print(f"    ✓ Found metrics:")
                print(f"      Accuracy: {metrics['accuracy']:.2f}%")
                print(f"      Precision: {metrics['precision']:.2f}%")
                print(f"      Recall: {metrics['recall']:.2f}%")
                print(f"      F1: {metrics['f1']:.2f}%")
                print(f"      AUC Macro: {metrics['auc_macro']:.4f}")
                print(f"      AUC Micro: {metrics['auc_micro']:.4f}")
                
                result = {
                    'model_name': model_short_name,
                    'fold': fold_num,
                    'checkpoint_iter': checkpoint_iter,
                    'accuracy': metrics['accuracy'] / 100.0,  # Convert to 0-1 range
                    'precision': metrics['precision'] / 100.0,
                    'recall': metrics['recall'] / 100.0,
                    'f1': metrics['f1'] / 100.0,
                    'auc_macro': metrics['auc_macro'],
                    'auc_micro': metrics['auc_micro'],
                }
                model_results.append(result)
                all_detailed_results.append(result)
            else:
                print(f"    ⚠️  Could not find metrics at checkpoint {checkpoint_iter}")
        
        # Aggregate results for this model
        if model_results:
            summary = aggregate_results(model_results)
            all_model_results[model_short_name] = {
                'per_fold': model_results,
                'aggregated': summary
            }
            
            # Print detailed results table for this model
            print(f"\n  {'='*70}")
            print(f"  DETAILED RESULTS BY FOLD - {model_short_name}")
            print(f"  {'='*70}")
            print(f"\n  {'Fold':<6} {'Checkpoint':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC Macro':<12} {'AUC Micro':<12}")
            print("  " + "-" * 98)
            
            for result in sorted(model_results, key=lambda x: x['fold']):
                fold = result['fold']
                checkpoint = result['checkpoint_iter']
                accuracy = result['accuracy'] * 100
                precision = result['precision'] * 100
                recall = result['recall'] * 100
                f1 = result['f1'] * 100
                auc_macro = result['auc_macro']
                auc_micro = result['auc_micro']
                
                print(f"  {fold:<6} {checkpoint:<12} {accuracy:<12.2f} {precision:<12.2f} {recall:<12.2f} {f1:<12.2f} {auc_macro:<12.4f} {auc_micro:<12.4f}")
            
            print(f"\n  {'='*70}")
            print(f"  AGGREGATED RESULTS FOR {model_short_name}")
            print(f"  {'='*70}")
            print(f"\n  Mean ± Std across folds:")
            metric_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_macro', 'auc_micro']
            for key in metric_order:
                mean_key = f'{key}_mean'
                std_key = f'{key}_std'
                if mean_key in summary:
                    mean_val = summary[mean_key]
                    std_val = summary[std_key]
                    if key in ['accuracy', 'precision', 'recall', 'f1']:
                        print(f"    {key:15s}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")
                    else:
                        print(f"    {key:15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print summary table for all models (after processing all models)
    if all_model_results:
        print(f"\n\n{'='*80}")
        print("SUMMARY TABLE - ALL MODELS (Mean ± Std)")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        summary_rows = []
        for model_name, data in all_model_results.items():
            agg = data['aggregated']
            summary_rows.append({
                'Model': model_name,
                'Accuracy (%)': f"{agg.get('accuracy_mean', 0)*100:.2f} ± {agg.get('accuracy_std', 0)*100:.2f}",
                'Precision (%)': f"{agg.get('precision_mean', 0)*100:.2f} ± {agg.get('precision_std', 0)*100:.2f}",
                'Recall (%)': f"{agg.get('recall_mean', 0)*100:.2f} ± {agg.get('recall_std', 0)*100:.2f}",
                'F1 (%)': f"{agg.get('f1_mean', 0)*100:.2f} ± {agg.get('f1_std', 0)*100:.2f}",
                'AUC Macro': f"{agg.get('auc_macro_mean', 0):.4f} ± {agg.get('auc_macro_std', 0):.4f}",
                'AUC Micro': f"{agg.get('auc_micro_mean', 0):.4f} ± {agg.get('auc_micro_std', 0):.4f}",
            })
        
        df_summary = pd.DataFrame(summary_rows)
        print("\n" + df_summary.to_string(index=False))
        
        # Save results
        # 1. Full results JSON
        full_results_file = os.path.join(output_dir, 'three_models_aggregated_results.json')
        with open(full_results_file, 'w') as f:
            json.dump(all_model_results, f, indent=2)
        print(f"\n✓ Full results saved to: {full_results_file}")
        
        # 2. Summary CSV
        summary_csv = os.path.join(output_dir, 'three_models_summary.csv')
        df_summary.to_csv(summary_csv, index=False)
        print(f"✓ Summary CSV saved to: {summary_csv}")
        
        # 3. Detailed results CSV (all folds)
        detailed_csv = os.path.join(output_dir, 'three_models_detailed_results.csv')
        df_detailed = pd.DataFrame(all_detailed_results)
        df_detailed.to_csv(detailed_csv, index=False)
        print(f"✓ Detailed results CSV saved to: {detailed_csv}")
        
        # 4. Per-model aggregated JSON
        for model_name, data in all_model_results.items():
            model_file = os.path.join(output_dir, f'{model_name}_aggregated.json')
            with open(model_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"✓ Per-model aggregated JSON files saved to: {output_dir}")
    else:
        print("\n❌ ERROR: No results collected!")
        sys.exit(1)
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
