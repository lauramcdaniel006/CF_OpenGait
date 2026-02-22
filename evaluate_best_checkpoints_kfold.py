#!/usr/bin/env python3
"""
Script to evaluate TEST sets for each fold using manually identified best checkpoints.
This runs evaluation on TEST_SET (not validation) for each fold.
"""

import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path

# ============================================================================
# EDIT THIS: Map fold number to your best checkpoint iteration
# ============================================================================
BEST_CHECKPOINTS = {
    1: 500,  # Replace with your best iteration for fold 1
    2: 2500,  # Replace with your best iteration for fold 2
    3: 3000,  # Replace with your best iteration for fold 3
    4: 1500,  # Replace with your best iteration for fold 4
    5: 1500,  # Replace with your best iteration for fold 5
}

def extract_metrics_from_output(output_text):
    """Extract metrics from evaluation output"""
    metrics = {}
    
    # Extract accuracy
    acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', output_text)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1)) / 100.0
    
    # Extract AUC
    auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', output_text)
    if auc_macro_match:
        metrics['auc_macro'] = float(auc_macro_match.group(1))
    
    auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', output_text)
    if auc_micro_match:
        metrics['auc_micro'] = float(auc_micro_match.group(1))
    
    return metrics

def run_test_evaluation(config_file, checkpoint_iter, fold_num, device='0,1', nproc=2):
    """Run TEST evaluation for a fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}: TEST Evaluation (checkpoint iter {checkpoint_iter})")
    print(f"{'='*70}")
    
    cmd = [
        'python', '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',  # This evaluates on TEST_SET
        '--iter', str(checkpoint_iter),
        '--log_to_file'
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    
    # Fix OpenCV library compatibility issues
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        lib_path = f"{conda_prefix}/lib"
        env['LD_LIBRARY_PATH'] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
        env['LD_PRELOAD'] = f"{lib_path}/libstdc++.so.6:{lib_path}/libgcc_s.so.1"
    
    # Run evaluation
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
        sys.stdout.flush()
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ ERROR: TEST evaluation failed for fold {fold_num}")
        return None
    
    print(f"\n✅ TEST evaluation completed for fold {fold_num}")
    return ''.join(output_lines)

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate TEST sets for each fold using best checkpoints'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/swingait/swin_part1_p_CNN.yaml',  # p+CNN config as default
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
        '--device', 
        type=str, 
        default='0,1',
        help='CUDA devices (default: 0,1)'
    )
    parser.add_argument(
        '--nproc', 
        type=int, 
        default=2,
        help='Number of processes (default: 2)'
    )
    parser.add_argument(
        '--folds', 
        type=str, 
        default=None,
        help='Comma-separated list of folds to evaluate (e.g., "1,2,3"). If None, evaluates all folds.'
    )
    
    args = parser.parse_args()
    
    # Determine which folds to evaluate
    if args.folds:
        folds_to_eval = [int(f.strip()) for f in args.folds.split(',')]
    else:
        folds_to_eval = list(range(1, args.k + 1))
    
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION WITH BEST CHECKPOINTS")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Folds to evaluate: {folds_to_eval}")
    print(f"Best checkpoints: {BEST_CHECKPOINTS}")
    print(f"{'='*70}\n")
    
    results = {}
    all_results = []
    
    for fold_num in folds_to_eval:
        if fold_num not in BEST_CHECKPOINTS:
            print(f"⚠️  Warning: No best checkpoint specified for fold {fold_num}, skipping...")
            continue
        
        checkpoint_iter = BEST_CHECKPOINTS[fold_num]
        
        # Find fold config file
        fold_config = os.path.join(args.output_dir, f'config_fold{fold_num}.yaml')
        
        if not os.path.exists(fold_config):
            print(f"❌ Error: Config file not found: {fold_config}")
            print(f"   Make sure you've run kfold cross-validation first to generate config files.")
            continue
        
        # Run test evaluation
        output = run_test_evaluation(
            fold_config, 
            checkpoint_iter, 
            fold_num, 
            args.device, 
            args.nproc
        )
        
        if output is None:
            print(f"❌ Failed to evaluate fold {fold_num}")
            results[fold_num] = None
        else:
            # Extract metrics from output
            metrics = extract_metrics_from_output(output)
            if metrics:
                metrics['fold'] = fold_num
                metrics['checkpoint_iter'] = checkpoint_iter
                all_results.append(metrics)
                
                # Save individual fold results (with _best_checkpoint suffix to avoid overwriting)
                fold_results_file = os.path.join(args.output_dir, f'fold_{fold_num}_results_best_checkpoint.json')
                with open(fold_results_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"✓ Saved results to: {fold_results_file}")
                
                print(f"\nFold {fold_num} Results (checkpoint iter {checkpoint_iter}):")
                for key, value in metrics.items():
                    if key not in ['fold', 'checkpoint_iter']:
                        print(f"  {key}: {value:.4f}")
            
            results[fold_num] = {
                'fold': fold_num,
                'checkpoint_iter': checkpoint_iter,
                'status': 'success',
                'metrics': metrics if metrics else None
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    for fold_num, result in results.items():
        if result:
            print(f"Fold {fold_num}: ✅ Success (checkpoint iter {result['checkpoint_iter']})")
        else:
            print(f"Fold {fold_num}: ❌ Failed")
    
    # Aggregate results if we have any
    if all_results:
        print(f"\n{'='*70}")
        print("DETAILED RESULTS BY FOLD")
        print(f"{'='*70}")
        
        # Print table header
        print(f"\n{'Fold':<6} {'Checkpoint':<12} {'Accuracy':<12} {'AUC Macro':<12} {'AUC Micro':<12}")
        print("-" * 70)
        
        # Print each fold's results
        for result in sorted(all_results, key=lambda x: x['fold']):
            fold_num = result['fold']
            checkpoint_iter = result['checkpoint_iter']
            accuracy = result.get('accuracy', 0.0)
            auc_macro = result.get('auc_macro', 0.0)
            auc_micro = result.get('auc_micro', 0.0)
            print(f"{fold_num:<6} {checkpoint_iter:<12} {accuracy:<12.4f} {auc_macro:<12.4f} {auc_micro:<12.4f}")
        
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS ACROSS ALL FOLDS")
        print(f"{'='*70}")
        
        # Calculate mean and std for each metric
        summary = {}
        metric_keys = ['accuracy', 'auc_macro', 'auc_micro']
        
        # Try to use numpy, fallback to manual calculation
        try:
            import numpy as np
            use_numpy = True
        except ImportError:
            use_numpy = False
        
        for key in metric_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                if use_numpy:
                    summary[f'{key}_mean'] = float(np.mean(values))
                    summary[f'{key}_std'] = float(np.std(values))
                else:
                    # Manual calculation
                    mean_val = sum(values) / len(values)
                    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                    std_val = variance ** 0.5
                    summary[f'{key}_mean'] = mean_val
                    summary[f'{key}_std'] = std_val
                summary[f'{key}_values'] = values
        
        # Print summary
        print("\nMean ± Std across folds:")
        for key in metric_keys:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in summary:
                mean_val = summary[mean_key]
                std_val = summary[std_key]
                print(f"  {key:15s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save summary
        summary_file = os.path.join(args.output_dir, 'kfold_summary_best_checkpoints.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_file}")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
