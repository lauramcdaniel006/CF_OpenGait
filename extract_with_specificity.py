#!/usr/bin/env python3
"""
Extract all metrics including per-class specificity from log files.
Since specificity is logged but not saved to TensorBoard, we need to parse log files.
"""

import os
import re
import glob
import json
from pathlib import Path

def extract_from_log_file(log_file):
    """Extract all evaluation metrics including per-class specificity from log files."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Split by evaluation sections (look for "EVALUATION RESULTS")
            eval_sections = re.split(r'EVALUATION RESULTS', content)
            
            all_evaluations = []
            
            for section in eval_sections[1:]:  # Skip first empty part
                eval_metrics = {}
                
                # Extract iteration number (look backwards for "Iteration" or "Running test")
                iter_match = re.search(r'Iteration\s+(\d+)', section)
                if iter_match:
                    eval_metrics['iteration'] = int(iter_match.group(1))
                
                # Overall Accuracy
                acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
                if acc_match:
                    eval_metrics['overall_accuracy'] = float(acc_match.group(1))
                
                # Per-class metrics
                class_names = ['Frail', 'Prefrail', 'Nonfrail']
                for class_name in class_names:
                    # Sensitivity (Recall)
                    sens_match = re.search(
                        rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
                        section
                    )
                    if sens_match:
                        eval_metrics[f'{class_name.lower()}_sensitivity'] = float(sens_match.group(1))
                    
                    # Specificity
                    spec_match = re.search(
                        rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%', 
                        section
                    )
                    if spec_match:
                        eval_metrics[f'{class_name.lower()}_specificity'] = float(spec_match.group(1))
                    
                    # Precision
                    prec_match = re.search(
                        rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
                        section
                    )
                    if prec_match:
                        eval_metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1))
                
                # Confusion Matrix
                cm_match = re.search(
                    r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
                    section
                )
                if cm_match:
                    eval_metrics['confusion_matrix'] = [
                        [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                        [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                        [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
                    ]
                
                if eval_metrics:
                    all_evaluations.append(eval_metrics)
            
            metrics['evaluations'] = all_evaluations
            
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return metrics

def find_log_files(exp_dir):
    """Find log files in experiment directory."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return []
    
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    return sorted(log_files, key=os.path.getmtime)

def main():
    print("=" * 80)
    print("Extract All Metrics Including Specificity from Log Files")
    print("=" * 80)
    
    redo_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_smooth',
        'REDO_log',
        'REDO_uniform'
    ]
    
    all_results = {}
    
    for exp_name in redo_experiments:
        exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
        
        if not os.path.exists(exp_path):
            print(f"✗ {exp_name}: Not found")
            continue
        
        log_files = find_log_files(exp_path)
        
        if not log_files:
            print(f"✗ {exp_name}: No log files found")
            continue
        
        # Use the most recent log file
        log_file = log_files[-1]
        print(f"✓ {exp_name}: Processing {os.path.basename(log_file)}")
        
        metrics = extract_from_log_file(log_file)
        
        if metrics and metrics.get('evaluations'):
            all_results[exp_name] = metrics
            print(f"  Found {len(metrics['evaluations'])} evaluations")
        else:
            print(f"  ✗ No evaluation data found")
    
    # Save to JSON
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'all_metrics_with_specificity.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved to: {json_file}")
    
    # Create CSV with all metrics
    import csv
    csv_file = os.path.join(output_dir, 'all_metrics_with_specificity.csv')
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Experiment', 'Iteration', 'Overall_Accuracy']
        for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
            header.extend([
                f'{class_name}_Sensitivity',
                f'{class_name}_Specificity',
                f'{class_name}_Precision'
            ])
        
        writer.writerow(header)
        
        # Data rows
        for exp_name, exp_data in all_results.items():
            for eval_data in exp_data.get('evaluations', []):
                row = [
                    exp_name,
                    eval_data.get('iteration', ''),
                    eval_data.get('overall_accuracy', '')
                ]
                
                for class_name in ['frail', 'prefrail', 'nonfrail']:
                    row.append(eval_data.get(f'{class_name}_sensitivity', ''))
                    row.append(eval_data.get(f'{class_name}_specificity', ''))
                    row.append(eval_data.get(f'{class_name}_precision', ''))
                
                writer.writerow(row)
    
    print(f"✓ Saved CSV to: {csv_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for exp_name, exp_data in all_results.items():
        evals = exp_data.get('evaluations', [])
        if not evals:
            continue
        
        print(f"\n{exp_name}: {len(evals)} evaluations found")
        
        # Show first and last evaluation
        if len(evals) > 0:
            first = evals[0]
            last = evals[-1]
            
            print(f"  First (iter {first.get('iteration', '?')}): "
                  f"Acc={first.get('overall_accuracy', 0):.2f}%")
            print(f"  Last (iter {last.get('iteration', '?')}): "
                  f"Acc={last.get('overall_accuracy', 0):.2f}%")
            
            # Show per-class metrics for last evaluation
            if 'frail_sensitivity' in last:
                print(f"    Frail - Sens: {last.get('frail_sensitivity', 0):.2f}%, "
                      f"Spec: {last.get('frail_specificity', 0):.2f}%, "
                      f"Prec: {last.get('frail_precision', 0):.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

