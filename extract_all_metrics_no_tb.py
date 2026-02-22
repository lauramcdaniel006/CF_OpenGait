#!/usr/bin/env python3
"""
Extract all metrics from log files or by re-running evaluation on checkpoints.
No TensorBoard required!
"""

import os
import re
import glob
import subprocess
from pathlib import Path
import json

def extract_from_log_file(log_file):
    """Extract all evaluation metrics from log text files."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract Overall Accuracy
            acc_matches = re.findall(r'Overall Accuracy:\s*(\d+\.?\d*)%', content)
            if acc_matches:
                metrics['accuracies'] = [float(x) for x in acc_matches]
                metrics['final_accuracy'] = float(acc_matches[-1])
            
            # Extract per-class metrics (get all occurrences)
            class_names = ['Frail', 'Prefrail', 'Nonfrail']
            for class_name in class_names:
                # Sensitivity
                sens_matches = re.findall(
                    rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
                    content
                )
                if sens_matches:
                    metrics[f'{class_name.lower()}_sensitivity'] = [float(x) for x in sens_matches]
                    metrics[f'{class_name.lower()}_sensitivity_final'] = float(sens_matches[-1])
                
                # Specificity
                spec_matches = re.findall(
                    rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%', 
                    content
                )
                if spec_matches:
                    metrics[f'{class_name.lower()}_specificity'] = [float(x) for x in spec_matches]
                    metrics[f'{class_name.lower()}_specificity_final'] = float(spec_matches[-1])
                
                # Precision
                prec_matches = re.findall(
                    rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
                    content
                )
                if prec_matches:
                    metrics[f'{class_name.lower()}_precision'] = [float(x) for x in prec_matches]
                    metrics[f'{class_name.lower()}_precision_final'] = float(prec_matches[-1])
            
            # Extract confusion matrices
            cm_matches = re.findall(
                r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
                content
            )
            if cm_matches:
                metrics['confusion_matrices'] = []
                for cm_match in cm_matches:
                    cm = [
                        [int(cm_match[0]), int(cm_match[1]), int(cm_match[2])],
                        [int(cm_match[3]), int(cm_match[4]), int(cm_match[5])],
                        [int(cm_match[6]), int(cm_match[7]), int(cm_match[8])]
                    ]
                    metrics['confusion_matrices'].append(cm)
                metrics['final_confusion_matrix'] = metrics['confusion_matrices'][-1]
            
            # Extract iteration numbers if available
            iter_matches = re.findall(r'Iteration\s+(\d+)', content)
            if iter_matches and acc_matches:
                # Try to match iterations with accuracies
                metrics['evaluation_iterations'] = [int(x) for x in iter_matches]
                
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return metrics

def find_checkpoints(exp_dir):
    """Find all checkpoint files."""
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    # Sort by iteration number
    def get_iter(cp):
        match = re.search(r'-(\d+)\.pt$', cp)
        return int(match.group(1)) if match else 0
    
    return sorted(checkpoints, key=get_iter)

def find_experiments(output_dir='output'):
    """Find all experiment directories."""
    experiments = {}
    
    for root, dirs, files in os.walk(output_dir):
        if 'checkpoints' in dirs:
            path_parts = Path(root).parts
            if len(path_parts) >= 4:
                dataset = path_parts[-3]
                model = path_parts[-2]
                save_name = path_parts[-1]
                
                exp_key = f"{dataset}/{model}/{save_name}"
                experiments[exp_key] = {
                    'full_path': root,
                    'dataset': dataset,
                    'model': model,
                    'save_name': save_name
                }
    
    return experiments

def main():
    print("=" * 70)
    print("Extract All Metrics Without TensorBoard")
    print("=" * 70)
    
    # Focus on REDO Part 2 experiments
    redo_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_smooth',
        'REDO_log',
        'REDO_uniform'
    ]
    
    print("\n1. Checking for log files...")
    all_results = {}
    
    for exp_name in redo_experiments:
        exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
        
        if not os.path.exists(exp_path):
            print(f"   ✗ {exp_name}: Experiment not found")
            continue
        
        # Check for log files
        logs_dir = os.path.join(exp_path, 'logs')
        metrics = {}
        
        if os.path.exists(logs_dir):
            log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
            if log_files:
                # Use the most recent log file
                log_file = max(log_files, key=os.path.getmtime)
                print(f"   ✓ {exp_name}: Found log file")
                metrics = extract_from_log_file(log_file)
                all_results[exp_name] = metrics
            else:
                print(f"   ✗ {exp_name}: No log files found")
        else:
            print(f"   ✗ {exp_name}: No logs directory")
        
        # Also check checkpoints
        checkpoints = find_checkpoints(exp_path)
        if checkpoints:
            print(f"      Found {len(checkpoints)} checkpoints")
            if exp_name not in all_results:
                all_results[exp_name] = {}
            all_results[exp_name]['checkpoints'] = [os.path.basename(cp) for cp in checkpoints]
    
    print("\n2. Extracted Metrics:")
    print("=" * 70)
    
    for exp_name, metrics in all_results.items():
        print(f"\n{exp_name}:")
        if 'accuracies' in metrics:
            print(f"  Accuracy History: {[f'{a:.2f}%' for a in metrics['accuracies']]}")
            print(f"  Final Accuracy: {metrics.get('final_accuracy', 'N/A'):.2f}%")
            print(f"  Number of Evaluations: {len(metrics['accuracies'])}")
            print(f"  Times ≥73.0%: {sum(1 for a in metrics['accuracies'] if a >= 73.0)}")
        
        if 'frail_sensitivity_final' in metrics:
            print(f"  Frail - Sens: {metrics['frail_sensitivity_final']:.2f}%, "
                  f"Spec: {metrics['frail_specificity_final']:.2f}%, "
                  f"Prec: {metrics['frail_precision_final']:.2f}%")
        
        if 'checkpoints' in metrics:
            print(f"  Checkpoints: {len(metrics['checkpoints'])} available")
    
    # Save to JSON
    output_file = 'results_visualization/redo_pt2_detailed_metrics.json'
    os.makedirs('results_visualization', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n3. Saved detailed metrics to: {output_file}")
    
    # Create CSV summary
    csv_file = 'results_visualization/redo_pt2_summary.csv'
    import csv
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Final Accuracy (%)', 'Num Evaluations', 
                        'Times ≥73%', 'Frail Sens (%)', 'Frail Spec (%)', 'Frail Prec (%)'])
        
        for exp_name, metrics in all_results.items():
            row = [exp_name]
            row.append(f"{metrics.get('final_accuracy', 0):.2f}")
            row.append(len(metrics.get('accuracies', [])))
            row.append(sum(1 for a in metrics.get('accuracies', []) if a >= 73.0))
            row.append(f"{metrics.get('frail_sensitivity_final', 0):.2f}")
            row.append(f"{metrics.get('frail_specificity_final', 0):.2f}")
            row.append(f"{metrics.get('frail_precision_final', 0):.2f}")
            writer.writerow(row)
    
    print(f"   Saved summary CSV to: {csv_file}")
    print("\n" + "=" * 70)
    print("Done! Check the JSON file for all metrics, CSV for summary.")
    print("=" * 70)

if __name__ == '__main__':
    main()

