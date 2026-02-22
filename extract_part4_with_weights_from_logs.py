#!/usr/bin/env python3
"""
Extract Part 4 class weights results from log files and update visualization.
"""

import os
import glob
import re
import csv
import json

def extract_from_log_file(log_file):
    """Extract evaluation metrics from log file."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract Overall Accuracy with iteration
            acc_pattern = r'Iteration\s+(\d+).*?Overall Accuracy:\s*(\d+\.?\d*)%'
            acc_matches = re.findall(acc_pattern, content, re.DOTALL)
            
            if acc_matches:
                metrics['evaluations'] = []
                for iter_num, acc in acc_matches:
                    metrics['evaluations'].append({
                        'iteration': int(iter_num),
                        'accuracy': float(acc)
                    })
                
                # Find best
                best_eval = max(metrics['evaluations'], key=lambda x: x['accuracy'])
                metrics['best_iteration'] = best_eval['iteration']
                metrics['best_accuracy'] = best_eval['accuracy']
            
            # Extract F1 scores
            f1_pattern = r'Iteration\s+(\d+).*?F1 Score.*?(\d+\.?\d*)%'
            f1_matches = re.findall(f1_pattern, content, re.DOTALL)
            
            if f1_matches:
                f1_dict = {int(iter_num): float(f1) for iter_num, f1 in f1_matches}
                metrics['f1_scores'] = f1_dict
                
                # Get F1 at best iteration
                if 'best_iteration' in metrics:
                    best_iter = metrics['best_iteration']
                    # Find closest iteration
                    if best_iter in f1_dict:
                        metrics['best_f1'] = f1_dict[best_iter]
                    else:
                        closest_iter = min(f1_dict.keys(), key=lambda x: abs(x - best_iter))
                        if abs(closest_iter - best_iter) <= 500:
                            metrics['best_f1'] = f1_dict[closest_iter]
            
            # Extract per-class metrics at best iteration
            if 'best_iteration' in metrics:
                best_iter = metrics['best_iteration']
                # Look for metrics around this iteration
                iter_section = re.search(
                    rf'Iteration\s+{best_iter}.*?(?=Iteration|\Z)',
                    content,
                    re.DOTALL
                )
                
                if iter_section:
                    section = iter_section.group(0)
                    
                    # Extract per-class metrics
                    for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
                        for metric_type in ['Precision', 'Recall', 'F1', 'Specificity']:
                            pattern = rf'{re.escape(class_name)}\s+{metric_type}:\s*(\d+\.?\d*)%'
                            match = re.search(pattern, section)
                            if match:
                                key = f'{class_name.lower()}_{metric_type.lower()}'
                                metrics[key] = float(match.group(1))
            
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    return metrics

def main():
    print("="*80)
    print("EXTRACTING PART 4 CLASS WEIGHTS RESULTS FROM LOG FILES")
    print("="*80)
    
    experiments = {
        'REDO_Frailty_ccpg_pt4_swingait_with_weights': {
            'name': 'SwinGait (Frozen CNN) + Class Weights',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_with_weights/logs'
        },
        'REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights': {
            'name': 'DeepGaitV2 (Frozen CNN) + Class Weights',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/logs'
        }
    }
    
    results = {}
    
    for exp_key, exp_info in experiments.items():
        print(f"\n{'='*80}")
        print(f"Processing: {exp_info['name']}")
        print(f"{'='*80}")
        
        log_dir = exp_info['log_dir']
        if not os.path.exists(log_dir):
            print(f"  ✗ Log directory not found: {log_dir}")
            continue
        
        log_files = glob.glob(os.path.join(log_dir, '*.txt'))
        if not log_files:
            print(f"  ✗ No log files found")
            continue
        
        log_file = max(log_files, key=os.path.getmtime)
        print(f"  ✓ Using log file: {os.path.basename(log_file)}")
        
        metrics = extract_from_log_file(log_file)
        
        if not metrics or 'best_accuracy' not in metrics:
            print(f"  ✗ Could not extract metrics")
            # Try alternative extraction
            with open(log_file, 'r') as f:
                content = f.read()
                # Look for any accuracy values
                all_acc = re.findall(r'(\d+\.?\d*)%', content)
                print(f"    Found {len(all_acc)} percentage values in log")
            continue
        
        print(f"\n  Results:")
        print(f"    Best Iteration: {metrics['best_iteration']}")
        print(f"    Best Accuracy: {metrics['best_accuracy']:.2f}%")
        if 'best_f1' in metrics:
            print(f"    Best F1 Score: {metrics['best_f1']:.2f}%")
        if 'evaluations' in metrics:
            print(f"    Total Evaluations: {len(metrics['evaluations'])}")
        
        results[exp_key] = {
            'name': exp_info['name'],
            **metrics
        }
    
    # Save to CSV
    os.makedirs('results_visualization', exist_ok=True)
    csv_file = 'results_visualization/part4_with_weights_results.csv'
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Best Iteration', 'Accuracy (%)', 'F1 Score (%)'])
        for exp_key, data in results.items():
            writer.writerow([
                data['name'],
                data.get('best_iteration', ''),
                f"{data.get('best_accuracy', 0):.2f}",
                f"{data.get('best_f1', 0):.2f}" if 'best_f1' in data else ''
            ])
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_file}")
    
    # Save JSON
    json_file = 'results_visualization/part4_with_weights_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    print("\nBaseline Part 4 (no class weights):")
    print("  SwinGait (Frozen CNN):     73.33% accuracy, 73.89% F1")
    print("  DeepGaitV2 (Frozen CNN):   60.00% accuracy, 60.33% F1")
    
    print("\nPart 4 with Class Weights:")
    for exp_key, data in results.items():
        baseline_acc = 73.33 if 'swingait' in exp_key else 60.00
        acc_change = data.get('best_accuracy', 0) - baseline_acc
        sign = '+' if acc_change >= 0 else ''
        f1_str = f", {data.get('best_f1', 0):.2f}% F1" if 'best_f1' in data else ""
        print(f"  {data['name']}:")
        print(f"    {data.get('best_accuracy', 0):.2f}% accuracy ({sign}{acc_change:.2f}%){f1_str}")
    
    return results

if __name__ == '__main__':
    main()

