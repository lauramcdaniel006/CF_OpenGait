#!/usr/bin/env python3
"""
Print F1, Precision, and Recall (macro-averaged) from log files or TensorBoard.

Usage:
    python print_metrics_from_logs.py <experiment_path>
    
Example:
    python print_metrics_from_logs.py output/REDO_Frailty_ccpg_pt1_p+CNN/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN
"""

import os
import sys
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_from_log_file(log_file):
    """Extract metrics from a log file."""
    results = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Split by evaluation sections
        eval_sections = re.split(r'EVALUATION RESULTS', content)
        
        for i, section in enumerate(eval_sections[1:], 1):
            # Find iteration number before this section
            before_idx = content.find(section) - 1000
            before_text = content[max(0, before_idx):content.find(section)]
            
            iter_match = re.search(r'Iteration\s+0*(\d+)', before_text)
            if not iter_match:
                continue
            
            iter_num = int(iter_match.group(1))
            
            # Extract overall accuracy
            acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
            if not acc_match:
                continue
            
            accuracy = float(acc_match.group(1)) / 100.0
            
            # Extract per-class metrics
            frail_recall = re.search(r'Frail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            frail_prec = re.search(r'Frail Precision:\s*(\d+\.?\d*)%', section)
            
            prefrail_recall = re.search(r'Prefrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            prefrail_prec = re.search(r'Prefrail Precision:\s*(\d+\.?\d*)%', section)
            
            nonfrail_recall = re.search(r'Nonfrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            nonfrail_prec = re.search(r'Nonfrail Precision:\s*(\d+\.?\d*)%', section)
            
            if not all([frail_recall, frail_prec, prefrail_recall, prefrail_prec, nonfrail_recall, nonfrail_prec]):
                continue
            
            # Convert to decimals
            frail_recall_val = float(frail_recall.group(1)) / 100.0
            frail_prec_val = float(frail_prec.group(1)) / 100.0
            frail_f1 = 2 * (frail_prec_val * frail_recall_val) / (frail_prec_val + frail_recall_val) if (frail_prec_val + frail_recall_val) > 0 else 0.0
            
            prefrail_recall_val = float(prefrail_recall.group(1)) / 100.0
            prefrail_prec_val = float(prefrail_prec.group(1)) / 100.0
            prefrail_f1 = 2 * (prefrail_prec_val * prefrail_recall_val) / (prefrail_prec_val + prefrail_recall_val) if (prefrail_prec_val + prefrail_recall_val) > 0 else 0.0
            
            nonfrail_recall_val = float(nonfrail_recall.group(1)) / 100.0
            nonfrail_prec_val = float(nonfrail_prec.group(1)) / 100.0
            nonfrail_f1 = 2 * (nonfrail_prec_val * nonfrail_recall_val) / (nonfrail_prec_val + nonfrail_recall_val) if (nonfrail_prec_val + nonfrail_recall_val) > 0 else 0.0
            
            # Calculate macro-averaged metrics
            macro_precision = (frail_prec_val + prefrail_prec_val + nonfrail_prec_val) / 3.0
            macro_recall = (frail_recall_val + prefrail_recall_val + nonfrail_recall_val) / 3.0
            macro_f1 = (frail_f1 + prefrail_f1 + nonfrail_f1) / 3.0
            
            results.append({
                'iteration': iter_num,
                'accuracy': accuracy,
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            })
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []
    
    return sorted(results, key=lambda x: x['iteration'])

def extract_from_tensorboard(summary_dir):
    """Extract metrics from TensorBoard event files."""
    try:
        ea = EventAccumulator(summary_dir)
        ea.Reload()
        
        if 'test_accuracy/' not in ea.Tags()['scalars']:
            return []
        
        accuracy_events = ea.Scalars('test_accuracy/')
        f1_events = ea.Scalars('test_f1/') if 'test_f1/' in ea.Tags()['scalars'] else []
        prec_events = ea.Scalars('test_precision/') if 'test_precision/' in ea.Tags()['scalars'] else []
        recall_events = ea.Scalars('test_recall/') if 'test_recall/' in ea.Tags()['scalars'] else []
        
        results = []
        for acc_event in accuracy_events:
            iter_num = acc_event.step
            acc_val = acc_event.value
            
            f1_val = None
            prec_val = None
            recall_val = None
            
            for f1_event in f1_events:
                if f1_event.step == iter_num:
                    f1_val = f1_event.value
                    break
            
            for prec_event in prec_events:
                if prec_event.step == iter_num:
                    prec_val = prec_event.value
                    break
            
            for recall_event in recall_events:
                if recall_event.step == iter_num:
                    recall_val = recall_event.value
                    break
            
            if f1_val is not None and prec_val is not None and recall_val is not None:
                results.append({
                    'iteration': iter_num,
                    'accuracy': acc_val,
                    'precision': prec_val,
                    'recall': recall_val,
                    'f1': f1_val
                })
        
        return sorted(results, key=lambda x: x['iteration'])
    
    except Exception as e:
        print(f"Error reading TensorBoard: {e}")
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python print_metrics_from_logs.py <experiment_path>")
        print("\nExample:")
        print("  python print_metrics_from_logs.py output/REDO_Frailty_ccpg_pt1_p+CNN/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN")
        sys.exit(1)
    
    exp_path = sys.argv[1]
    
    if not os.path.exists(exp_path):
        print(f"Error: Path not found: {exp_path}")
        sys.exit(1)
    
    # Try TensorBoard first (more reliable)
    summary_dir = os.path.join(exp_path, 'summary')
    if os.path.exists(summary_dir):
        print(f"Extracting from TensorBoard: {summary_dir}")
        results = extract_from_tensorboard(summary_dir)
    else:
        # Try log files
        logs_dir = os.path.join(exp_path, 'logs')
        if os.path.exists(logs_dir):
            log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.txt')])
            if log_files:
                log_file = os.path.join(logs_dir, log_files[-1])  # Most recent
                print(f"Extracting from log file: {log_file}")
                results = extract_from_log_file(log_file)
            else:
                print(f"Error: No log files found in {logs_dir}")
                sys.exit(1)
        else:
            print(f"Error: Neither summary/ nor logs/ directory found in {exp_path}")
            sys.exit(1)
    
    if not results:
        print("No metrics found!")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*80)
    print("MACRO-AVERAGED METRICS (F1, Precision, Recall)")
    print("="*80)
    print(f"{'Iteration':<12} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['iteration']:<12} {r['accuracy']*100:>10.2f}% {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in results]
    mean_acc = sum(accuracies) / len(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    
    # Find best accuracy iteration
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Evaluations: {len(results)}")
    print(f"Best Accuracy: {max_acc*100:.2f}% (iter {best_result['iteration']})")
    print(f"  - F1 Score: {best_result['f1']:.4f}")
    print(f"  - Precision: {best_result['precision']:.4f}")
    print(f"  - Recall: {best_result['recall']:.4f}")
    print(f"Mean Accuracy: {mean_acc*100:.2f}%")
    print(f"Lowest Accuracy: {min_acc*100:.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()
