#!/usr/bin/env python3
"""
View detailed iteration-by-iteration logs for all SwinGait Part 1 configs.
Shows all iterations with their accuracies, losses, and metrics.
"""

import os
import re
import glob
from pathlib import Path

def find_latest_log_with_evals(exp_dir):
    """Find the latest log file with evaluation results."""
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        return None
    
    log_files = glob.glob(os.path.join(logs_dir, '*.txt'))
    if not log_files:
        return None
    
    # Sort by modification time (newest first)
    log_files_sorted = sorted(log_files, key=os.path.getmtime, reverse=True)
    
    # Find the most recent log file that has evaluation results
    for log_file in log_files_sorted:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'EVALUATION RESULTS' in content:
                    return log_file
        except:
            continue
    
    # If no log file has evaluations, return the most recent one anyway
    return log_files_sorted[0] if log_files_sorted else None

def extract_all_iterations(log_file):
    """Extract all training iterations and evaluation results from log file."""
    iterations = []
    evaluations = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract training iterations
            train_pattern = r'Iteration\s+(\d+).*?Cost\s+([\d.]+)s.*?triplet_loss=([\d.]+).*?softmax_loss=([\d.]+).*?softmax_accuracy=([\d.]+)'
            for match in re.finditer(train_pattern, content):
                iterations.append({
                    'iteration': int(match.group(1)),
                    'type': 'training',
                    'cost': float(match.group(2)),
                    'triplet_loss': float(match.group(3)),
                    'softmax_loss': float(match.group(4)),
                    'softmax_accuracy': float(match.group(5))
                })
            
            # Extract evaluation results
            eval_sections = re.split(r'EVALUATION RESULTS', content)
            for section in eval_sections[1:]:
                eval_metrics = {}
                
                # Extract iteration number
                iter_match = re.search(r'Iteration\s+(\d+)', section)
                if iter_match:
                    eval_metrics['iteration'] = int(iter_match.group(1))
                else:
                    # Sometimes evaluation happens without explicit iteration
                    eval_metrics['iteration'] = None
                
                # Overall Accuracy
                acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
                if acc_match:
                    eval_metrics['overall_accuracy'] = float(acc_match.group(1))
                
                # Per-class metrics
                class_names = ['Frail', 'Prefrail', 'Nonfrail']
                for class_name in class_names:
                    # Recall
                    rec_match = re.search(
                        rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
                        section
                    )
                    if rec_match:
                        eval_metrics[f'{class_name.lower()}_recall'] = float(rec_match.group(1))
                    
                    # Precision
                    prec_match = re.search(
                        rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
                        section
                    )
                    if prec_match:
                        eval_metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1))
                    
                    # F1
                    f1_match = re.search(
                        rf'{re.escape(class_name)}\s+F1\s+Score:\s*(\d+\.?\d*)%', 
                        section
                    )
                    if not f1_match:
                        f1_match = re.search(
                            rf'{re.escape(class_name)}\s+F1:\s*(\d+\.?\d*)%', 
                            section
                        )
                    if f1_match:
                        eval_metrics[f'{class_name.lower()}_f1'] = float(f1_match.group(1))
                    # Calculate F1 if precision and recall exist
                    elif f'{class_name.lower()}_precision' in eval_metrics and f'{class_name.lower()}_recall' in eval_metrics:
                        prec = eval_metrics[f'{class_name.lower()}_precision']
                        rec = eval_metrics[f'{class_name.lower()}_recall']
                        if prec > 0 and rec > 0:
                            f1 = 2 * (prec * rec) / (prec + rec)
                            eval_metrics[f'{class_name.lower()}_f1'] = f1
                
                # Mean metrics
                mean_f1_match = re.search(r'Mean\s+F1:\s*(\d+\.?\d*)%', section)
                if mean_f1_match:
                    eval_metrics['mean_f1'] = float(mean_f1_match.group(1))
                else:
                    f1s = [eval_metrics.get(f'{c.lower()}_f1') for c in class_names 
                           if eval_metrics.get(f'{c.lower()}_f1') is not None]
                    if f1s:
                        eval_metrics['mean_f1'] = sum(f1s) / len(f1s)
                
                mean_prec_match = re.search(r'Mean\s+Precision:\s*(\d+\.?\d*)%', section)
                if mean_prec_match:
                    eval_metrics['mean_precision'] = float(mean_prec_match.group(1))
                else:
                    precs = [eval_metrics.get(f'{c.lower()}_precision') for c in class_names 
                            if eval_metrics.get(f'{c.lower()}_precision') is not None]
                    if precs:
                        eval_metrics['mean_precision'] = sum(precs) / len(precs)
                
                mean_rec_match = re.search(r'Mean\s+Recall:\s*(\d+\.?\d*)%', section)
                if mean_rec_match:
                    eval_metrics['mean_recall'] = float(mean_rec_match.group(1))
                else:
                    recalls = [eval_metrics.get(f'{c.lower()}_recall') for c in class_names 
                              if eval_metrics.get(f'{c.lower()}_recall') is not None]
                    if recalls:
                        eval_metrics['mean_recall'] = sum(recalls) / len(recalls)
                
                if eval_metrics:
                    eval_metrics['type'] = 'evaluation'
                    evaluations.append(eval_metrics)
                    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return iterations, evaluations

def format_evaluation(eval_dict):
    """Format evaluation metrics for display."""
    parts = []
    if 'overall_accuracy' in eval_dict:
        parts.append(f"Acc: {eval_dict['overall_accuracy']:.2f}%")
    if 'mean_f1' in eval_dict:
        parts.append(f"F1: {eval_dict['mean_f1']:.2f}%")
    if 'mean_precision' in eval_dict:
        parts.append(f"Prec: {eval_dict['mean_precision']:.2f}%")
    if 'mean_recall' in eval_dict:
        parts.append(f"Rec: {eval_dict['mean_recall']:.2f}%")
    return ", ".join(parts) if parts else "N/A"

def main():
    print("=" * 100)
    print("SWINGAIT PART 1 - DETAILED ITERATION LOGS")
    print("=" * 100)
    print()
    
    # Part 1 configs
    part1_configs = [
        ("Part 1: Pretrained Unfrozen", "REDO_Frailty_ccpg_pt1_pretrained(UF)", "SwinGait"),
        ("Part 1: p+CNN", "REDO_Frailty_ccpg_pt1_p+CNN", "SwinGait"),
        ("Part 1: p+CNN+Tintro", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro", "SwinGait"),
        ("Part 1: p+CNN+Tintro+T1", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1", "SwinGait"),
        ("Part 1: p+CNN+Tintro+T1+T2", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2", "SwinGait"),
    ]
    
    for config_name, exp_path, model_type in part1_configs:
        full_path = os.path.join("output", exp_path, model_type, exp_path)
        
        if not os.path.exists(full_path):
            print(f"\n{'='*100}")
            print(f"{config_name} - Directory not found")
            print(f"{'='*100}")
            continue
        
        log_file = find_latest_log_with_evals(full_path)
        if not log_file:
            print(f"\n{'='*100}")
            print(f"{config_name} - No log file found")
            print(f"{'='*100}")
            continue
        
        print(f"\n{'='*100}")
        print(f"{config_name}")
        print(f"Log file: {os.path.basename(log_file)}")
        print(f"{'='*100}")
        print()
        
        iterations, evaluations = extract_all_iterations(log_file)
        
        # Combine and sort by iteration
        all_events = []
        for it in iterations:
            all_events.append(it)
        for ev in evaluations:
            all_events.append(ev)
        
        # Sort by iteration
        all_events.sort(key=lambda x: x.get('iteration', 0) if x.get('iteration') is not None else 99999)
        
        print("ITERATION-BY-ITERATION LOG:")
        print("-" * 100)
        
        current_iter = None
        for event in all_events:
            iter_num = event.get('iteration', 'N/A')
            
            if event['type'] == 'training':
                if iter_num != current_iter:
                    print()
                    print(f"Iteration {iter_num}:")
                    current_iter = iter_num
                print(f"  Training - Cost: {event['cost']:.2f}s, "
                      f"Triplet Loss: {event['triplet_loss']:.4f}, "
                      f"Softmax Loss: {event['softmax_loss']:.4f}, "
                      f"Train Acc: {event['softmax_accuracy']:.4f}")
            
            elif event['type'] == 'evaluation':
                if iter_num != current_iter:
                    print()
                    if iter_num is not None:
                        print(f"Iteration {iter_num}:")
                    else:
                        print("Evaluation (no iteration number):")
                    current_iter = iter_num
                print(f"  Evaluation - {format_evaluation(event)}")
                
                # Show per-class if available
                for class_name in ['frail', 'prefrail', 'nonfrail']:
                    if f'{class_name}_precision' in event:
                        prec = event.get(f'{class_name}_precision', 0)
                        rec = event.get(f'{class_name}_recall', 0)
                        f1 = event.get(f'{class_name}_f1', 0)
                        print(f"    {class_name.capitalize()}: Prec={prec:.2f}%, Rec={rec:.2f}%, F1={f1:.2f}%")
        
        print()
        print(f"Summary: {len(iterations)} training iterations, {len(evaluations)} evaluations")
        print()

if __name__ == "__main__":
    main()
