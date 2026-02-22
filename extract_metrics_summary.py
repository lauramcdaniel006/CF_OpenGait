#!/usr/bin/env python3
"""
Extract metrics summary from SwinGait Part 1, Part 2, Part 4 and DeepGaitV2 Part 1, Part 4a logs.
Shows best, mean, and lowest accuracy with corresponding F1, precision, and recall for each config.
"""

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

def extract_evaluation_metrics(log_file):
    """Extract all evaluation metrics from log file."""
    evaluations = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Split by evaluation sections
            eval_sections = re.split(r'EVALUATION RESULTS', content)
            
            for section in eval_sections[1:]:  # Skip first empty part
                eval_metrics = {}
                
                # Extract iteration number
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
                        eval_metrics[f'{class_name.lower()}_recall'] = float(sens_match.group(1))
                    
                    # Precision
                    prec_match = re.search(
                        rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
                        section
                    )
                    if prec_match:
                        eval_metrics[f'{class_name.lower()}_precision'] = float(prec_match.group(1))
                    
                    # F1 Score (try multiple patterns)
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
                    # Calculate F1 from precision and recall if not found
                    elif f'{class_name.lower()}_precision' in eval_metrics and f'{class_name.lower()}_recall' in eval_metrics:
                        prec = eval_metrics[f'{class_name.lower()}_precision']
                        rec = eval_metrics[f'{class_name.lower()}_recall']
                        if prec > 0 and rec > 0:
                            f1 = 2 * (prec * rec) / (prec + rec)
                            eval_metrics[f'{class_name.lower()}_f1'] = f1
                
                # Mean metrics (if available)
                mean_recall_match = re.search(r'Mean\s+Recall:\s*(\d+\.?\d*)%', section)
                if mean_recall_match:
                    eval_metrics['mean_recall'] = float(mean_recall_match.group(1))
                
                mean_precision_match = re.search(r'Mean\s+Precision:\s*(\d+\.?\d*)%', section)
                if mean_precision_match:
                    eval_metrics['mean_precision'] = float(mean_precision_match.group(1))
                
                mean_f1_match = re.search(r'Mean\s+F1:\s*(\d+\.?\d*)%', section)
                if mean_f1_match:
                    eval_metrics['mean_f1'] = float(mean_f1_match.group(1))
                
                # Calculate mean F1 from per-class if not found
                if 'mean_f1' not in eval_metrics:
                    f1s = [eval_metrics.get(f'{c.lower()}_f1') for c in class_names 
                           if eval_metrics.get(f'{c.lower()}_f1') is not None]
                    if f1s:
                        eval_metrics['mean_f1'] = sum(f1s) / len(f1s)
                
                # Calculate mean precision from per-class if not found
                if 'mean_precision' not in eval_metrics:
                    precs = [eval_metrics.get(f'{c.lower()}_precision') for c in class_names 
                            if eval_metrics.get(f'{c.lower()}_precision') is not None]
                    if precs:
                        eval_metrics['mean_precision'] = sum(precs) / len(precs)
                
                # Calculate mean recall from per-class if not found
                if 'mean_recall' not in eval_metrics:
                    recalls = [eval_metrics.get(f'{c.lower()}_recall') for c in class_names 
                              if eval_metrics.get(f'{c.lower()}_recall') is not None]
                    if recalls:
                        eval_metrics['mean_recall'] = sum(recalls) / len(recalls)
                
                if eval_metrics:
                    evaluations.append(eval_metrics)
                    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return evaluations

def find_latest_log(exp_dir):
    """Find the latest log file with evaluation results in experiment directory."""
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

def calculate_mean_metrics(evaluations):
    """Calculate mean metrics from all evaluations."""
    if not evaluations:
        return {}
    
    metrics_to_avg = ['overall_accuracy', 'mean_recall', 'mean_precision', 'mean_f1',
                      'frail_recall', 'frail_precision', 'frail_f1',
                      'prefrail_recall', 'prefrail_precision', 'prefrail_f1',
                      'nonfrail_recall', 'nonfrail_precision', 'nonfrail_f1']
    
    mean_metrics = {}
    for metric in metrics_to_avg:
        values = [e.get(metric) for e in evaluations if e.get(metric) is not None]
        if values:
            mean_metrics[metric] = sum(values) / len(values)
    
    return mean_metrics

def get_best_and_lowest(evaluations):
    """Get best and lowest accuracy evaluations with corresponding metrics."""
    if not evaluations:
        return None, None
    
    # Filter evaluations with overall_accuracy
    valid_evals = [e for e in evaluations if 'overall_accuracy' in e]
    if not valid_evals:
        return None, None
    
    best_eval = max(valid_evals, key=lambda x: x['overall_accuracy'])
    lowest_eval = min(valid_evals, key=lambda x: x['overall_accuracy'])
    
    return best_eval, lowest_eval

def format_metrics(eval_dict, prefix=""):
    """Format metrics for display."""
    if not eval_dict:
        return "N/A"
    
    parts = []
    if 'overall_accuracy' in eval_dict:
        parts.append(f"Acc: {eval_dict['overall_accuracy']:.2f}%")
    if 'mean_f1' in eval_dict:
        parts.append(f"F1: {eval_dict['mean_f1']:.2f}%")
    elif any(k.endswith('_f1') for k in eval_dict.keys()):
        f1s = [eval_dict[k] for k in eval_dict.keys() if k.endswith('_f1')]
        if f1s:
            parts.append(f"F1: {sum(f1s)/len(f1s):.2f}%")
    if 'mean_precision' in eval_dict:
        parts.append(f"Prec: {eval_dict['mean_precision']:.2f}%")
    elif any(k.endswith('_precision') for k in eval_dict.keys()):
        precs = [eval_dict[k] for k in eval_dict.keys() if k.endswith('_precision')]
        if precs:
            parts.append(f"Prec: {sum(precs)/len(precs):.2f}%")
    if 'mean_recall' in eval_dict:
        parts.append(f"Rec: {eval_dict['mean_recall']:.2f}%")
    elif any(k.endswith('_recall') for k in eval_dict.keys()):
        recalls = [eval_dict[k] for k in eval_dict.keys() if k.endswith('_recall')]
        if recalls:
            parts.append(f"Rec: {sum(recalls)/len(recalls):.2f}%")
    
    return ", ".join(parts) if parts else "N/A"

def process_config(config_name, exp_path, model_type="SwinGait"):
    """Process a single config and return summary."""
    full_path = os.path.join("output", exp_path, model_type, exp_path)
    
    if not os.path.exists(full_path):
        # Try alternative path structure (some configs might have different save_name)
        alt_path = os.path.join("output", exp_path, model_type)
        if os.path.exists(alt_path):
            # Find the actual save_name directory
            subdirs = [d for d in os.listdir(alt_path) if os.path.isdir(os.path.join(alt_path, d))]
            if subdirs:
                full_path = os.path.join(alt_path, subdirs[0])
            else:
                return None
        else:
            return None
    
    log_file = find_latest_log(full_path)
    if not log_file:
        return None
    
    evaluations = extract_evaluation_metrics(log_file)
    if not evaluations:
        # Debug: check if log file has content
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'EVALUATION RESULTS' not in content:
                    return None
        except:
            return None
    
    best_eval, lowest_eval = get_best_and_lowest(evaluations)
    mean_metrics = calculate_mean_metrics(evaluations)
    
    return {
        'config_name': config_name,
        'evaluations': evaluations,
        'best': best_eval,
        'lowest': lowest_eval,
        'mean': mean_metrics,
        'count': len(evaluations)
    }

def main():
    print("=" * 100)
    print("METRICS SUMMARY: SwinGait Part 1, Part 2, Part 4 & DeepGaitV2 Part 1, Part 4a")
    print("=" * 100)
    print()
    
    # SwinGait Part 1 configs
    swingait_part1_configs = [
        ("Part 1: Pretrained Unfrozen", "REDO_Frailty_ccpg_pt1_pretrained(UF)", "SwinGait"),
        ("Part 1: p+CNN", "REDO_Frailty_ccpg_pt1_p+CNN", "SwinGait"),
        ("Part 1: p+CNN+Tintro", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro", "SwinGait"),
        ("Part 1: p+CNN+Tintro+T1", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1", "SwinGait"),
        ("Part 1: p+CNN+Tintro+T1+T2", "REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2", "SwinGait"),
    ]
    
    # SwinGait Part 2 configs
    swingait_part2_configs = [
        ("Part 2: Uniform Weights", "REDO_Frailty_ccpg_pt2_classweight_uniform", "SwinGait"),
        ("Part 2: Balanced Normal", "REDO_Frailty_ccpg_pt2_classweight_balanced_normal", "SwinGait"),
        ("Part 2: Inverse Sqrt", "REDO_Frailty_ccpg_pt2_classweight_inverse_sqrt", "SwinGait"),
        ("Part 2: Logarithmic", "REDO_Frailty_ccpg_pt2_classweight_logarithmic", "SwinGait"),
        ("Part 2: Smooth Effective", "REDO_Frailty_ccpg_pt2_classweight_smooth_effective", "SwinGait"),
    ]
    
    # SwinGait Part 4 configs
    swingait_part4_configs = [
        ("Part 4a: B1 Frozen CNN", "REDO_Frailty_ccpg_pt4a_swingait_B1_frozen_cnn", "SwinGait"),
        ("Part 4a: B2 Frozen CNN + Weights", "REDO_Frailty_ccpg_pt4a_swingait_B2_frozen_cnn_with_weights", "SwinGait"),
        ("Part 4a: B3 Unfrozen CNN", "REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn", "SwinGait"),
        ("Part 4a: B4 Unfrozen CNN + Weights", "REDO_Frailty_ccpg_pt4a_swingait_B4_unfrozen_cnn_with_weights", "SwinGait"),
    ]
    
    # DeepGaitV2 Part 1 configs
    deepgait_part1_configs = [
        ("Part 1: All Trainable", "REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable", "DeepGaitV2"),
        ("Part 1: All Frozen", "REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen", "DeepGaitV2"),
        ("Part 1: Early Frozen", "REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen", "DeepGaitV2"),
        ("Part 1: First Layer Frozen", "REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen", "DeepGaitV2"),
        ("Part 1: First Two Frozen", "REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen", "DeepGaitV2"),
        ("Part 1: Heavy Frozen", "REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen", "DeepGaitV2"),
    ]
    
    # DeepGaitV2 Part 4a configs
    deepgait_part4a_configs = [
        ("Part 4a: B1 Partially Frozen", "REDO_Frailty_ccpg_pt4a_deepgaitv2_B1_partially_frozen", "DeepGaitV2"),
        ("Part 4a: B2 Partially Frozen + Weights", "REDO_Frailty_ccpg_pt4a_deepgaitv2_B2_partially_frozen_with_weights", "DeepGaitV2"),
        ("Part 4a: B3 Unfrozen", "REDO_Frailty_ccpg_pt4a_deepgaitv2_B3_unfrozen", "DeepGaitV2"),
        ("Part 4a: B4 Unfrozen + Weights", "REDO_Frailty_ccpg_pt4a_deepgaitv2_B4_unfrozen_with_weights", "DeepGaitV2"),
    ]
    
    all_configs = [
        ("SWINGAIT PART 1", swingait_part1_configs),
        ("SWINGAIT PART 2", swingait_part2_configs),
        ("SWINGAIT PART 4", swingait_part4_configs),
        ("DEEPGaitV2 PART 1", deepgait_part1_configs),
        ("DEEPGaitV2 PART 4a", deepgait_part4a_configs),
    ]
    
    results = {}
    
    for section_name, configs in all_configs:
        print(f"\n{'='*100}")
        print(f"{section_name}")
        print(f"{'='*100}")
        print()
        
        section_results = {}
        
        for config_name, exp_path, model_type in configs:
            result = process_config(config_name, exp_path, model_type)
            if result:
                section_results[config_name] = result
                
                print(f"Config: {config_name}")
                print(f"  Evaluations found: {result['count']}")
                
                if result['best']:
                    print(f"  Best Accuracy (Iter {result['best'].get('iteration', 'N/A')}): {format_metrics(result['best'])}")
                
                if result['mean'] and 'overall_accuracy' in result['mean']:
                    print(f"  Mean Accuracy: {result['mean']['overall_accuracy']:.2f}%")
                    mean_f1 = result['mean'].get('mean_f1') or result['mean'].get('frail_f1', 0)
                    mean_prec = result['mean'].get('mean_precision') or result['mean'].get('frail_precision', 0)
                    mean_rec = result['mean'].get('mean_recall') or result['mean'].get('frail_recall', 0)
                    if mean_f1 or mean_prec or mean_rec:
                        print(f"  Mean Metrics: F1: {mean_f1:.2f}%, Prec: {mean_prec:.2f}%, Rec: {mean_rec:.2f}%")
                
                if result['lowest']:
                    print(f"  Lowest Accuracy (Iter {result['lowest'].get('iteration', 'N/A')}): {format_metrics(result['lowest'])}")
                
                print()
            else:
                print(f"Config: {config_name} - No log file or evaluations found")
                print()
        
        results[section_name] = section_results
    
    print("\n" + "=" * 100)
    print("SUMMARY COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
