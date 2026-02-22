#!/usr/bin/env python3
"""
Create a summary table for Part 4 (DeepGait vs SwinGait) experiments.
Shows best accuracy, F1, precision, and recall for 6 experiments with frozen/unfrozen and class weights indicators.
"""

import csv
import os
import re
import glob
from collections import defaultdict

def load_part4_data(csv_file):
    """Load Part 4 and Part 4b experiment data from CSV."""
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Part 4' in row['Part']:  # This will match both "Part 4" and "Part 4b"
                exp_name = row['Experiment']
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['Accuracy (%)']),
                    'f1': float(row['F1']) if row['F1'] and row['F1'].strip() else None,
                    'precision': float(row['Precision']) if row['Precision'] and row['Precision'].strip() else None,
                    'recall': float(row['Recall']) if row['Recall'] and row['Recall'].strip() else None
                })
    
    # Sort by iteration for each experiment
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['iteration'])
    
    return experiments

def extract_with_weights_metrics():
    """Extract metrics for with_weights experiments from TensorBoard events (new training with Inverse Sqrt weights)."""
    results = {}
    
    # Try to use TensorBoard first (for new training results)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        TB_AVAILABLE = True
    except ImportError:
        TB_AVAILABLE = False
    
    experiments = {
        'REDO_Frailty_ccpg_pt4_swingait_with_weights': {
            'name': 'SwinGait (Frozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_with_weights/logs'
        },
        'REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights': {
            'name': 'DeepGaitV2 (Frozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/logs'
        },
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights': {
            'name': 'SwinGait (Unfrozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights/logs'
        },
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights': {
            'name': 'DeepGaitV2 (Unfrozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights/logs'
        },
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn': {
            'name': 'DeepGaitV2 (Half-Frozen CNN)',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn/logs'
        },
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights': {
            'name': 'DeepGaitV2 (Half-Frozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/logs'
        }
    }
    
    # Try TensorBoard extraction first (for new results)
    # Only use event files created after backup (Nov 20, 15:45) to get Inverse Sqrt weights results
    import tempfile
    import shutil
    backup_time = 1734709500  # Nov 20, 2025 15:45:00 UTC timestamp
    
    if TB_AVAILABLE:
        for exp_key, exp_info in experiments.items():
            summary_dir = exp_info.get('summary_dir')
            if summary_dir and os.path.exists(summary_dir):
                try:
                    # Filter to only use event files newer than backup
                    event_files = glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))
                    if event_files:
                        # Only use event files created after backup
                        new_event_files = [ef for ef in event_files if os.path.getmtime(ef) > backup_time]
                        if new_event_files:
                            # Use only the newest event file to avoid merging old and new results
                            newest_event = max(new_event_files, key=os.path.getmtime)
                            print(f"  Using newest event file for {exp_key}: {os.path.basename(newest_event)}")
                            
                            # Create temp directory with only the new event file
                            temp_dir = tempfile.mkdtemp()
                            extracted_from_tb = False
                            try:
                                temp_event = os.path.join(temp_dir, os.path.basename(newest_event))
                                shutil.copy2(newest_event, temp_event)
                                
                                ea = EventAccumulator(temp_dir)
                                ea.Reload()
                                scalar_tags = ea.Tags()['scalars']
                                
                                # Find test accuracy
                                test_acc_tags = [tag for tag in scalar_tags if 'test' in tag.lower() and 'accuracy' in tag.lower() and not any(c in tag for c in ['Frail', 'Prefrail', 'Nonfrail'])]
                                
                                if test_acc_tags:
                                    tag = test_acc_tags[0]
                                    scalar_events = ea.Scalars(tag)
                                    
                                    if scalar_events:
                                        # Get all accuracies
                                        accuracies = [(int(e.step), e.value * 100) for e in scalar_events]
                                        best_iter, best_acc = max(accuracies, key=lambda x: x[1])
                                        mean_acc = sum(a[1] for a in accuracies) / len(accuracies)
                                        
                                        # Get F1, Precision, Recall at best iteration
                                        f1_tag = [tag for tag in scalar_tags if 'test' in tag.lower() and 'f1' in tag.lower() and not any(c in tag for c in ['Frail', 'Prefrail', 'Nonfrail'])]
                                        prec_tag = [tag for tag in scalar_tags if 'test' in tag.lower() and 'precision' in tag.lower() and not any(c in tag for c in ['Frail', 'Prefrail', 'Nonfrail'])]
                                        rec_tag = [tag for tag in scalar_tags if 'test' in tag.lower() and 'recall' in tag.lower() and not any(c in tag for c in ['Frail', 'Prefrail', 'Nonfrail'])]
                                        
                                        best_f1 = None
                                        best_prec = None
                                        best_rec = None
                                        
                                        if f1_tag:
                                            f1_events = ea.Scalars(f1_tag[0])
                                            f1_dict = {int(e.step): e.value for e in f1_events}
                                            if best_iter in f1_dict:
                                                best_f1 = f1_dict[best_iter]
                                            else:
                                                closest = min(f1_dict.keys(), key=lambda x: abs(x - best_iter))
                                                if abs(closest - best_iter) <= 500:
                                                    best_f1 = f1_dict[closest]
                                        
                                        if prec_tag:
                                            prec_events = ea.Scalars(prec_tag[0])
                                            prec_dict = {int(e.step): e.value for e in prec_events}
                                            if best_iter in prec_dict:
                                                best_prec = prec_dict[best_iter]
                                            else:
                                                closest = min(prec_dict.keys(), key=lambda x: abs(x - best_iter))
                                                if abs(closest - best_iter) <= 500:
                                                    best_prec = prec_dict[closest]
                                        
                                        if rec_tag:
                                            rec_events = ea.Scalars(rec_tag[0])
                                            rec_dict = {int(e.step): e.value for e in rec_events}
                                            if best_iter in rec_dict:
                                                best_rec = rec_dict[best_iter]
                                            else:
                                                closest = min(rec_dict.keys(), key=lambda x: abs(x - best_iter))
                                                if abs(closest - best_iter) <= 500:
                                                    best_rec = rec_dict[closest]
                                        
                                        # Build all_evaluations list
                                        all_evaluations = []
                                        for iter_num, acc in accuracies:
                                            f1_val = None
                                            prec_val = None
                                            rec_val = None
                                            
                                            if f1_tag:
                                                f1_dict = {int(e.step): e.value for e in ea.Scalars(f1_tag[0])}
                                                f1_val = f1_dict.get(iter_num)
                                            if prec_tag:
                                                prec_dict = {int(e.step): e.value for e in ea.Scalars(prec_tag[0])}
                                                prec_val = prec_dict.get(iter_num)
                                            if rec_tag:
                                                rec_dict = {int(e.step): e.value for e in ea.Scalars(rec_tag[0])}
                                                rec_val = rec_dict.get(iter_num)
                                            
                                            all_evaluations.append({
                                                'iteration': iter_num,
                                                'accuracy': acc,
                                                'f1': f1_val,
                                                'precision': prec_val,
                                                'recall': rec_val
                                            })
                                        
                                        results[exp_key] = {
                                            'best_iteration': best_iter,
                                            'best_accuracy': best_acc,
                                            'mean_accuracy': mean_acc,
                                            'f1': best_f1,
                                            'precision': best_prec,
                                            'recall': best_rec,
                                            'all_evaluations': all_evaluations
                                        }
                                        print(f"  ✓ Extracted from TensorBoard: {exp_key} - Best Acc: {best_acc:.2f}%")
                                        extracted_from_tb = True
                            finally:
                                # Clean up temp directory
                                shutil.rmtree(temp_dir, ignore_errors=True)
                            
                            if extracted_from_tb:
                                continue  # Successfully extracted from TensorBoard, skip log file extraction
                        else:
                            # No new event files, skip TensorBoard extraction
                            continue
                    else:
                        continue
                except Exception as e:
                    print(f"  ⚠ TensorBoard extraction failed for {exp_key}: {e}")
                    pass  # Fall back to log file extraction
    
    # Only process log files for experiments that weren't successfully extracted from TensorBoard
    for exp_key, exp_info in experiments.items():
        if exp_key in results:
            continue  # Already extracted from TensorBoard, skip log files
    
    # Fall back to log file extraction if TensorBoard failed or not available
    # Only process experiments that weren't successfully extracted from TensorBoard
    for exp_key, exp_info in experiments.items():
        # Skip if already extracted from TensorBoard
        if exp_key in results:
            continue
            
        log_dir = exp_info['log_dir']
        if not os.path.exists(log_dir):
            continue
        
        log_files = glob.glob(os.path.join(log_dir, '*.txt'))
        if not log_files:
            continue
        
        log_file = max(log_files, key=os.path.getmtime)
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Extract all evaluation sections - look for "Running test" followed by iteration and accuracy
                # Pattern: "Running test" -> "EVALUATION RESULTS" -> "Overall Accuracy: X%"
                eval_pattern = r'Running test.*?EVALUATION RESULTS.*?Overall Accuracy:\s*(\d+\.?\d*)%'
                
                # Find all accuracy values and their positions
                acc_matches = list(re.finditer(r'Overall Accuracy:\s*(\d+\.?\d*)%', content))
                
                best_acc = 0
                best_iter = None
                best_metrics = {}
                all_accuracies = []  # Store all accuracies for mean calculation
                
                # For each accuracy match, find the iteration number before it
                for acc_match in acc_matches:
                    acc = float(acc_match.group(1))
                    
                    # Find the iteration number before this accuracy
                    # Look backwards for "Iteration XXXX" or the test section
                    before_text = content[:acc_match.start()]
                    
                    # Find the most recent iteration number before this accuracy
                    iter_matches = list(re.finditer(r'Iteration\s+(\d+)', before_text))
                    if not iter_matches:
                        continue
                    
                    # Get the iteration number from the section containing this evaluation
                    # Look for the evaluation results section
                    eval_start = before_text.rfind('EVALUATION RESULTS')
                    if eval_start == -1:
                        continue
                    
                    # Find iteration in the section before EVALUATION RESULTS
                    section_before_eval = before_text[:eval_start]
                    iter_match = list(re.finditer(r'Iteration\s+(\d+)', section_before_eval))
                    if not iter_match:
                        continue
                    
                    iter_num = int(iter_match[-1].group(1))
                    
                    # Store all accuracies for mean calculation
                    all_accuracies.append(acc)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_iter = iter_num
                        
                        # Extract metrics for this iteration - find the evaluation section
                        eval_section_start = before_text.rfind('EVALUATION RESULTS')
                        eval_section_end = acc_match.end()
                        section_text = content[eval_section_start:eval_section_end]
                        
                        # Extract per-class metrics and calculate macro-averaged
                        class_names = ['Frail', 'Prefrail', 'Nonfrail']
                        precisions = []
                        recalls = []
                        
                        for class_name in class_names:
                            # Extract Precision
                            prec_pattern = rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%'
                            prec_match = re.search(prec_pattern, section_text)
                            if prec_match:
                                precisions.append(float(prec_match.group(1)) / 100.0)
                            
                            # Extract Recall (Sensitivity)
                            rec_pattern = rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%'
                            rec_match = re.search(rec_pattern, section_text)
                            if rec_match:
                                recalls.append(float(rec_match.group(1)) / 100.0)
                        
                        # Calculate macro-averaged metrics
                        macro_precision = sum(precisions) / len(precisions) if precisions else None
                        macro_recall = sum(recalls) / len(recalls) if recalls else None
                        
                        # Calculate F1 from precision and recall
                        if macro_precision is not None and macro_recall is not None:
                            if macro_precision + macro_recall > 0:
                                macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
                            else:
                                macro_f1 = 0.0
                        else:
                            macro_f1 = None
                        
                        best_metrics = {
                            'f1': macro_f1,
                            'precision': macro_precision,
                            'recall': macro_recall
                        }
                
                if best_iter is not None:
                    mean_acc = sum(all_accuracies) / len(all_accuracies) if all_accuracies else None
                    results[exp_key] = {
                        'best_iteration': best_iter,
                        'best_accuracy': best_acc,
                        'mean_accuracy': mean_acc,
                        'f1': best_metrics.get('f1'),
                        'precision': best_metrics.get('precision'),
                        'recall': best_metrics.get('recall'),
                        'all_evaluations': []  # Will be populated for CSV
                    }
                    
                    # Extract all evaluations for CSV
                    for acc_match in acc_matches:
                        acc = float(acc_match.group(1))
                        before_text = content[:acc_match.start()]
                        eval_start = before_text.rfind('EVALUATION RESULTS')
                        if eval_start == -1:
                            continue
                        section_before_eval = before_text[:eval_start]
                        iter_match = list(re.finditer(r'Iteration\s+(\d+)', section_before_eval))
                        if not iter_match:
                            continue
                        iter_num = int(iter_match[-1].group(1))
                        
                        # Extract metrics for this iteration
                        eval_section_start = before_text.rfind('EVALUATION RESULTS')
                        eval_section_end = acc_match.end()
                        section_text = content[eval_section_start:eval_section_end]
                        
                        # Extract per-class metrics
                        class_names = ['Frail', 'Prefrail', 'Nonfrail']
                        precisions = []
                        recalls = []
                        
                        for class_name in class_names:
                            prec_pattern = rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%'
                            prec_match = re.search(prec_pattern, section_text)
                            if prec_match:
                                precisions.append(float(prec_match.group(1)) / 100.0)
                            
                            rec_pattern = rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%'
                            rec_match = re.search(rec_pattern, section_text)
                            if rec_match:
                                recalls.append(float(rec_match.group(1)) / 100.0)
                        
                        macro_precision = sum(precisions) / len(precisions) if precisions else None
                        macro_recall = sum(recalls) / len(recalls) if recalls else None
                        
                        if macro_precision is not None and macro_recall is not None:
                            if macro_precision + macro_recall > 0:
                                macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
                            else:
                                macro_f1 = 0.0
                        else:
                            macro_f1 = None
                        
                        results[exp_key]['all_evaluations'].append({
                            'iteration': iter_num,
                            'accuracy': acc,
                            'f1': macro_f1,
                            'precision': macro_precision,
                            'recall': macro_recall
                        })
        except Exception as e:
            print(f"Error extracting metrics for {exp_key}: {e}")
            continue
    
    return results

def get_clean_experiment_name(exp_name):
    """Get a cleaner name for display."""
    name_map = {
        'REDO_Frailty_ccpg_pt4_swingait': 'SwinGait (Frozen CNN)',
        'REDO_Frailty_ccpg_pt4_swingait_with_weights': 'SwinGait (Frozen CNN) + Class Weights',
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn': 'SwinGait (Unfrozen CNN)',
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights': 'SwinGait (Unfrozen CNN) + Class Weights',
        'REDO_Frailty_ccpg_pt4_deepgaitv2': 'DeepGaitV2 (Frozen CNN)',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights': 'DeepGaitV2 (Frozen CNN) + Class Weights',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn': 'DeepGaitV2 (Unfrozen CNN)',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights': 'DeepGaitV2 (Unfrozen CNN) + Class Weights',
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn': 'DeepGaitV2 (Half-Frozen CNN)',
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights': 'DeepGaitV2 (Half-Frozen CNN) + Class Weights'
    }
    return name_map.get(exp_name, exp_name.replace('REDO_Frailty_ccpg_pt4_', ''))

def main():
    csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    experiments = load_part4_data(csv_file)
    
    # Extract with_weights metrics from TensorBoard (new Inverse Sqrt weights results)
    with_weights_metrics = extract_with_weights_metrics()
    
    # REPLACE (not merge) with_weights data - TensorBoard results take precedence
    for exp_key, metrics in with_weights_metrics.items():
        # Completely replace old data with new TensorBoard results
        if 'all_evaluations' in metrics and metrics['all_evaluations']:
            experiments[exp_key] = metrics['all_evaluations']
        else:
            # Fallback: just add the best iteration data
            experiments[exp_key] = [{
                'iteration': metrics['best_iteration'],
                'accuracy': metrics['best_accuracy'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }]
    
    if not experiments:
        print("❌ No Part 4 data found!")
        return
    
    print("=" * 100)
    print("PART 4: DEEPGAIT VS SWINGAIT - COMPARISON TABLE")
    print("=" * 100)
    
    # Define all experiments in order (8 Part 4 + 2 Part 4b fairness extension)
    exp_order = [
        'REDO_Frailty_ccpg_pt4_swingait',
        'REDO_Frailty_ccpg_pt4_swingait_with_weights',
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn',
        'REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights',
        'REDO_Frailty_ccpg_pt4_deepgaitv2',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn',
        'REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights',
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn',
        'REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights'
    ]
    
    # Calculate statistics for each experiment
    summary_data = []
    
    for exp_name in exp_order:
        if exp_name not in experiments:
            print(f"⚠️  Warning: {exp_name} not found in data - will show as N/A")
            # Add placeholder entry for missing experiments
            clean_name = get_clean_experiment_name(exp_name)
            is_frozen = 'unfrozen' not in exp_name.lower() and 'half_frozen' not in exp_name.lower()
            has_weights = 'with_weights' in exp_name.lower()
            summary_data.append({
                'Experiment': clean_name,
                'Original Name': exp_name,
                'Best Accuracy (%)': None,
                'Mean Accuracy (%)': None,
                'F1 at Best Acc': None,
                'Precision at Best Acc': None,
                'Recall at Best Acc': None,
                'Best Acc Iteration': None,
                'Frozen': is_frozen,
                'Class Weights': has_weights
            })
            continue
        
        data = experiments[exp_name]
        
        if not data:
            continue
        
        accuracies = [d['accuracy'] for d in data]
        best_acc = max(accuracies)
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else None
        best_acc_row = max(data, key=lambda x: x['accuracy'])
        
        # Get metrics at best accuracy
        f1_at_best = best_acc_row['f1'] if best_acc_row['f1'] is not None else None
        precision_at_best = best_acc_row['precision'] if best_acc_row['precision'] is not None else None
        recall_at_best = best_acc_row['recall'] if best_acc_row['recall'] is not None else None
        
        clean_name = get_clean_experiment_name(exp_name)
        
        # Determine frozen/unfrozen and class weights status
        is_frozen = 'unfrozen' not in exp_name.lower() and 'half_frozen' not in exp_name.lower()
        has_weights = 'with_weights' in exp_name.lower()
        
        summary_data.append({
            'Experiment': clean_name,
            'Original Name': exp_name,
            'Best Accuracy (%)': best_acc,
            'Mean Accuracy (%)': mean_acc,
            'F1 at Best Acc': f1_at_best,
            'Precision at Best Acc': precision_at_best,
            'Recall at Best Acc': recall_at_best,
            'Best Acc Iteration': best_acc_row['iteration'],
            'Frozen': is_frozen,
            'Class Weights': has_weights
        })
    
    # Print table
    print(f"\n{'Experiment':<45} {'Best Acc (%)':<12} {'Mean Acc (%)':<12} {'F1':<8} {'Precision':<10} {'Recall':<10} {'Frozen':<8} {'Weights':<8}")
    print("-" * 125)
    
    for row in summary_data:
        acc_str = f"{row['Best Accuracy (%)']:.2f}" if row['Best Accuracy (%)'] is not None else "N/A"
        mean_acc_str = f"{row['Mean Accuracy (%)']:.2f}" if row['Mean Accuracy (%)'] is not None else "N/A"
        f1_str = f"{row['F1 at Best Acc']:.3f}" if row['F1 at Best Acc'] is not None else "N/A"
        prec_str = f"{row['Precision at Best Acc']:.3f}" if row['Precision at Best Acc'] is not None else "N/A"
        rec_str = f"{row['Recall at Best Acc']:.3f}" if row['Recall at Best Acc'] is not None else "N/A"
        frozen_str = "✓" if row['Frozen'] else "✗"
        weights_str = "✓" if row['Class Weights'] else "✗"
        
        print(f"{row['Experiment']:<50} {acc_str:<12} {mean_acc_str:<12} {f1_str:<8} "
              f"{prec_str:<10} {rec_str:<10} {frozen_str:<8} {weights_str:<8}")
    
    # Save to CSV
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_output = os.path.join(output_dir, 'part4_comparison_table.csv')
    
    with open(csv_output, 'w', newline='') as f:
        fieldnames = ['Experiment', 'Original Name', 'Best Accuracy (%)', 'Mean Accuracy (%)', 
                     'F1 at Best Acc', 'Precision at Best Acc', 'Recall at Best Acc', 
                     'Best Acc Iteration', 'Frozen', 'Class Weights']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in summary_data:
            # Convert None to empty string for CSV
            csv_row = {k: (v if v is not None else '') for k, v in row.items()}
            writer.writerow(csv_row)
    
    # Also update the main CSV file with all evaluations
    print(f"\nUpdating main CSV file with all evaluations...")
    main_csv_file = 'results_visualization/all_redo_parts1-4_all_iterations.csv'
    if os.path.exists(main_csv_file):
        # Read existing CSV
        existing_rows = []
        with open(main_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        
        # Add new rows for unfrozen + weights experiments
        new_rows = []
        for exp_key, metrics in with_weights_metrics.items():
            if 'all_evaluations' in metrics:
                for eval_data in metrics['all_evaluations']:
                    new_rows.append({
                        'Part': 'Part 4: Architecture Comparison',
                        'Experiment': exp_key,
                        'Iteration': str(eval_data['iteration']),
                        'Accuracy (%)': str(eval_data['accuracy']),
                        'F1': str(eval_data['f1']) if eval_data['f1'] is not None else '',
                        'Precision': str(eval_data['precision']) if eval_data['precision'] is not None else '',
                        'Recall': str(eval_data['recall']) if eval_data['recall'] is not None else ''
                    })
        
        # Check which rows are new (not already in CSV)
        existing_exp_iters = {(row['Experiment'], row['Iteration']) for row in existing_rows}
        rows_to_add = [row for row in new_rows 
                      if (row['Experiment'], row['Iteration']) not in existing_exp_iters]
        
        if rows_to_add:
            # Append new rows
            with open(main_csv_file, 'a', newline='') as f:
                fieldnames = ['Part', 'Experiment', 'Iteration', 'Accuracy (%)', 'F1', 'Precision', 'Recall']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(rows_to_add)
            print(f"✓ Added {len(rows_to_add)} new evaluation rows to {main_csv_file}")
        else:
            print(f"✓ All evaluations already in {main_csv_file}")
    
    print(f"\n{'='*100}")
    print(f"✓ Saved summary table to: {csv_output}")
    print(f"{'='*100}")
    
    # Create LaTeX table - separate Part 4a (original) from Part 4b (fairness extension)
    latex_file = os.path.join(output_dir, 'part4_comparison_table_latex.txt')
    with open(latex_file, 'w') as f:
        f.write("% IMPORTANT: To prevent text from appearing over the table, add these to your LaTeX preamble:\n")
        f.write("% \\usepackage{float}\n")
        f.write("% \\usepackage{placeins}\n")
        f.write("%\n")
        f.write("% The \\FloatBarrier clears all pending floats before the table\n")
        f.write("% The [H] option forces the table to appear exactly where you place it\n")
        f.write("\\FloatBarrier\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Part 4: DeepGait vs SwinGait - Performance Comparison}\n")
        f.write("\\label{tab:part4_comparison}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write("Experiment & Best Acc (\\%) & Mean Acc (\\%) & F1 & Precision & Recall & Frozen & Weights \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{8}{l}{\\textit{Part 4a: Original Fair Comparison (2x2x2 Design)}} \\\\\n")
        f.write("\\midrule\n")
        
        # Part 4a experiments (first 8)
        part4a_experiments = summary_data[:8]
        for row in part4a_experiments:
            clean_name = row['Experiment'].replace('&', '\\&')
            acc_str = f"{row['Best Accuracy (%)']:.2f}" if row['Best Accuracy (%)'] is not None else "N/A"
            mean_acc_str = f"{row['Mean Accuracy (%)']:.2f}" if row['Mean Accuracy (%)'] is not None else "N/A"
            f1_str = f"{row['F1 at Best Acc']:.3f}" if row['F1 at Best Acc'] is not None else "N/A"
            prec_str = f"{row['Precision at Best Acc']:.3f}" if row['Precision at Best Acc'] is not None else "N/A"
            rec_str = f"{row['Recall at Best Acc']:.3f}" if row['Recall at Best Acc'] is not None else "N/A"
            
            # Convert checkmarks to LaTeX
            frozen_latex = "$\\checkmark$" if row['Frozen'] else "$\\times$"
            weights_latex = "$\\checkmark$" if row['Class Weights'] else "$\\times$"
            
            f.write(f"{clean_name} & {acc_str} & {mean_acc_str} & {f1_str} & "
                   f"{prec_str} & {rec_str} & {frozen_latex} & {weights_latex} \\\\\n")
        
        # Part 4b experiments (fairness extension) - if they exist
        if len(summary_data) > 8:
            f.write("\\midrule\n")
            f.write("\\multicolumn{8}{l}{\\textit{Part 4b: Fairness Extension (Half-Frozen DeepGaitV2)}} \\\\\n")
            f.write("\\midrule\n")
            
            part4b_experiments = summary_data[8:]
            for row in part4b_experiments:
                clean_name = row['Experiment'].replace('&', '\\&')
                acc_str = f"{row['Best Accuracy (%)']:.2f}" if row['Best Accuracy (%)'] is not None else "N/A"
                mean_acc_str = f"{row['Mean Accuracy (%)']:.2f}" if row['Mean Accuracy (%)'] is not None else "N/A"
                f1_str = f"{row['F1 at Best Acc']:.3f}" if row['F1 at Best Acc'] is not None else "N/A"
                prec_str = f"{row['Precision at Best Acc']:.3f}" if row['Precision at Best Acc'] is not None else "N/A"
                rec_str = f"{row['Recall at Best Acc']:.3f}" if row['Recall at Best Acc'] is not None else "N/A"
                
                # Convert checkmarks to LaTeX
                frozen_latex = "$\\checkmark$" if row['Frozen'] else "$\\times$"
                weights_latex = "$\\checkmark$" if row['Class Weights'] else "$\\times$"
                
                f.write(f"{clean_name} & {acc_str} & {mean_acc_str} & {f1_str} & "
                       f"{prec_str} & {rec_str} & {frozen_latex} & {weights_latex} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to: {latex_file}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()

