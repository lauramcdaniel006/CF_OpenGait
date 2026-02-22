#!/usr/bin/env python3
"""
Extract metrics from DeepGaitV2 Part 1 and Part 4a experiments.
"""

import os
import glob
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_metrics_from_tensorboard(log_dir):
    """Extract all metrics from TensorBoard event files."""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        
        # Extract test metrics (tags don't have "scalar/" prefix)
        for tag in scalar_tags:
            if tag == 'test_accuracy/':
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    metrics['accuracy'] = [s.value * 100 for s in scalar_events]  # Convert to percentage
                    metrics['accuracy_iterations'] = [s.step for s in scalar_events]
            
            if tag == 'test_f1/':
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    metrics['f1'] = [s.value for s in scalar_events]
                    metrics['f1_iterations'] = [s.step for s in scalar_events]
            
            if tag == 'test_precision/':
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    metrics['precision'] = [s.value for s in scalar_events]
                    metrics['precision_iterations'] = [s.step for s in scalar_events]
            
            if tag == 'test_recall/':
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    metrics['recall'] = [s.value for s in scalar_events]
                    metrics['recall_iterations'] = [s.step for s in scalar_events]
        
        return metrics
    except Exception as e:
        print(f"Error reading TensorBoard from {log_dir}: {e}")
        return {}

def find_tensorboard_dir(exp_dir):
    """Find TensorBoard event file directory."""
    # Look for event files
    event_files = glob.glob(f"{exp_dir}/**/events.out.tfevents*", recursive=True)
    if event_files:
        # Return the directory containing the event file
        from pathlib import Path
        return str(Path(event_files[0]).parent)
    return None

def extract_all_experiments():
    """Extract metrics from all DeepGaitV2 Part 1 and Part 4a experiments."""
    
    # Part 1 experiments
    part1_experiments = {
        'All Trainable (0 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable',
        'First Layer Frozen (1 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen',
        'First Two Frozen (2 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen',
        'Early Layers Frozen (3 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen',
        'Heavy Frozen (4 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen',
        'All Frozen (5 frozen)': 'output/REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen',
    }
    
    # Part 4a experiments
    part4a_experiments = {
        'B1: Partially Frozen (no weights)': 'output/REDO_Frailty_ccpg_pt4a_deepgaitv2_B1_partially_frozen',
        'B2: Partially Frozen (with weights)': 'output/REDO_Frailty_ccpg_pt4a_deepgaitv2_B2_partially_frozen_with_weights',
        'B3: Unfrozen (no weights)': 'output/REDO_Frailty_ccpg_pt4a_deepgaitv2_B3_unfrozen',
        'B4: Unfrozen (with weights)': 'output/REDO_Frailty_ccpg_pt4a_deepgaitv2_B4_unfrozen_with_weights',
    }
    
    results = {}
    
    print("=" * 80)
    print("DEEPGaitV2 PART 1 - FREEZING STRATEGIES")
    print("=" * 80)
    
    for exp_name, exp_dir in part1_experiments.items():
        if not os.path.exists(exp_dir):
            print(f"\n⚠️  {exp_name}: Directory not found")
            continue
        
        tb_dir = find_tensorboard_dir(exp_dir)
        if not tb_dir:
            print(f"\n⚠️  {exp_name}: No TensorBoard files found")
            continue
        
        metrics = extract_metrics_from_tensorboard(tb_dir)
        
        if not metrics or 'accuracy' not in metrics:
            print(f"\n⚠️  {exp_name}: No metrics found")
            continue
        
        acc_values = metrics['accuracy']
        f1_values = metrics.get('f1', [])
        prec_values = metrics.get('precision', [])
        recall_values = metrics.get('recall', [])
        
        # Find best accuracy and corresponding metrics
        best_idx = acc_values.index(max(acc_values))
        best_acc = acc_values[best_idx]
        best_f1 = f1_values[best_idx] if f1_values else None
        best_prec = prec_values[best_idx] if prec_values else None
        best_recall = recall_values[best_idx] if recall_values else None
        best_iter = metrics['accuracy_iterations'][best_idx]
        
        # Calculate statistics
        mean_acc = sum(acc_values) / len(acc_values) if acc_values else 0
        min_acc = min(acc_values) if acc_values else 0
        
        results[exp_name] = {
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_precision': best_prec,
            'best_recall': best_recall,
            'best_iter': best_iter,
            'mean_acc': mean_acc,
            'min_acc': min_acc,
            'all_acc': acc_values,
            'all_f1': f1_values,
            'all_precision': prec_values,
            'all_recall': recall_values,
        }
        
        print(f"\n{exp_name}:")
        print(f"  Best Accuracy: {best_acc:.2f}% (iter {best_iter})")
        if best_f1 is not None:
            print(f"  Best F1: {best_f1:.4f}")
        if best_prec is not None:
            print(f"  Best Precision: {best_prec:.4f}")
        if best_recall is not None:
            print(f"  Best Recall: {best_recall:.4f}")
        print(f"  Mean Accuracy: {mean_acc:.2f}%")
        print(f"  Lowest Accuracy: {min_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("DEEPGaitV2 PART 4A - BEST FREEZING STRATEGY EXPERIMENTS")
    print("=" * 80)
    
    for exp_name, exp_dir in part4a_experiments.items():
        if not os.path.exists(exp_dir):
            print(f"\n⚠️  {exp_name}: Directory not found")
            continue
        
        tb_dir = find_tensorboard_dir(exp_dir)
        if not tb_dir:
            print(f"\n⚠️  {exp_name}: No TensorBoard files found")
            continue
        
        metrics = extract_metrics_from_tensorboard(tb_dir)
        
        if not metrics or 'accuracy' not in metrics:
            print(f"\n⚠️  {exp_name}: No metrics found")
            continue
        
        acc_values = metrics['accuracy']
        f1_values = metrics.get('f1', [])
        prec_values = metrics.get('precision', [])
        recall_values = metrics.get('recall', [])
        
        # Find best accuracy and corresponding metrics
        best_idx = acc_values.index(max(acc_values))
        best_acc = acc_values[best_idx]
        best_f1 = f1_values[best_idx] if f1_values else None
        best_prec = prec_values[best_idx] if prec_values else None
        best_recall = recall_values[best_idx] if recall_values else None
        best_iter = metrics['accuracy_iterations'][best_idx]
        
        # Calculate statistics
        mean_acc = sum(acc_values) / len(acc_values) if acc_values else 0
        min_acc = min(acc_values) if acc_values else 0
        
        results[exp_name] = {
            'best_acc': best_acc,
            'best_f1': best_f1,
            'best_precision': best_prec,
            'best_recall': best_recall,
            'best_iter': best_iter,
            'mean_acc': mean_acc,
            'min_acc': min_acc,
            'all_acc': acc_values,
            'all_f1': f1_values,
            'all_precision': prec_values,
            'all_recall': recall_values,
        }
        
        print(f"\n{exp_name}:")
        print(f"  Best Accuracy: {best_acc:.2f}% (iter {best_iter})")
        if best_f1 is not None:
            print(f"  Best F1: {best_f1:.4f}")
        if best_prec is not None:
            print(f"  Best Precision: {best_prec:.4f}")
        if best_recall is not None:
            print(f"  Best Recall: {best_recall:.4f}")
        print(f"  Mean Accuracy: {mean_acc:.2f}%")
        print(f"  Lowest Accuracy: {min_acc:.2f}%")
    
    return results

if __name__ == '__main__':
    results = extract_all_experiments()
    
    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Experiment':<50} {'Best Acc':<12} {'Best F1':<12} {'Precision':<12} {'Recall':<12} {'Mean Acc':<12} {'Low Acc':<12}")
    print("-" * 80)
    
    for exp_name, metrics in results.items():
        best_acc = f"{metrics['best_acc']:.2f}%"
        best_f1 = f"{metrics['best_f1']:.4f}" if metrics['best_f1'] is not None else "N/A"
        best_prec = f"{metrics['best_precision']:.4f}" if metrics['best_precision'] is not None else "N/A"
        best_recall = f"{metrics['best_recall']:.4f}" if metrics['best_recall'] is not None else "N/A"
        mean_acc = f"{metrics['mean_acc']:.2f}%"
        min_acc = f"{metrics['min_acc']:.2f}%"
        
        print(f"{exp_name:<50} {best_acc:<12} {best_f1:<12} {best_prec:<12} {best_recall:<12} {mean_acc:<12} {min_acc:<12}")
