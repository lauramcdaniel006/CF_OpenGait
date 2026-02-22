#!/usr/bin/env python3
"""
Extract results from Part 4 class weights experiments and update visualization.
"""

import os
import glob
import csv
import json
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("Warning: TensorBoard not available. Results may be limited.")

def extract_from_tensorboard(summary_dir):
    """Extract all metrics from TensorBoard."""
    if not TB_AVAILABLE or not os.path.isdir(summary_dir):
        return {}
    
    try:
        event_files = glob.glob(os.path.join(summary_dir, "events.out.tfevents.*"))
        if not event_files:
            return {}
        
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            metrics[tag] = [(event.step, event.value) for event in scalar_events]
        
        return metrics
    except Exception as e:
        print(f"Error extracting from TensorBoard: {e}")
        return {}

def find_best_checkpoint(metrics):
    """Find best checkpoint based on accuracy."""
    # Find overall test accuracy
    acc_key = None
    for key in metrics.keys():
        if 'test_accuracy' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
            acc_key = key
            break
    
    if not acc_key or not metrics[acc_key]:
        return None, None, None
    
    # Find max accuracy
    max_acc = max(v for _, v in metrics[acc_key])
    max_iter = max(step for step, v in metrics[acc_key] if v == max_acc)
    
    # Get corresponding F1
    f1_key = None
    for key in metrics.keys():
        if 'test_f1' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
            f1_key = key
            break
    
    f1_value = None
    if f1_key and metrics[f1_key]:
        # Find F1 at the same iteration (or closest)
        f1_at_iter = [v for step, v in metrics[f1_key] if step == max_iter]
        if f1_at_iter:
            f1_value = f1_at_iter[0]
        else:
            # Find closest iteration
            closest = min(metrics[f1_key], key=lambda x: abs(x[0] - max_iter))
            f1_value = closest[1]
    
    return max_iter, max_acc * 100, f1_value * 100 if f1_value is not None else None

def extract_all_metrics_at_iteration(metrics, target_iter):
    """Extract all metrics at a specific iteration."""
    result = {}
    
    for key, values in metrics.items():
        # Find value at target iteration or closest
        matching = [v for step, v in values if step == target_iter]
        if matching:
            result[key] = matching[0]
        else:
            closest = min(values, key=lambda x: abs(x[0] - target_iter))
            if abs(closest[0] - target_iter) <= 100:  # Within 100 iterations
                result[key] = closest[1]
    
    return result

def main():
    print("="*80)
    print("EXTRACTING PART 4 CLASS WEIGHTS EXPERIMENTS RESULTS")
    print("="*80)
    
    if not TB_AVAILABLE:
        print("\n❌ TensorBoard library not available!")
        print("   Please activate your conda environment with TensorBoard installed.")
        return
    
    experiments = {
        'REDO_Frailty_ccpg_pt4_swingait_with_weights': {
            'name': 'SwinGait (Frozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_swingait_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_with_weights/summary'
        },
        'REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights': {
            'name': 'DeepGaitV2 (Frozen CNN) + Class Weights',
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/summary'
        }
    }
    
    results = {}
    
    for exp_key, exp_info in experiments.items():
        print(f"\n{'='*80}")
        print(f"Processing: {exp_info['name']}")
        print(f"{'='*80}")
        
        summary_dir = exp_info['summary_dir']
        if not os.path.exists(summary_dir):
            print(f"  ✗ Summary directory not found: {summary_dir}")
            continue
        
        metrics = extract_from_tensorboard(summary_dir)
        
        if not metrics:
            print(f"  ✗ No metrics found")
            continue
        
        print(f"  ✓ Found {len(metrics)} metric(s)")
        
        # Find best checkpoint
        best_iter, best_acc, best_f1 = find_best_checkpoint(metrics)
        
        if best_iter is None:
            print(f"  ✗ Could not find best checkpoint")
            continue
        
        print(f"\n  Best Checkpoint:")
        print(f"    Iteration: {best_iter}")
        print(f"    Accuracy: {best_acc:.2f}%")
        if best_f1 is not None:
            print(f"    F1 Score: {best_f1:.2f}%")
        
        # Extract all metrics at best iteration
        all_metrics = extract_all_metrics_at_iteration(metrics, best_iter)
        
        results[exp_key] = {
            'name': exp_info['name'],
            'best_iteration': best_iter,
            'best_accuracy': best_acc,
            'best_f1': best_f1,
            'all_metrics': all_metrics
        }
        
        # Show per-class metrics if available
        print(f"\n  Per-Class Metrics:")
        for class_name in ['Frail', 'Prefrail', 'Nonfrail']:
            for metric_type in ['precision', 'recall', 'f1', 'specificity']:
                key = f'scalar/test_{metric_type}/{class_name}'
                if key in all_metrics:
                    value = all_metrics[key] * 100
                    print(f"    {class_name} {metric_type.capitalize()}: {value:.2f}%")
    
    # Save results to CSV
    csv_file = 'results_visualization/part4_with_weights_results.csv'
    os.makedirs('results_visualization', exist_ok=True)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment', 'Best Iteration', 'Accuracy (%)', 'F1 Score (%)'])
        for exp_key, data in results.items():
            writer.writerow([
                data['name'],
                data['best_iteration'],
                f"{data['best_accuracy']:.2f}",
                f"{data['best_f1']:.2f}" if data['best_f1'] is not None else ''
            ])
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_file}")
    print(f"{'='*80}")
    
    # Save detailed JSON
    json_file = 'results_visualization/part4_with_weights_results.json'
    json_results = {}
    for exp_key, data in results.items():
        json_results[exp_key] = {
            'name': data['name'],
            'best_iteration': data['best_iteration'],
            'best_accuracy': data['best_accuracy'],
            'best_f1': data['best_f1'],
            'all_metrics': {k: float(v) for k, v in data['all_metrics'].items()}
        }
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {json_file}")
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print("\nBaseline Part 4 (no class weights):")
    print("  SwinGait (Frozen CNN):     73.33% accuracy, 73.89% F1")
    print("  DeepGaitV2 (Frozen CNN):   60.00% accuracy, 60.33% F1")
    
    print("\nPart 4 with Class Weights:")
    for exp_key, data in results.items():
        acc_change = data['best_accuracy'] - (73.33 if 'swingait' in exp_key else 60.00)
        sign = '+' if acc_change >= 0 else ''
        print(f"  {data['name']}:")
        print(f"    {data['best_accuracy']:.2f}% accuracy ({sign}{acc_change:.2f}%), {data['best_f1']:.2f}% F1" if data['best_f1'] else f"    {data['best_accuracy']:.2f}% accuracy ({sign}{acc_change:.2f}%)")
    
    return results

if __name__ == '__main__':
    main()

