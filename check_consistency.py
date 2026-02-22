#!/usr/bin/env python3
"""Check evaluation consistency for REDO Part 2 experiments."""

import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_evaluation_history(summary_dir):
    """Get all evaluation accuracy values from TensorBoard."""
    if not os.path.isdir(summary_dir):
        return []
    
    try:
        event_files = glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))
        if not event_files:
            return []
        
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        # Look for test accuracy
        for tag in scalar_tags:
            if 'test_accuracy' in tag and tag.endswith('/'):
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    # Return all (step, value) pairs
                    return [(s.step, s.value * 100) for s in scalar_events]
        
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# REDO Part 2 experiments
redo_experiments = [
    'REDO_insqrt',
    'REDO_balnormal', 
    'REDO_smooth',
    'REDO_log',
    'REDO_uniform'
]

print("=" * 70)
print("Checking Consistency of REDO Part 2 Experiments")
print("=" * 70)

results = {}

for exp_name in redo_experiments:
    summary_dir = f"output/{exp_name}/SwinGait/{exp_name}/summary"
    history = get_evaluation_history(summary_dir)
    
    if history:
        steps, accuracies = zip(*history)
        final_acc = accuracies[-1]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        avg_acc = sum(accuracies) / len(accuracies)
        
        # Count how many times it hit 73.3% or higher
        count_73_plus = sum(1 for acc in accuracies if acc >= 73.0)
        
        # Count how many evaluations
        num_evals = len(accuracies)
        
        results[exp_name] = {
            'final': final_acc,
            'max': max_acc,
            'min': min_acc,
            'avg': avg_acc,
            'count_73_plus': count_73_plus,
            'num_evals': num_evals,
            'consistency': count_73_plus / num_evals if num_evals > 0 else 0,
            'history': accuracies
        }
        
        print(f"\n{exp_name}:")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Max Accuracy: {max_acc:.2f}%")
        print(f"  Min Accuracy: {min_acc:.2f}%")
        print(f"  Average Accuracy: {avg_acc:.2f}%")
        print(f"  Evaluations: {num_evals}")
        print(f"  Times ≥73.0%: {count_73_plus} ({count_73_plus/num_evals*100:.1f}%)")
        print(f"  Accuracy History: {[f'{a:.1f}' for a in accuracies]}")

print("\n" + "=" * 70)
print("Summary - Best Consistency:")
print("=" * 70)

# Sort by consistency (times hitting 73%+)
sorted_results = sorted(results.items(), key=lambda x: (x[1]['count_73_plus'], x[1]['final']), reverse=True)

for exp_name, data in sorted_results:
    print(f"{exp_name}:")
    print(f"  - Final: {data['final']:.2f}%")
    print(f"  - Hit 73%+ in {data['count_73_plus']}/{data['num_evals']} evaluations ({data['consistency']*100:.1f}%)")
    print(f"  - Average: {data['avg']:.2f}%")

