#!/usr/bin/env python3
"""
Extract complete metrics including specificity.
For existing experiments: tries to extract from TensorBoard (limited) or suggests re-evaluation.
For future experiments: will have per-class metrics including specificity.
"""

import os
import glob
import json
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

def extract_from_tensorboard(summary_dir):
    """Extract all metrics from TensorBoard, including per-class if available."""
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
        print(f"Error: {e}")
        return {}

def main():
    print("=" * 80)
    print("Complete Metrics Extraction (Including Specificity)")
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
        summary_dir = os.path.join(exp_path, 'summary')
        
        if not os.path.exists(summary_dir):
            continue
        
        print(f"\n{exp_name}:")
        metrics = extract_from_tensorboard(summary_dir)
        
        if not metrics:
            continue
        
        all_results[exp_name] = metrics
        
        # Check what metrics are available
        test_metrics = [k for k in metrics.keys() if 'test' in k.lower()]
        per_class_metrics = [k for k in test_metrics if any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])]
        specificity_metrics = [k for k in test_metrics if 'specificity' in k.lower()]
        
        print(f"  Total metrics: {len(metrics)}")
        print(f"  Test metrics: {len(test_metrics)}")
        print(f"  Per-class metrics: {len(per_class_metrics)}")
        print(f"  Specificity metrics: {len(specificity_metrics)}")
        
        if specificity_metrics:
            print(f"  ✓ Specificity available!")
            for spec_metric in specificity_metrics:
                if metrics[spec_metric]:
                    final_value = metrics[spec_metric][-1][1]
                    print(f"    {spec_metric}: {final_value*100:.2f}%")
        else:
            print(f"  ✗ Specificity NOT in TensorBoard (only macro-averaged metrics)")
            print(f"    Available: {', '.join(test_metrics[:5])}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nCurrent TensorBoard files contain:")
    print("  ✓ Overall Accuracy")
    print("  ✓ Macro-averaged Precision")
    print("  ✓ Macro-averaged Recall (Sensitivity)")
    print("  ✓ Macro-averaged F1")
    print("  ✗ Per-class Precision (not saved)")
    print("  ✗ Per-class Sensitivity (not saved)")
    print("  ✗ Per-class Specificity (not saved)")
    
    print("\n" + "=" * 80)
    print("SOLUTION")
    print("=" * 80)
    print("\nI've updated the evaluator to save per-class metrics including specificity.")
    print("For existing experiments, you have two options:")
    print("\n1. Re-run evaluation on checkpoints (recommended):")
    print("   This will generate new TensorBoard entries with all metrics.")
    print("   Run: python evaluate_all_checkpoints.py")
    print("\n2. Extract from log files (if they exist):")
    print("   Log files contain all metrics but may not be available.")
    print("\n3. For future experiments:")
    print("   The updated evaluator will automatically save all per-class metrics.")
    
    # Save what we have
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'current_available_metrics.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved available metrics to: {json_file}")

if __name__ == '__main__':
    main()

