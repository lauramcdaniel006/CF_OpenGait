#!/usr/bin/env python3
"""
Extract metrics from every checkpoint evaluation using TensorBoard event files.
Works without TensorBoard installed by using tensorboard's event_accumulator directly.
"""

import os
import sys
import glob
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("Warning: tensorboard not available. Trying alternative method...")

def extract_from_tensorboard_events(summary_dir):
    """Extract all scalar metrics from TensorBoard event files."""
    if not TB_AVAILABLE:
        return {}
    
    try:
        # Find event files
        event_files = glob.glob(os.path.join(summary_dir, "events.out.tfevents.*"))
        if not event_files:
            return {}
        
        # Use the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        
        # Create EventAccumulator
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        # Get all scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        for tag in scalar_tags:
            # Get all scalar events for this tag
            scalar_events = ea.Scalars(tag)
            
            # Store as list of (step, value) tuples
            metrics[tag] = [(event.step, event.value) for event in scalar_events]
        
        return metrics
    except Exception as e:
        print(f"Error reading TensorBoard events: {e}")
        return {}

def extract_metrics_simple(summary_dir):
    """Alternative: Try to read event files using protobuf directly."""
    import struct
    
    event_files = glob.glob(os.path.join(summary_dir, "events.out.tfevents.*"))
    if not event_files:
        return {}
    
    metrics = {}
    
    for event_file in sorted(event_files):
        try:
            with open(event_file, 'rb') as f:
                # This is a simplified parser - TensorBoard format is complex
                # For now, we'll rely on the EventAccumulator method
                pass
        except:
            pass
    
    return metrics

def find_experiments():
    """Find all REDO Part 2 experiments."""
    experiments = {}
    
    redo_experiments = [
        'REDO_insqrt',
        'REDO_balnormal',
        'REDO_smooth',
        'REDO_log',
        'REDO_uniform'
    ]
    
    for exp_name in redo_experiments:
        exp_path = f"output/{exp_name}/SwinGait/{exp_name}"
        summary_dir = os.path.join(exp_path, 'summary')
        
        if os.path.exists(summary_dir):
            experiments[exp_name] = {
                'path': exp_path,
                'summary_dir': summary_dir
            }
    
    return experiments

def format_metrics_table(metrics_dict):
    """Format metrics into a readable table."""
    if not metrics_dict:
        return "No metrics found"
    
    # Focus on test accuracy metrics
    test_metrics = {k: v for k, v in metrics_dict.items() if 'test' in k.lower()}
    
    if not test_metrics:
        return "No test metrics found"
    
    lines = []
    lines.append("=" * 80)
    
    # Get all unique steps
    all_steps = set()
    for values in test_metrics.values():
        all_steps.update([step for step, _ in values])
    all_steps = sorted(all_steps)
    
    # Separate macro-averaged and per-class metrics
    macro_metrics = {k: v for k, v in test_metrics.items() if not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
    per_class_metrics = {k: v for k, v in test_metrics.items() if any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
    
    # Header - show macro metrics first
    header = f"{'Iteration':<12}"
    for metric_name in sorted(macro_metrics.keys()):
        short_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
        if short_name == '':
            short_name = 'accuracy'
        header += f"{short_name:<15}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Data rows for macro metrics
    for step in all_steps:
        row = f"{step:<12}"
        for metric_name in sorted(macro_metrics.keys()):
            values = macro_metrics[metric_name]
            # Find value for this step
            value = None
            for s, v in values:
                if s == step:
                    value = v
                    break
            
            if value is not None:
                if 'accuracy' in metric_name.lower():
                    row += f"{value*100:>13.2f}%"
                else:
                    row += f"{value:>13.4f}"
            else:
                row += f"{'N/A':>13}"
        lines.append(row)
    
    # Show per-class metrics if available
    if per_class_metrics:
        lines.append("")
        lines.append("PER-CLASS METRICS (if available):")
        lines.append("-" * 80)
        
        # Group by class
        classes = ['Frail', 'Prefrail', 'Nonfrail']
        metric_types = ['precision', 'recall', 'specificity', 'f1']
        
        for class_name in classes:
            lines.append(f"\n{class_name}:")
            header = f"{'Iteration':<12}"
            for mt in metric_types:
                header += f"{mt.capitalize():<15}"
            lines.append(header)
            lines.append("-" * 80)
            
            for step in all_steps:
                row = f"{step:<12}"
                for mt in metric_types:
                    metric_key = f"scalar/test_{mt}/{class_name}"
                    if metric_key in per_class_metrics:
                        values = per_class_metrics[metric_key]
                        value = None
                        for s, v in values:
                            if s == step:
                                value = v
                                break
                        if value is not None:
                            row += f"{value*100:>13.2f}%"
                        else:
                            row += f"{'N/A':>13}"
                    else:
                        row += f"{'N/A':>13}"
                lines.append(row)
    
    lines.append("=" * 80)
    return "\n".join(lines)

def main():
    print("=" * 80)
    print("Extract All Checkpoint Evaluation Metrics")
    print("=" * 80)
    
    if not TB_AVAILABLE:
        print("\n❌ TensorBoard library not available.")
        print("   Installing tensorboard...")
        print("   Run: pip install tensorboard")
        print("\n   Or use the alternative method: re-run evaluation on checkpoints")
        return
    
    experiments = find_experiments()
    
    if not experiments:
        print("\n❌ No experiments found!")
        return
    
    print(f"\n✓ Found {len(experiments)} experiments\n")
    
    all_results = {}
    
    for exp_name, exp_info in experiments.items():
        print(f"Processing {exp_name}...")
        summary_dir = exp_info['summary_dir']
        
        metrics = extract_from_tensorboard_events(summary_dir)
        
        if metrics:
            all_results[exp_name] = metrics
            print(f"  ✓ Found {len(metrics)} metric types")
            
        # Show summary
        test_acc_metrics = {k: v for k, v in metrics.items() if 'test_accuracy' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
        specificity_metrics = {k: v for k, v in metrics.items() if 'specificity' in k.lower()}
        
        if test_acc_metrics:
            for metric_name, values in test_acc_metrics.items():
                if values:
                    final_value = values[-1][1]
                    if 'accuracy' in metric_name.lower():
                        print(f"    Final {metric_name}: {final_value*100:.2f}%")
        
        if specificity_metrics:
            print(f"    ✓ Specificity metrics found: {len(specificity_metrics)}")
            for spec_metric, values in list(specificity_metrics.items())[:3]:  # Show first 3
                if values:
                    final_value = values[-1][1]
                    class_name = spec_metric.split('/')[-1]
                    print(f"      {class_name}: {final_value*100:.2f}%")
        else:
            print(f"    ⚠ Specificity not in TensorBoard (need to re-run evaluation)")
        
        if not metrics:
            print(f"  ✗ No metrics found in {summary_dir}")
    
    # Save to JSON
    import json
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full data
    json_file = os.path.join(output_dir, 'all_checkpoint_metrics.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved full metrics to: {json_file}")
    
    # Create CSV with all evaluations
    import csv
    csv_file = os.path.join(output_dir, 'all_checkpoint_metrics.csv')
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Experiment', 'Iteration']
        
        # Get all unique metric names
        all_metric_names = set()
        for exp_metrics in all_results.values():
            all_metric_names.update(exp_metrics.keys())
        
        # Focus on test metrics - include both macro and per-class
        test_metric_names = sorted([m for m in all_metric_names if 'test' in m.lower()])
        header.extend(test_metric_names)
        
        writer.writerow(header)
        
        # Data rows
        for exp_name, metrics in all_results.items():
            # Get all unique steps
            all_steps = set()
            for values in metrics.values():
                all_steps.update([step for step, _ in values])
            all_steps = sorted(all_steps)
            
            for step in all_steps:
                row = [exp_name, step]
                for metric_name in test_metric_names:
                    if metric_name in metrics:
                        # Find value for this step
                        value = None
                        for s, v in metrics[metric_name]:
                            if s == step:
                                value = v
                                break
                        
                        if value is not None:
                            # Convert accuracy to percentage
                            if 'accuracy' in metric_name.lower():
                                row.append(f"{value*100:.2f}")
                            else:
                                row.append(f"{value:.4f}")
                        else:
                            row.append("")
                    else:
                        row.append("")
                
                writer.writerow(row)
    
    print(f"✓ Saved CSV to: {csv_file}")
    
    # Print summary tables
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)
    
    for exp_name, metrics in all_results.items():
        print(f"\n{exp_name}:")
        print(format_metrics_table(metrics))
    
    print("\n" + "=" * 80)
    print("Done! Check the JSON and CSV files for all metrics.")
    print("=" * 80)

if __name__ == '__main__':
    main()

