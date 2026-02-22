#!/usr/bin/env python3
"""
Simple script to extract key metrics from TensorBoard files into a clean CSV
for manual visualization in Excel/Google Sheets/etc.
"""

import os
import glob
import json
from pathlib import Path
import ast

# Try to import tensorboard, but make it optional
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

def parse_metric_value(metric):
    """Parse metric value from dict, string, or direct value."""
    if metric is None:
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    if isinstance(metric, dict):
        return metric.get('value', None)
    if isinstance(metric, str):
        try:
            parsed = ast.literal_eval(metric)
            if isinstance(parsed, dict):
                return parsed.get('value', None)
        except:
            pass
    return None

def extract_from_tensorboard(summary_dir):
    """Extract test accuracy from TensorBoard."""
    if not TENSORBOARD_AVAILABLE or not os.path.isdir(summary_dir):
        return None
    
    try:
        event_files = glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))
        if not event_files:
            return None
        
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        # Look for test accuracy
        for tag in scalar_tags:
            if 'test_accuracy' in tag and tag.endswith('/'):
                scalar_events = ea.Scalars(tag)
                if scalar_events:
                    # Get the last (most recent) value
                    return scalar_events[-1].value
        
        return None
    except Exception as e:
        return None

def find_experiments(output_dir='output'):
    """Find all experiment directories."""
    experiments = {}
    
    for root, dirs, files in os.walk(output_dir):
        if 'summary' in dirs:
            path_parts = Path(root).parts
            if len(path_parts) >= 4:
                dataset = path_parts[-3]
                model = path_parts[-2]
                save_name = path_parts[-1]
                
                exp_key = f"{dataset}/{model}/{save_name}"
                summary_dir = os.path.join(root, 'summary')
                
                experiments[exp_key] = {
                    'summary_dir': summary_dir,
                    'dataset': dataset,
                    'model': model,
                    'save_name': save_name
                }
    
    return experiments

def categorize_experiment(exp_name):
    """Categorize experiment into Part 1-4."""
    if 'pt1' in exp_name.lower() or 'part1' in exp_name.lower():
        return 'Part 1: Freezing Strategy'
    elif 'pt2' in exp_name.lower() or 'part2' in exp_name.lower():
        return 'Part 2: Class Weights'
    elif 'pt3' in exp_name.lower() or 'part3' in exp_name.lower():
        return 'Part 3: Loss Functions'
    elif 'pt4' in exp_name.lower() or 'part4' in exp_name.lower():
        return 'Part 4: Architecture Comparison'
    return 'Other'

def main():
    print("=" * 70)
    print("Simple Metrics Extraction for Manual Visualization")
    print("=" * 70)
    
    if not TENSORBOARD_AVAILABLE:
        print("\nERROR: TensorBoard is required for this script.")
        print("Install with: pip install tensorboard")
        return
    
    print("\n1. Finding experiments...")
    experiments = find_experiments('output')
    print(f"   Found {len(experiments)} experiments")
    
    print("\n2. Extracting test accuracy...")
    results = []
    
    for exp_name, info in experiments.items():
        accuracy = extract_from_tensorboard(info['summary_dir'])
        if accuracy is not None:
            part = categorize_experiment(exp_name)
            results.append({
                'Part': part,
                'Experiment': info['save_name'],
                'Model': info['model'],
                'Dataset': info['dataset'],
                'Test Accuracy (%)': accuracy * 100
            })
            print(f"   ✓ {info['save_name']}: {accuracy*100:.2f}%")
        else:
            print(f"   ✗ {info['save_name']}: No metrics found")
    
    if not results:
        print("\nNo results found. Make sure TensorBoard files exist in output/ directories.")
        return
    
    # Write CSV
    output_file = 'results_visualization/simple_metrics.csv'
    os.makedirs('results_visualization', exist_ok=True)
    
    import csv
    with open(output_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n3. Saved results to: {output_file}")
    print(f"   Total experiments with metrics: {len(results)}")
    
    # Print summary by part
    print("\n4. Summary by Part:")
    from collections import defaultdict
    by_part = defaultdict(list)
    for r in results:
        by_part[r['Part']].append(r['Test Accuracy (%)'])
    
    for part, accuracies in sorted(by_part.items()):
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            max_acc = max(accuracies)
            print(f"   {part}:")
            print(f"     - Experiments: {len(accuracies)}")
            print(f"     - Average Accuracy: {avg_acc:.2f}%")
            print(f"     - Best Accuracy: {max_acc:.2f}%")
    
    print("\n" + "=" * 70)
    print("Done! Open 'results_visualization/simple_metrics.csv' in Excel/Sheets")
    print("=" * 70)

if __name__ == '__main__':
    main()

