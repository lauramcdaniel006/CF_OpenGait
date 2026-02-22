#!/usr/bin/env python3
"""
Extract per-class metrics (Frail, Prefrail, Nonfrail) for each iteration from TensorBoard.
Adds them to the existing CSV or creates a new one with per-class breakdown.
"""

import csv
import os
import glob
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

def extract_from_tensorboard(summary_dir):
    """Extract all metrics including per-class from TensorBoard."""
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
        return {}

def find_all_redo_experiments():
    """Find all REDO experiments with summary directories."""
    experiments = {}
    
    for root, dirs, files in os.walk('output'):
        if 'summary' in dirs:
            path_parts = Path(root).parts
            if len(path_parts) >= 2:
                save_name = path_parts[-1]
                
                if 'REDO' in save_name:
                    summary_dir = os.path.join(root, 'summary')
                    if glob.glob(os.path.join(summary_dir, "events.out.tfevents.*")):
                        experiments[save_name] = {
                            'full_path': root,
                            'save_name': save_name,
                            'summary_dir': summary_dir
                        }
    
    return experiments

def categorize_experiment(exp_name):
    """Categorize experiment into Part 1-4."""
    exp_lower = exp_name.lower()
    
    if 'pt1' in exp_lower or 'part1' in exp_lower:
        return 'Part 1: Freezing Strategy'
    elif 'pt2' in exp_lower or 'part2' in exp_lower:
        return 'Part 2: Class Weights'
    elif 'pt3' in exp_lower or 'part3' in exp_lower:
        return 'Part 3: Loss Functions'
    elif 'pt4' in exp_lower or 'part4' in exp_lower:
        return 'Part 4: Architecture Comparison'
    elif any(x in exp_lower for x in ['insqrt', 'balnormal', 'log', 'smooth', 'uniform']):
        return 'Part 2: Class Weights'
    else:
        return 'Other'

def main():
    print("=" * 120)
    print("EXTRACTING PER-CLASS METRICS FROM TENSORBOARD")
    print("=" * 120)
    
    if not TB_AVAILABLE:
        print("\n❌ TensorBoard library not available.")
        print("   Run: conda activate myGait38")
        print("   Then: pip install tensorboard")
        return
    
    experiments = find_all_redo_experiments()
    
    if not experiments:
        print("\n❌ No REDO experiments found with TensorBoard data!")
        return
    
    print(f"\nFound {len(experiments)} REDO experiment(s) with TensorBoard data\n")
    
    # Extract per-class metrics
    all_data = []
    classes = ['Frail', 'Prefrail', 'Nonfrail']
    metric_types = ['precision', 'recall', 'specificity', 'f1']
    
    for exp_name, exp_info in experiments.items():
        part = categorize_experiment(exp_name)
        metrics = extract_from_tensorboard(exp_info['summary_dir'])
        
        if not metrics:
            print(f"⚠️  No metrics found for {exp_name}")
            continue
        
        # Check if per-class metrics exist
        per_class_found = False
        for class_name in classes:
            for mt in metric_types:
                key = f"scalar/test_{mt}/{class_name}"
                if key in metrics:
                    per_class_found = True
                    break
            if per_class_found:
                break
        
        if not per_class_found:
            print(f"⚠️  No per-class metrics found for {exp_name}")
            print(f"    Available metrics: {[k for k in metrics.keys() if 'test' in k][:5]}...")
            continue
        
        print(f"✓ Extracting per-class metrics for {exp_name}...")
        
        # Get all iterations from any metric
        all_iterations = set()
        for key in metrics.keys():
            if 'test_accuracy' in key.lower() and not any(c in key for c in classes):
                all_iterations.update(step for step, _ in metrics[key])
        
        # Extract per-class metrics for each iteration
        for iteration in sorted(all_iterations):
            row = {
                'Part': part,
                'Experiment': exp_name,
                'Iteration': iteration
            }
            
            # Extract per-class metrics
            for class_name in classes:
                for mt in metric_types:
                    key = f"scalar/test_{mt}/{class_name}"
                    if key in metrics:
                        # Find value for this iteration
                        value = None
                        for step, val in metrics[key]:
                            if step == iteration:
                                value = val
                                break
                        row[f"{class_name}_{mt.capitalize()}"] = value if value is not None else ''
                    else:
                        row[f"{class_name}_{mt.capitalize()}"] = ''
            
            all_data.append(row)
    
    if not all_data:
        print("\n❌ No per-class metrics extracted!")
        print("   This might mean:")
        print("   1. Per-class metrics weren't saved to TensorBoard (need to re-run evaluation)")
        print("   2. The experiments were run before per-class metrics were added to evaluator")
        return
    
    # Save to CSV
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, 'per_class_metrics_all_iterations.csv')
    
    # Get all column names
    fieldnames = ['Part', 'Experiment', 'Iteration']
    for class_name in classes:
        for mt in metric_types:
            fieldnames.append(f"{class_name}_{mt.capitalize()}")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"\n{'='*120}")
    print(f"✓ Saved per-class metrics to: {csv_file}")
    print(f"  Total rows: {len(all_data)}")
    print(f"  Columns: {len(fieldnames)}")
    print(f"{'='*120}")
    
    # Show sample
    print("\nSample data (first 3 rows):")
    print("-" * 120)
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"  {row['Part']} | {row['Experiment']} | Iter {row['Iteration']} | "
                  f"Frail_Precision: {row.get('Frail_Precision', 'N/A')[:6] if row.get('Frail_Precision') else 'N/A'}")
    
    print(f"\n{'='*120}")
    print("Per-class metrics extracted!")
    print("=" * 120)

if __name__ == '__main__':
    main()

