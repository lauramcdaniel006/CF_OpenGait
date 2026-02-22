#!/usr/bin/env python3
"""
Extract all REDO experiments from Parts 1, 2, 3, and 4.
"""

import os
import glob
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

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
        return {}

def find_all_redo_experiments():
    """Find all REDO experiments across all parts."""
    experiments = {}
    
    # Search for all directories with REDO in the name
    for root, dirs, files in os.walk('output'):
        if 'summary' in dirs:
            path_parts = Path(root).parts
            if len(path_parts) >= 4:
                dataset = path_parts[-3]
                model = path_parts[-2]
                save_name = path_parts[-1]
                
                # Only include REDO experiments
                if 'REDO' in save_name or 'REDO' in dataset:
                    exp_key = f"{dataset}/{model}/{save_name}"
                    summary_dir = os.path.join(root, 'summary')
                    
                    experiments[exp_key] = {
                        'full_path': root,
                        'dataset': dataset,
                        'model': model,
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
    print("=" * 100)
    print("EXTRACTING ALL REDO EXPERIMENTS FROM PARTS 1-4")
    print("=" * 100)
    
    if not TB_AVAILABLE:
        print("\n❌ TensorBoard library not available.")
        print("   Run: conda activate myGait38")
        return
    
    experiments = find_all_redo_experiments()
    
    if not experiments:
        print("\n❌ No REDO experiments found!")
        return
    
    print(f"\n✓ Found {len(experiments)} REDO experiment(s)\n")
    
    # Group by part
    by_part = {}
    for exp_key, exp_info in experiments.items():
        part = categorize_experiment(exp_info['save_name'])
        if part not in by_part:
            by_part[part] = []
        by_part[part].append((exp_key, exp_info))
    
    # Extract metrics for each experiment
    all_results = {}
    
    for part_name in sorted(by_part.keys()):
        print(f"\n{'='*100}")
        print(f"{part_name}")
        print(f"{'='*100}")
        
        for exp_key, exp_info in sorted(by_part[part_name]):
            exp_name = exp_info['save_name']
            print(f"\nProcessing: {exp_name}")
            
            metrics = extract_from_tensorboard(exp_info['summary_dir'])
            
            if not metrics:
                print(f"  ✗ No metrics found")
                continue
            
            # Get test accuracy
            test_acc_metrics = {k: v for k, v in metrics.items() if 'test_accuracy' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
            
            if test_acc_metrics:
                for metric_name, values in test_acc_metrics.items():
                    if values:
                        final_value = values[-1][1]
                        num_evals = len(values)
                        print(f"  ✓ Found {num_evals} evaluations")
                        print(f"    Final accuracy: {final_value*100:.2f}%")
                        
                        # Get max accuracy
                        max_value = max(v for _, v in values)
                        max_iter = max(step for step, v in values if v == max_value)
                        print(f"    Max accuracy: {max_value*100:.2f}% at iteration {max_iter}")
                        
                        all_results[exp_name] = {
                            'part': part_name,
                            'final_acc': final_value*100,
                            'max_acc': max_value*100,
                            'max_iter': max_iter,
                            'num_evals': num_evals,
                            'metrics': metrics
                        }
                        break
            else:
                print(f"  ✗ No test accuracy metrics found")
    
    # Summary by part
    print("\n" + "=" * 100)
    print("SUMMARY BY PART")
    print("=" * 100)
    
    for part_name in sorted(by_part.keys()):
        part_exps = {k: v for k, v in all_results.items() if v['part'] == part_name}
        if not part_exps:
            continue
        
        print(f"\n{part_name}:")
        print(f"  {'Experiment':<50} {'Final Acc':<12} {'Max Acc':<12} {'Max Iter':<10} {'Evals':<8}")
        print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
        
        for exp_name in sorted(part_exps.keys()):
            exp_data = part_exps[exp_name]
            print(f"  {exp_name:<50} {exp_data['final_acc']:>10.2f}%  {exp_data['max_acc']:>10.2f}%  {exp_data['max_iter']:>8}  {exp_data['num_evals']:>6}")
    
    # Save to JSON
    import json
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save simplified results
    simplified_results = {}
    for exp_name, exp_data in all_results.items():
        simplified_results[exp_name] = {
            'part': exp_data['part'],
            'final_acc': exp_data['final_acc'],
            'max_acc': exp_data['max_acc'],
            'max_iter': exp_data['max_iter'],
            'num_evals': exp_data['num_evals']
        }
    
    json_file = os.path.join(output_dir, 'all_parts_redo_summary.json')
    with open(json_file, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"✓ Saved summary to: {json_file}")
    print(f"{'='*100}")

if __name__ == '__main__':
    main()

