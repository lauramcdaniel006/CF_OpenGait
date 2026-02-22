#!/usr/bin/env python3
"""
View all iterations for all REDO experiments from Parts 1-4.
"""

import os
import glob
import pandas as pd
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
    """Find all REDO experiments."""
    experiments = {}
    
    for root, dirs, files in os.walk('output'):
        if 'summary' in dirs:
            path_parts = Path(root).parts
            if len(path_parts) >= 4:
                dataset = path_parts[-3]
                model = path_parts[-2]
                save_name = path_parts[-1]
                
                if 'REDO' in save_name or 'REDO' in dataset:
                    exp_key = f"{dataset}/{model}/{save_name}"
                    summary_dir = os.path.join(root, 'summary')
                    
                    experiments[exp_key] = {
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
    print("ALL REDO EXPERIMENTS - PARTS 1-4 - COMPLETE ITERATION HISTORY")
    print("=" * 120)
    
    if not TB_AVAILABLE:
        print("\n❌ TensorBoard library not available.")
        print("   Run: conda activate myGait38")
        return
    
    experiments = find_all_redo_experiments()
    
    if not experiments:
        print("\n❌ No REDO experiments found!")
        return
    
    print(f"\nFound {len(experiments)} REDO experiment(s)\n")
    
    # Group by part and extract all data
    all_data = []
    
    for exp_key, exp_info in experiments.items():
        exp_name = exp_info['save_name']
        part = categorize_experiment(exp_name)
        
        metrics = extract_from_tensorboard(exp_info['summary_dir'])
        
        if not metrics:
            continue
        
        # Get test accuracy
        test_acc_metrics = {k: v for k, v in metrics.items() if 'test_accuracy' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
        
        if test_acc_metrics:
            for metric_name, values in test_acc_metrics.items():
                for step, value in values:
                    all_data.append({
                        'Part': part,
                        'Experiment': exp_name,
                        'Iteration': step,
                        'Accuracy': value * 100,
                        'F1': None,
                        'Precision': None,
                        'Recall': None
                    })
                
                # Get other metrics if available
                f1_metrics = {k: v for k, v in metrics.items() if 'test_f1' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
                prec_metrics = {k: v for k, v in metrics.items() if 'test_precision' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
                recall_metrics = {k: v for k, v in metrics.items() if 'test_recall' in k.lower() and not any(c in k for c in ['Frail', 'Prefrail', 'Nonfrail'])}
                
                # Match by iteration
                for i, row in enumerate(all_data):
                    if row['Experiment'] == exp_name:
                        iter_num = row['Iteration']
                        
                        if f1_metrics:
                            for f1_values in f1_metrics.values():
                                for step, val in f1_values:
                                    if step == iter_num:
                                        row['F1'] = val
                                        break
                        
                        if prec_metrics:
                            for prec_values in prec_metrics.values():
                                for step, val in prec_values:
                                    if step == iter_num:
                                        row['Precision'] = val
                                        break
                        
                        if recall_metrics:
                            for recall_values in recall_metrics.values():
                                for step, val in recall_values:
                                    if step == iter_num:
                                        row['Recall'] = val
                                        break
                break
    
    if not all_data:
        print("❌ No data found!")
        return
    
    df = pd.DataFrame(all_data)
    
    # Group by part
    by_part = df.groupby('Part')
    
    for part_name, part_df in sorted(by_part):
        print(f"\n{'='*120}")
        print(f"{part_name}")
        print(f"{'='*120}")
        
        experiments_in_part = sorted(part_df['Experiment'].unique())
        
        for exp_name in experiments_in_part:
            exp_data = part_df[part_df['Experiment'] == exp_name].sort_values('Iteration').copy()
            
            print(f"\n{exp_name}:")
            print(f"{'─'*120}")
            
            # Display table
            display_df = exp_data[['Iteration', 'Accuracy', 'F1', 'Precision', 'Recall']].copy()
            display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            print(f"\n{display_df.to_string(index=False)}")
            
            # Statistics
            acc_values = exp_data['Accuracy'].values
            print(f"\n  Statistics:")
            print(f"    Total Evaluations: {len(exp_data)}")
            print(f"    Max Accuracy: {acc_values.max():.2f}% at iteration {exp_data.loc[exp_data['Accuracy'].idxmax(), 'Iteration']}")
            print(f"    Final Accuracy: {acc_values[-1]:.2f}%")
            print(f"    Mean Accuracy: {acc_values.mean():.2f}%")
            print(f"    Times ≥73%: {(acc_values >= 73.0).sum()} / {len(exp_data)}")
            print(f"    Times ≥66.67%: {(acc_values >= 66.67).sum()} / {len(exp_data)}")
    
    # Save to CSV
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, 'all_parts_redo_all_iterations.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*120}")
    print(f"✓ Saved all data to: {csv_file}")
    print(f"{'='*120}")

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available. Install with: pip install pandas")
        exit(1)
    
    main()

