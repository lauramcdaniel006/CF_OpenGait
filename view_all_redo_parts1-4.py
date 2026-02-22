#!/usr/bin/env python3
"""
View all metric values for each iteration of all REDO experiments from Parts 1-4.
Reads from CSV file if available, otherwise tries TensorBoard.
"""

import os
import glob
import csv
from pathlib import Path
from collections import defaultdict

# Try to import TensorBoard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

def extract_from_tensorboard(summary_dir):
    """Extract all metrics from TensorBoard event files."""
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

def read_from_csv(csv_file):
    """Read metrics from CSV file."""
    experiments = defaultdict(list)
    
    if not os.path.exists(csv_file):
        return experiments
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row['Experiment']
            if 'REDO' in exp_name:
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['test_accuracy/']),
                    'f1': float(row.get('test_f1/', 0)) if row.get('test_f1/') else None,
                    'precision': float(row.get('test_precision/', 0)) if row.get('test_precision/') else None,
                    'recall': float(row.get('test_recall/', 0)) if row.get('test_recall/') else None
                })
    
    return experiments

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
    print("ALL REDO EXPERIMENTS - PARTS 1-4 - METRIC VALUES BY ITERATION")
    print("=" * 120)
    
    # Try to read from CSV first
    csv_file = 'results_visualization/checkpoint_evaluations_only.csv'
    csv_data = read_from_csv(csv_file)
    
    # Try to get additional data from TensorBoard
    tb_data = defaultdict(list)
    
    if TB_AVAILABLE:
        print("\n✓ TensorBoard available - extracting from event files...")
        experiments = find_all_redo_experiments()
        
        for exp_name, exp_info in experiments.items():
            part = categorize_experiment(exp_name)
            metrics = extract_from_tensorboard(exp_info['summary_dir'])
            
            if not metrics:
                continue
            
            # Get test accuracy
            test_acc_key = None
            for key in metrics.keys():
                if 'test_accuracy' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
                    test_acc_key = key
                    break
            
            if not test_acc_key:
                continue
            
            # Get other metrics
            f1_key = None
            prec_key = None
            recall_key = None
            
            for key in metrics.keys():
                if 'test_f1' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
                    f1_key = key
                if 'test_precision' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
                    prec_key = key
                if 'test_recall' in key.lower() and not any(c in key for c in ['Frail', 'Prefrail', 'Nonfrail']):
                    recall_key = key
            
            # Extract all iterations
            for step, acc_value in metrics[test_acc_key]:
                row = {
                    'part': part,
                    'experiment': exp_name,
                    'iteration': step,
                    'accuracy': acc_value * 100,
                    'f1': None,
                    'precision': None,
                    'recall': None
                }
                
                if f1_key:
                    for f1_step, f1_val in metrics[f1_key]:
                        if f1_step == step:
                            row['f1'] = f1_val
                            break
                
                if prec_key:
                    for prec_step, prec_val in metrics[prec_key]:
                        if prec_step == step:
                            row['precision'] = prec_val
                            break
                
                if recall_key:
                    for recall_step, recall_val in metrics[recall_key]:
                        if recall_step == step:
                            row['recall'] = recall_val
                            break
                
                tb_data[part].append(row)
    else:
        print("\n⚠️  TensorBoard not available - using CSV data only")
        print("   To get all experiments, run: conda activate myGait38")
    
    # Merge CSV and TensorBoard data (TB takes precedence if both exist)
    all_data = defaultdict(lambda: defaultdict(list))
    
    # Add CSV data
    for exp_name, rows in csv_data.items():
        part = categorize_experiment(exp_name)
        for row in rows:
            all_data[part][exp_name].append({
                'iteration': row['iteration'],
                'accuracy': row['accuracy'],
                'f1': row['f1'],
                'precision': row['precision'],
                'recall': row['recall']
            })
    
    # Add/override with TensorBoard data
    for part, rows in tb_data.items():
        for row in rows:
            all_data[part][row['experiment']].append({
                'iteration': row['iteration'],
                'accuracy': row['accuracy'],
                'f1': row['f1'],
                'precision': row['precision'],
                'recall': row['recall']
            })
    
    if not all_data:
        print("\n❌ No data found!")
        return
    
    # Display by part
    part_order = [
        'Part 1: Freezing Strategy',
        'Part 2: Class Weights',
        'Part 3: Loss Functions',
        'Part 4: Architecture Comparison',
        'Other'
    ]
    
    for part_name in part_order:
        if part_name not in all_data:
            continue
        
        part_experiments = all_data[part_name]
        
        print(f"\n{'='*120}")
        print(f"{part_name}")
        print(f"{'='*120}")
        
        for exp_name in sorted(part_experiments.keys()):
            exp_data = sorted(part_experiments[exp_name], key=lambda x: x['iteration'])
            
            print(f"\n{exp_name}:")
            print(f"{'─'*120}")
            print(f"{'Iteration':<12} {'Accuracy (%)':<15} {'F1':<12} {'Precision':<12} {'Recall':<12}")
            print(f"{'-'*120}")
            
            for row in exp_data:
                f1_str = f"{row['f1']:.4f}" if row['f1'] is not None else "N/A"
                prec_str = f"{row['precision']:.4f}" if row['precision'] is not None else "N/A"
                recall_str = f"{row['recall']:.4f}" if row['recall'] is not None else "N/A"
                
                print(f"{row['iteration']:<12} {row['accuracy']:<15.2f} {f1_str:<12} {prec_str:<12} {recall_str:<12}")
            
            # Statistics
            accuracies = [r['accuracy'] for r in exp_data]
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            mean_acc = sum(accuracies) / len(accuracies)
            final_acc = accuracies[-1]
            max_iter = max(exp_data, key=lambda x: x['accuracy'])['iteration']
            
            times_73 = sum(1 for a in accuracies if a >= 73.0)
            times_66 = sum(1 for a in accuracies if a >= 66.67)
            
            print(f"\n  Statistics:")
            print(f"    Total Evaluations: {len(exp_data)}")
            print(f"    Iteration Range: {exp_data[0]['iteration']} - {exp_data[-1]['iteration']}")
            print(f"    Max Accuracy: {max_acc:.2f}% at iteration {max_iter}")
            print(f"    Min Accuracy: {min_acc:.2f}%")
            print(f"    Mean Accuracy: {mean_acc:.2f}%")
            print(f"    Final Accuracy: {final_acc:.2f}%")
            print(f"    Times ≥73%: {times_73} / {len(exp_data)} ({times_73/len(exp_data)*100:.1f}%)")
            print(f"    Times ≥66.67%: {times_66} / {len(exp_data)} ({times_66/len(exp_data)*100:.1f}%)")
            
            # Show high accuracy iterations
            high_acc = [r for r in exp_data if r['accuracy'] >= 73.0]
            if high_acc:
                print(f"\n  Iterations with ≥73% accuracy:")
                for r in high_acc:
                    f1_str = f"{r['f1']:.4f}" if r['f1'] is not None else "N/A"
                    prec_str = f"{r['precision']:.4f}" if r['precision'] is not None else "N/A"
                    recall_str = f"{r['recall']:.4f}" if r['recall'] is not None else "N/A"
                    print(f"    Iteration {r['iteration']}: {r['accuracy']:.2f}% (F1: {f1_str}, Precision: {prec_str}, Recall: {recall_str})")
    
    # Overall comparison
    print(f"\n{'='*120}")
    print("OVERALL COMPARISON - ALL EXPERIMENTS")
    print(f"{'='*120}")
    print(f"\n{'Experiment':<50} {'Part':<30} {'Max Acc':<12} {'Mean Acc':<12} {'Final Acc':<12} {'Best Iter':<12}")
    print(f"{'-'*120}")
    
    for part_name in part_order:
        if part_name not in all_data:
            continue
        
        part_experiments = all_data[part_name]
        
        for exp_name in sorted(part_experiments.keys()):
            exp_data = sorted(part_experiments[exp_name], key=lambda x: x['iteration'])
            accuracies = [r['accuracy'] for r in exp_data]
            max_acc = max(accuracies)
            mean_acc = sum(accuracies) / len(accuracies)
            final_acc = accuracies[-1]
            max_iter = max(exp_data, key=lambda x: x['accuracy'])['iteration']
            
            print(f"{exp_name:<50} {part_name:<30} {max_acc:<12.2f} {mean_acc:<12.2f} {final_acc:<12.2f} {max_iter:<12}")
    
    print(f"\n{'='*120}")
    print("Done! All metric values displayed above.")
    print(f"{'='*120}")
    
    # Save to CSV file
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, 'all_redo_parts1-4_all_iterations.csv')
    
    # Flatten all data for CSV
    csv_rows = []
    for part_name in part_order:
        if part_name not in all_data:
            continue
        
        part_experiments = all_data[part_name]
        for exp_name in sorted(part_experiments.keys()):
            exp_data = sorted(part_experiments[exp_name], key=lambda x: x['iteration'])
            for row in exp_data:
                csv_rows.append({
                    'Part': part_name,
                    'Experiment': exp_name,
                    'Iteration': row['iteration'],
                    'Accuracy (%)': row['accuracy'],
                    'F1': row['f1'] if row['f1'] is not None else '',
                    'Precision': row['precision'] if row['precision'] is not None else '',
                    'Recall': row['recall'] if row['recall'] is not None else ''
                })
    
    # Write to CSV
    if csv_rows:
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['Part', 'Experiment', 'Iteration', 'Accuracy (%)', 'F1', 'Precision', 'Recall']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n{'='*120}")
        print(f"✓ Saved all data to CSV: {csv_file}")
        print(f"  Total rows: {len(csv_rows)}")
        print(f"{'='*120}")

if __name__ == '__main__':
    main()
