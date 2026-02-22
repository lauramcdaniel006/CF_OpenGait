#!/usr/bin/env python3
"""
Simple script to view all metric values for each iteration of all REDO experiments.
No pandas required - uses only standard library.
"""

import csv
import os
from collections import defaultdict

def main():
    csv_file = 'results_visualization/checkpoint_evaluations_only.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("   Run: python extract_all_checkpoint_metrics.py")
        return
    
    # Read all data
    experiments = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row['Experiment']
            if 'REDO' in exp_name:
                experiments[exp_name].append({
                    'iteration': int(row['Iteration']),
                    'accuracy': float(row['test_accuracy/']),
                    'f1': float(row['test_f1/']),
                    'precision': float(row['test_precision/']),
                    'recall': float(row['test_recall/'])
                })
    
    if not experiments:
        print("❌ No REDO experiments found!")
        return
    
    print("=" * 120)
    print("ALL REDO EXPERIMENTS - METRIC VALUES BY ITERATION")
    print("=" * 120)
    
    # Sort experiments
    sorted_experiments = sorted(experiments.keys())
    
    for exp_name in sorted_experiments:
        data = experiments[exp_name]
        data.sort(key=lambda x: x['iteration'])
        
        print(f"\n{'='*120}")
        print(f"{exp_name}")
        print(f"{'='*120}")
        print(f"\n{'Iteration':<12} {'Accuracy (%)':<15} {'F1':<12} {'Precision':<12} {'Recall':<12}")
        print(f"{'-'*120}")
        
        for row in data:
            print(f"{row['iteration']:<12} {row['accuracy']:<15.2f} {row['f1']:<12.4f} {row['precision']:<12.4f} {row['recall']:<12.4f}")
        
        # Statistics
        accuracies = [r['accuracy'] for r in data]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        mean_acc = sum(accuracies) / len(accuracies)
        final_acc = accuracies[-1]
        max_iter = max(data, key=lambda x: x['accuracy'])['iteration']
        
        times_73 = sum(1 for a in accuracies if a >= 73.0)
        times_66 = sum(1 for a in accuracies if a >= 66.67)
        
        print(f"\n  Statistics:")
        print(f"    Total Evaluations: {len(data)}")
        print(f"    Iteration Range: {data[0]['iteration']} - {data[-1]['iteration']}")
        print(f"    Max Accuracy: {max_acc:.2f}% at iteration {max_iter}")
        print(f"    Min Accuracy: {min_acc:.2f}%")
        print(f"    Mean Accuracy: {mean_acc:.2f}%")
        print(f"    Final Accuracy: {final_acc:.2f}%")
        print(f"    Times ≥73%: {times_73} / {len(data)} ({times_73/len(data)*100:.1f}%)")
        print(f"    Times ≥66.67%: {times_66} / {len(data)} ({times_66/len(data)*100:.1f}%)")
        
        # Show high accuracy iterations
        high_acc = [r for r in data if r['accuracy'] >= 73.0]
        if high_acc:
            print(f"\n  Iterations with ≥73% accuracy:")
            for r in high_acc:
                print(f"    Iteration {r['iteration']}: {r['accuracy']:.2f}% (F1: {r['f1']:.4f}, Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f})")
    
    # Comparison summary
    print(f"\n{'='*120}")
    print("COMPARISON SUMMARY")
    print(f"{'='*120}")
    print(f"\n{'Experiment':<40} {'Max Acc':<12} {'Mean Acc':<12} {'Final Acc':<12} {'Best Iter':<12}")
    print(f"{'-'*120}")
    
    for exp_name in sorted_experiments:
        data = experiments[exp_name]
        accuracies = [r['accuracy'] for r in data]
        max_acc = max(accuracies)
        mean_acc = sum(accuracies) / len(accuracies)
        final_acc = accuracies[-1]
        max_iter = max(data, key=lambda x: x['accuracy'])['iteration']
        
        print(f"{exp_name:<40} {max_acc:<12.2f} {mean_acc:<12.2f} {final_acc:<12.2f} {max_iter:<12}")
    
    print(f"\n{'='*120}")
    print("Done! All metric values displayed above.")
    print(f"{'='*120}")

if __name__ == '__main__':
    main()

