#!/usr/bin/env python3
"""
View all checkpoint evaluation metrics in a clean, readable format.
"""

import pandas as pd
import os

def main():
    csv_file = 'results_visualization/all_checkpoint_metrics.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("   Run extract_all_checkpoint_metrics.py first!")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Filter to only rows with evaluation metrics (non-empty test_accuracy)
    df_eval = df[df['test_accuracy/'].notna()].copy()
    
    # Convert accuracy to percentage for display
    df_eval['Accuracy (%)'] = df_eval['test_accuracy/'].apply(lambda x: f"{x:.2f}%")
    df_eval['F1'] = df_eval['test_f1/'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    df_eval['Precision'] = df_eval['test_precision/'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    df_eval['Recall'] = df_eval['test_recall/'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    # Display by experiment
    experiments = df_eval['Experiment'].unique()
    
    print("=" * 100)
    print("ALL CHECKPOINT EVALUATION METRICS")
    print("=" * 100)
    
    for exp in sorted(experiments):
        exp_data = df_eval[df_eval['Experiment'] == exp].copy()
        exp_data = exp_data.sort_values('Iteration')
        
        print(f"\n{exp}:")
        print("-" * 100)
        print(f"{'Iteration':<12} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 100)
        
        for _, row in exp_data.iterrows():
            print(f"{row['Iteration']:<12} {row['Accuracy (%)']:<12} {row['F1']:<12} "
                  f"{row['Precision']:<12} {row['Recall']:<12}")
        
        # Summary stats
        max_acc = exp_data['test_accuracy/'].max()
        max_acc_iter = exp_data.loc[exp_data['test_accuracy/'].idxmax(), 'Iteration']
        times_73 = (exp_data['test_accuracy/'] >= 73.0).sum()
        times_66 = (exp_data['test_accuracy/'] >= 66.67).sum()
        
        print("-" * 100)
        print(f"  Max Accuracy: {max_acc:.2f}% at iteration {max_acc_iter}")
        print(f"  Times ≥73%: {times_73} / {len(exp_data)} evaluations")
        print(f"  Times ≥66.67%: {times_66} / {len(exp_data)} evaluations")
        print()
    
    # Overall comparison
    print("=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<20} {'Max Acc (%)':<15} {'Final Acc (%)':<15} {'Times ≥73%':<15}")
    print("-" * 100)
    
    for exp in sorted(experiments):
        exp_data = df_eval[df_eval['Experiment'] == exp].copy()
        exp_data = exp_data.sort_values('Iteration')
        
        max_acc = exp_data['test_accuracy/'].max()
        final_acc = exp_data.iloc[-1]['test_accuracy/']
        times_73 = (exp_data['test_accuracy/'] >= 73.0).sum()
        total_evals = len(exp_data)
        
        print(f"{exp:<20} {max_acc:<15.2f} {final_acc:<15.2f} {times_73}/{total_evals}")
    
    print("=" * 100)
    
    # Save filtered CSV (only evaluation checkpoints)
    output_file = 'results_visualization/checkpoint_evaluations_only.csv'
    df_eval[['Experiment', 'Iteration', 'test_accuracy/', 'test_f1/', 'test_precision/', 'test_recall/']].to_csv(
        output_file, index=False
    )
    print(f"\n✓ Saved evaluation-only CSV to: {output_file}")

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available. Install with: pip install pandas")
        print("\nAlternatively, view the CSV file directly:")
        print("  results_visualization/all_checkpoint_metrics.csv")
        exit(1)
    
    main()

