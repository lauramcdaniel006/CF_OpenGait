#!/usr/bin/env python3
"""
View all REDO experiments with detailed iteration-by-iteration results.
"""

import pandas as pd
import os

def main():
    print("=" * 120)
    print("ALL REDO EXPERIMENTS - COMPLETE ITERATION HISTORY")
    print("=" * 120)
    
    csv_file = 'results_visualization/checkpoint_evaluations_only.csv'
    
    if not os.path.exists(csv_file):
        print(f"\n❌ File not found: {csv_file}")
        print("   Run: python extract_all_checkpoint_metrics.py")
        return
    
    df = pd.read_csv(csv_file)
    
    # Filter to only REDO experiments
    redo_experiments = [exp for exp in df['Experiment'].unique() if 'REDO' in exp]
    
    if not redo_experiments:
        print("\n❌ No REDO experiments found!")
        return
    
    print(f"\nFound {len(redo_experiments)} REDO experiment(s): {', '.join(sorted(redo_experiments))}\n")
    
    # Show detailed table for each experiment
    for exp_name in sorted(redo_experiments):
        exp_data = df[df['Experiment'] == exp_name].sort_values('Iteration').copy()
        
        print("=" * 120)
        print(f"{exp_name} - All Iterations")
        print("=" * 120)
        
        # Format the data for display
        display_df = exp_data[['Iteration', 'test_accuracy/', 'test_f1/', 'test_precision/', 'test_recall/']].copy()
        display_df.columns = ['Iteration', 'Accuracy (%)', 'F1', 'Precision', 'Recall']
        display_df['Accuracy (%)'] = display_df['Accuracy (%)'].apply(lambda x: f"{x:.2f}%")
        display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.4f}")
        display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.4f}")
        display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.4f}")
        
        print(f"\n{display_df.to_string(index=False)}")
        
        # Statistics
        print(f"\n{'─' * 120}")
        print("STATISTICS:")
        print(f"{'─' * 120}")
        
        acc_values = exp_data['test_accuracy/'].values
        max_acc = acc_values.max()
        min_acc = acc_values.min()
        mean_acc = acc_values.mean()
        final_acc = acc_values[-1]
        max_iter = exp_data.loc[exp_data['test_accuracy/'].idxmax(), 'Iteration']
        
        times_73 = (acc_values >= 73.0).sum()
        times_66 = (acc_values >= 66.67).sum()
        times_60 = (acc_values >= 60.0).sum()
        
        print(f"  Total Evaluations: {len(exp_data)}")
        print(f"  Iteration Range: {exp_data['Iteration'].min()} - {exp_data['Iteration'].max()}")
        print(f"  Max Accuracy: {max_acc:.2f}% at iteration {max_iter}")
        print(f"  Min Accuracy: {min_acc:.2f}%")
        print(f"  Mean Accuracy: {mean_acc:.2f}%")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Times ≥73%: {times_73} / {len(exp_data)} ({times_73/len(exp_data)*100:.1f}%)")
        print(f"  Times ≥66.67%: {times_66} / {len(exp_data)} ({times_66/len(exp_data)*100:.1f}%)")
        print(f"  Times ≥60%: {times_60} / {len(exp_data)} ({times_60/len(exp_data)*100:.1f}%)")
        
        # Show iterations where accuracy was ≥73%
        high_acc_iters = exp_data[exp_data['test_accuracy/'] >= 73.0][['Iteration', 'test_accuracy/']]
        if len(high_acc_iters) > 0:
            print(f"\n  Iterations with ≥73% accuracy:")
            for _, row in high_acc_iters.iterrows():
                print(f"    Iteration {row['Iteration']}: {row['test_accuracy/']:.2f}%")
        
        print()
    
    # Comparison table
    print("=" * 120)
    print("COMPARISON TABLE - All REDO Experiments")
    print("=" * 120)
    
    comparison_data = []
    for exp_name in sorted(redo_experiments):
        exp_data = df[df['Experiment'] == exp_name].sort_values('Iteration')
        acc_values = exp_data['test_accuracy/'].values
        
        comparison_data.append({
            'Experiment': exp_name,
            'Max Acc (%)': f"{acc_values.max():.2f}",
            'Min Acc (%)': f"{acc_values.min():.2f}",
            'Mean Acc (%)': f"{acc_values.mean():.2f}",
            'Final Acc (%)': f"{acc_values[-1]:.2f}",
            'Times ≥73%': f"{(acc_values >= 73.0).sum()}/{len(exp_data)}",
            'Times ≥66.67%': f"{(acc_values >= 66.67).sum()}/{len(exp_data)}",
            'Best Iter': exp_data.loc[exp_data['test_accuracy/'].idxmax(), 'Iteration']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Side-by-side iteration comparison
    print("\n" + "=" * 120)
    print("SIDE-BY-SIDE ITERATION COMPARISON")
    print("=" * 120)
    
    # Create pivot table
    pivot = df[df['Experiment'].isin(redo_experiments)].pivot_table(
        values='test_accuracy/',
        index='Iteration',
        columns='Experiment',
        aggfunc='first'
    )
    
    # Format for display
    pivot_display = pivot.copy()
    for col in pivot_display.columns:
        pivot_display[col] = pivot_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    print(f"\n{pivot_display.to_string()}")
    
    print("\n" + "=" * 120)
    print("Done! All REDO experiment data displayed above.")
    print("=" * 120)

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available. Install with: pip install pandas")
        print("\nAlternatively, view the CSV file directly:")
        print("  results_visualization/checkpoint_evaluations_only.csv")
        exit(1)
    
    main()

