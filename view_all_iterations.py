#!/usr/bin/env python3
"""
View all previous iteration results from your experiments.
Shows metrics for every evaluation checkpoint.
"""

import pandas as pd
import os
import sys

def main():
    print("=" * 100)
    print("VIEW ALL PREVIOUS ITERATION RESULTS")
    print("=" * 100)
    
    # Check which files are available
    csv_file = 'results_visualization/checkpoint_evaluations_only.csv'
    full_csv = 'results_visualization/all_checkpoint_metrics.csv'
    
    if os.path.exists(csv_file):
        print(f"\n✓ Found: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"  Contains {len(df)} evaluation checkpoints")
    elif os.path.exists(full_csv):
        print(f"\n✓ Found: {full_csv}")
        df = pd.read_csv(full_csv)
        # Filter to only evaluation checkpoints
        df = df[df['test_accuracy/'].notna()]
        print(f"  Contains {len(df)} evaluation checkpoints")
    else:
        print("\n❌ No CSV files found!")
        print("   Run: python extract_all_checkpoint_metrics.py")
        return
    
    print("\n" + "=" * 100)
    print("OPTIONS:")
    print("=" * 100)
    print("1. View all iterations for a specific experiment")
    print("2. View all experiments at a specific iteration")
    print("3. View summary table (all experiments, all iterations)")
    print("4. Export to Excel-readable format")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        # View all iterations for one experiment
        experiments = sorted(df['Experiment'].unique())
        print("\nAvailable experiments:")
        for i, exp in enumerate(experiments, 1):
            count = len(df[df['Experiment'] == exp])
            print(f"  {i}. {exp} ({count} evaluations)")
        
        exp_idx = int(input("\nEnter experiment number: ")) - 1
        if 0 <= exp_idx < len(experiments):
            exp_name = experiments[exp_idx]
            exp_data = df[df['Experiment'] == exp_name].sort_values('Iteration')
            
            print(f"\n{'='*100}")
            print(f"{exp_name} - All Iterations")
            print(f"{'='*100}")
            print(f"\n{exp_data.to_string(index=False)}")
            
            # Summary stats
            print(f"\n{'='*100}")
            print("SUMMARY STATISTICS")
            print(f"{'='*100}")
            print(f"Total evaluations: {len(exp_data)}")
            print(f"First iteration: {exp_data['Iteration'].min()}")
            print(f"Last iteration: {exp_data['Iteration'].max()}")
            print(f"Max accuracy: {exp_data['test_accuracy/'].max():.2f}% at iteration {exp_data.loc[exp_data['test_accuracy/'].idxmax(), 'Iteration']}")
            print(f"Final accuracy: {exp_data.iloc[-1]['test_accuracy/']:.2f}%")
            print(f"Times ≥73%: {(exp_data['test_accuracy/'] >= 73.0).sum()}")
            print(f"Times ≥66.67%: {(exp_data['test_accuracy/'] >= 66.67).sum()}")
    
    elif choice == "2":
        # View all experiments at specific iteration
        iterations = sorted(df['Iteration'].unique())
        print("\nAvailable iterations:")
        for i, it in enumerate(iterations, 1):
            count = len(df[df['Iteration'] == it])
            print(f"  {i}. Iteration {it} ({count} experiments)")
        
        it_idx = int(input("\nEnter iteration number: ")) - 1
        if 0 <= it_idx < len(iterations):
            iter_num = iterations[it_idx]
            iter_data = df[df['Iteration'] == iter_num].sort_values('test_accuracy/', ascending=False)
            
            print(f"\n{'='*100}")
            print(f"All Experiments at Iteration {iter_num}")
            print(f"{'='*100}")
            print(f"\n{iter_data.to_string(index=False)}")
    
    elif choice == "3":
        # Summary table
        print(f"\n{'='*100}")
        print("SUMMARY TABLE - All Experiments, All Iterations")
        print(f"{'='*100}")
        
        # Pivot table: experiments x iterations
        pivot = df.pivot_table(
            values='test_accuracy/',
            index='Experiment',
            columns='Iteration',
            aggfunc='first'
        )
        
        print("\nAccuracy by Experiment and Iteration:")
        print(pivot.to_string())
        
        # Summary by experiment
        print(f"\n{'='*100}")
        print("EXPERIMENT SUMMARIES")
        print(f"{'='*100}")
        summary = df.groupby('Experiment').agg({
            'test_accuracy/': ['max', 'min', 'mean', 'last'],
            'Iteration': ['min', 'max', 'count']
        })
        print(summary.to_string())
    
    elif choice == "4":
        # Export to Excel format
        output_file = 'results_visualization/all_iterations_excel.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Exported to: {output_file}")
        print("  You can open this in Excel, Google Sheets, or any spreadsheet program")
        print(f"  Contains {len(df)} rows with all evaluation metrics")
    
    else:
        print("Invalid choice")

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available. Install with: pip install pandas")
        print("\nAlternatively, view the CSV files directly:")
        print("  results_visualization/checkpoint_evaluations_only.csv")
        print("  results_visualization/all_checkpoint_metrics.csv")
        sys.exit(1)
    
    main()

