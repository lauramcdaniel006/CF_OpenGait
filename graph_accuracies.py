#!/usr/bin/env python3
"""
Script to extract and graph accuracies from multiple experiments.

This script reads TensorBoard event files from different experiments
and creates comparison graphs of accuracies over training iterations.
"""

import os
import re
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_from_tensorboard(log_dir):
    """Extract accuracy metrics from TensorBoard event files."""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Look for accuracy scalars
        scalar_tags = ea.Tags()['scalars']
        
        # Find test accuracy tags
        accuracy_data = {}
        for tag in scalar_tags:
            if 'test_accuracy' in tag.lower() or 'accuracy' in tag.lower():
                scalar_events = ea.Scalars(tag)
                iterations = [s.step for s in scalar_events]
                values = [s.value for s in scalar_events]
                accuracy_data[tag] = {'iterations': iterations, 'values': values}
        
        return accuracy_data
    except Exception as e:
        print(f"Error reading TensorBoard from {log_dir}: {e}")
        return {}

def extract_from_log_file(log_file):
    """Extract accuracy from log text files if they exist."""
    accuracies = []
    iterations = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for "Overall Accuracy: XX.XX%"
                match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', line)
                if match:
                    acc = float(match.group(1))
                    accuracies.append(acc)
                    # Try to find iteration number in nearby lines
                    # This is a simple approach - you may need to adjust
                    iter_match = re.search(r'Iteration\s+(\d+)', line)
                    if iter_match:
                        iterations.append(int(iter_match.group(1)))
                    else:
                        iterations.append(len(accuracies) * 500)  # Default assumption
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return iterations, accuracies

def find_experiments(output_dir):
    """Find all experiment directories."""
    experiments = {}
    
    # Look for pattern: output/<dataset>/<model>/<save_name>/
    for dataset_dir in glob.glob(os.path.join(output_dir, '*')):
        if not os.path.isdir(dataset_dir):
            continue
        
        dataset_name = os.path.basename(dataset_dir)
        model_dir = os.path.join(dataset_dir, 'SwinGait')
        
        if not os.path.isdir(model_dir):
            continue
        
        for exp_dir in glob.glob(os.path.join(model_dir, '*')):
            if not os.path.isdir(exp_dir):
                continue
            
            exp_name = os.path.basename(exp_dir)
            summary_dir = os.path.join(exp_dir, 'summary')
            logs_dir = os.path.join(exp_dir, 'logs')
            
            if os.path.isdir(summary_dir) or os.path.isdir(logs_dir):
                # Create a readable name
                readable_name = exp_name.replace('Frailty_', '').replace('_', ' ').title()
                experiments[readable_name] = {
                    'summary_dir': summary_dir,
                    'logs_dir': logs_dir,
                    'full_path': exp_dir
                }
    
    return experiments

def plot_accuracies(experiments_data, output_file='accuracy_comparison.png'):
    """Plot accuracies from all experiments."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for (exp_name, data), color in zip(experiments_data.items(), colors):
        if 'iterations' in data and 'values' in data:
            iterations = data['iterations']
            values = data['values']
            
            if len(iterations) > 0:
                ax.plot(iterations, values, label=exp_name, color=color, linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {output_file}")
    plt.show()

def main():
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found!")
        return
    
    print("Finding experiments...")
    experiments = find_experiments(output_dir)
    
    if len(experiments) == 0:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for name in experiments.keys():
        print(f"  - {name}")
    
    print("\nExtracting accuracy data...")
    experiments_data = {}
    
    for exp_name, paths in experiments.items():
        print(f"  Processing: {exp_name}")
        
        # Try TensorBoard first
        if os.path.isdir(paths['summary_dir']):
            tb_data = extract_from_tensorboard(paths['summary_dir'])
            
            # Look for the main accuracy metric
            if 'scalar/test_accuracy/' in tb_data:
                experiments_data[exp_name] = tb_data['scalar/test_accuracy/']
            elif len(tb_data) > 0:
                # Use the first available accuracy metric
                first_key = list(tb_data.keys())[0]
                experiments_data[exp_name] = tb_data[first_key]
                print(f"    Using metric: {first_key}")
        
        # Fallback to log files if TensorBoard didn't work
        if exp_name not in experiments_data and os.path.isdir(paths['logs_dir']):
            log_files = glob.glob(os.path.join(paths['logs_dir'], '*.txt'))
            if log_files:
                # Use the most recent log file
                log_file = max(log_files, key=os.path.getmtime)
                iterations, accuracies = extract_from_log_file(log_file)
                if len(iterations) > 0:
                    experiments_data[exp_name] = {
                        'iterations': iterations,
                        'values': accuracies
                    }
                    print(f"    Extracted from log file: {len(accuracies)} points")
    
    if len(experiments_data) == 0:
        print("No accuracy data found in any experiments!")
        return
    
    print(f"\nSuccessfully extracted data from {len(experiments_data)} experiments")
    
    # Plot
    print("\nGenerating graph...")
    plot_accuracies(experiments_data)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Final Accuracies:")
    print("="*70)
    for exp_name, data in experiments_data.items():
        if 'values' in data and len(data['values']) > 0:
            final_acc = data['values'][-1]
            max_acc = max(data['values'])
            print(f"{exp_name:40s} Final: {final_acc:.2f}%  Max: {max_acc:.2f}%")
    print("="*70)

if __name__ == '__main__':
    main()

