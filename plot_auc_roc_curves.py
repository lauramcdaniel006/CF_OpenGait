#!/usr/bin/env python3
"""
Extract AUC-ROC values from TensorBoard logs and plot curves for SwinGait M1 and DeepGaitV2 M6.
This script reads TensorBoard event files and creates AUC-ROC vs iteration plots.
"""

import os
import sys
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("WARNING: TensorBoard not available. Install with: pip install tensorboard")

def extract_auc_from_tensorboard(summary_dir, metric_name='scalar/test_auc_macro/'):
    """Extract AUC-ROC values from TensorBoard event files."""
    if not TB_AVAILABLE:
        return None, None
    
    if not os.path.exists(summary_dir):
        return None, None
    
    try:
        # Find all event files (may have multiple)
        event_files = glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))
        if not event_files:
            return None, None
        
        # Load event accumulator from directory
        ea = EventAccumulator(summary_dir)
        ea.Reload()
        
        # Get scalar values
        scalar_tags = ea.Tags().get('scalars', [])
        if metric_name not in scalar_tags:
            # Try alternative names
            alt_names = [
                'scalar/test_auc_macro',
                'test_auc_macro/',
                'test_auc_macro'
            ]
            found = False
            for alt in alt_names:
                if alt in scalar_tags:
                    metric_name = alt
                    found = True
                    break
            if not found:
                return None, None
        
        scalar_events = ea.Scalars(metric_name)
        iterations = [s.step for s in scalar_events]
        values = [s.value for s in scalar_events]
        
        return iterations, values
    except Exception as e:
        print(f"Error extracting from TensorBoard: {e}")
        return None, None

def extract_auc_from_logs(log_dir):
    """Extract AUC-ROC values from log files as fallback."""
    log_files = glob.glob(os.path.join(log_dir, '*.txt'))
    if not log_files:
        return None, None
    
    # Use most recent log file
    log_file = max(log_files, key=os.path.getmtime)
    
    iterations = []
    auc_macro_values = []
    auc_micro_values = []
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Look for evaluation sections with iteration numbers
        # Pattern: "Iteration XXXXX" followed by AUC values
        lines = content.split('\n')
        current_iter = None
        
        for i, line in enumerate(lines):
            # Find iteration numbers
            iter_match = re.search(r'Iteration\s+(\d+)', line)
            if iter_match:
                current_iter = int(iter_match.group(1))
            
            # Find AUC values
            if 'ROC AUC (macro)' in line:
                match = re.search(r'(\d+\.\d+)', line)
                if match and current_iter is not None:
                    iterations.append(current_iter)
                    auc_macro_values.append(float(match.group(1)))
            
            if 'ROC AUC (micro)' in line:
                match = re.search(r'(\d+\.\d+)', line)
                if match and current_iter is not None:
                    auc_micro_values.append(float(match.group(1)))
    
    if iterations and auc_macro_values:
        return iterations, auc_macro_values
    return None, None

def plot_auc_curves(models_data, output_file='auc_roc_curves.png'):
    """Plot AUC-ROC curves for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        if data['iterations'] is None or data['values'] is None:
            print(f"Warning: No data for {model_name}")
            continue
        
        iterations = data['iterations']
        values = data['values']
        
        # Plot macro AUC
        ax1.plot(iterations, values, 
                marker='o', markersize=4, linewidth=2, 
                label=model_name, color=colors[idx % len(colors)])
        
        # Also plot micro if available
        if 'micro_values' in data and data['micro_values']:
            ax1.plot(iterations, data['micro_values'],
                    marker='s', markersize=3, linewidth=1.5, linestyle='--',
                    label=f"{model_name} (micro)", alpha=0.7, color=colors[idx % len(colors)])
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('AUC-ROC (Macro)', fontsize=12)
    ax1.set_title('AUC-ROC (Macro) vs Iteration', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot comparison (if both models have data)
    if len(models_data) == 2:
        model_names = list(models_data.keys())
        if (models_data[model_names[0]]['iterations'] is not None and 
            models_data[model_names[1]]['iterations'] is not None):
            
            ax2.plot(models_data[model_names[0]]['iterations'], 
                    models_data[model_names[0]]['values'],
                    marker='o', markersize=4, linewidth=2, 
                    label=model_names[0], color=colors[0])
            ax2.plot(models_data[model_names[1]]['iterations'], 
                    models_data[model_names[1]]['values'],
                    marker='s', markersize=4, linewidth=2, 
                    label=model_names[1], color=colors[1])
            
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('AUC-ROC (Macro)', fontsize=12)
            ax2.set_title('SwinGait M1 vs DeepGaitV2 M6', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_file}")
    
    return fig

def main():
    print("="*80)
    print("Extracting AUC-ROC Curves from Training Logs")
    print("="*80)
    
    # Define model paths
    models = {
        'SwinGait M1': {
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4_swingait/SwinGait/REDO_Frailty_ccpg_pt4_swingait/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4_swingait/SwinGait/REDO_Frailty_ccpg_pt4_swingait/logs'
        },
        'DeepGaitV2 M6': {
            'summary_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/summary',
            'log_dir': 'output/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4b_deepgaitv2_half_frozen_cnn_with_weights/logs'
        }
    }
    
    models_data = {}
    
    for model_name, paths in models.items():
        print(f"\nProcessing {model_name}...")
        
        # Try TensorBoard first
        iterations, values = extract_auc_from_tensorboard(
            paths['summary_dir'], 
            'scalar/test_auc_macro/'
        )
        
        # Fallback to log files
        if iterations is None:
            print(f"  Trying log files...")
            iterations, values = extract_auc_from_logs(paths['log_dir'])
        
        if iterations and values:
            print(f"  ✓ Found {len(iterations)} AUC-ROC values")
            print(f"    Iterations: {min(iterations)} to {max(iterations)}")
            print(f"    AUC range: {min(values):.4f} to {max(values):.4f}")
            
            # Try to get micro AUC too
            micro_iterations, micro_values = extract_auc_from_tensorboard(
                paths['summary_dir'],
                'scalar/test_auc_micro/'
            )
            
            models_data[model_name] = {
                'iterations': iterations,
                'values': values,
                'micro_values': micro_values if micro_iterations else None
            }
        else:
            print(f"  ⚠️  No AUC-ROC data found")
            models_data[model_name] = {
                'iterations': None,
                'values': None
            }
    
    # Plot curves
    if any(data['iterations'] is not None for data in models_data.values()):
        print("\n" + "="*80)
        print("Creating AUC-ROC curves...")
        print("="*80)
        plot_auc_curves(models_data, 'auc_roc_curves_part4.png')
    else:
        print("\n⚠️  No AUC-ROC data found. Make sure:")
        print("   1. Training was run with save_iter: 500")
        print("   2. Training was run with with_test: true")
        print("   3. The evaluator has been updated to compute AUC-ROC")

if __name__ == '__main__':
    main()

