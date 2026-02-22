#!/usr/bin/env python3
"""
Extract and visualize evaluation metrics from Parts 1-4 experiments.
Extracts: Overall Accuracy, Sensitivity, Precision, Specificity for each class.
Creates publication-ready visualizations.
"""

import os
import re
import glob
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Plots will be skipped.")
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Will use log files only.")
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. CSV export will be limited.")

# Set publication-quality style
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

# Define experiment parts and their expected directories
EXPERIMENT_PARTS = {
    'Part 1': {
        'patterns': ['REDO_Frailty_ccpg_pt1', 'Dataset_Frailty_ccpg_pt1'],
        'description': 'Freezing Strategy Ablation'
    },
    'Part 2': {
        'patterns': ['REDO_Frailty_ccpg_pt2', 'REDO_balnormal', 'REDO_insqrt', 'REDO_log', 'REDO_smooth', 'REDO_uniform'],
        'description': 'Class Weight Tuning'
    },
    'Part 3': {
        'patterns': ['REDO_Frailty_ccpg_pt3'],
        'description': 'Loss Function Comparison'
    },
    'Part 4': {
        'patterns': ['REDO_Frailty_ccpg_pt4'],
        'description': 'Architecture Comparison (SwinGait vs DeepGaitV2)'
    }
}

CLASS_NAMES = ['Frail', 'Prefrail', 'Nonfrail']


def parse_metric_value(metric):
    """Parse metric value from dict, string, or direct value."""
    if metric is None:
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    if isinstance(metric, dict):
        return metric.get('value', None)
    if isinstance(metric, str):
        # Try to parse string representation of dict
        try:
            import ast
            parsed = ast.literal_eval(metric)
            if isinstance(parsed, dict):
                return parsed.get('value', None)
        except:
            pass
    return None


def extract_from_tensorboard(summary_dir):
    """Extract metrics from TensorBoard event files."""
    if not os.path.isdir(summary_dir):
        return {}
    
    if not TENSORBOARD_AVAILABLE:
        return {}
    
    try:
        event_files = glob.glob(os.path.join(summary_dir, 'events.out.tfevents.*'))
        if not event_files:
            return {}
        
        # Use the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(os.path.dirname(event_file))
        ea.Reload()
        
        metrics = {}
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            if scalar_events:
                # Get the last value (most recent evaluation)
                last_value = scalar_events[-1].value
                metrics[tag] = {
                    'value': last_value,
                    'step': scalar_events[-1].step,
                    'all_values': [(s.step, s.value) for s in scalar_events]
                }
        
        return metrics
    except Exception as e:
        print(f"Error reading TensorBoard from {summary_dir}: {e}")
        return {}


def extract_from_log_file(log_file):
    """Extract evaluation metrics from log text files."""
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Extract Overall Accuracy
            acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', content)
            if acc_match:
                metrics['scalar/test_accuracy/'] = float(acc_match.group(1)) / 100.0
            
            # Extract per-class metrics
            for i, class_name in enumerate(CLASS_NAMES):
                # Sensitivity (Recall)
                sens_match = re.search(
                    rf'{re.escape(class_name)}\s+Sensitivity\s+\(Recall\):\s*(\d+\.?\d*)%', 
                    content
                )
                if sens_match:
                    metrics[f'scalar/test_sensitivity/{class_name}'] = float(sens_match.group(1)) / 100.0
                
                # Specificity
                spec_match = re.search(
                    rf'{re.escape(class_name)}\s+Specificity:\s*(\d+\.?\d*)%', 
                    content
                )
                if spec_match:
                    metrics[f'scalar/test_specificity/{class_name}'] = float(spec_match.group(1)) / 100.0
                
                # Precision
                prec_match = re.search(
                    rf'{re.escape(class_name)}\s+Precision:\s*(\d+\.?\d*)%', 
                    content
                )
                if prec_match:
                    metrics[f'scalar/test_precision/{class_name}'] = float(prec_match.group(1)) / 100.0
            
            # Extract confusion matrix
            cm_match = re.search(
                r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\s*\[(\d+)\s+(\d+)\s+(\d+)\]\]',
                content
            )
            if cm_match:
                cm = np.array([
                    [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                    [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                    [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
                ])
                metrics['confusion_matrix'] = cm
            
            # Extract iteration number if available
            iter_match = re.search(r'Iteration\s+(\d+)', content)
            if iter_match:
                metrics['iteration'] = int(iter_match.group(1))
                
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
    
    return metrics


def find_experiments(output_dir='output'):
    """Find all experiment directories and categorize them."""
    experiments = defaultdict(dict)
    
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found!")
        return experiments
    
    # Walk through output directory
    for root, dirs, files in os.walk(output_dir):
        # Look for summary directories (indicates an experiment)
        if 'summary' in dirs:
            # Extract experiment name from path
            # Path format: output/<dataset>/<model>/<save_name>/
            path_parts = Path(root).parts
            if len(path_parts) >= 4:
                dataset = path_parts[-3]
                model = path_parts[-2]
                save_name = path_parts[-1]
                
                exp_key = f"{dataset}/{model}/{save_name}"
                
                summary_dir = os.path.join(root, 'summary')
                logs_dir = os.path.join(root, 'logs')
                
                experiments[exp_key] = {
                    'summary_dir': summary_dir,
                    'logs_dir': logs_dir if os.path.exists(logs_dir) else None,
                    'dataset': dataset,
                    'model': model,
                    'save_name': save_name,
                    'full_path': root
                }
    
    return experiments


def categorize_experiment(exp_name):
    """Categorize experiment into Part 1-4."""
    for part_name, part_info in EXPERIMENT_PARTS.items():
        for pattern in part_info['patterns']:
            if pattern in exp_name:
                return part_name, part_info['description']
    return 'Other', 'Unknown'


def extract_all_metrics(experiments):
    """Extract metrics from all experiments."""
    all_results = defaultdict(dict)
    
    for exp_name, paths in experiments.items():
        print(f"Processing: {exp_name}")
        
        metrics = {}
        
        # Try TensorBoard first
        if os.path.isdir(paths['summary_dir']):
            tb_metrics = extract_from_tensorboard(paths['summary_dir'])
            metrics.update(tb_metrics)
        
        # Try log files as fallback/supplement
        if paths['logs_dir'] and os.path.isdir(paths['logs_dir']):
            log_files = glob.glob(os.path.join(paths['logs_dir'], '*.txt'))
            if log_files:
                # Use the most recent log file
                log_file = max(log_files, key=os.path.getmtime)
                log_metrics = extract_from_log_file(log_file)
                # Log file metrics take precedence (more detailed)
                for k, v in log_metrics.items():
                    if k not in metrics or isinstance(v, dict):
                        metrics[k] = v
        
        if metrics:
            part, description = categorize_experiment(exp_name)
            all_results[part][exp_name] = {
                'metrics': metrics,
                'info': paths,
                'description': description
            }
            print(f"  ✓ Extracted {len(metrics)} metrics")
        else:
            print(f"  ✗ No metrics found")
    
    return all_results


def parse_metric_value(metric):
    """Parse metric value from dict, string, or direct value."""
    if metric is None:
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    if isinstance(metric, dict):
        return metric.get('value', None)
    if isinstance(metric, str):
        # Try to parse string representation of dict
        try:
            import ast
            parsed = ast.literal_eval(metric)
            if isinstance(parsed, dict):
                return parsed.get('value', None)
        except:
            pass
    return None


def create_summary_table(all_results):
    """Create a summary table of all results."""
    rows = []
    
    for part in ['Part 1', 'Part 2', 'Part 3', 'Part 4']:
        if part not in all_results:
            continue
        
        for exp_name, exp_data in all_results[part].items():
            metrics = exp_data['metrics']
            info = exp_data['info']
            
            # Extract key metrics - try both with and without scalar/ prefix
            overall_acc = parse_metric_value(metrics.get('scalar/test_accuracy/', None))
            if overall_acc is None:
                overall_acc = parse_metric_value(metrics.get('test_accuracy/', None))
            if overall_acc is None:
                continue
            
            row = {
                'Part': part,
                'Experiment': exp_name.split('/')[-1],  # Just the save_name
                'Model': info['model'],
                'Overall Accuracy (%)': overall_acc * 100 if overall_acc else None,
            }
            
            # Add per-class metrics if available
            for class_name in CLASS_NAMES:
                sens = parse_metric_value(metrics.get(f'scalar/test_sensitivity/{class_name}', None))
                if sens is None:
                    sens = parse_metric_value(metrics.get(f'test_sensitivity/{class_name}', None))
                spec = parse_metric_value(metrics.get(f'scalar/test_specificity/{class_name}', None))
                if spec is None:
                    spec = parse_metric_value(metrics.get(f'test_specificity/{class_name}', None))
                prec = parse_metric_value(metrics.get(f'scalar/test_precision/{class_name}', None))
                if prec is None:
                    prec = parse_metric_value(metrics.get(f'test_precision/{class_name}', None))
                
                if sens is not None:
                    row[f'{class_name} Sens (%)'] = sens * 100
                if spec is not None:
                    row[f'{class_name} Spec (%)'] = spec * 100
                if prec is not None:
                    row[f'{class_name} Prec (%)'] = prec * 100
            
            rows.append(row)
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        return df
    else:
        # Return as list of dicts if pandas not available
        return rows


def plot_part_comparison(all_results, output_dir='results_visualization'):
    """Create comparison plots for each part."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots (matplotlib not available)")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    for part in ['Part 1', 'Part 2', 'Part 3', 'Part 4']:
        if part not in all_results or not all_results[part]:
            continue
        
        part_data = all_results[part]
        
        # Extract experiment names and accuracies
        exp_names = []
        accuracies = []
        sensitivities = {name: [] for name in CLASS_NAMES}
        specificities = {name: [] for name in CLASS_NAMES}
        precisions = {name: [] for name in CLASS_NAMES}
        
        for exp_name, exp_data in part_data.items():
            metrics = exp_data['metrics']
            
            acc = parse_metric_value(metrics.get('scalar/test_accuracy/', None))
            if acc is None:
                acc = parse_metric_value(metrics.get('test_accuracy/', None))
            if acc is None:
                continue
            
            # Shorten experiment name for display
            short_name = exp_name.split('/')[-1]
            # Further shorten if too long
            if len(short_name) > 30:
                short_name = short_name[:27] + '...'
            
            exp_names.append(short_name)
            accuracies.append(acc * 100)
            
            for class_name in CLASS_NAMES:
                sens = parse_metric_value(metrics.get(f'scalar/test_sensitivity/{class_name}', None))
                if sens is None:
                    sens = parse_metric_value(metrics.get(f'test_sensitivity/{class_name}', None))
                spec = parse_metric_value(metrics.get(f'scalar/test_specificity/{class_name}', None))
                if spec is None:
                    spec = parse_metric_value(metrics.get(f'test_specificity/{class_name}', None))
                prec = parse_metric_value(metrics.get(f'scalar/test_precision/{class_name}', None))
                if prec is None:
                    prec = parse_metric_value(metrics.get(f'test_precision/{class_name}', None))
                
                sensitivities[class_name].append(sens * 100 if sens else None)
                specificities[class_name].append(spec * 100 if spec else None)
                precisions[class_name].append(prec * 100 if prec else None)
        
        if not exp_names:
            continue
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{part}: {EXPERIMENT_PARTS[part]["description"]}', fontsize=14, fontweight='bold')
        
        # Plot 1: Overall Accuracy
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(exp_names)), accuracies, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(exp_names)))
        ax1.set_yticklabels(exp_names, fontsize=8)
        ax1.set_xlabel('Overall Accuracy (%)', fontweight='bold')
        ax1.set_title('Overall Accuracy Comparison', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim([0, 100])
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax1.text(acc + 1, i, f'{acc:.2f}%', va='center', fontsize=8)
        
        # Plot 2: Sensitivity (Recall) by Class
        ax2 = axes[0, 1]
        x = np.arange(len(exp_names))
        width = 0.25
        for i, class_name in enumerate(CLASS_NAMES):
            sens_vals = [s if s is not None else 0 for s in sensitivities[class_name]]
            ax2.bar(x + i*width, sens_vals, width, label=class_name, alpha=0.7)
        ax2.set_xlabel('Experiment', fontweight='bold')
        ax2.set_ylabel('Sensitivity (%)', fontweight='bold')
        ax2.set_title('Sensitivity (Recall) by Class', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=7)
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Plot 3: Specificity by Class
        ax3 = axes[1, 0]
        for i, class_name in enumerate(CLASS_NAMES):
            spec_vals = [s if s is not None else 0 for s in specificities[class_name]]
            ax3.bar(x + i*width, spec_vals, width, label=class_name, alpha=0.7)
        ax3.set_xlabel('Experiment', fontweight='bold')
        ax3.set_ylabel('Specificity (%)', fontweight='bold')
        ax3.set_title('Specificity by Class', fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=7)
        ax3.legend(fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 100])
        
        # Plot 4: Precision by Class
        ax4 = axes[1, 1]
        for i, class_name in enumerate(CLASS_NAMES):
            prec_vals = [p if p is not None else 0 for p in precisions[class_name]]
            ax4.bar(x + i*width, prec_vals, width, label=class_name, alpha=0.7)
        ax4.set_xlabel('Experiment', fontweight='bold')
        ax4.set_ylabel('Precision (%)', fontweight='bold')
        ax4.set_title('Precision by Class', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=7)
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim([0, 100])
        
        plt.tight_layout()
        
        # Save figure
        safe_part_name = part.replace(' ', '_').lower()
        output_file = os.path.join(output_dir, f'{safe_part_name}_comparison.png')
        plt.savefig(output_file, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()


def plot_confusion_matrices(all_results, output_dir='results_visualization'):
    """Plot confusion matrices for experiments that have them."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping confusion matrices (matplotlib not available)")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    confusion_matrices = []
    exp_labels = []
    
    for part in ['Part 1', 'Part 2', 'Part 3', 'Part 4']:
        if part not in all_results:
            continue
        
        for exp_name, exp_data in all_results[part].items():
            metrics = exp_data['metrics']
            if 'confusion_matrix' in metrics:
                confusion_matrices.append(metrics['confusion_matrix'])
                short_name = exp_name.split('/')[-1]
                if len(short_name) > 40:
                    short_name = short_name[:37] + '...'
                exp_labels.append(f"{part}\n{short_name}")
    
    if not confusion_matrices:
        print("No confusion matrices found in results.")
        return
    
    # Create a grid of confusion matrices
    n = len(confusion_matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (cm, label) in enumerate(zip(confusion_matrices, exp_labels)):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   cbar_kws={'label': 'Count'})
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_best_results_summary(all_results, output_dir='results_visualization'):
    """Create a summary plot showing best results from each part."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping best results summary (matplotlib not available)")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    best_results = {}
    
    for part in ['Part 1', 'Part 2', 'Part 3', 'Part 4']:
        if part not in all_results or not all_results[part]:
            continue
        
        best_acc = -1
        best_exp = None
        best_metrics = None
        
        for exp_name, exp_data in all_results[part].items():
            metrics = exp_data['metrics']
            acc = parse_metric_value(metrics.get('scalar/test_accuracy/', None))
            if acc is None:
                acc = parse_metric_value(metrics.get('test_accuracy/', None))
            if acc and acc > best_acc:
                best_acc = acc
                best_exp = exp_name
                best_metrics = metrics
        
        if best_exp:
            best_results[part] = {
                'experiment': best_exp,
                'accuracy': best_acc * 100,
                'metrics': best_metrics
            }
    
    if not best_results:
        print("No results to summarize.")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Best Results Summary by Part', fontsize=14, fontweight='bold')
    
    parts = list(best_results.keys())
    accuracies = [best_results[p]['accuracy'] for p in parts]
    exp_names = [best_results[p]['experiment'].split('/')[-1][:30] for p in parts]
    
    # Plot 1: Best Accuracy by Part
    ax1 = axes[0, 0]
    bars = ax1.bar(parts, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(parts)], alpha=0.7)
    ax1.set_ylabel('Overall Accuracy (%)', fontweight='bold')
    ax1.set_title('Best Overall Accuracy by Part', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])
    for bar, acc, exp in zip(bars, accuracies, exp_names):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 1, 
                f'{acc:.2f}%\n{exp}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2-4: Per-class metrics for best experiments
    metrics_to_plot = [
        ('Sensitivity', 'scalar/test_sensitivity/'),
        ('Specificity', 'scalar/test_specificity/'),
        ('Precision', 'scalar/test_precision/')
    ]
    
    for idx, (metric_name, metric_key_prefix) in enumerate(metrics_to_plot):
        ax = axes[(idx+1)//2, (idx+1)%2]
        x = np.arange(len(parts))
        width = 0.25
        
        for i, class_name in enumerate(CLASS_NAMES):
            values = []
            for part in parts:
                metric_key = f'{metric_key_prefix}{class_name}'
                val = parse_metric_value(best_results[part]['metrics'].get(f'scalar/{metric_key}', None))
                if val is None:
                    val = parse_metric_value(best_results[part]['metrics'].get(metric_key, None))
                values.append(val * 100 if val else 0)
            ax.bar(x + i*width, values, width, label=class_name, alpha=0.7)
        
        ax.set_xlabel('Part', fontweight='bold')
        ax.set_ylabel(f'{metric_name} (%)', fontweight='bold')
        ax.set_title(f'Best {metric_name} by Class and Part', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(parts)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'best_results_summary.png')
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    print("=" * 70)
    print("Extracting and Visualizing Experiment Results")
    print("=" * 70)
    
    # Find all experiments
    print("\n1. Finding experiments...")
    experiments = find_experiments('output')
    print(f"   Found {len(experiments)} experiments")
    
    # Extract metrics
    print("\n2. Extracting metrics...")
    all_results = extract_all_metrics(experiments)
    
    # Print summary
    print("\n3. Results Summary:")
    for part in ['Part 1', 'Part 2', 'Part 3', 'Part 4']:
        if part in all_results:
            print(f"   {part}: {len(all_results[part])} experiments")
    
    # Create summary table
    print("\n4. Creating summary table...")
    table_data = create_summary_table(all_results)
    output_dir = 'results_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    if PANDAS_AVAILABLE:
        csv_file = os.path.join(output_dir, 'results_summary.csv')
        table_data.to_csv(csv_file, index=False)
        print(f"   Saved: {csv_file}")
    else:
        # Write CSV manually
        csv_file = os.path.join(output_dir, 'results_summary.csv')
        if table_data:
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=table_data[0].keys())
                writer.writeheader()
                writer.writerows(table_data)
            print(f"   Saved: {csv_file}")
    
    # Save JSON for programmatic access
    json_file = os.path.join(output_dir, 'results_summary.json')
    # Convert to JSON-serializable format
    json_results = {}
    for part, part_data in all_results.items():
        json_results[part] = {}
        for exp_name, exp_data in part_data.items():
            json_results[part][exp_name] = {
                'metrics': {k: (float(v) if isinstance(v, (int, float, np.number)) else str(v)) 
                           for k, v in exp_data['metrics'].items() 
                           if not isinstance(v, np.ndarray)},
                'info': exp_data['info']
            }
            if 'confusion_matrix' in exp_data['metrics']:
                json_results[part][exp_name]['metrics']['confusion_matrix'] = \
                    exp_data['metrics']['confusion_matrix'].tolist()
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   Saved: {json_file}")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    plot_part_comparison(all_results, output_dir)
    plot_confusion_matrices(all_results, output_dir)
    plot_best_results_summary(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("Done! All results saved to 'results_visualization/' directory")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - results_summary.csv (spreadsheet)")
    print(f"  - results_summary.json (programmatic access)")
    print(f"  - part_*_comparison.png (detailed comparisons)")
    print(f"  - confusion_matrices.png (confusion matrices)")
    print(f"  - best_results_summary.png (best results overview)")


if __name__ == '__main__':
    main()

