#!/usr/bin/env python3
"""
Script to analyze Part 1 experiment results and find the best performing configuration.
"""

import os
import glob
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_from_tensorboard(summary_dir):
    """Extract all metrics from TensorBoard event files."""
    try:
        ea = EventAccumulator(summary_dir)
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            iterations = [s.step for s in scalar_events]
            values = [s.value for s in scalar_events]
            metrics[tag] = {
                'iterations': iterations,
                'values': values
            }
        
        return metrics
    except Exception as e:
        print(f"Error reading TensorBoard from {summary_dir}: {e}")
        return {}

def find_part1_experiments(output_dir):
    """Find all Part 1 experiment directories."""
    experiments = {}
    
    # Look for Part 1 experiments
    part1_patterns = [
        '*pt1*',
        '*part1*',
        '*Part1*',
        '*PART1*'
    ]
    
    for pattern in part1_patterns:
        for dataset_dir in glob.glob(os.path.join(output_dir, pattern)):
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
                checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
                
                # Only include experiments that start with "REDO"
                if os.path.isdir(summary_dir) and exp_name.startswith('REDO'):
                    experiments[exp_name] = {
                        'summary_dir': summary_dir,
                        'checkpoints_dir': checkpoints_dir,
                        'full_path': exp_dir,
                        'dataset_name': dataset_name
                    }
    
    return experiments

def get_best_checkpoint(checkpoints_dir):
    """Get the checkpoint with the highest iteration number."""
    if not os.path.isdir(checkpoints_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
    if not checkpoint_files:
        return None
    
    # Extract iteration numbers and find the highest
    best_checkpoint = None
    best_iter = -1
    
    for ckpt_file in checkpoint_files:
        # Extract iteration number from filename (e.g., "name-01000.pt" -> 1000)
        basename = os.path.basename(ckpt_file)
        try:
            # Find the number before .pt
            iter_str = basename.split('-')[-1].replace('.pt', '')
            iter_num = int(iter_str)
            if iter_num > best_iter:
                best_iter = iter_num
                best_checkpoint = ckpt_file
        except:
            continue
    
    return best_checkpoint, best_iter

def main():
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found!")
        return
    
    print("="*80)
    print("ANALYZING PART 1 EXPERIMENTS (REDO ONLY)")
    print("="*80)
    print("\nFinding Part 1 REDO experiments...")
    
    experiments = find_part1_experiments(output_dir)
    
    if len(experiments) == 0:
        print("No Part 1 REDO experiments found!")
        return
    
    print(f"Found {len(experiments)} Part 1 REDO experiments:\n")
    
    results = []
    
    for exp_name, paths in experiments.items():
        print(f"Processing: {exp_name}")
        print(f"  Path: {paths['full_path']}")
        
        # Extract metrics from TensorBoard
        metrics = extract_from_tensorboard(paths['summary_dir'])
        
        if not metrics:
            print(f"  ⚠️  No metrics found in TensorBoard")
            continue
        
        # Find test accuracy metrics
        test_acc_metrics = {k: v for k, v in metrics.items() if 'test_accuracy' in k.lower()}
        
        if not test_acc_metrics:
            print(f"  ⚠️  No test accuracy metrics found")
            continue
        
        # Get best checkpoint
        best_ckpt, best_iter = get_best_checkpoint(paths['checkpoints_dir'])
        
        # Extract best accuracies for each metric
        exp_result = {
            'name': exp_name,
            'dataset': paths['dataset_name'],
            'path': paths['full_path'],
            'best_checkpoint': best_ckpt,
            'best_iteration': best_iter,
            'metrics': {}
        }
        
        for metric_name, metric_data in test_acc_metrics.items():
            if len(metric_data['values']) > 0:
                max_acc = max(metric_data['values'])
                max_iter = metric_data['iterations'][metric_data['values'].index(max_acc)]
                final_acc = metric_data['values'][-1]
                final_iter = metric_data['iterations'][-1]
                
                exp_result['metrics'][metric_name] = {
                    'max_acc': max_acc,
                    'max_iter': max_iter,
                    'final_acc': final_acc,
                    'final_iter': final_iter
                }
        
        results.append(exp_result)
        
        # Print summary for this experiment
        print(f"  Best checkpoint: {best_iter} iterations")
        for metric_name, metric_info in exp_result['metrics'].items():
            print(f"  {metric_name}:")
            print(f"    Max: {metric_info['max_acc']:.2f}% @ iter {metric_info['max_iter']}")
            print(f"    Final: {metric_info['final_acc']:.2f}% @ iter {metric_info['final_iter']}")
        print()
    
    # Find the best performing experiment
    print("="*80)
    print("SUMMARY - BEST PERFORMING CONFIGURATIONS")
    print("="*80)
    
    if not results:
        print("No results to summarize!")
        return
    
    # Group by metric type and find best
    all_metric_names = set()
    for result in results:
        all_metric_names.update(result['metrics'].keys())
    
    for metric_name in sorted(all_metric_names):
        print(f"\n📊 Metric: {metric_name}")
        print("-" * 80)
        
        # Find best for this metric
        best_exp = None
        best_max_acc = -1
        
        metric_results = []
        for result in results:
            if metric_name in result['metrics']:
                metric_info = result['metrics'][metric_name]
                metric_results.append({
                    'name': result['name'],
                    'max_acc': metric_info['max_acc'],
                    'max_iter': metric_info['max_iter'],
                    'final_acc': metric_info['final_acc'],
                    'path': result['path'],
                    'checkpoint': result['best_checkpoint']
                })
                
                if metric_info['max_acc'] > best_max_acc:
                    best_max_acc = metric_info['max_acc']
                    best_exp = result
        
        # Sort by max accuracy
        metric_results.sort(key=lambda x: x['max_acc'], reverse=True)
        
        # Print top 3
        print(f"{'Rank':<6} {'Experiment':<40} {'Max Acc':<12} {'Final Acc':<12} {'Iteration':<12}")
        print("-" * 80)
        for i, res in enumerate(metric_results[:5], 1):
            rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{rank_symbol:<6} {res['name']:<40} {res['max_acc']:>10.2f}%  {res['final_acc']:>10.2f}%  {res['max_iter']:>10}")
        
        if best_exp:
            print(f"\n🏆 BEST: {best_exp['name']}")
            print(f"   Max Accuracy: {best_max_acc:.2f}% @ iteration {best_exp['metrics'][metric_name]['max_iter']}")
            print(f"   Checkpoint: {best_exp['best_checkpoint']}")
            print(f"   Full Path: {best_exp['path']}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR PART 2")
    print("="*80)
    
    # Find overall best (using the most common metric or Rank-1)
    rank1_metric = None
    for metric_name in sorted(all_metric_names):
        if 'rank' in metric_name.lower() and '1' in metric_name:
            rank1_metric = metric_name
            break
    
    if not rank1_metric and all_metric_names:
        # Use first available metric
        rank1_metric = sorted(all_metric_names)[0]
    
    if rank1_metric:
        # Find best by MAX accuracy
        best_max_for_part2 = None
        best_max_acc = -1
        for result in results:
            if rank1_metric in result['metrics']:
                acc = result['metrics'][rank1_metric]['max_acc']
                if acc > best_max_acc:
                    best_max_acc = acc
                    best_max_for_part2 = result
        
        # Find best by FINAL/CONSISTENT accuracy
        best_final_for_part2 = None
        best_final_acc = -1
        for result in results:
            if rank1_metric in result['metrics']:
                acc = result['metrics'][rank1_metric]['final_acc']
                if acc > best_final_acc:
                    best_final_acc = acc
                    best_final_for_part2 = result
        
        # Find best by STABILITY (smallest drop from max to final)
        best_stable_for_part2 = None
        best_stability = float('inf')
        for result in results:
            if rank1_metric in result['metrics']:
                max_acc = result['metrics'][rank1_metric]['max_acc']
                final_acc = result['metrics'][rank1_metric]['final_acc']
                drop = max_acc - final_acc
                # Prefer higher final accuracy, but penalize large drops
                stability_score = drop - (final_acc * 0.01)  # Lower is better
                if stability_score < best_stability:
                    best_stability = stability_score
                    best_stable_for_part2 = result
        
        print(f"\n📊 Analysis based on: {rank1_metric}")
        print("-" * 80)
        
        if best_max_for_part2:
            max_info = best_max_for_part2['metrics'][rank1_metric]
            print(f"\n🥇 BEST BY MAX ACCURACY:")
            print(f"   Experiment: {best_max_for_part2['name']}")
            print(f"   Max Accuracy: {max_info['max_acc']:.2f}% @ iter {max_info['max_iter']}")
            print(f"   Final Accuracy: {max_info['final_acc']:.2f}% @ iter {max_info['final_iter']}")
            drop = max_info['max_acc'] - max_info['final_acc']
            if drop > 0.01:
                print(f"   ⚠️  Dropped by {drop:.2f}% (possible overfitting)")
            print(f"   Checkpoint: {best_max_for_part2['best_checkpoint']}")
        
        if best_final_for_part2:
            final_info = best_final_for_part2['metrics'][rank1_metric]
            print(f"\n✅ BEST BY FINAL/CONSISTENT ACCURACY (RECOMMENDED):")
            print(f"   Experiment: {best_final_for_part2['name']}")
            print(f"   Max Accuracy: {final_info['max_acc']:.2f}% @ iter {final_info['max_iter']}")
            print(f"   Final Accuracy: {final_info['final_acc']:.2f}% @ iter {final_info['final_iter']}")
            drop = final_info['max_acc'] - final_info['final_acc']
            if abs(drop) < 0.01:
                print(f"   ✓ Stable performance (no significant drop)")
            print(f"   Checkpoint: {best_final_for_part2['best_checkpoint']}")
            print(f"   Full path: {best_final_for_part2['path']}")
        
        if best_stable_for_part2 and best_stable_for_part2['name'] != best_final_for_part2['name']:
            stable_info = best_stable_for_part2['metrics'][rank1_metric]
            print(f"\n⚖️  BEST BY STABILITY (smallest drop):")
            print(f"   Experiment: {best_stable_for_part2['name']}")
            print(f"   Max Accuracy: {stable_info['max_acc']:.2f}% @ iter {stable_info['max_iter']}")
            print(f"   Final Accuracy: {stable_info['final_acc']:.2f}% @ iter {stable_info['final_iter']}")
            drop = stable_info['max_acc'] - stable_info['final_acc']
            print(f"   Drop: {drop:.2f}%")
            print(f"   Checkpoint: {best_stable_for_part2['best_checkpoint']}")
        
        print(f"\n💡 RECOMMENDATION:")
        if best_final_for_part2:
            final_info = best_final_for_part2['metrics'][rank1_metric]
            print(f"   Use: {best_final_for_part2['name']} (Final: {final_info['final_acc']:.2f}%)")
            print(f"   Reason: Consistent performance is better for Part 2 transfer learning")
            print(f"\n   To use this checkpoint in Part 2, update your config:")
            print(f"   trainer_cfg:")
            print(f"     restore_hint: {best_final_for_part2['best_iteration']}")
            print(f"   Or use the checkpoint path directly in your training script.")

if __name__ == '__main__':
    main()

