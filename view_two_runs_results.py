#!/usr/bin/env python3
"""
View results for the two specific training runs:
1. swin_part1_pretrained_unfrozen.yaml
2. swin_part4a_B3_unfrozen_cnn.yaml

Usage:
    python view_two_runs_results.py
"""

import os
import sys
import re
from datetime import datetime

# Paths for the two runs
RUN1_PATH = "output/REDO_Frailty_ccpg_pt1_pretrained(UF)/SwinGait/REDO_Frailty_ccpg_pt1_pretrained(UF)"
RUN2_PATH = "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn"

RUN1_NAME = "Part 1 (Pretrained Unfrozen)"
RUN2_NAME = "Part 4a B3 (Unfrozen CNN)"

def extract_from_log_file(log_file):
    """Extract all evaluation metrics from a log file."""
    results = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Split by evaluation sections
        eval_sections = re.split(r'EVALUATION RESULTS', content)
        
        for i, section in enumerate(eval_sections[1:], 1):
            # Find iteration number before this section
            # Look for "Running test..." which appears right before EVALUATION RESULTS
            section_start = content.find(section)
            before_idx = max(0, section_start - 500)  # Look back 500 chars (enough for one log line)
            before_text = content[before_idx:section_start]
            
            # First try to find "Running test..." and get iteration right before it
            running_test_match = re.search(r'Iteration\s+0*(\d+).*?Running test', before_text, re.DOTALL)
            if running_test_match:
                iter_num = int(running_test_match.group(1))
            else:
                # Fallback: find the last iteration number before this section
                iter_matches = list(re.finditer(r'Iteration\s+0*(\d+)', before_text))
                if iter_matches:
                    iter_num = int(iter_matches[-1].group(1))  # Use the last (most recent) iteration
                else:
                    continue
            
            # Extract overall accuracy
            acc_match = re.search(r'Overall Accuracy:\s*(\d+\.?\d*)%', section)
            if not acc_match:
                continue
            
            accuracy = float(acc_match.group(1)) / 100.0
            
            # Extract ROC AUC
            auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', section)
            auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', section)
            
            auc_macro = float(auc_macro_match.group(1)) if auc_macro_match else None
            auc_micro = float(auc_micro_match.group(1)) if auc_micro_match else None
            
            # Extract per-class metrics
            frail_recall = re.search(r'Frail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            frail_prec = re.search(r'Frail Precision:\s*(\d+\.?\d*)%', section)
            frail_spec = re.search(r'Frail Specificity:\s*(\d+\.?\d*)%', section)
            
            prefrail_recall = re.search(r'Prefrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            prefrail_prec = re.search(r'Prefrail Precision:\s*(\d+\.?\d*)%', section)
            prefrail_spec = re.search(r'Prefrail Specificity:\s*(\d+\.?\d*)%', section)
            
            nonfrail_recall = re.search(r'Nonfrail Sensitivity \(Recall\):\s*(\d+\.?\d*)%', section)
            nonfrail_prec = re.search(r'Nonfrail Precision:\s*(\d+\.?\d*)%', section)
            nonfrail_spec = re.search(r'Nonfrail Specificity:\s*(\d+\.?\d*)%', section)
            
            # Extract confusion matrix
            cm_match = re.search(r'Confusion Matrix:\s*\[\[(\d+)\s+(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\s+(\d+)\]\]', section)
            
            result = {
                'iteration': iter_num,
                'accuracy': accuracy,
                'auc_macro': auc_macro,
                'auc_micro': auc_micro,
            }
            
            if all([frail_recall, frail_prec, prefrail_recall, prefrail_prec, nonfrail_recall, nonfrail_prec]):
                frail_recall_val = float(frail_recall.group(1)) / 100.0
                frail_prec_val = float(frail_prec.group(1)) / 100.0
                frail_f1 = 2 * (frail_prec_val * frail_recall_val) / (frail_prec_val + frail_recall_val) if (frail_prec_val + frail_recall_val) > 0 else 0.0
                
                prefrail_recall_val = float(prefrail_recall.group(1)) / 100.0
                prefrail_prec_val = float(prefrail_prec.group(1)) / 100.0
                prefrail_f1 = 2 * (prefrail_prec_val * prefrail_recall_val) / (prefrail_prec_val + prefrail_recall_val) if (prefrail_prec_val + prefrail_recall_val) > 0 else 0.0
                
                nonfrail_recall_val = float(nonfrail_recall.group(1)) / 100.0
                nonfrail_prec_val = float(nonfrail_prec.group(1)) / 100.0
                nonfrail_f1 = 2 * (nonfrail_prec_val * nonfrail_recall_val) / (nonfrail_prec_val + nonfrail_recall_val) if (nonfrail_prec_val + nonfrail_recall_val) > 0 else 0.0
                
                result.update({
                    'frail': {'recall': frail_recall_val, 'precision': frail_prec_val, 'f1': frail_f1},
                    'prefrail': {'recall': prefrail_recall_val, 'precision': prefrail_prec_val, 'f1': prefrail_f1},
                    'nonfrail': {'recall': nonfrail_recall_val, 'precision': nonfrail_prec_val, 'f1': nonfrail_f1},
                    'macro_f1': (frail_f1 + prefrail_f1 + nonfrail_f1) / 3.0,
                    'macro_precision': (frail_prec_val + prefrail_prec_val + nonfrail_prec_val) / 3.0,
                    'macro_recall': (frail_recall_val + prefrail_recall_val + nonfrail_recall_val) / 3.0,
                })
            
            if frail_spec and prefrail_spec and nonfrail_spec:
                result['frail_spec'] = float(frail_spec.group(1)) / 100.0
                result['prefrail_spec'] = float(prefrail_spec.group(1)) / 100.0
                result['nonfrail_spec'] = float(nonfrail_spec.group(1)) / 100.0
            
            if cm_match:
                result['confusion_matrix'] = [
                    [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3))],
                    [int(cm_match.group(4)), int(cm_match.group(5)), int(cm_match.group(6))],
                    [int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9))]
                ]
            
            results.append(result)
    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return []
    
    return sorted(results, key=lambda x: x['iteration'])

def get_latest_log_file(exp_path):
    """Get the most recent log file."""
    logs_dir = os.path.join(exp_path, 'logs')
    if not os.path.exists(logs_dir):
        return None
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]
    if not log_files:
        return None
    
    # Sort by modification time
    log_files.sort(key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)))
    return os.path.join(logs_dir, log_files[-1])

def print_comparison(run1_results, run2_results):
    """Print side-by-side comparison of results."""
    print("\n" + "="*120)
    print("COMPARISON: Part 1 vs Part 4a B3")
    print("="*120)
    
    # Get all unique iterations
    all_iters = sorted(set([r['iteration'] for r in run1_results] + [r['iteration'] for r in run2_results]))
    
    if not all_iters:
        print("No evaluation results found!")
        return
    
    # Create lookup dictionaries
    run1_dict = {r['iteration']: r for r in run1_results}
    run2_dict = {r['iteration']: r for r in run2_results}
    
    # Print header
    print(f"\n{'Iter':<8} {'Run 1 Acc':<12} {'Run 1 F1':<12} {'Run 1 AUC':<12} {'Run 2 Acc':<12} {'Run 2 F1':<12} {'Run 2 AUC':<12} {'Diff Acc':<12}")
    print("-"*120)
    
    for iter_num in all_iters:
        r1 = run1_dict.get(iter_num, {})
        r2 = run2_dict.get(iter_num, {})
        
        r1_acc = f"{r1.get('accuracy', 0)*100:.2f}%" if r1 else "N/A"
        r1_f1 = f"{r1.get('macro_f1', 0):.4f}" if r1 and 'macro_f1' in r1 else "N/A"
        r1_auc = f"{r1.get('auc_macro', 0):.4f}" if r1 and r1.get('auc_macro') else "N/A"
        
        r2_acc = f"{r2.get('accuracy', 0)*100:.2f}%" if r2 else "N/A"
        r2_f1 = f"{r2.get('macro_f1', 0):.4f}" if r2 and 'macro_f1' in r2 else "N/A"
        r2_auc = f"{r2.get('auc_macro', 0):.4f}" if r2 and r2.get('auc_macro') else "N/A"
        
        diff_acc = "N/A"
        if r1 and r2 and 'accuracy' in r1 and 'accuracy' in r2:
            diff = (r2['accuracy'] - r1['accuracy']) * 100
            diff_acc = f"{diff:+.2f}%"
        
        print(f"{iter_num:<8} {r1_acc:<12} {r1_f1:<12} {r1_auc:<12} {r2_acc:<12} {r2_f1:<12} {r2_auc:<12} {diff_acc:<12}")
    
    # Print best results
    print("\n" + "="*120)
    print("BEST OVERALL ACCURACY (with corresponding F1, Recall, Precision)")
    print("="*120)
    
    if run1_results:
        best_r1 = max(run1_results, key=lambda x: x.get('accuracy', 0))
        print(f"\n{RUN1_NAME}:")
        print(f"  Best Accuracy: {best_r1.get('accuracy', 0)*100:.2f}% (iter {best_r1['iteration']})")
        if 'macro_f1' in best_r1:
            print(f"  ✅ Macro F1: {best_r1['macro_f1']:.4f}")
            print(f"  ✅ Macro Precision: {best_r1['macro_precision']:.4f}")
            print(f"  ✅ Macro Recall: {best_r1['macro_recall']:.4f}")
        else:
            print(f"  ⚠️  F1, Precision, Recall metrics not available for this iteration")
        if best_r1.get('auc_macro'):
            print(f"  ROC AUC (macro): {best_r1['auc_macro']:.4f}")
        if 'confusion_matrix' in best_r1:
            print(f"  Confusion Matrix:")
            for row in best_r1['confusion_matrix']:
                print(f"    {row}")
    
    if run2_results:
        best_r2 = max(run2_results, key=lambda x: x.get('accuracy', 0))
        print(f"\n{RUN2_NAME}:")
        print(f"  Best Accuracy: {best_r2.get('accuracy', 0)*100:.2f}% (iter {best_r2['iteration']})")
        if 'macro_f1' in best_r2:
            print(f"  ✅ Macro F1: {best_r2['macro_f1']:.4f}")
            print(f"  ✅ Macro Precision: {best_r2['macro_precision']:.4f}")
            print(f"  ✅ Macro Recall: {best_r2['macro_recall']:.4f}")
        else:
            print(f"  ⚠️  F1, Precision, Recall metrics not available for this iteration")
        if best_r2.get('auc_macro'):
            print(f"  ROC AUC (macro): {best_r2['auc_macro']:.4f}")
        if 'confusion_matrix' in best_r2:
            print(f"  Confusion Matrix:")
            for row in best_r2['confusion_matrix']:
                print(f"    {row}")
    
    print("="*120)
    
    # Also show best F1, Precision, Recall separately
    print("\n" + "="*120)
    print("BEST METRICS BY INDIVIDUAL METRIC")
    print("="*120)
    
    if run1_results:
        results_with_metrics = [r for r in run1_results if 'macro_f1' in r]
        if results_with_metrics:
            best_f1_r1 = max(results_with_metrics, key=lambda x: x['macro_f1'])
            best_prec_r1 = max(results_with_metrics, key=lambda x: x['macro_precision'])
            best_recall_r1 = max(results_with_metrics, key=lambda x: x['macro_recall'])
            
            print(f"\n{RUN1_NAME}:")
            print(f"  Best F1: {best_f1_r1['macro_f1']:.4f} (iter {best_f1_r1['iteration']}, Acc: {best_f1_r1['accuracy']*100:.2f}%)")
            print(f"  Best Precision: {best_prec_r1['macro_precision']:.4f} (iter {best_prec_r1['iteration']}, Acc: {best_prec_r1['accuracy']*100:.2f}%)")
            print(f"  Best Recall: {best_recall_r1['macro_recall']:.4f} (iter {best_recall_r1['iteration']}, Acc: {best_recall_r1['accuracy']*100:.2f}%)")
    
    if run2_results:
        results_with_metrics = [r for r in run2_results if 'macro_f1' in r]
        if results_with_metrics:
            best_f1_r2 = max(results_with_metrics, key=lambda x: x['macro_f1'])
            best_prec_r2 = max(results_with_metrics, key=lambda x: x['macro_precision'])
            best_recall_r2 = max(results_with_metrics, key=lambda x: x['macro_recall'])
            
            print(f"\n{RUN2_NAME}:")
            print(f"  Best F1: {best_f1_r2['macro_f1']:.4f} (iter {best_f1_r2['iteration']}, Acc: {best_f1_r2['accuracy']*100:.2f}%)")
            print(f"  Best Precision: {best_prec_r2['macro_precision']:.4f} (iter {best_prec_r2['iteration']}, Acc: {best_prec_r2['accuracy']*100:.2f}%)")
            print(f"  Best Recall: {best_recall_r2['macro_recall']:.4f} (iter {best_recall_r2['iteration']}, Acc: {best_recall_r2['accuracy']*100:.2f}%)")
    
    print("="*120)

def main():
    print("="*120)
    print("VIEWING RESULTS FOR TWO TRAINING RUNS")
    print("="*120)
    
    # Check if paths exist
    if not os.path.exists(RUN1_PATH):
        print(f"\n❌ Run 1 path not found: {RUN1_PATH}")
        print("   Make sure training has completed and logs were generated.")
    else:
        print(f"\n✓ Found Run 1: {RUN1_PATH}")
    
    if not os.path.exists(RUN2_PATH):
        print(f"\n❌ Run 2 path not found: {RUN2_PATH}")
        print("   Make sure training has completed and logs were generated.")
    else:
        print(f"\n✓ Found Run 2: {RUN2_PATH}")
    
    # Extract results
    run1_results = []
    run2_results = []
    
    if os.path.exists(RUN1_PATH):
        log_file1 = get_latest_log_file(RUN1_PATH)
        if log_file1:
            print(f"\n📄 Reading Run 1 log: {os.path.basename(log_file1)}")
            run1_results = extract_from_log_file(log_file1)
            print(f"   Found {len(run1_results)} evaluation(s)")
        else:
            print(f"\n⚠️  No log files found for Run 1")
    
    if os.path.exists(RUN2_PATH):
        log_file2 = get_latest_log_file(RUN2_PATH)
        if log_file2:
            print(f"\n📄 Reading Run 2 log: {os.path.basename(log_file2)}")
            run2_results = extract_from_log_file(log_file2)
            print(f"   Found {len(run2_results)} evaluation(s)")
        else:
            print(f"\n⚠️  No log files found for Run 2")
    
    # Print comparison
    if run1_results or run2_results:
        print_comparison(run1_results, run2_results)
    else:
        print("\n❌ No results found for either run!")
        print("\nMake sure:")
        print("  1. Training has completed")
        print("  2. You used --log_to_file flag during training")
        print("  3. Log files exist in the output directories")
    
    # Print detailed per-class metrics for best results
    if run1_results or run2_results:
        print("\n" + "="*120)
        print("DETAILED PER-CLASS METRICS (Best Iteration)")
        print("="*120)
        
        if run1_results:
            best_r1 = max(run1_results, key=lambda x: x.get('accuracy', 0))
            print(f"\n{RUN1_NAME} - Iteration {best_r1['iteration']}:")
            if 'frail' in best_r1:
                print(f"  Frail:      Recall={best_r1['frail']['recall']:.4f}, Precision={best_r1['frail']['precision']:.4f}, F1={best_r1['frail']['f1']:.4f}")
                print(f"  Prefrail:   Recall={best_r1['prefrail']['recall']:.4f}, Precision={best_r1['prefrail']['precision']:.4f}, F1={best_r1['prefrail']['f1']:.4f}")
                print(f"  Nonfrail:   Recall={best_r1['nonfrail']['recall']:.4f}, Precision={best_r1['nonfrail']['precision']:.4f}, F1={best_r1['nonfrail']['f1']:.4f}")
        
        if run2_results:
            best_r2 = max(run2_results, key=lambda x: x.get('accuracy', 0))
            print(f"\n{RUN2_NAME} - Iteration {best_r2['iteration']}:")
            if 'frail' in best_r2:
                print(f"  Frail:      Recall={best_r2['frail']['recall']:.4f}, Precision={best_r2['frail']['precision']:.4f}, F1={best_r2['frail']['f1']:.4f}")
                print(f"  Prefrail:   Recall={best_r2['prefrail']['recall']:.4f}, Precision={best_r2['prefrail']['precision']:.4f}, F1={best_r2['prefrail']['f1']:.4f}")
                print(f"  Nonfrail:   Recall={best_r2['nonfrail']['recall']:.4f}, Precision={best_r2['nonfrail']['precision']:.4f}, F1={best_r2['nonfrail']['f1']:.4f}")
        
        print("="*120)

if __name__ == '__main__':
    main()
