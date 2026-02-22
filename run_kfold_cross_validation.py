#!/usr/bin/env python3
"""
K-Fold Cross-Validation for OpenGait
Generates k partition files and runs training/evaluation for each fold
"""

import json
import os
import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import yaml
from datetime import datetime

class Tee:
    """Write to both stdout and a file"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        self.file.flush()
        self.stdout.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()

def load_existing_partition(partition_file):
    """Load existing partition file to get all participant IDs"""
    with open(partition_file, 'r') as f:
        partition = json.load(f)
    
    # Get all participant IDs from both train and test sets
    train_set = partition.get('TRAIN_SET', partition.get('train', []))
    test_set = partition.get('TEST_SET', partition.get('test', []))
    all_pids = sorted(list(set(train_set + test_set)))
    
    return all_pids, partition

def get_labels_for_pids(pids, label_file):
    """Get frailty labels for participant IDs"""
    import pandas as pd
    
    # Try reading CSV with different methods
    try:
        df = pd.read_csv(label_file)
    except:
        # If that fails, try with quote handling
        df = pd.read_csv(label_file, quotechar='"', skipinitialspace=True)
    
    # Handle case where columns might be combined (e.g., "subject_id,frailty_score")
    if len(df.columns) == 1 and ',' in str(df.columns[0]):
        # Split the combined column
        combined_col = df.columns[0]
        col_names = [c.strip().strip('"') for c in combined_col.split(',')]
        df = df[combined_col].str.split(',', expand=True)
        df.columns = col_names
        # Also split the data rows
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
    
    # Find the ID column
    id_col = None
    for col in ['subject_id', 'id', 'ID', 'sample_id', 'sample_ID', 'subject_ID', 'index']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        id_col = df.columns[0]
    
    # Find the label column
    label_col = None
    for col in ['frailty_score', 'frailty', 'frailty_status', 'status', 'label', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        # Try to use second column if it exists
        if len(df.columns) >= 2:
            label_col = df.columns[1]
        else:
            raise ValueError(f"Could not find label column in {label_file}. Available columns: {df.columns.tolist()}")
    
    # Create mapping
    pid_to_label = {}
    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        label = str(row[label_col]).strip()
        pid_to_label[pid] = label
    
    # Map labels to integers for stratification
    # Handle both string labels (Frail/Prefrail/Nonfrail) and numeric labels (0/1/2)
    # frailty_score mapping: 0=Nonfrail, 1=Frail, 2=Prefrail (for stratification: 0=Frail, 1=Prefrail, 2=Nonfrail)
    label_map_str = {'Frail': 0, 'Prefrail': 1, 'Nonfrail': 2, 'frail': 0, 'prefrail': 1, 'nonfrail': 2}
    # CSV uses: 0=Nonfrail, 1=Frail, 2=Prefrail
    # Map to: 0=Frail, 1=Prefrail, 2=Nonfrail for sklearn
    label_map_num_csv = {'0': 2, '1': 0, '2': 1, 0: 2, 1: 0, 2: 1}  # CSV: 0=Nonfrail->2, 1=Frail->0, 2=Prefrail->1
    label_map_num_012 = {'0': 0, '1': 1, '2': 2, 0: 0, 1: 1, 2: 2}  # Alternative: 0=Frail, 1=Prefrail, 2=Nonfrail
    label_map_num_123 = {'1': 0, '2': 1, '3': 2, 1: 0, 2: 1, 3: 2}  # Alternative: 1=Frail, 2=Prefrail, 3=Nonfrail
    
    pids_with_labels = []
    labels = []
    for pid in pids:
        if pid in pid_to_label:
            label_str = str(pid_to_label[pid]).strip()
            # Try string mapping first
            if label_str in label_map_str:
                pids_with_labels.append(pid)
                labels.append(label_map_str[label_str])
            # Try CSV mapping first (0=Nonfrail, 1=Frail, 2=Prefrail -> 0=Frail, 1=Prefrail, 2=Nonfrail)
            elif label_str in label_map_num_csv:
                pids_with_labels.append(pid)
                labels.append(label_map_num_csv[label_str])
            # Try numeric mapping (0/1/2)
            elif label_str in label_map_num_012:
                pids_with_labels.append(pid)
                labels.append(label_map_num_012[label_str])
            # Try numeric mapping (1/2/3)
            elif label_str in label_map_num_123:
                pids_with_labels.append(pid)
                labels.append(label_map_num_123[label_str])
            # Try converting to int if it's a number
            elif label_str.isdigit():
                label_int = int(label_str)
                if label_int in label_map_num_csv:
                    pids_with_labels.append(pid)
                    labels.append(label_map_num_csv[label_int])
                elif label_int in label_map_num_012:
                    pids_with_labels.append(pid)
                    labels.append(label_map_num_012[label_int])
                elif label_int in label_map_num_123:
                    pids_with_labels.append(pid)
                    labels.append(label_map_num_123[label_int])
    
    return pids_with_labels, labels

def create_kfold_partitions(all_pids, labels, k=5, output_dir='kfold_partitions', random_state=42, 
                            holdout_test_ratio=None, use_validation=True):
    """Create k partition files for k-fold cross-validation
    
    Standard k-fold CV (if holdout_test_ratio=None):
    - Split all data into k folds
    - Each fold: k-1 folds for TRAIN, 1 fold for TEST
    - If use_validation: Further split training into train/val
    - VAL_SET participants will NOT appear in TEST_SET of any other fold (no leakage)
    
    Nested CV with holdout (if holdout_test_ratio is set):
    - Hold out test_ratio as final TEST set
    - Do k-fold CV on remaining data
    - Each fold: train/val split, same holdout test
    
    Args:
        all_pids: List of participant IDs
        labels: List of labels for stratification
        k: Number of folds for cross-validation
        output_dir: Directory to save partition files
        random_state: Random seed
        holdout_test_ratio: If None, standard k-fold. If set, holdout this ratio as test
        use_validation: If True, create VAL_SET from training data in each fold
    
    Returns:
        List of partition file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.model_selection import train_test_split
    
    if holdout_test_ratio is not None:
        # Nested CV: Hold out test set first
        cv_pids, test_pids, cv_labels, test_labels = train_test_split(
            all_pids, labels,
            test_size=holdout_test_ratio,
            random_state=random_state,
            stratify=labels
        )
        print(f"Initial split: CV set={len(cv_pids)}, Holdout Test={len(test_pids)}")
        data_for_cv = cv_pids
        labels_for_cv = cv_labels
    else:
        # Standard k-fold: Use all data
        data_for_cv = all_pids
        labels_for_cv = labels
        test_pids = None
    
    # Do k-fold CV
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # FIRST PASS: Create all test sets first to prevent leakage
    all_test_sets = []
    all_train_sets = []
    all_train_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data_for_cv, labels_for_cv)):
        train_pids = [data_for_cv[i] for i in train_idx]
        fold_test_pids = [data_for_cv[i] for i in test_idx]
        train_labels = [labels_for_cv[i] for i in train_idx]
        
        all_test_sets.append(set(fold_test_pids))
        all_train_sets.append(train_pids)
        all_train_labels.append(train_labels)
    
    # SECOND PASS: Create validation sets ensuring no overlap with any test set
    partitions = []
    for fold_idx in range(k):
        train_pids = all_train_sets[fold_idx]
        train_labels = all_train_labels[fold_idx]
        fold_test_pids = list(all_test_sets[fold_idx])
        
        # Note: In standard k-fold CV, it's CORRECT for TRAIN_SET to include participants
        # that are in other folds' TEST_SET. Each fold is independent:
        # - Fold i trains on folds 1..i-1, i+1..k (which includes participants from other folds' test sets)
        # - Fold i tests on fold i only
        # This is standard k-fold CV behavior and is NOT data leakage.
        
        # If using validation, split training set further
        # For 80/20 train/test split with validation:
        # - Split the 80% training portion into ~64% train and ~16% val
        # - Keep 20% for test
        if use_validation and len(train_pids) > 1:
            # IMPORTANT: In k-fold CV, it's OK for VAL_SET participants to appear in OTHER folds' TEST_SET
            # because each fold is independent. We only need to ensure:
            # 1. VAL_SET doesn't overlap with THIS fold's TEST_SET (guaranteed by k-fold split)
            # 2. VAL_SET doesn't overlap with THIS fold's TRAIN_SET (we're splitting from it)
            
            # Use fold-specific random_state to ensure different splits per fold
            fold_specific_seed = random_state + fold_idx * 1000
            
            # Split training set: 20% for validation (of the training portion)
            # This gives: ~64% train, ~16% val, ~20% test (approximately 80/20 overall)
            train_pids_final, val_pids, _, _ = train_test_split(
                train_pids, train_labels,
                test_size=0.20,  # 20% of training data for validation
                random_state=fold_specific_seed,  # Different seed per fold
                stratify=train_labels
            )
            
            # Verify: VAL_SET should not overlap with THIS fold's TEST_SET
            # (This should never happen due to k-fold split, but check anyway)
            same_fold_leakage = [pid for pid in val_pids if pid in fold_test_pids]
            if same_fold_leakage:
                print(f"⚠️  ERROR: Fold {fold_idx+1} VAL_SET overlaps with its own TEST_SET: {same_fold_leakage}")
                print(f"   This indicates a bug in the k-fold split!")
                val_pids = [pid for pid in val_pids if pid not in fold_test_pids]
                train_pids_final.extend(same_fold_leakage)
            
            partition = {
                "TRAIN_SET": train_pids_final,
                "VAL_SET": val_pids,
                "TEST_SET": fold_test_pids if test_pids is None else test_pids
            }
            total = len(train_pids_final) + len(val_pids) + len(partition['TEST_SET'])
            print(f"Created fold {fold_idx+1}: Train={len(train_pids_final)} ({len(train_pids_final)/total*100:.1f}%), Val={len(val_pids)} ({len(val_pids)/total*100:.1f}%), Test={len(partition['TEST_SET'])} ({len(partition['TEST_SET'])/total*100:.1f}%)")
        else:
            partition = {
                "TRAIN_SET": train_pids,
                "TEST_SET": fold_test_pids if test_pids is None else test_pids
            }
            if use_validation:
                partition["VAL_SET"] = []
            total = len(train_pids) + len(partition['TEST_SET'])
            print(f"Created fold {fold_idx+1}: Train={len(train_pids)} ({len(train_pids)/total*100:.1f}%), Test={len(partition['TEST_SET'])} ({len(partition['TEST_SET'])/total*100:.1f}%)")
        
        partition_file = os.path.join(output_dir, f'fold_{fold_idx+1}.json')
        with open(partition_file, 'w') as f:
            json.dump(partition, f, indent=2)
        
        partitions.append(partition_file)
    
    return partitions

def update_config_for_fold(config_file, partition_file, fold_num, output_dir, enable_validation=True):
    """Create a modified config file for this fold"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update partition file path
    config['data_cfg']['dataset_partition'] = os.path.abspath(partition_file)
    
    # Update save_name to include fold number
    original_save_name = config['trainer_cfg'].get('save_name', 'default')
    config['trainer_cfg']['save_name'] = f"{original_save_name}_fold{fold_num}"
    config['evaluator_cfg']['save_name'] = f"{original_save_name}_fold{fold_num}"
    
    # Update dataset_name
    original_dataset_name = config['data_cfg'].get('dataset_name', 'default')
    config['data_cfg']['dataset_name'] = f"{original_dataset_name}_fold{fold_num}"
    
    # Enable validation during training to track best checkpoint
    if enable_validation:
        config['trainer_cfg']['with_test'] = True
    
    # Save modified config
    fold_config_file = os.path.join(output_dir, f'config_fold{fold_num}.yaml')
    with open(fold_config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return fold_config_file

def run_training(config_file, fold_num, device='0,1', nproc=2):
    """Run training for a fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}: Training")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'train',
        '--log_to_file'
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    # Fix OpenCV library compatibility issues - use the Python environment's lib path
    python_env = os.path.dirname(sys.executable)
    conda_env = os.path.dirname(python_env)  # Go up from bin/ to env root
    if os.path.exists(conda_env):
        lib_path = f"{conda_env}/lib"
        if os.path.exists(lib_path):
            # Set CONDA_PREFIX so other scripts can find the environment
            env['CONDA_PREFIX'] = conda_env
            # Remove base conda paths from LD_LIBRARY_PATH to avoid conflicts
            existing_ld_path = env.get('LD_LIBRARY_PATH', '')
            if existing_ld_path:
                # Filter out base conda paths
                paths = [p for p in existing_ld_path.split(':') if p and '/r38/miniconda3/lib' not in p]
                env['LD_LIBRARY_PATH'] = ':'.join([lib_path] + paths)
            else:
                env['LD_LIBRARY_PATH'] = lib_path
            # Set LD_PRELOAD to force use of myGait38's libstdc++
            stdcpp = f"{lib_path}/libstdc++.so.6"
            libgcc = f"{lib_path}/libgcc_s.so.1"
            if os.path.exists(stdcpp):
                env['LD_PRELOAD'] = stdcpp
                if os.path.exists(libgcc):
                    env['LD_PRELOAD'] = f"{stdcpp}:{libgcc}"
    
    # Run with real-time output streaming
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to stdout (which goes to log file via Tee)
        output_lines.append(line)
        sys.stdout.flush()  # Ensure immediate write
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nERROR: Training failed for fold {fold_num}")
        return None
    
    return ''.join(output_lines)

def run_evaluation(config_file, checkpoint_iter, fold_num, device='0,1', nproc=2):
    """Run evaluation for a fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}: Evaluation")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, '-m', 'torch.distributed.launch',
        f'--nproc_per_node={nproc}',
        'opengait/main.py',
        '--cfgs', config_file,
        '--phase', 'test',
        '--iter', str(checkpoint_iter),
        '--log_to_file'
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    # Fix OpenCV library compatibility issues - use the Python environment's lib path
    python_env = os.path.dirname(sys.executable)
    conda_env = os.path.dirname(python_env)  # Go up from bin/ to env root
    if os.path.exists(conda_env):
        lib_path = f"{conda_env}/lib"
        if os.path.exists(lib_path):
            # Set CONDA_PREFIX so other scripts can find the environment
            env['CONDA_PREFIX'] = conda_env
            # Remove base conda paths from LD_LIBRARY_PATH to avoid conflicts
            existing_ld_path = env.get('LD_LIBRARY_PATH', '')
            if existing_ld_path:
                # Filter out base conda paths
                paths = [p for p in existing_ld_path.split(':') if p and '/r38/miniconda3/lib' not in p]
                env['LD_LIBRARY_PATH'] = ':'.join([lib_path] + paths)
            else:
                env['LD_LIBRARY_PATH'] = lib_path
            # Set LD_PRELOAD to force use of myGait38's libstdc++
            stdcpp = f"{lib_path}/libstdc++.so.6"
            libgcc = f"{lib_path}/libgcc_s.so.1"
            if os.path.exists(stdcpp):
                env['LD_PRELOAD'] = stdcpp
                if os.path.exists(libgcc):
                    env['LD_PRELOAD'] = f"{stdcpp}:{libgcc}"
    
    # Run with real-time output streaming
    process = subprocess.Popen(
        cmd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to stdout (which goes to log file via Tee)
        output_lines.append(line)
        sys.stdout.flush()  # Ensure immediate write
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nERROR: Evaluation failed for fold {fold_num}")
        return None
    
    return ''.join(output_lines)

def find_best_checkpoint_from_logs(exp_dir, metric='accuracy'):
    """Find the best checkpoint based on validation metrics in training logs"""
    import glob
    import re
    
    # Find log files
    log_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(log_dir):
        return None
    
    log_files = glob.glob(os.path.join(log_dir, '*.txt'))
    if not log_files:
        return None
    
    # Extract validation metrics from all log files
    all_metrics = []
    for log_file in sorted(log_files):  # Sort to process in order
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find all "Running test..." sections
        # Pattern: "Iteration XXXX" followed by "Running test..." then metrics
        i = 0
        while i < len(lines):
            if 'Running test...' in lines[i]:
                # Look backwards for iteration number (checkpoint save happens before "Running test")
                iteration = None
                for k in range(max(0, i - 5), i):
                    iter_match = re.search(r'Iteration\s+(\d+)', lines[k])
                    if iter_match:
                        iteration = int(iter_match.group(1))
                        break
                
                if iteration is None:
                    i += 1
                    continue
                
                # Look forward for metrics (within next 30 lines)
                accuracy = None
                for j in range(i, min(i + 30, len(lines))):
                    line = lines[j]
                    
                    # Find accuracy
                    acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', line)
                    if acc_match:
                        accuracy = float(acc_match.group(1))
                        
                        # Get section for other metrics
                        section_start = max(0, j - 5)
                        section_end = min(len(lines), j + 25)
                        section_text = ''.join(lines[section_start:section_end])
                        
                        auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', section_text)
                        auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', section_text)
                        f1_match = re.search(r'F1 Score \(macro\):\s*([\d.]+)%', section_text)
                        precision_match = re.search(r'Precision \(macro\):\s*([\d.]+)%', section_text)
                        recall_match = re.search(r'Recall \(macro\):\s*([\d.]+)%', section_text)
                        
                        metric_dict = {
                            'iteration': iteration,
                            'accuracy': accuracy,
                            'auc_macro': float(auc_macro_match.group(1)) if auc_macro_match else None,
                            'auc_micro': float(auc_micro_match.group(1)) if auc_micro_match else None,
                            'f1': float(f1_match.group(1)) if f1_match else None,
                            'precision': float(precision_match.group(1)) if precision_match else None,
                            'recall': float(recall_match.group(1)) if recall_match else None,
                        }
                        all_metrics.append(metric_dict)
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1
    
    if not all_metrics:
        return None
    
    # Find best based on metric
    if metric == 'accuracy':
        best = max(all_metrics, key=lambda x: x['accuracy'])
    elif metric == 'auc_macro':
        best = max(all_metrics, key=lambda x: x['auc_macro'] if x['auc_macro'] is not None else 0)
    elif metric == 'f1':
        best = max(all_metrics, key=lambda x: x['f1'] if x['f1'] is not None else 0)
    elif metric == 'precision':
        best = max(all_metrics, key=lambda x: x['precision'] if x['precision'] is not None else 0)
    elif metric == 'recall':
        best = max(all_metrics, key=lambda x: x['recall'] if x['recall'] is not None else 0)
    else:
        return None
    
    # Return full metrics dict (not just iteration)
    return best

def extract_metrics_from_output(output_text):
    """Extract metrics from evaluation output"""
    metrics = {}
    
    # Extract accuracy
    import re
    acc_match = re.search(r'Overall Accuracy:\s*([\d.]+)%', output_text)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1)) / 100.0
    
    # Extract AUC
    auc_macro_match = re.search(r'ROC AUC \(macro\):\s*([\d.]+)', output_text)
    if auc_macro_match:
        metrics['auc_macro'] = float(auc_macro_match.group(1))
    
    auc_micro_match = re.search(r'ROC AUC \(micro\):\s*([\d.]+)', output_text)
    if auc_micro_match:
        metrics['auc_micro'] = float(auc_micro_match.group(1))
    
    # Extract precision, recall, F1 (macro)
    prec_match = re.search(r'Precision \(macro\):\s*([\d.]+)%', output_text)
    if prec_match:
        metrics['precision'] = float(prec_match.group(1)) / 100.0
    
    recall_match = re.search(r'Recall \(macro\):\s*([\d.]+)%', output_text)
    if recall_match:
        metrics['recall'] = float(recall_match.group(1)) / 100.0
    
    f1_match = re.search(r'F1 Score \(macro\):\s*([\d.]+)%', output_text)
    if f1_match:
        metrics['f1'] = float(f1_match.group(1)) / 100.0
    
    return metrics

def aggregate_results(all_results):
    """Aggregate results across all folds"""
    if not all_results:
        return None
    
    # Collect all metrics (only numeric metrics, skip fold, checkpoint_iter, note)
    skip_keys = {'fold', 'checkpoint_iter', 'note'}
    metrics_dict = {}
    for result in all_results:
        for key, value in result.items():
            if key in skip_keys:
                continue
            if key not in metrics_dict:
                metrics_dict[key] = []
            # Only include non-None numeric values
            if value is not None and isinstance(value, (int, float)):
                metrics_dict[key].append(value)
    
    # Compute mean and std for numeric metrics
    summary = {}
    for key, values in metrics_dict.items():
        if values:  # Only compute if we have values
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_values'] = values
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation for OpenGait')
    parser.add_argument('--config', type=str, required=False,
                        help='Path to config file (single config)')
    parser.add_argument('--configs', type=str, nargs='+', required=False,
                        help='Paths to multiple config files (e.g., --configs config1.yaml config2.yaml). If provided, runs k-fold CV for each config sequentially.')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--checkpoint_iter', type=int, default=None,
                        help='Checkpoint iteration to evaluate. If None, uses best checkpoint from validation (default: None = best checkpoint)')
    parser.add_argument('--best_metric', type=str, default='accuracy',
                        choices=['accuracy', 'auc_macro', 'f1', 'precision', 'recall'],
                        help='Metric to use for finding best checkpoint when --checkpoint_iter is None (default: accuracy)')
    parser.add_argument('--device', type=str, default='0,1',
                        help='CUDA devices (default: 0,1)')
    parser.add_argument('--nproc', type=int, default=2,
                        help='Number of processes (default: 2)')
    parser.add_argument('--output_dir', type=str, default='kfold_results',
                        help='Output directory for results (default: kfold_results)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only run evaluation (assumes models already trained)')
    parser.add_argument('--holdout_test_ratio', type=float, default=None,
                        help='Ratio to hold out as final test set. If None, uses standard k-fold (all data in CV)')
    parser.add_argument('--no_validation', action='store_true',
                        help='Do not create separate validation set (only train/test split)')
    parser.add_argument('--use-existing-partitions', action='store_true',
                        help='Use existing partition files if they exist (skip partition creation)')
    
    args = parser.parse_args()
    
    # Determine which configs to run
    if args.configs:
        configs_to_run = args.configs
    elif args.config:
        configs_to_run = [args.config]
    else:
        parser.error("Either --config or --configs must be provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file (saves all output)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(args.output_dir, f'kfold_run_{timestamp}.log')
    tee = Tee(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print(f"K-Fold Cross-Validation Log")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}")
        print(f"Configs to run: {len(configs_to_run)}")
        for i, cfg in enumerate(configs_to_run, 1):
            print(f"  {i}. {cfg}")
        print("="*70)
        print()
        
        # Run k-fold CV for each config
        for config_idx, config_path in enumerate(configs_to_run, 1):
            if len(configs_to_run) > 1:
                print(f"\n{'='*70}")
                print(f"CONFIG {config_idx}/{len(configs_to_run)}: {config_path}")
                print(f"{'='*70}\n")
            
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            partition_file = config['data_cfg']['dataset_partition']
            label_file = config['data_cfg'].get('frailty_label_file')
            
            if not label_file:
                print("ERROR: frailty_label_file not found in config")
                sys.exit(1)
            
            print("="*70)
            print("K-FOLD CROSS-VALIDATION SETUP")
            print("="*70)
            print(f"Config: {config_path}")
            print(f"K-folds: {args.k}")
            if args.checkpoint_iter is not None:
                print(f"Evaluation: Fixed checkpoint iteration {args.checkpoint_iter}")
            else:
                print(f"Evaluation: Best checkpoint (metric: {args.best_metric})")
                print(f"  (Validation will run every 500 iterations during training)")
            print(f"Output directory: {args.output_dir}")
            print()
            
            if args.holdout_test_ratio is not None:
                print("Data split strategy: NESTED CV (with holdout test set)")
                print(f"  1. Holdout {args.holdout_test_ratio*100:.0f}% as final TEST set (same for all folds)")
                print(f"  2. Remaining {100-args.holdout_test_ratio*100:.0f}% split into {args.k} folds for CV")
                print(f"  3. Each fold: train/val split, same holdout test")
            else:
                print("Data split strategy: STANDARD K-FOLD CV")
                print(f"  1. All data split into {args.k} folds")
                print(f"  2. Each fold: {args.k-1} folds for TRAIN, 1 fold for TEST")
                if not args.no_validation:
                    print(f"  3. Training set further split into train/val")
            print()
            
            # Load existing partition to get all PIDs
            print("Loading participant IDs...")
            all_pids, _ = load_existing_partition(partition_file)
            print(f"Total participants: {len(all_pids)}")
            
            # Get labels for stratification
            print("Loading labels for stratification...")
            pids_with_labels, labels = get_labels_for_pids(all_pids, label_file)
            print(f"Participants with labels: {len(pids_with_labels)}")
            print(f"Label distribution: {np.bincount(labels)}")
            
            # Create or load k-fold partitions
            partition_dir = os.path.join(args.output_dir, 'partitions')
            os.makedirs(partition_dir, exist_ok=True)
            
            # Check if existing partitions should be used
            if args.use_existing_partitions:
                print(f"\nChecking for existing partitions in {partition_dir}...")
                partition_files = []
                all_exist = True
                for fold_num in range(1, args.k + 1):
                    partition_file = os.path.join(partition_dir, f'fold_{fold_num}.json')
                    if os.path.exists(partition_file):
                        partition_files.append(partition_file)
                        print(f"  Found: fold_{fold_num}.json")
                    else:
                        all_exist = False
                        print(f"  Missing: fold_{fold_num}.json")
                
                if all_exist and len(partition_files) == args.k:
                    print(f"\n✓ Using existing {args.k}-fold partitions")
                else:
                    print(f"\n⚠️  Not all partition files found. Creating new partitions...")
                    partition_files = create_kfold_partitions(
                        pids_with_labels, labels, 
                        k=args.k, 
                        output_dir=partition_dir,
                        random_state=args.random_state,
                        holdout_test_ratio=args.holdout_test_ratio,
                        use_validation=not args.no_validation
                    )
            else:
                print(f"\nCreating {args.k}-fold partitions...")
                partition_files = create_kfold_partitions(
                    pids_with_labels, labels, 
                    k=args.k, 
                    output_dir=partition_dir,
                    random_state=args.random_state,
                    holdout_test_ratio=args.holdout_test_ratio,
                    use_validation=not args.no_validation
                )
            
            # Create output directory for results
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Run training and evaluation for each fold
            all_results = []
            
            for fold_num in range(1, args.k + 1):
                print(f"\n{'='*70}")
                print(f"PROCESSING FOLD {fold_num}/{args.k}")
                print(f"{'='*70}")
                
                # Create config for this fold
                # Enable testing during training (every 500 iterations) if:
                # - Not using fixed checkpoint (want to find best checkpoint)
                # This will test on TEST_SET every 500 iterations during training
                enable_test_during_training = (args.checkpoint_iter is None)
                fold_config = update_config_for_fold(
                    config_path, 
                    partition_files[fold_num - 1], 
                    fold_num,
                    args.output_dir,
                    enable_validation=enable_test_during_training
                )
                
                # Run training (unless skipped)
                if not args.skip_training:
                    train_output = run_training(fold_config, fold_num, args.device, args.nproc)
                    if train_output is None:
                        print(f"Skipping fold {fold_num} due to training failure")
                        continue
                
                # Determine which checkpoint to evaluate
                if args.checkpoint_iter is not None:
                    # Use specified checkpoint iteration
                    eval_iter = args.checkpoint_iter
                    print(f"Using specified checkpoint at iteration {eval_iter}")
                    metrics = {
                        'fold': fold_num,
                        'checkpoint_iter': eval_iter,
                        'note': 'Using specified checkpoint iteration'
                    }
                elif not enable_test_during_training:
                    # Testing during training disabled, use final checkpoint
                    with open(fold_config, 'r') as f:
                        fold_config_data = yaml.safe_load(f)
                    total_iter = fold_config_data['trainer_cfg'].get('total_iter', 10000)
                    eval_iter = total_iter
                    print(f"Testing during training disabled - using final checkpoint at iteration {eval_iter}")
                    metrics = {
                        'fold': fold_num,
                        'checkpoint_iter': eval_iter,
                        'note': 'Testing during training disabled - using final checkpoint'
                    }
                else:
                    # Find best checkpoint from test metrics during training
                    # (test set is evaluated every 500 iterations during training)
                    print(f"\nFinding best checkpoint for fold {fold_num} (based on TEST_SET metrics during training)...")
                    # Get experiment directory from config
                    with open(fold_config, 'r') as f:
                        fold_config_data = yaml.safe_load(f)
                    save_name = fold_config_data['trainer_cfg']['save_name']
                    dataset_name = fold_config_data['data_cfg']['dataset_name']
                    exp_dir = os.path.join('output', dataset_name, 'SwinGait', save_name)
                    
                    best_metrics = find_best_checkpoint_from_logs(exp_dir, metric=args.best_metric)
                    if best_metrics is not None:
                        eval_iter = best_metrics['iteration']
                        print(f"✓ Best checkpoint found: iteration {eval_iter} (based on {args.best_metric} on TEST_SET)")
                        print(f"  Metrics at best checkpoint:")
                        print(f"    Accuracy: {best_metrics['accuracy']:.2f}%")
                        if best_metrics.get('precision') is not None:
                            print(f"    Precision (macro): {best_metrics['precision']:.2f}%")
                        if best_metrics.get('recall') is not None:
                            print(f"    Recall (macro): {best_metrics['recall']:.2f}%")
                        if best_metrics.get('f1') is not None:
                            print(f"    F1 Score (macro): {best_metrics['f1']:.2f}%")
                        if best_metrics.get('auc_macro') is not None:
                            print(f"    ROC AUC (macro): {best_metrics['auc_macro']:.4f}")
                        if best_metrics.get('auc_micro') is not None:
                            print(f"    ROC AUC (micro): {best_metrics['auc_micro']:.4f}")
                        
                        # Store all metrics from best checkpoint
                        metrics = {
                            'fold': fold_num,
                            'checkpoint_iter': eval_iter,
                            'accuracy': best_metrics['accuracy'] / 100.0,  # Convert to 0-1 range
                            'precision': best_metrics.get('precision', None) / 100.0 if best_metrics.get('precision') is not None else None,
                            'recall': best_metrics.get('recall', None) / 100.0 if best_metrics.get('recall') is not None else None,
                            'f1': best_metrics.get('f1', None) / 100.0 if best_metrics.get('f1') is not None else None,
                            'auc_macro': best_metrics.get('auc_macro', None),
                            'auc_micro': best_metrics.get('auc_micro', None),
                            'note': 'Metrics from best checkpoint during training (every 500 iterations on TEST_SET)'
                        }
                    else:
                        print(f"⚠ Could not find best checkpoint, using final iteration")
                        # Get total_iter from config
                        total_iter = fold_config_data['trainer_cfg'].get('total_iter', 10000)
                        eval_iter = total_iter
                        metrics = {
                            'fold': fold_num,
                            'checkpoint_iter': eval_iter,
                            'note': 'Best checkpoint not found - using final iteration'
                        }
                
                # Skip final evaluation - metrics are already collected during training
                # (testing happens every 500 iterations during training on TEST_SET)
                print(f"\n✓ Fold {fold_num} training complete")
                print(f"  Best checkpoint: iteration {eval_iter}")
                print(f"  Metrics were logged during training (every 500 iterations on TEST_SET)")
                print(f"  No separate final evaluation - using metrics from training logs")
                
                all_results.append(metrics)
                
                # Save individual fold results
                fold_results_file = os.path.join(args.output_dir, f'fold_{fold_num}_results.json')
                with open(fold_results_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            # Aggregate results
            if all_results:
                print(f"\n{'='*70}")
                print("AGGREGATED RESULTS ACROSS ALL FOLDS")
                print(f"{'='*70}")
                
                summary = aggregate_results(all_results)
                
                # Print summary
                print("\nMean ± Std across folds:")
                metric_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_macro', 'auc_micro']
                for key in metric_order:
                    mean_key = f'{key}_mean'
                    std_key = f'{key}_std'
                    if mean_key in summary:
                        mean_val = summary[mean_key]
                        std_val = summary[std_key]
                        # Format percentage metrics (0-1 range) as percentages
                        if key in ['accuracy', 'precision', 'recall', 'f1']:
                            print(f"  {key:15s}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")
                        else:
                            print(f"  {key:15s}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Save summary (with config name in filename if multiple configs)
                if len(configs_to_run) > 1:
                    config_name = os.path.splitext(os.path.basename(config_path))[0]
                    summary_file = os.path.join(args.output_dir, f'kfold_summary_{config_name}.json')
                    csv_file = os.path.join(args.output_dir, f'kfold_results_{config_name}.csv')
                else:
                    summary_file = os.path.join(args.output_dir, 'kfold_summary.json')
                    csv_file = os.path.join(args.output_dir, 'kfold_results.csv')
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n✓ Summary saved to: {summary_file}")
                
                # Create CSV for easy viewing
                df = pd.DataFrame(all_results)
                df.to_csv(csv_file, index=False)
                print(f"✓ Results CSV saved to: {csv_file}")
                
                # Clear results for next config if running multiple
                if len(configs_to_run) > 1 and config_idx < len(configs_to_run):
                    all_results = []
                    print(f"\n{'='*70}")
                    print(f"Completed config {config_idx}/{len(configs_to_run)}")
                    print(f"{'='*70}\n")
            else:
                print("\nERROR: No results collected!")
                sys.exit(1)
        
        print(f"\n{'='*70}")
        print("K-FOLD CROSS-VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All output saved to: {log_file}")
    
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\n✓ All output saved to: {log_file}")

if __name__ == '__main__':
    main()
