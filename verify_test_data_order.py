#!/usr/bin/env python3
"""
Script to verify that test data is loaded in the same order across runs.
This helps ensure evaluation uses the same data samples.
"""

import os
import sys
import json
import pickle
from pathlib import Path

def load_dataset_info(output_dir):
    """Load dataset information from a training run."""
    # Try to find log files or dataset info
    log_dir = Path(output_dir) / 'logs'
    if not log_dir.exists():
        return None
    
    # Look for most recent log file
    log_files = sorted(log_dir.glob('*.txt'), key=os.path.getmtime, reverse=True)
    if not log_files:
        return None
    
    # Try to extract dataset info from logs or find dataset partition file
    dataset_info = {
        'log_file': str(log_files[0]),
        'test_pids': None,
        'test_order': None
    }
    
    return dataset_info

def get_test_pids_from_config(config_path):
    """Extract test PIDs from config or dataset partition file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset partition file
    dataset_partition = config.get('data_cfg', {}).get('dataset_partition', '')
    if not dataset_partition or not os.path.exists(dataset_partition):
        return None
    
    # Load partition file
    with open(dataset_partition, 'r') as f:
        partition = json.load(f)
    
    test_set = partition.get('test', [])
    return sorted(test_set) if test_set else None

def compare_test_orders(config1_path, config2_path):
    """Compare test data order between two configs."""
    print("="*70)
    print("TEST DATA ORDER VERIFICATION")
    print("="*70)
    
    print(f"\n📂 Loading configs...")
    print(f"   Config 1: {config1_path}")
    print(f"   Config 2: {config2_path}")
    
    test_pids1 = get_test_pids_from_config(config1_path)
    test_pids2 = get_test_pids_from_config(config2_path)
    
    if test_pids1 is None or test_pids2 is None:
        print("❌ Could not extract test PIDs from configs")
        return False
    
    print(f"\n📊 Test PIDs:")
    print(f"   Run 1: {len(test_pids1)} test samples")
    print(f"   Run 2: {len(test_pids2)} test samples")
    
    if len(test_pids1) != len(test_pids2):
        print(f"❌ Different number of test samples!")
        return False
    
    if test_pids1 == test_pids2:
        print(f"✅ Test PIDs are IDENTICAL and in SAME ORDER")
        return True
    else:
        print(f"❌ Test PIDs differ!")
        
        # Find differences
        set1 = set(test_pids1)
        set2 = set(test_pids2)
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        
        if only_in_1:
            print(f"   Only in Run 1: {only_in_1}")
        if only_in_2:
            print(f"   Only in Run 2: {only_in_2}")
        
        # Check if same set, different order
        if set1 == set2:
            print(f"   ⚠️  Same PIDs but different order!")
            # Find first difference
            for i, (p1, p2) in enumerate(zip(test_pids1, test_pids2)):
                if p1 != p2:
                    print(f"   First difference at index {i}: {p1} vs {p2}")
                    break
        
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify test data order between configs')
    parser.add_argument('--config1', type=str, required=True, help='Path to first config file')
    parser.add_argument('--config2', type=str, required=True, help='Path to second config file')
    
    args = parser.parse_args()
    
    match = compare_test_orders(args.config1, args.config2)
    
    if match:
        print(f"\n✅ Test data order is consistent!")
    else:
        print(f"\n❌ Test data order differs - this could cause evaluation differences!")

if __name__ == "__main__":
    main()
