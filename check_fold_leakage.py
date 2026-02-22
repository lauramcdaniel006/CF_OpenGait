#!/usr/bin/env python3
"""
Script to check for data leakage between folds in k-fold cross-validation.
Checks that VAL_SET participants don't appear in TEST_SET of other folds.
"""

import json
import os
import sys
from pathlib import Path

def check_leakage(partitions_dir='kfold_results/partitions'):
    """Check for data leakage between folds"""
    partitions_dir = Path(partitions_dir)
    
    if not partitions_dir.exists():
        print(f"❌ Partitions directory not found: {partitions_dir}")
        return False
    
    # Load all partition files
    folds = {}
    for fold_file in sorted(partitions_dir.glob('fold_*.json')):
        fold_num = int(fold_file.stem.split('_')[1])
        with open(fold_file, 'r') as f:
            folds[fold_num] = json.load(f)
    
    if not folds:
        print(f"❌ No partition files found in {partitions_dir}")
        return False
    
    print(f"✓ Found {len(folds)} fold partition files")
    print(f"\n{'='*70}")
    print("CHECKING FOR DATA LEAKAGE")
    print(f"{'='*70}\n")
    
    leakage_found = False
    
    # Check each fold's VAL_SET against all other folds' TEST_SET
    for fold_num, partition in folds.items():
        val_set = set(partition.get('VAL_SET', []))
        test_set = set(partition.get('TEST_SET', []))
        
        print(f"Fold {fold_num}:")
        print(f"  VAL_SET: {len(val_set)} participants")
        print(f"  TEST_SET: {len(test_set)} participants")
        
        # Check against all other folds
        for other_fold_num, other_partition in folds.items():
            if other_fold_num == fold_num:
                continue
            
            other_test_set = set(other_partition.get('TEST_SET', []))
            
            # Check if VAL_SET participants appear in other fold's TEST_SET
            leakage = val_set.intersection(other_test_set)
            if leakage:
                leakage_found = True
                print(f"  ❌ LEAKAGE DETECTED: {len(leakage)} VAL_SET participants appear in Fold {other_fold_num} TEST_SET:")
                for pid in sorted(leakage):
                    print(f"      - Participant {pid}")
        
        # Also check that VAL_SET and TEST_SET don't overlap within same fold
        same_fold_overlap = val_set.intersection(test_set)
        if same_fold_overlap:
            leakage_found = True
            print(f"  ❌ LEAKAGE DETECTED: {len(same_fold_overlap)} participants in both VAL_SET and TEST_SET of same fold:")
            for pid in sorted(same_fold_overlap):
                print(f"      - Participant {pid}")
        
        if not leakage_found or fold_num == len(folds):
            print(f"  ✓ No leakage detected for Fold {fold_num}")
        print()
    
    print(f"{'='*70}")
    if leakage_found:
        print("❌ DATA LEAKAGE FOUND - Results may be invalid!")
        return False
    else:
        print("✅ NO DATA LEAKAGE - All folds are independent!")
        return True
    print(f"{'='*70}\n")

if __name__ == '__main__':
    partitions_dir = sys.argv[1] if len(sys.argv) > 1 else 'kfold_results/partitions'
    success = check_leakage(partitions_dir)
    sys.exit(0 if success else 1)
