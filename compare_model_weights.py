#!/usr/bin/env python3
"""
Script to compare model weights between two training runs.
This helps verify if model weights are identical at the same iteration.
"""

import torch
import os
import sys
import argparse
from pathlib import Path

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return the model state dict."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            return checkpoint['model']
        elif 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        else:
            # Assume the checkpoint IS the state dict
            return checkpoint
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def compare_weights(weights1, weights2, tolerance=1e-6):
    """Compare two model weight dictionaries."""
    if weights1 is None or weights2 is None:
        return False, {}
    
    # Get all parameter names
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    # Check if keys match
    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        print(f"❌ Parameter keys don't match!")
        if missing_in_2:
            print(f"   Missing in checkpoint 2: {missing_in_2}")
        if missing_in_1:
            print(f"   Missing in checkpoint 1: {missing_in_1}")
        return False, {}
    
    differences = {}
    all_match = True
    
    for key in keys1:
        w1 = weights1[key]
        w2 = weights2[key]
        
        # Check shape
        if w1.shape != w2.shape:
            differences[key] = {
                'match': False,
                'reason': f'Shape mismatch: {w1.shape} vs {w2.shape}'
            }
            all_match = False
            continue
        
        # Check values
        diff = torch.abs(w1 - w2)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff > tolerance:
            differences[key] = {
                'match': False,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': w1.shape
            }
            all_match = False
        else:
            differences[key] = {
                'match': True,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'shape': w1.shape
            }
    
    return all_match, differences

def find_checkpoints(output_dir, iteration=None):
    """Find checkpoint files in output directory."""
    checkpoints_dir = Path(output_dir) / 'checkpoints'
    if not checkpoints_dir.exists():
        return []
    
    checkpoints = []
    for ckpt_file in sorted(checkpoints_dir.glob('*.pt')):
        if iteration is None or f'iter-{iteration}' in ckpt_file.name:
            checkpoints.append(ckpt_file)
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description='Compare model weights between two checkpoints')
    parser.add_argument('--ckpt1', type=str, required=True, help='Path to first checkpoint')
    parser.add_argument('--ckpt2', type=str, required=True, help='Path to second checkpoint')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Tolerance for weight comparison')
    parser.add_argument('--output-dir1', type=str, help='Output directory for run 1 (to find checkpoints)')
    parser.add_argument('--output-dir2', type=str, help='Output directory for run 2 (to find checkpoints)')
    parser.add_argument('--iteration', type=int, help='Specific iteration to compare')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODEL WEIGHT COMPARISON")
    print("="*70)
    
    # If output dirs provided, find checkpoints
    if args.output_dir1 and args.output_dir2:
        print(f"\n📁 Searching for checkpoints...")
        ckpts1 = find_checkpoints(args.output_dir1, args.iteration)
        ckpts2 = find_checkpoints(args.output_dir2, args.iteration)
        
        if not ckpts1:
            print(f"❌ No checkpoints found in {args.output_dir1}")
            return
        if not ckpts2:
            print(f"❌ No checkpoints found in {args.output_dir2}")
            return
        
        print(f"   Found {len(ckpts1)} checkpoint(s) in run 1")
        print(f"   Found {len(ckpts2)} checkpoint(s) in run 2")
        
        # Use most recent or specified iteration
        if args.iteration:
            ckpt1 = [c for c in ckpts1 if f'iter-{args.iteration}' in c.name]
            ckpt2 = [c for c in ckpts2 if f'iter-{args.iteration}' in c.name]
            if not ckpt1 or not ckpt2:
                print(f"❌ Checkpoint for iteration {args.iteration} not found")
                return
            args.ckpt1 = str(ckpt1[0])
            args.ckpt2 = str(ckpt2[0])
        else:
            args.ckpt1 = str(ckpts1[-1])  # Most recent
            args.ckpt2 = str(ckpts2[-1])
    
    print(f"\n📂 Loading checkpoints...")
    print(f"   Checkpoint 1: {args.ckpt1}")
    print(f"   Checkpoint 2: {args.ckpt2}")
    
    weights1 = load_checkpoint(args.ckpt1)
    weights2 = load_checkpoint(args.ckpt2)
    
    if weights1 is None or weights2 is None:
        return
    
    print(f"\n🔍 Comparing weights (tolerance: {args.tolerance})...")
    all_match, differences = compare_weights(weights1, weights2, args.tolerance)
    
    if all_match:
        print(f"\n✅ ALL WEIGHTS MATCH! (within tolerance {args.tolerance})")
        print(f"   Model weights are identical between the two checkpoints.")
    else:
        print(f"\n❌ WEIGHTS DIFFER!")
        print(f"   Found differences in {sum(1 for d in differences.values() if not d['match'])} parameter(s)")
        
        # Show top differences
        mismatches = [(k, d) for k, d in differences.items() if not d['match']]
        mismatches.sort(key=lambda x: x[1].get('max_diff', 0), reverse=True)
        
        print(f"\n📊 Top 10 differences:")
        for i, (key, diff_info) in enumerate(mismatches[:10]):
            if 'reason' in diff_info:
                print(f"   {i+1}. {key}: {diff_info['reason']}")
            else:
                print(f"   {i+1}. {key}: max_diff={diff_info['max_diff']:.6e}, mean_diff={diff_info['mean_diff']:.6e}, shape={diff_info['shape']}")
        
        if len(mismatches) > 10:
            print(f"   ... and {len(mismatches) - 10} more differences")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    total_params = len(differences)
    matching_params = sum(1 for d in differences.values() if d['match'])
    mismatching_params = total_params - matching_params
    
    print(f"   Total parameters: {total_params}")
    print(f"   Matching: {matching_params} ({100*matching_params/total_params:.1f}%)")
    print(f"   Differing: {mismatching_params} ({100*mismatching_params/total_params:.1f}%)")
    
    if mismatching_params > 0:
        max_diffs = [d.get('max_diff', 0) for d in differences.values() if not d.get('match', True)]
        print(f"   Max difference: {max(max_diffs):.6e}")
        print(f"   Mean difference: {sum(max_diffs)/len(max_diffs):.6e}")

if __name__ == "__main__":
    main()
