#!/usr/bin/env python3
"""
Quick verification script to check DeepGaitV2 layer freezing status.
This can be run before training to verify the config is correct.
"""

import sys
import os
import yaml
import torch

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opengait.modeling.models.deepgaitv2 import DeepGaitV2
from opengait.utils import get_msg_mgr, init_logger

def verify_freezing(config_path):
    """Load model and verify which layers are frozen"""
    
    print("=" * 70)
    print(f"Verifying freezing configuration: {config_path}")
    print("=" * 70)
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get freeze_layers setting
    freeze_layers = cfg['model_cfg']['Backbone'].get('freeze_layers', False)
    print(f"\nConfig setting: freeze_layers = {freeze_layers}")
    print(f"Type: {type(freeze_layers).__name__}")
    
    # Initialize minimal environment for model loading
    if not os.path.exists('output'):
        os.makedirs('output', exist_ok=True)
    
    # Initialize logger (minimal setup)
    init_logger('output/test_freezing', log_to_file=False)
    msg_mgr = get_msg_mgr()
    
    # Create model
    print("\nCreating model...")
    model = DeepGaitV2(cfg, training=False)
    
    # Check each layer
    print("\n" + "=" * 70)
    print("LAYER FREEZING VERIFICATION")
    print("=" * 70)
    
    layer_info = {
        0: ('layer0', 'First Conv'),
        1: ('layer1', 'BasicBlock2D'),
        2: ('layer2', 'P3D Block'),
        3: ('layer3', 'P3D Block'),
        4: ('layer4', 'P3D Block')
    }
    
    frozen_count = 0
    trainable_count = 0
    
    for idx, (layer_name, layer_desc) in layer_info.items():
        layer = getattr(model, layer_name)
        
        # Check if any parameters are trainable
        has_trainable = any(p.requires_grad for p in layer.parameters())
        has_frozen = any(not p.requires_grad for p in layer.parameters())
        
        # Count parameters
        total_params = sum(p.numel() for p in layer.parameters())
        trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        status = "🔥 TRAINABLE" if has_trainable and not has_frozen else \
                 "❄️ FROZEN" if has_frozen and not has_trainable else \
                 "⚠️ MIXED (ERROR!)"
        
        print(f"{status} | {layer_name:8s} ({layer_desc:15s}) | "
              f"Trainable: {trainable_params:>10,} | Frozen: {frozen_params:>10,} | Total: {total_params:>10,}")
        
        if has_frozen and not has_trainable:
            frozen_count += 1
        elif has_trainable and not has_frozen:
            trainable_count += 1
        else:
            print(f"  ⚠️ WARNING: Layer has mixed trainable/frozen parameters!")
    
    # Check FCs and BNNecks (should always be trainable)
    print("\n" + "-" * 70)
    print("Classification Heads (should always be trainable):")
    print("-" * 70)
    
    for head_name in ['FCs', 'BNNecks']:
        if hasattr(model, head_name):
            head = getattr(model, head_name)
            total_params = sum(p.numel() for p in head.parameters())
            trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            if frozen_params == 0:
                print(f"✅ {head_name:8s} | All {trainable_params:>10,} parameters are TRAINABLE")
            else:
                print(f"⚠️ {head_name:8s} | ERROR: {frozen_params:>10,} parameters are FROZEN!")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Frozen CNN layers: {frozen_count}/5")
    print(f"Trainable CNN layers: {trainable_count}/5")
    print(f"Expected based on config: {freeze_layers}")
    
    # Verify expected vs actual
    if isinstance(freeze_layers, bool):
        if freeze_layers:
            expected_frozen = 5
            expected_trainable = 0
        else:
            expected_frozen = 0
            expected_trainable = 5
    elif isinstance(freeze_layers, list):
        expected_frozen = len(freeze_layers)
        expected_trainable = 5 - len(freeze_layers)
    else:
        expected_frozen = 0
        expected_trainable = 5
    
    if frozen_count == expected_frozen and trainable_count == expected_trainable:
        print("✅ VERIFICATION PASSED: Freezing matches config!")
    else:
        print(f"❌ VERIFICATION FAILED:")
        print(f"   Expected: {expected_frozen} frozen, {expected_trainable} trainable")
        print(f"   Actual:   {frozen_count} frozen, {trainable_count} trainable")
    
    print("=" * 70)
    
    return frozen_count == expected_frozen and trainable_count == expected_trainable


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_freezing.py <config_path>")
        print("\nExample:")
        print("  python verify_freezing.py configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    success = verify_freezing(config_path)
    sys.exit(0 if success else 1)

