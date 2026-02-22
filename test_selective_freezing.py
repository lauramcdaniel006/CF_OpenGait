#!/usr/bin/env python3
"""
Test script to verify selective layer freezing works correctly for DeepGaitV2.
This script loads the model with selective freezing and checks which layers are frozen.
"""

import sys
import os
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize distributed (required for get_msg_mgr)
if not torch.distributed.is_initialized():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    try:
        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
    except:
        # If already initialized, that's fine
        pass

from opengait.modeling.models.deepgaitv2 import DeepGaitV2
from opengait.utils import get_msg_mgr, config_loader

def count_trainable_params(model):
    """Count trainable and frozen parameters for each layer."""
    layer_info = {}
    
    for name, module in model.named_modules():
        if name == '':
            continue
        
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        frozen = total - trainable
        
        if total > 0:
            layer_info[name] = {
                'trainable': trainable,
                'frozen': frozen,
                'total': total,
                'is_trainable': trainable > 0
            }
    
    return layer_info

def test_selective_freezing(config_path):
    """Test selective freezing with the given config."""
    print(f"\n{'='*70}")
    print(f"Testing Selective Freezing: {config_path}")
    print(f"{'='*70}\n")
    
    # Load config
    cfgs = config_loader(config_path)
    
    # Initialize message manager (for logging)
    try:
        msg_mgr = get_msg_mgr()
    except:
        # Create a simple mock message manager if get_msg_mgr fails
        class MockMsgMgr:
            def log_info(self, *args, **kwargs):
                print(*args, **kwargs)
            def log_warning(self, *args, **kwargs):
                print("WARNING:", *args, **kwargs)
        msg_mgr = MockMsgMgr()
    
    # Create model
    print("Creating model...")
    model = DeepGaitV2(cfgs, training=True)
    model.eval()
    
    # Get freeze_layers setting
    freeze_setting = cfgs['model_cfg']['Backbone'].get('freeze_layers', False)
    print(f"Config freeze_layers setting: {freeze_setting}")
    print(f"Type: {type(freeze_setting)}")
    print()
    
    # Count parameters
    layer_info = count_trainable_params(model)
    
    # Check each CNN layer
    cnn_layers = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
    print("CNN Layer Freezing Status:")
    print("-" * 70)
    
    for layer_name in cnn_layers:
        if layer_name in layer_info:
            info = layer_info[layer_name]
            status = "🔥 TRAINABLE" if info['is_trainable'] else "❄️ FROZEN"
            print(f"{layer_name:10s} | {status:15s} | Trainable: {info['trainable']:>10,} | Frozen: {info['frozen']:>10,} | Total: {info['total']:>10,}")
        else:
            print(f"{layer_name:10s} | NOT FOUND")
    
    print("-" * 70)
    
    # Check FCs and BNNecks (should always be trainable)
    print("\nClassification Head Status:")
    print("-" * 70)
    for head_name in ['FCs', 'BNNecks']:
        if head_name in layer_info:
            info = layer_info[head_name]
            status = "🔥 TRAINABLE" if info['is_trainable'] else "❄️ FROZEN (ERROR!)"
            print(f"{head_name:10s} | {status:15s} | Trainable: {info['trainable']:>10,} | Frozen: {info['frozen']:>10,} | Total: {info['total']:>10,}")
        else:
            print(f"{head_name:10s} | NOT FOUND")
    print("-" * 70)
    
    # Verify expected behavior
    print("\nVerification:")
    print("-" * 70)
    
    expected_frozen = []
    if isinstance(freeze_setting, bool):
        if freeze_setting:
            expected_frozen = [0, 1, 2, 3, 4]  # All layers
    elif isinstance(freeze_setting, list):
        expected_frozen = freeze_setting
    
    layer_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
    all_correct = True
    
    for i, layer_name in enumerate(layer_names):
        if layer_name in layer_info:
            is_trainable = layer_info[layer_name]['is_trainable']
            should_be_frozen = i in expected_frozen
            
            if should_be_frozen and is_trainable:
                print(f"❌ ERROR: {layer_name} should be FROZEN but is TRAINABLE")
                all_correct = False
            elif not should_be_frozen and not is_trainable:
                print(f"❌ ERROR: {layer_name} should be TRAINABLE but is FROZEN")
                all_correct = False
            else:
                status = "FROZEN" if should_be_frozen else "TRAINABLE"
                print(f"✅ {layer_name} is correctly {status}")
    
    # Check FCs and BNNecks
    for head_name in ['FCs', 'BNNecks']:
        if head_name in layer_info:
            if not layer_info[head_name]['is_trainable']:
                print(f"❌ ERROR: {head_name} should be TRAINABLE but is FROZEN")
                all_correct = False
            else:
                print(f"✅ {head_name} is correctly TRAINABLE")
    
    print("-" * 70)
    
    if all_correct:
        print("\n🎉 All checks passed! Selective freezing is working correctly.")
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
    
    return all_correct

if __name__ == '__main__':
    # Test with selective freezing (first 2 layers)
    test_config = 'configs/deepgaitv2/DeepGaitV2_part4_test_selective_freezing.yaml'
    
    if not os.path.exists(test_config):
        print(f"Error: Config file not found: {test_config}")
        sys.exit(1)
    
    success = test_selective_freezing(test_config)
    sys.exit(0 if success else 1)

