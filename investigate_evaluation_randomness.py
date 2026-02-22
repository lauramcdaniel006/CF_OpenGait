#!/usr/bin/env python3
"""
Investigation script to find sources of randomness during evaluation.
This will help identify why evaluation metrics might not be identical.
"""

import os
import sys
import torch
import numpy as np
import random

# Add opengait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_seed_state(location=""):
    """Check current state of all random number generators."""
    print(f"\n{'='*70}")
    print(f"SEED STATE CHECK: {location}")
    print(f"{'='*70}")
    
    # Get current random states (we can't directly read them, but we can check if they're set)
    print(f"Python random module: Available")
    print(f"NumPy random state: Available")
    print(f"PyTorch random state: Available")
    print(f"PyTorch CUDA random state: Available")
    
    # Check environment variables
    cublas_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET')
    print(f"\nCUBLAS_WORKSPACE_CONFIG: {cublas_config}")
    
    # Check PyTorch deterministic settings
    print(f"\nPyTorch deterministic settings:")
    print(f"  torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
    try:
        # Try to check if deterministic algorithms are enabled
        # This is a bit tricky - we'll just note it
        print(f"  torch.use_deterministic_algorithms: Checked in set_seed()")
    except:
        pass

def trace_evaluation_path():
    """Trace through the evaluation path to identify potential randomness sources."""
    print("\n" + "="*70)
    print("TRACING EVALUATION PATH")
    print("="*70)
    
    print("\n1. EVALUATION ENTRY POINT:")
    print("   - BaseModel.run_test(model)")
    print("   - Called from: opengait/modeling/base_model.py:650")
    
    print("\n2. INFERENCE PROCESS:")
    print("   - model.inference(rank)")
    print("   - Location: opengait/modeling/base_model.py:567")
    print("   - Uses: test_loader (InferenceSampler)")
    
    print("\n3. DATA LOADING:")
    print("   - InferenceSampler: Sequential, no random sampling")
    print("   - Location: opengait/data/sampler.py:199")
    print("   - ✅ Should be deterministic (sequential indices)")
    
    print("\n4. COLLATE FUNCTION:")
    print("   - CollateFn.__call__(batch)")
    print("   - Location: opengait/data/collate_fn.py:47")
    print("   - Uses: get_batch_seed() -> Should return 42 during evaluation")
    print("   - Frame sampling: Uses random.choice, np.random.choice")
    print("   - ⚠️  CHECK: Is batch_counter = -1 during evaluation?")
    
    print("\n5. DATA TRANSFORMS:")
    print("   - inputs_pretreament(inputs)")
    print("   - Location: opengait/modeling/base_model.py:479")
    print("   - During evaluation: Uses seed 42 (fixed)")
    print("   - Transforms: RandomPerspective, RandomHorizontalFlip, RandomRotate")
    print("   - ⚠️  CHECK: Are transforms seeded before each batch?")
    
    print("\n6. MODEL FORWARD PASS:")
    print("   - model.forward(ipts)")
    print("   - Uses: autocast (if enabled)")
    print("   - ⚠️  CHECK: Are there any random operations in forward()?")
    
    print("\n7. EVALUATION METRICS:")
    print("   - evaluate_scoliosis() or other eval function")
    print("   - Location: opengait/evaluation/evaluator.py")
    print("   - Uses: numpy operations (softmax, argmax)")
    print("   - ⚠️  CHECK: Are numpy operations deterministic?")

def check_critical_points():
    """Check critical points where randomness might occur."""
    print("\n" + "="*70)
    print("CRITICAL POINTS TO CHECK")
    print("="*70)
    
    issues = []
    
    print("\n1. BATCH COUNTER DURING EVALUATION:")
    print("   - InferenceSampler does NOT increment batch_counter")
    print("   - get_batch_seed() should return 42 when counter = -1")
    print("   - ⚠️  VERIFY: Check if counter is properly -1 during eval")
    
    print("\n2. COLLATE FUNCTION SEEDING:")
    print("   - CollateFn uses get_batch_seed() at start of __call__")
    print("   - Should get seed 42 during evaluation")
    print("   - ⚠️  VERIFY: Is set_seed(42) called in CollateFn during eval?")
    
    print("\n3. TRANSFORM SEEDING:")
    print("   - inputs_pretreament sets seed 42 during evaluation")
    print("   - ⚠️  VERIFY: Is this called BEFORE transforms are applied?")
    
    print("\n4. NUMPY OPERATIONS IN EVALUATION:")
    print("   - evaluate_scoliosis uses np.exp, np.max, np.sum for softmax")
    print("   - These should be deterministic if inputs are identical")
    print("   - ⚠️  VERIFY: Are inputs identical between runs?")
    
    print("\n5. MODEL WEIGHTS:")
    print("   - If model weights differ, evaluation will differ")
    print("   - ⚠️  VERIFY: Are model weights identical at same iteration?")
    
    print("\n6. CUDA OPERATIONS:")
    print("   - Matrix multiplications should be deterministic with CUBLAS_WORKSPACE_CONFIG")
    print("   - ⚠️  VERIFY: Is CUBLAS_WORKSPACE_CONFIG set before torch import?")

def create_test_script():
    """Create a test script to verify evaluation determinism."""
    script = """
# Test script to verify evaluation determinism
# Run this to check if evaluation is deterministic

import torch
import numpy as np
import random
import os

# Set seeds
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# Test 1: Check if softmax is deterministic
print("Test 1: Softmax determinism")
set_seed(42)
logits1 = torch.randn(10, 3)
probs1 = torch.softmax(logits1, dim=1)

set_seed(42)
logits2 = torch.randn(10, 3)
probs2 = torch.softmax(logits2, dim=1)

print(f"Softmax identical: {torch.allclose(probs1, probs2)}")
print(f"Max difference: {(probs1 - probs2).abs().max().item()}")

# Test 2: Check numpy softmax
print("\\nTest 2: NumPy softmax determinism")
np.random.seed(42)
logits_np1 = np.random.randn(10, 3)
exp1 = np.exp(logits_np1 - np.max(logits_np1, axis=1, keepdims=True))
probs_np1 = exp1 / np.sum(exp1, axis=1, keepdims=True)

np.random.seed(42)
logits_np2 = np.random.randn(10, 3)
exp2 = np.exp(logits_np2 - np.max(logits_np2, axis=1, keepdims=True))
probs_np2 = exp2 / np.sum(exp2, axis=1, keepdims=True)

print(f"NumPy softmax identical: {np.allclose(probs_np1, probs_np2)}")
print(f"Max difference: {np.abs(probs_np1 - probs_np2).max()}")

# Test 3: Check CUBLAS setting
print("\\nTest 3: CUBLAS setting")
cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'NOT SET')
print(f"CUBLAS_WORKSPACE_CONFIG: {cublas}")
"""
    
    with open('test_evaluation_determinism.py', 'w') as f:
        f.write(script)
    
    print("\n✅ Created test_evaluation_determinism.py")
    print("   Run: python test_evaluation_determinism.py")

def main():
    print("="*70)
    print("EVALUATION RANDOMNESS INVESTIGATION")
    print("="*70)
    
    check_seed_state("Initial")
    trace_evaluation_path()
    check_critical_points()
    create_test_script()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python test_evaluation_determinism.py")
    print("2. Check evaluation logs for seed initialization messages")
    print("3. Compare model weights at same iteration between runs")
    print("4. Verify batch_counter is -1 during evaluation")
    print("5. Check if CollateFn is called with same seed during eval")

if __name__ == "__main__":
    main()
