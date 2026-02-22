#!/bin/bash
# ================================================================================
# TRAINING COMMANDS FOR ALL DEEPGaitV2 CONFIGS
# All configs now have seed: 42 for full reproducibility
# ================================================================================

# ================================================================================
# PART 1 CONFIGS
# ================================================================================

# Part 1 Baseline All Frozen
echo "Running: DeepGaitV2 Part 1 Baseline All Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_baseline_all_frozen.yaml --phase train --log_to_file

# Part 1 All Trainable
echo "Running: DeepGaitV2 Part 1 All Trainable"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml --phase train --log_to_file

# Part 1 First Layer Frozen
echo "Running: DeepGaitV2 Part 1 First Layer Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml --phase train --log_to_file

# Part 1 First Two Frozen
echo "Running: DeepGaitV2 Part 1 First Two Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml --phase train --log_to_file

# Part 1 Early Layers Frozen
echo "Running: DeepGaitV2 Part 1 Early Layers Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml --phase train --log_to_file

# Part 1 Heavy Frozen
echo "Running: DeepGaitV2 Part 1 Heavy Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml --phase train --log_to_file

# ================================================================================
# PART 4 CONFIGS
# ================================================================================

# Part 4a B1 Partially Frozen
echo "Running: DeepGaitV2 Part 4a B1 Partially Frozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4a_B1_partially_frozen.yaml --phase train --log_to_file

# Part 4a B2 Partially Frozen with Weights
echo "Running: DeepGaitV2 Part 4a B2 Partially Frozen with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4a_B2_partially_frozen_with_weights.yaml --phase train --log_to_file

# Part 4a B3 Unfrozen
echo "Running: DeepGaitV2 Part 4a B3 Unfrozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4a_B3_unfrozen.yaml --phase train --log_to_file

# Part 4a B4 Unfrozen with Weights
echo "Running: DeepGaitV2 Part 4a B4 Unfrozen with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4a_B4_unfrozen_with_weights.yaml --phase train --log_to_file

# Part 4a Updated Best Freezing
echo "Running: DeepGaitV2 Part 4a Updated Best Freezing"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4a_updated_best_freezing.yaml --phase train --log_to_file

# Part 4b Half Frozen CNN
echo "Running: DeepGaitV2 Part 4b Half Frozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn.yaml --phase train --log_to_file

# Part 4b Half Frozen CNN with Weights
echo "Running: DeepGaitV2 Part 4b Half Frozen CNN with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml --phase train --log_to_file

# Part 4 Test Selective Freezing
echo "Running: DeepGaitV2 Part 4 Test Selective Freezing"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4_test_selective_freezing.yaml --phase train --log_to_file

# Part 4 SwinGait Comparison
echo "Running: DeepGaitV2 Part 4 SwinGait Comparison"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4_swingait_comparison.yaml --phase train --log_to_file

# Part 4 SwinGait Comparison with Weights
echo "Running: DeepGaitV2 Part 4 SwinGait Comparison with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4_swingait_comparison_with_weights.yaml --phase train --log_to_file

# Part 4 SwinGait Comparison Unfrozen CNN
echo "Running: DeepGaitV2 Part 4 SwinGait Comparison Unfrozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4_swingait_comparison_unfrozen_cnn.yaml --phase train --log_to_file

# Part 4 SwinGait Comparison Unfrozen CNN with Weights
echo "Running: DeepGaitV2 Part 4 SwinGait Comparison Unfrozen CNN with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_part4_swingait_comparison_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

# ================================================================================
# DATASET-SPECIFIC CONFIGS
# ================================================================================

# CASIA-B
echo "Running: DeepGaitV2 CASIA-B"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_casiab.yaml --phase train --log_to_file

# CCPG
echo "Running: DeepGaitV2 CCPG"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_ccpg.yaml --phase train --log_to_file

# Gait3D
echo "Running: DeepGaitV2 Gait3D"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_gait3d.yaml --phase train --log_to_file

# GREW
echo "Running: DeepGaitV2 GREW"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_grew.yaml --phase train --log_to_file

# OUMVLP
echo "Running: DeepGaitV2 OUMVLP"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_oumvlp.yaml --phase train --log_to_file

# SUSTech1K
echo "Running: DeepGaitV2 SUSTech1K"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_sustech1k.yaml --phase train --log_to_file

