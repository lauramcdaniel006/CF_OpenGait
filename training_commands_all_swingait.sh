#!/bin/bash
# ================================================================================
# TRAINING COMMANDS FOR ALL SWINGAIT CONFIGS (Part 1-4)
# All configs now have seed: 42 for full reproducibility
# ================================================================================

# ================================================================================
# PART 1 CONFIGS
# ================================================================================

# Part 1 Baseline
echo "Running: Part 1 Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_baseline.yaml --phase train --log_to_file

# Part 1 p+CNN
echo "Running: Part 1 p+CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN.yaml --phase train --log_to_file

# Part 1 p+CNN Tintro
echo "Running: Part 1 p+CNN Tintro"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro.yaml --phase train --log_to_file

# Part 1 p+CNN Tintro T1
echo "Running: Part 1 p+CNN Tintro T1"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml --phase train --log_to_file

# Part 1 p+CNN Tintro T1 T2
echo "Running: Part 1 p+CNN Tintro T1 T2"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml --phase train --log_to_file

# Part 1 Pretrained Unfrozen
echo "Running: Part 1 Pretrained Unfrozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_pretrained_unfrozen.yaml --phase train --log_to_file

# ================================================================================
# PART 2 CONFIGS
# ================================================================================

# Part 2 Baseline
echo "Running: Part 2 Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_baseline.yaml --phase train --log_to_file

# ================================================================================
# PART 3 CONFIGS
# ================================================================================

# Part 3 Baseline
echo "Running: Part 3 Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_baseline.yaml --phase train --log_to_file

# Part 3 Triplet Focal
echo "Running: Part 3 Triplet Focal"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_triplet_focal.yaml --phase train --log_to_file

# Part 3 Contrastive Focal
echo "Running: Part 3 Contrastive Focal"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_contrastive_focal.yaml --phase train --log_to_file

# Part 3 CE Contrastive
echo "Running: Part 3 CE Contrastive"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_ce_contrastive.yaml --phase train --log_to_file

# ================================================================================
# PART 4 CONFIGS
# ================================================================================

# Part 4a B1 Frozen CNN
echo "Running: Part 4a B1 Frozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B1_frozen_cnn.yaml --phase train --log_to_file

# Part 4a B2 Frozen CNN with Weights
echo "Running: Part 4a B2 Frozen CNN with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B2_frozen_cnn_with_weights.yaml --phase train --log_to_file

# Part 4a B3 Unfrozen CNN
echo "Running: Part 4a B3 Unfrozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml --phase train --log_to_file

# Part 4a B4 Unfrozen CNN with Weights
echo "Running: Part 4a B4 Unfrozen CNN with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B4_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

# Part 4 DeepGaitV2 Comparison
echo "Running: Part 4 DeepGaitV2 Comparison"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison.yaml --phase train --log_to_file

# Part 4 DeepGaitV2 Comparison with Weights
echo "Running: Part 4 DeepGaitV2 Comparison with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_with_weights.yaml --phase train --log_to_file

# Part 4 DeepGaitV2 Comparison Unfrozen CNN
echo "Running: Part 4 DeepGaitV2 Comparison Unfrozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_unfrozen_cnn.yaml --phase train --log_to_file

# Part 4 DeepGaitV2 Comparison Unfrozen CNN with Weights
echo "Running: Part 4 DeepGaitV2 Comparison Unfrozen CNN with Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

