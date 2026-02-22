#!/bin/bash
# Training commands for SwinGait Part 1-4 configs
# All configs use: CUDA_VISIBLE_DEVICES=0,1, torch.distributed.launch with 2 GPUs

# ============================================================================
# PART 1 CONFIGS
# ============================================================================

echo "Starting Part 1: Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_baseline.yaml --phase train --log_to_file

echo "Starting Part 1: p+CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN.yaml --phase train --log_to_file

echo "Starting Part 1: p+CNN Tintro"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro.yaml --phase train --log_to_file

echo "Starting Part 1: p+CNN Tintro T1"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml --phase train --log_to_file

echo "Starting Part 1: p+CNN Tintro T1 T2"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml --phase train --log_to_file

echo "Starting Part 1: Pretrained Unfrozen"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_pretrained_unfrozen.yaml --phase train --log_to_file

# ============================================================================
# PART 2 CONFIGS
# ============================================================================

echo "Starting Part 2: Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_baseline.yaml --phase train --log_to_file

# ============================================================================
# PART 3 CONFIGS
# ============================================================================

echo "Starting Part 3: Baseline"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_baseline.yaml --phase train --log_to_file

echo "Starting Part 3: CE Contrastive"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_ce_contrastive.yaml --phase train --log_to_file

echo "Starting Part 3: Contrastive Focal"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_contrastive_focal.yaml --phase train --log_to_file

echo "Starting Part 3: Triplet Focal"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_triplet_focal.yaml --phase train --log_to_file

# ============================================================================
# PART 4 CONFIGS
# ============================================================================

echo "Starting Part 4: DeepGaitV2 Comparison"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison.yaml --phase train --log_to_file

echo "Starting Part 4: DeepGaitV2 Comparison Unfrozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_unfrozen_cnn.yaml --phase train --log_to_file

echo "Starting Part 4: DeepGaitV2 Comparison Unfrozen CNN With Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

echo "Starting Part 4: DeepGaitV2 Comparison With Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4_deepgaitv2_comparison_with_weights.yaml --phase train --log_to_file

echo "Starting Part 4a: B1 Frozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B1_frozen_cnn.yaml --phase train --log_to_file

echo "Starting Part 4a: B2 Frozen CNN With Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B2_frozen_cnn_with_weights.yaml --phase train --log_to_file

echo "Starting Part 4a: B3 Unfrozen CNN"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml --phase train --log_to_file

echo "Starting Part 4a: B4 Unfrozen CNN With Weights"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B4_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

echo "All training commands completed!"
