#!/bin/bash

# # Part 1 - Tintro Configs (3 commands)
# echo "=========================================="
# echo "Starting Part 1 - Tintro Configs"
# echo "=========================================="

# # Part 1 - Tintro
# echo "Running: swin_part1_p_CNN_Tintro.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro.yaml --phase train --log_to_file

# # Part 1 - Tintro T1
# echo "Running: swin_part1_p_CNN_Tintro_T1.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml --phase train --log_to_file

# # Part 1 - Tintro T1 T2
# echo "Running: swin_part1_p_CNN_Tintro_T1_T2.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml --phase train --log_to_file

echo "=========================================="
echo "Starting Part 2 - Class Weight Configs 2"
echo "=========================================="

# # Part 2 - Baseline
# echo "Running: swin_part2_baseline.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_baseline.yaml --phase train --log_to_file

# # Part 2 - Class Weight Uniform
# echo "Running: swin_part2_classweight_uniform.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_uniform.yaml --phase train --log_to_file

# Part 2 - Class Weight Balanced Normal
# SKIPPED - Currently running manually
# echo "Running: swin_part2_classweight_balanced_normal.yaml"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_balanced_normal.yaml --phase train --log_to_file

# Part 2 - Class Weight Inverse Sqrt
echo "Running: swin_part2_classweight_inverse_sqrt.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_inverse_sqrt.yaml --phase train --log_to_file

# Part 2 - Class Weight Logarithmic
echo "Running: swin_part2_classweight_logarithmic.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_logarithmic.yaml --phase train --log_to_file

# Part 2 - Class Weight Smooth Effective
echo "Running: swin_part2_classweight_smooth_effective.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_smooth_effective.yaml --phase train --log_to_file

echo "=========================================="
echo "Starting Part 3 - Loss Function Configs"
echo "=========================================="

# Part 3 - Baseline
echo "Running: swin_part3_baseline.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_baseline.yaml --phase train --log_to_file

# Part 3 - Triplet Focal
echo "Running: swin_part3_triplet_focal.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_triplet_focal.yaml --phase train --log_to_file

# Part 3 - Contrastive Focal
echo "Running: swin_part3_contrastive_focal.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_contrastive_focal.yaml --phase train --log_to_file

# Part 3 - CE Contrastive
echo "Running: swin_part3_ce_contrastive.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_ce_contrastive.yaml --phase train --log_to_file

echo "=========================================="
echo "Starting Part 4a - With Weights Configs"
echo "=========================================="

# Part 4a - B2 Frozen CNN With Weights
echo "Running: swin_part4a_B2_frozen_cnn_with_weights.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B2_frozen_cnn_with_weights.yaml --phase train --log_to_file

# Part 4a - B4 Unfrozen CNN With Weights
echo "Running: swin_part4a_B4_unfrozen_cnn_with_weights.yaml"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B4_unfrozen_cnn_with_weights.yaml --phase train --log_to_file

echo "=========================================="
echo "All configs completed!"
echo "=========================================="
