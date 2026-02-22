#!/bin/bash
# Sequential training script for all SwinGait configs
# This script runs all 19 training jobs one after another

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo '=========================================='
echo 'SwinGait Sequential Training - All Configs'
echo '=========================================='
echo ''
echo 'Total configs: 19'
echo 'Mode: Training only (no testing)'
echo ''

# ================================================================================
# PART 1 CONFIGS
# ================================================================================

echo '=========================================='
echo 'PART 1: CNN Architecture Experiments'
echo '=========================================='
echo ''

echo '[1/19] Part 1: p+CNN'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 1 p+CNN failed!'
    exit 1
fi
echo '✓ Completed: Part 1 p+CNN'
echo ''

echo '[2/19] Part 1: p+CNN Tintro'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 1 p+CNN Tintro failed!'
    exit 1
fi
echo '✓ Completed: Part 1 p+CNN Tintro'
echo ''

echo '[3/19] Part 1: p+CNN Tintro T1'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 1 p+CNN Tintro T1 failed!'
    exit 1
fi
echo '✓ Completed: Part 1 p+CNN Tintro T1'
echo ''

echo '[4/19] Part 1: p+CNN Tintro T1 T2'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 1 p+CNN Tintro T1 T2 failed!'
    exit 1
fi
echo '✓ Completed: Part 1 p+CNN Tintro T1 T2'
echo ''

echo '[5/19] Part 1: Pretrained Unfrozen'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_pretrained_unfrozen.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 1 Pretrained Unfrozen failed!'
    exit 1
fi
echo '✓ Completed: Part 1 Pretrained Unfrozen'
echo ''

# ================================================================================
# PART 2 CONFIGS
# ================================================================================

echo '=========================================='
echo 'PART 2: Class Weight Experiments'
echo '=========================================='
echo ''

echo '[6/19] Part 2: Class Weight Balanced Normal'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_balanced_normal.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 2 Balanced Normal failed!'
    exit 1
fi
echo '✓ Completed: Part 2 Balanced Normal'
echo ''

echo '[7/19] Part 2: Class Weight Logarithmic'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_logarithmic.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 2 Logarithmic failed!'
    exit 1
fi
echo '✓ Completed: Part 2 Logarithmic'
echo ''

echo '[8/19] Part 2: Class Weight Smooth Effective'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_smooth_effective.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 2 Smooth Effective failed!'
    exit 1
fi
echo '✓ Completed: Part 2 Smooth Effective'
echo ''

echo '[9/19] Part 2: Class Weight Uniform'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_uniform.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 2 Uniform failed!'
    exit 1
fi
echo '✓ Completed: Part 2 Uniform'
echo ''

echo '[10/19] Part 2: Class Weight Inverse Square Root'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part2_classweight_inverse_sqrt.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 2 Inverse Square Root failed!'
    exit 1
fi
echo '✓ Completed: Part 2 Inverse Square Root'
echo ''

# ================================================================================
# PART 3 CONFIGS
# ================================================================================

echo '=========================================='
echo 'PART 3: Loss Function Experiments'
echo '=========================================='
echo ''

echo '[11/19] Part 3: Baseline'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_baseline.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 3 Baseline failed!'
    exit 1
fi
echo '✓ Completed: Part 3 Baseline'
echo ''

echo '[12/19] Part 3: CE Contrastive'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_ce_contrastive.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 3 CE Contrastive failed!'
    exit 1
fi
echo '✓ Completed: Part 3 CE Contrastive'
echo ''

echo '[13/19] Part 3: Triplet Focal'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_triplet_focal.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 3 Triplet Focal failed!'
    exit 1
fi
echo '✓ Completed: Part 3 Triplet Focal'
echo ''

echo '[14/19] Part 3: Contrastive Focal'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part3_contrastive_focal.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 3 Contrastive Focal failed!'
    exit 1
fi
echo '✓ Completed: Part 3 Contrastive Focal'
echo ''

# ================================================================================
# PART 4 CONFIGS
# ================================================================================

echo '=========================================='
echo 'PART 4: Final Model Variants'
echo '=========================================='
echo ''

echo '[15/19] Part 4: B1 (Frozen CNN)'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B1_frozen_cnn.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 4 B1 failed!'
    exit 1
fi
echo '✓ Completed: Part 4 B1'
echo ''

echo '[16/19] Part 4: B2 (Frozen CNN with Weights)'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B2_frozen_cnn_with_weights.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 4 B2 failed!'
    exit 1
fi
echo '✓ Completed: Part 4 B2'
echo ''

echo '[17/19] Part 4: B3 (Unfrozen CNN)'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 4 B3 failed!'
    exit 1
fi
echo '✓ Completed: Part 4 B3'
echo ''

echo '[18/19] Part 4: B4 (Unfrozen CNN with Weights)'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B4_unfrozen_cnn_with_weights.yaml --phase train --log_to_file
if [ $? -ne 0 ]; then
    echo 'ERROR: Part 4 B4 failed!'
    exit 1
fi
echo '✓ Completed: Part 4 B4'
echo ''

echo '=========================================='
echo 'ALL TRAINING JOBS COMPLETED SUCCESSFULLY!'
echo '=========================================='
echo ''
echo 'Total configs trained: 19'
echo 'All checkpoints saved in respective output directories'
echo ''
