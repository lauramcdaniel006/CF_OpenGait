#!/bin/bash
# Run all DeepGaitV2 Part 1 freezing experiments
# Usage: bash run_part1_deepgaitv2_experiments.sh

# Set GPU configuration (adjust as needed)
GPUS="0,1"
NUM_GPUS=2

# Base command
BASE_CMD="CUDA_VISIBLE_DEVICES=${GPUS} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} opengait/main.py --phase train --log_to_file"

echo "=========================================="
echo "DeepGaitV2 Part 1 Freezing Experiments"
echo "=========================================="
echo ""

# Experiment 1: All Trainable (0 frozen)
echo "Starting Experiment 1: All Trainable (0 frozen)"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml
echo "Experiment 1 completed."
echo ""

# Experiment 2: Layer 0 Frozen (1 frozen)
echo "Starting Experiment 2: Layer 0 Frozen (1 frozen)"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml
echo "Experiment 2 completed."
echo ""

# Experiment 3: Layers 0-1 Frozen (2 frozen)
echo "Starting Experiment 3: Layers 0-1 Frozen (2 frozen)"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml
echo "Experiment 3 completed."
echo ""

# Experiment 4: Layers 0-2 Frozen (3 frozen) - Matches SwinGait
echo "Starting Experiment 4: Layers 0-2 Frozen (3 frozen) - Matches SwinGait"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml
echo "Experiment 4 completed."
echo ""

# Experiment 5: Layers 0-3 Frozen (4 frozen)
echo "Starting Experiment 5: Layers 0-3 Frozen (4 frozen)"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml
echo "Experiment 5 completed."
echo ""

# Experiment 6: All Frozen (5 frozen)
echo "Starting Experiment 6: All Frozen (5 frozen)"
echo "----------------------------------------"
${BASE_CMD} --cfgs configs/deepgaitv2/DeepGaitV2_part1_baseline_all_frozen.yaml
echo "Experiment 6 completed."
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

