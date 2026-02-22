#!/bin/bash
# Run DeepGaitV2 Part 1 and Part 4a B1-B4 configs sequentially
# All configs use: CUDA_VISIBLE_DEVICES=0,1, torch.distributed.launch with 2 GPUs
# Each config runs one after another automatically

CONDA_ENV="myGait38"
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Initialize conda if needed
if [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
fi

# Activate conda environment
conda activate "$CONDA_ENV"

# Change to project directory
cd /cis/home/lmcdan11/Documents_Swin/OpenGait

echo "=========================================="
echo "DeepGaitV2 Part 1 and Part 4a B1-B4 Sequential Training"
echo "=========================================="
echo ""
echo "All configs will run one after another automatically"
echo "Each config uses 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)"
echo ""

# Array of configs to run
declare -a configs=(
    "configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml|Part 1: All Trainable"
    "configs/deepgaitv2/DeepGaitV2_part1_baseline_all_frozen.yaml|Part 1: Baseline All Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml|Part 1: Early Layers Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml|Part 1: First Layer Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml|Part 1: First Two Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml|Part 1: Heavy Frozen"
    "configs/deepgaitv2/DeepGaitV2_part4a_B1_partially_frozen.yaml|Part 4a: B1 Partially Frozen"
    "configs/deepgaitv2/DeepGaitV2_part4a_B2_partially_frozen_with_weights.yaml|Part 4a: B2 Partially Frozen With Weights"
    "configs/deepgaitv2/DeepGaitV2_part4a_B3_unfrozen.yaml|Part 4a: B3 Unfrozen"
    "configs/deepgaitv2/DeepGaitV2_part4a_B4_unfrozen_with_weights.yaml|Part 4a: B4 Unfrozen With Weights"
)

total=${#configs[@]}
current=0

for config_entry in "${configs[@]}"; do
    IFS='|' read -r config_file config_name <<< "$config_entry"
    current=$((current + 1))
    
    echo "=========================================="
    echo "[$current/$total] Starting: $config_name"
    echo "File: $config_file"
    echo "=========================================="
    echo ""
    
    # Run the training command
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs "$config_file" --phase train --log_to_file
    
    # Check if training completed successfully
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ [$current/$total] Completed: $config_name"
        echo ""
    else
        echo ""
        echo "✗ [$current/$total] Failed: $config_name (exit code: $exit_code)"
        echo "Continuing with next config..."
        echo ""
    fi
    
    # Small delay before starting next config
    sleep 2
done

echo "=========================================="
echo "All training jobs completed!"
echo "=========================================="
