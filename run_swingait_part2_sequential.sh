#!/bin/bash

# Script to run SwinGait Part 2 class weighting configs sequentially
# Part 2: Class Weighting Strategies (frozen CNN, unfrozen transformer)

CONDA_ENV="myGait38"
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Activate conda environment
if [ -f "$CONDA_INIT" ]; then
    source "$CONDA_INIT"
fi
conda activate "$CONDA_ENV"

# Change to project directory
cd /cis/home/lmcdan11/Documents_Swin/OpenGait

# Array of configs: "config_file|Description"
declare -a configs=(
    "configs/swingait/swin_part2_classweight_uniform.yaml|Part 2: Uniform Weights (1.0, 1.0, 1.0)"
    "configs/swingait/swin_part2_classweight_balanced_normal.yaml|Part 2: Balanced Normal Weights (1.27, 0.87, 0.87)"
    "configs/swingait/swin_part2_classweight_inverse_sqrt.yaml|Part 2: Inverse Square Root Weights (1.13, 0.93, 0.93)"
    "configs/swingait/swin_part2_classweight_logarithmic.yaml|Part 2: Logarithmic Weights (1.14, 0.93, 0.93)"
    "configs/swingait/swin_part2_classweight_smooth_effective.yaml|Part 2: Smooth Effective Weights (0.78, 1.11, 1.11)"
)

total_configs=${#configs[@]}
current=0

echo "=========================================="
echo "SwinGait Part 2: Class Weighting Strategies"
echo "Total configs: $total_configs"
echo "=========================================="
echo ""

for config_entry in "${configs[@]}"; do
    IFS='|' read -r config_file config_name <<< "$config_entry"
    current=$((current + 1))
    
    echo ""
    echo "=========================================="
    echo "[$current/$total_configs] Starting: $config_name"
    echo "Config: $config_file"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Run training
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
        --nproc_per_node=2 \
        opengait/main.py \
        --cfgs "$config_file" \
        --phase train \
        --log_to_file
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Successfully completed: $config_name"
    else
        echo ""
        echo "✗ ERROR: Failed to complete: $config_name (exit code: $exit_code)"
        echo "Stopping sequential execution."
        exit $exit_code
    fi
    
    echo ""
    echo "Waiting 5 seconds before next config..."
    sleep 5
done

echo ""
echo "=========================================="
echo "All Part 2 class weighting configs completed!"
echo "Total: $total_configs configs"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
