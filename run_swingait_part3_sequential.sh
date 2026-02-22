#!/bin/bash
# Run SwinGait Part 3 configs sequentially
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
echo "SwinGait Part 3 Sequential Training"
echo "=========================================="
echo ""
echo "All configs will run one after another automatically"
echo "Each config uses 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)"
echo ""

# Array of configs to run
declare -a configs=(
    "configs/swingait/swin_part3_baseline.yaml|Part 3: Baseline (Triplet + CrossEntropy)"
    "configs/swingait/swin_part3_triplet_focal.yaml|Part 3: Triplet + Focal"
    "configs/swingait/swin_part3_ce_contrastive.yaml|Part 3: CrossEntropy + Contrastive"
    "configs/swingait/swin_part3_contrastive_focal.yaml|Part 3: Contrastive + Focal"
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
