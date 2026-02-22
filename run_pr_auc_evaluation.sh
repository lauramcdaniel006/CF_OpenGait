#!/bin/bash
# Script to evaluate both models and create PR-AUC curves
# Run this in your conda environment with PyTorch installed

echo "=========================================="
echo "PR-AUC Curve Generation"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Evaluate SwinGait M1 and save probabilities"
echo "2. Evaluate DeepGaitV2 M6 and save probabilities"
echo "3. Create PR-AUC curves for both models"
echo ""
echo "Make sure you're in the conda environment with PyTorch!"
echo ""

# Check if torch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Please activate your conda environment first."
    echo "Example: conda activate <your_env>"
    exit 1
}

# Note: Environment variables are set inline with each command
# to ensure they're passed to the subprocess correctly

# Fix LD_LIBRARY_PATH for OpenCV compatibility
# Prioritize conda environment's lib directory
CONDA_ENV_LIB="$CONDA_PREFIX/lib"
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_ENV_LIB" ]; then
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$CONDA_ENV_LIB:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$CONDA_ENV_LIB"
    fi
    echo "Set LD_LIBRARY_PATH to prioritize: $CONDA_ENV_LIB"
fi

# Device and process settings
DEVICE="${1:-0,1}"
NPROC="${2:-2}"

echo "Using GPUs: $DEVICE"
echo "Number of processes: $NPROC"
echo ""

# SwinGait M1
echo "=========================================="
echo "Evaluating SwinGait M1..."
echo "=========================================="
SAVE_PROBS_FOR_ROC=1 SAVE_PROBS_MODEL_NAME=SwinGait_M1 python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    --master_port=29500 \
    opengait/main.py \
    --cfgs configs/swingait/swin_part4_deepgaitv2_comparison.yaml \
    --phase test \
    --iter 10000

# DeepGaitV2 M6
echo ""
echo "=========================================="
echo "Evaluating DeepGaitV2 M6..."
echo "=========================================="
SAVE_PROBS_FOR_ROC=1 SAVE_PROBS_MODEL_NAME=DeepGaitV2_M6 python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    --master_port=29501 \
    opengait/main.py \
    --cfgs configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml \
    --phase test \
    --iter 10000

# Create PR curves from saved probabilities
echo ""
echo "=========================================="
echo "Creating PR-AUC curves..."
echo "=========================================="
python create_pr_curves_from_existing_probs.py

echo ""
echo "=========================================="
echo "Done! Check the generated PNG files:"
echo "  - pr_curves_SwinGait_M1.png"
echo "  - pr_curves_DeepGaitV2_M6.png"
echo "=========================================="

