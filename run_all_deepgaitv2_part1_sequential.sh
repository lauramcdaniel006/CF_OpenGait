#!/bin/bash
# Run all DeepGaitV2 Part 1 models sequentially with k-fold cross-validation
# Each model will complete all 5 folds before the next model starts

set -e  # Exit on error

cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait

# Activate conda environment
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

# All 5 DeepGaitV2 Part 1 configs (excluding baseline_all_frozen which already completed)
MODELS=(
    "configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml:All_Trainable"
    "configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml:First_Layer_Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml:First_Two_Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml:Early_Layers_Frozen"
    "configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml:Heavy_Frozen"
)

echo "=========================================="
echo "Starting Sequential DeepGaitV2 Part 1 Training"
echo "Using partitions: TRAIN_SET and TEST_SET only (no VAL_SET)"
echo "Total models: ${#MODELS[@]}"
echo "=========================================="
echo ""

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r config_file model_name <<< "$model_config"
    
    echo "=========================================="
    echo "Starting: $model_name"
    echo "Config: $config_file"
    echo "Time: $(date)"
    echo "=========================================="
    
    # Run k-fold cross-validation for this model
    python run_kfold_cross_validation.py \
        --config "$config_file" \
        --k 5 \
        --use-existing-partitions \
        --device 0,1 \
        --nproc 2
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $model_name completed successfully"
        echo "Time: $(date)"
        echo ""
    else
        echo ""
        echo "✗ $model_name failed with exit code $?"
        echo "Stopping sequential execution."
        exit 1
    fi
    
    echo "Waiting 5 seconds before starting next model..."
    sleep 5
    echo ""
done

echo "=========================================="
echo "All DeepGaitV2 Part 1 models completed successfully!"
echo "Final time: $(date)"
echo "=========================================="
