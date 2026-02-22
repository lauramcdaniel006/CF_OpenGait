#!/bin/bash
# Run all models sequentially (Tintro, T1, T2, UF) with k-fold cross-validation
# Each model will complete all 5 folds before the next model starts

set -e  # Exit on error

cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait

# Activate conda environment
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

# Models to run (excluding p+CNN as requested)
MODELS=(
    "configs/swingait/swin_part1_p_CNN_Tintro.yaml:Tintro"
    "configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml:T1"
    "configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml:T2"
    "configs/swingait/swin_part1_pretrained_unfrozen.yaml:UF"
)

echo "=========================================="
echo "Starting Sequential Model Training"
echo "Using partitions: TRAIN_SET and TEST_SET only (no VAL_SET)"
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
echo "All models completed successfully!"
echo "Final time: $(date)"
echo "=========================================="
