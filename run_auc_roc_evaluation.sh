#!/bin/bash
# Script to compute AUC-ROC for SwinGait M1 and DeepGaitV2 M6

echo "=========================================="
echo "Computing AUC-ROC for Part 4 Models"
echo "=========================================="

# SwinGait M1
echo ""
echo "Evaluating SwinGait M1..."
python opengait/main.py \
    --cfgs configs/swingait/swin_part4_deepgaitv2_comparison.yaml \
    --phase test \
    --iter 60000

echo ""
echo "=========================================="

# DeepGaitV2 M6
echo ""
echo "Evaluating DeepGaitV2 M6..."
python opengait/main.py \
    --cfgs configs/deepgaitv2/DeepGaitV2_part4b_half_frozen_cnn_with_weights.yaml \
    --phase test \
    --iter 60000

echo ""
echo "=========================================="
echo "Done! Check the output logs for AUC-ROC values."
echo "=========================================="

