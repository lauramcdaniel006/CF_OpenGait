#!/bin/bash
# Test single evaluation to verify AUC-ROC is computed

echo "Testing single evaluation for SwinGait M1 at iteration 10000..."
echo "=========================================="

# Use the conda environment's Python
# Adjust this to your conda environment path if needed
python opengait/main.py \
    --cfgs configs/swingait/swin_part4_deepgaitv2_comparison.yaml \
    --phase test \
    --iter 10000 2>&1 | grep -A 5 -B 5 "ROC AUC\|AUC\|auc"

echo ""
echo "=========================================="
echo "Check the output above for AUC-ROC values"
echo "If you see 'ROC AUC (macro): X.XXXX', it's working!"

