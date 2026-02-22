#!/bin/bash
# Script to run both PR-AUC scripts (Type 1 and Type 2)

echo "=========================================="
echo "Running PR-AUC Scripts"
echo "=========================================="

# Default values
DEVICE="${1:-0,1}"
NPROC="${2:-2}"

echo "Using GPUs: $DEVICE"
echo "Number of processes: $NPROC"
echo ""

# Type 1: PR-AUC over training iterations
echo "=========================================="
echo "Type 1: PR-AUC Over Training Iterations"
echo "=========================================="
python reevaluate_for_pr_auc.py --device "$DEVICE" --nproc "$NPROC"

echo ""
echo "=========================================="
echo "Type 2: Traditional PR Curves (Precision vs Recall)"
echo "=========================================="
python extract_probabilities_for_pr_auc.py --device "$DEVICE" --nproc "$NPROC"

echo ""
echo "=========================================="
echo "Done! Check the generated PNG files:"
echo "  - pr_auc_curve_SwinGait_M1.png"
echo "  - pr_auc_curve_DeepGaitV2_M6.png"
echo "  - pr_auc_comparison_part4.png"
echo "  - pr_curves_SwinGait_M1.png"
echo "  - pr_curves_DeepGaitV2_M6.png"
echo "=========================================="

