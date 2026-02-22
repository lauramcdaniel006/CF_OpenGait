#!/bin/bash
# Script to run AUC-ROC graph generation with CUDA device specification

# Default settings
DEVICE=${1:-"0,1"}  # Use GPU 0 and 1 by default, or specify: ./RUN_AUC_ROC_GRAPHS.sh "0"
NPROC=${2:-2}        # Number of processes (should match number of GPUs)

echo "=========================================="
echo "Generating AUC-ROC Graphs"
echo "=========================================="
echo "CUDA Devices: $DEVICE"
echo "Number of Processes: $NPROC"
echo "=========================================="

# Step 1: Get Type 1 graphs (AUC over iterations)
echo ""
echo "Step 1: Creating AUC-ROC over iterations..."
echo "------------------------------------------"
python reevaluate_for_auc_roc.py --device "$DEVICE" --nproc $NPROC

# Step 2: Get Type 2 graphs (Traditional ROC curves)
echo ""
echo "Step 2: Creating traditional ROC curves..."
echo "------------------------------------------"
python extract_probabilities_for_roc.py --device "$DEVICE" --nproc $NPROC

echo ""
echo "=========================================="
echo "Done! Check the output PNG files."
echo "=========================================="

