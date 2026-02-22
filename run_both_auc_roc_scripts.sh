#!/bin/bash
# Run both AUC-ROC scripts (Type 1 and Type 2) sequentially

echo "=================================================================================="
echo "Running Both AUC-ROC Scripts"
echo "=================================================================================="
echo ""
echo "Type 1: AUC-ROC over training iterations (ALL checkpoints)"
echo "Type 2: Traditional ROC curves (final checkpoint)"
echo ""
echo "This will take a while - output will be saved to logs"
echo "=================================================================================="
echo ""

# Activate conda environment
source ~/r38_conda_envs/myGait38/bin/activate 2>/dev/null || conda activate myGait38

# Change to OpenGait directory
cd /cis/home/lmcdan11/Documents_Swin/OpenGait

# Create logs directory
mkdir -p logs

# Run Type 1 script
echo ""
echo "=================================================================================="
echo "STEP 1: Running Type 1 - AUC-ROC Over Training Iterations"
echo "=================================================================================="
echo "Started at: $(date)"
python reevaluate_for_auc_roc.py --device "0,1" --nproc 2 2>&1 | tee logs/type1_auc_roc.log

TYPE1_EXIT_CODE=${PIPESTATUS[0]}

if [ $TYPE1_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Type 1 completed successfully!"
    echo "Finished at: $(date)"
else
    echo ""
    echo "⚠️  Type 1 had errors (exit code: $TYPE1_EXIT_CODE)"
    echo "Check logs/type1_auc_roc.log for details"
fi

echo ""
echo "=================================================================================="
echo "STEP 2: Running Type 2 - Traditional ROC Curves"
echo "=================================================================================="
echo "Started at: $(date)"

# Run Type 2 script
python extract_probabilities_for_roc.py --device "0,1" --nproc 2 2>&1 | tee logs/type2_roc_curves.log

TYPE2_EXIT_CODE=${PIPESTATUS[0]}

if [ $TYPE2_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Type 2 completed successfully!"
    echo "Finished at: $(date)"
else
    echo ""
    echo "⚠️  Type 2 had errors (exit code: $TYPE2_EXIT_CODE)"
    echo "Check logs/type2_roc_curves.log for details"
fi

echo ""
echo "=================================================================================="
echo "SUMMARY"
echo "=================================================================================="
echo "Type 1 (AUC-ROC over iterations): $([ $TYPE1_EXIT_CODE -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo "Type 2 (Traditional ROC curves): $([ $TYPE2_EXIT_CODE -eq 0 ] && echo '✓ Success' || echo '✗ Failed')"
echo ""
echo "Logs saved to:"
echo "  - logs/type1_auc_roc.log"
echo "  - logs/type2_roc_curves.log"
echo ""
echo "Graphs should be in current directory:"
echo "  - auc_roc_curve_*.png"
echo "  - roc_curves_*.png"
echo "  - auc_roc_comparison_part4.png"
echo ""
echo "Completed at: $(date)"
echo "=================================================================================="

