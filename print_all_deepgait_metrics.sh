#!/bin/bash
# ============================================================================
# DEEPGaitV2 METRICS SCRIPT
# ============================================================================
# Print metrics for all DeepGaitV2 Part 1 and Part 4 experiments
# Shows ALL evaluation runs for each experiment
#
# Usage: ./print_all_deepgait_metrics.sh
# ============================================================================

echo "=================================================================================="
echo "DEEPGaitV2 PART 1 - FREEZING STRATEGIES - METRICS"
echo "=================================================================================="

# Part 1 experiments
for exp in "output/REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable" \
           "output/REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen" \
           "output/REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen" \
           "output/REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen" \
           "output/REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen" \
           "output/REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen/DeepGaitV2/REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen"; do
    if [ -d "$exp" ]; then
        exp_name=$(basename "$exp")
        echo ""
        echo "=================================================================================="
        echo "EXPERIMENT: $exp_name"
        echo "=================================================================================="
        python3 print_metrics_from_logs.py "$exp" 2>/dev/null
        echo ""
    fi
done

echo ""
echo "=================================================================================="
echo "DEEPGaitV2 PART 4 - CLASS WEIGHTS EXPERIMENTS - METRICS"
echo "=================================================================================="

# Part 4 experiments
for exp in "output/REDO_Frailty_ccpg_pt4_deepgaitv2/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2" \
           "output/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_with_weights" \
           "output/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn" \
           "output/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights/DeepGaitV2/REDO_Frailty_ccpg_pt4_deepgaitv2_unfrozen_cnn_with_weights"; do
    if [ -d "$exp" ]; then
        exp_name=$(basename "$exp")
        echo ""
        echo "=================================================================================="
        echo "EXPERIMENT: $exp_name"
        echo "=================================================================================="
        python3 print_metrics_from_logs.py "$exp" 2>/dev/null
        echo ""
    fi
done
