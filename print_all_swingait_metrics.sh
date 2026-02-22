#!/bin/bash
# ============================================================================
# SWINGAIT METRICS SCRIPT
# ============================================================================
# Print metrics for all SwinGait Part 1 and Part 4 experiments
# Shows ALL evaluation runs for each experiment
#
# Usage: ./print_all_swingait_metrics.sh
# ============================================================================

echo "=================================================================================="
echo "SWINGAIT PART 1 - METRICS"
echo "=================================================================================="

# Part 1 experiments
for exp in "output/REDO_Frailty_ccpg_pt1_pretrained(UF)/SwinGait/REDO_Frailty_ccpg_pt1_pretrained(UF)" \
           "output/REDO_Frailty_ccpg_pt1_p+CNN/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN" \
           "output/REDO_Frailty_ccpg_pt1_p+CNN+Tintro/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN+Tintro" \
           "output/REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1" \
           "output/REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2/SwinGait/REDO_Frailty_ccpg_pt1_p+CNN+Tintro+T1+T2"; do
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
echo "SWINGAIT PART 4 - METRICS"
echo "=================================================================================="

# Part 4 experiments
for exp in "output/REDO_Frailty_ccpg_pt4_swingait/SwinGait/REDO_Frailty_ccpg_pt4_swingait" \
           "output/REDO_Frailty_ccpg_pt4_swingait_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_with_weights" \
           "output/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn" \
           "output/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights/SwinGait/REDO_Frailty_ccpg_pt4_swingait_unfrozen_cnn_with_weights"; do
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
echo "SWINGAIT PART 4A - METRICS"
echo "=================================================================================="

# Part 4a experiments
for exp in "output/REDO_Frailty_ccpg_pt4a_swingait_B1_frozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B1_frozen_cnn" \
           "output/REDO_Frailty_ccpg_pt4a_swingait_B2_frozen_cnn_with_weights/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B2_frozen_cnn_with_weights" \
           "output/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B3_unfrozen_cnn" \
           "output/REDO_Frailty_ccpg_pt4a_swingait_B4_unfrozen_cnn_with_weights/SwinGait/REDO_Frailty_ccpg_pt4a_swingait_B4_unfrozen_cnn_with_weights"; do
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
