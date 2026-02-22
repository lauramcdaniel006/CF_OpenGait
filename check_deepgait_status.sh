#!/bin/bash
# Check status of DeepGaitV2 Part 1 and Part 4a configs

echo "=========================================="
echo "DeepGaitV2 Part 1 & Part 4a Status Check"
echo "=========================================="
echo ""

# Part 1 configs
echo "=== PART 1 CONFIGS ==="
declare -a part1_configs=(
    "DeepGaitV2_part1_all_trainable|REDO_Frailty_ccpg_pt1_deepgaitv2_all_trainable|Part 1: All Trainable"
    "DeepGaitV2_part1_baseline_all_frozen|REDO_Frailty_ccpg_pt1_deepgaitv2_all_frozen|Part 1: Baseline All Frozen"
    "DeepGaitV2_part1_early_layers_frozen|REDO_Frailty_ccpg_pt1_deepgaitv2_early_frozen|Part 1: Early Layers Frozen"
    "DeepGaitV2_part1_first_layer_frozen|REDO_Frailty_ccpg_pt1_deepgaitv2_first_layer_frozen|Part 1: First Layer Frozen"
    "DeepGaitV2_part1_first_two_frozen|REDO_Frailty_ccpg_pt1_deepgaitv2_first_two_frozen|Part 1: First Two Frozen"
    "DeepGaitV2_part1_heavy_frozen|REDO_Frailty_ccpg_pt1_deepgaitv2_heavy_frozen|Part 1: Heavy Frozen"
)

part1_done=0
part1_total=${#part1_configs[@]}

for config_entry in "${part1_configs[@]}"; do
    IFS='|' read -r config_file save_name config_desc <<< "$config_entry"
    output_dir="output/$save_name/DeepGaitV2/$save_name"
    
    if [ -d "$output_dir/checkpoints" ]; then
        # Check for final checkpoint (10000 iterations)
        if [ -f "$output_dir/checkpoints/$save_name-10000.pt" ]; then
            echo "✓ $config_desc"
            part1_done=$((part1_done + 1))
        else
            # Check latest checkpoint
            latest=$(ls -1 "$output_dir/checkpoints" 2>/dev/null | tail -1 | sed 's/.*-\([0-9]*\)\.pt/\1/')
            if [ -n "$latest" ]; then
                echo "⚠ $config_desc (partial: iter $latest)"
            else
                echo "✗ $config_desc (no checkpoints)"
            fi
        fi
    else
        echo "✗ $config_desc (not started)"
    fi
done

echo ""
echo "Part 1: $part1_done/$part1_total completed"
echo ""

# Part 4a configs
echo "=== PART 4a CONFIGS ==="
declare -a part4a_configs=(
    "DeepGaitV2_part4a_B1_partially_frozen|REDO_Frailty_ccpg_pt4a_deepgaitv2_B1_partially_frozen|Part 4a: B1 Partially Frozen"
    "DeepGaitV2_part4a_B2_partially_frozen_with_weights|REDO_Frailty_ccpg_pt4a_deepgaitv2_B2_partially_frozen_with_weights|Part 4a: B2 Partially Frozen With Weights"
    "DeepGaitV2_part4a_B3_unfrozen|REDO_Frailty_ccpg_pt4a_deepgaitv2_B3_unfrozen|Part 4a: B3 Unfrozen"
    "DeepGaitV2_part4a_B4_unfrozen_with_weights|REDO_Frailty_ccpg_pt4a_deepgaitv2_B4_unfrozen_with_weights|Part 4a: B4 Unfrozen With Weights"
)

part4a_done=0
part4a_total=${#part4a_configs[@]}

for config_entry in "${part4a_configs[@]}"; do
    IFS='|' read -r config_file save_name config_desc <<< "$config_entry"
    output_dir="output/$save_name/DeepGaitV2/$save_name"
    
    if [ -d "$output_dir/checkpoints" ]; then
        # Check for final checkpoint (10000 iterations)
        if [ -f "$output_dir/checkpoints/$save_name-10000.pt" ]; then
            echo "✓ $config_desc"
            part4a_done=$((part4a_done + 1))
        else
            # Check latest checkpoint
            latest=$(ls -1 "$output_dir/checkpoints" 2>/dev/null | tail -1 | sed 's/.*-\([0-9]*\)\.pt/\1/')
            if [ -n "$latest" ]; then
                echo "⚠ $config_desc (partial: iter $latest)"
            else
                echo "✗ $config_desc (no checkpoints)"
            fi
        fi
    else
        echo "✗ $config_desc (not started)"
    fi
done

echo ""
echo "Part 4a: $part4a_done/$part4a_total completed"
echo ""

# Summary
total_done=$((part1_done + part4a_done))
total_total=$((part1_total + part4a_total))

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total: $total_done/$total_total completed"
echo ""

if [ $total_done -eq $total_total ]; then
    echo "🎉 All configs are completed!"
else
    remaining=$((total_total - total_done))
    echo "⚠️  $remaining config(s) remaining"
fi
