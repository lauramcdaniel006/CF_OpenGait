#!/bin/bash
# Create tmux session to run DeepGaitV2 Part 1 and Part 4a B1-B4 configs
# All configs use: CUDA_VISIBLE_DEVICES=0,1, torch.distributed.launch with 2 GPUs
# NOTE: Each config uses 2 GPUs, so run them one at a time or manually manage windows

SESSION_NAME="deepgait_part1_part4a"
CONDA_ENV="myGait38"
# Path to conda initialization (adjust if needed)
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session (detached)
tmux new-session -d -s "$SESSION_NAME"

# Function to create a new window with a command ready to run
setup_config_window() {
    local config_name=$1
    local config_file=$2
    local window_name=$3
    
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Config: $config_name'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'File: $config_file'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '==========================================='" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Ready to start training. Press Enter to run...'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" C-m
    # Initialize conda and activate environment
    if [ -f "$CONDA_INIT" ]; then
        tmux send-keys -t "$SESSION_NAME:$window_name" "source $CONDA_INIT" C-m
    fi
    tmux send-keys -t "$SESSION_NAME:$window_name" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
    # Prepare the command but don't execute it (no C-m at the end)
    tmux send-keys -t "$SESSION_NAME:$window_name" "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs $config_file --phase train --log_to_file"
}

# Start with a main window
tmux rename-window -t "$SESSION_NAME:0" "main"
if [ -f "$CONDA_INIT" ]; then
    tmux send-keys -t "$SESSION_NAME:main" "source $CONDA_INIT" C-m
fi
tmux send-keys -t "$SESSION_NAME:main" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:main" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'DeepGaitV2 Part 1 and Part 4a B1-B4 Training Session'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '==========================================='" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'IMPORTANT: Each config uses 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Run configs ONE AT A TIME to avoid GPU/port conflicts'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Navigate to a window and press Enter to start that training job'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Tmux Commands:'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then n = next window'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then p = previous window'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then w = list windows'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  Ctrl+b then d = detach from session'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '  tmux attach -t $SESSION_NAME = reattach'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo ''" C-m

# ============================================================================
# PART 1 CONFIGS
# ============================================================================

echo "Setting up Part 1 configs..."

setup_config_window "Part 1: All Trainable" \
    "configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml" \
    "pt1_all_trainable"

setup_config_window "Part 1: Baseline All Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part1_baseline_all_frozen.yaml" \
    "pt1_baseline"

setup_config_window "Part 1: Early Layers Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml" \
    "pt1_early_frozen"

setup_config_window "Part 1: First Layer Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml" \
    "pt1_first_frozen"

setup_config_window "Part 1: First Two Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml" \
    "pt1_first_two"

setup_config_window "Part 1: Heavy Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml" \
    "pt1_heavy_frozen"

# ============================================================================
# PART 4a CONFIGS (B1-B4)
# ============================================================================

echo "Setting up Part 4a B1-B4 configs..."

setup_config_window "Part 4a: B1 Partially Frozen" \
    "configs/deepgaitv2/DeepGaitV2_part4a_B1_partially_frozen.yaml" \
    "pt4a_B1"

setup_config_window "Part 4a: B2 Partially Frozen With Weights" \
    "configs/deepgaitv2/DeepGaitV2_part4a_B2_partially_frozen_with_weights.yaml" \
    "pt4a_B2"

setup_config_window "Part 4a: B3 Unfrozen" \
    "configs/deepgaitv2/DeepGaitV2_part4a_B3_unfrozen.yaml" \
    "pt4a_B3"

setup_config_window "Part 4a: B4 Unfrozen With Weights" \
    "configs/deepgaitv2/DeepGaitV2_part4a_B4_unfrozen_with_weights.yaml" \
    "pt4a_B4"

# Select the first training window
tmux select-window -t "$SESSION_NAME:pt1_all_trainable"

echo ""
echo "=========================================="
echo "Tmux session '$SESSION_NAME' created!"
echo "=========================================="
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (while inside):"
echo "  Ctrl+b then d"
echo ""
echo "To navigate between windows:"
echo "  Ctrl+b then n (next) or p (previous)"
echo ""
echo "To list windows:"
echo "  Ctrl+b then w"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
