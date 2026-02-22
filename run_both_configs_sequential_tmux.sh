#!/bin/bash

# Script to run both Part 1 and Part 4a B1 configs sequentially in tmux
# Runs Part 1 first, then Part 4a B1 after Part 1 completes
# Usage: ./run_both_configs_sequential_tmux.sh

SESSION_NAME="swingait_training_seq"
CONDA_ENV="myGait38"
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create a new tmux session (detached)
tmux new-session -d -s "$SESSION_NAME"

# Initialize conda and activate environment
if [ -f "$CONDA_INIT" ]; then
    tmux send-keys -t "$SESSION_NAME:0" "source $CONDA_INIT" C-m
fi
tmux send-keys -t "$SESSION_NAME:0" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:0" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
# Fix library compatibility issues (GLIBCXX)
tmux send-keys -t "$SESSION_NAME:0" "export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH}\"" C-m
tmux send-keys -t "$SESSION_NAME:0" "if [ -f \"\$CONDA_PREFIX/lib/libstdc++.so.6\" ]; then export LD_PRELOAD=\"\$CONDA_PREFIX/lib/libstdc++.so.6\"; fi" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Starting Part 1 Training (swin_part1_p_CNN.yaml) ==='" C-m
tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN.yaml --phase train --log_to_file" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Part 1 Training Complete ==='" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Starting Part 4a B1 Training (swin_part4a_B1_frozen_cnn.yaml) ==='" C-m
tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B1_frozen_cnn.yaml --phase train --log_to_file" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo '=== Both Training Runs Complete ==='" C-m

# Attach to the session
echo "Tmux session '$SESSION_NAME' created!"
echo "This will run Part 1 first, then Part 4a B1 sequentially."
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Attaching now..."
sleep 2
tmux attach -t "$SESSION_NAME"
