#!/bin/bash

# Script to run both Part 1 and Part 4a B1 configs in tmux
# Usage: ./run_both_configs_tmux.sh

SESSION_NAME="swingait_training"

# Create a new tmux session (detached)
tmux new-session -d -s "$SESSION_NAME"

# Split window into two panes (horizontal split)
tmux split-window -h -t "$SESSION_NAME:0"

# Run Part 1 config in left pane (pane 0)
tmux send-keys -t "$SESSION_NAME:0.0" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part1_p_CNN.yaml --phase train --log_to_file" C-m

# Run Part 4a B1 config in right pane (pane 1)
tmux send-keys -t "$SESSION_NAME:0.1" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs configs/swingait/swin_part4a_B1_frozen_cnn.yaml --phase train --log_to_file" C-m

# Attach to the session
echo "Tmux session '$SESSION_NAME' created with both training runs!"
echo "Left pane: Part 1 (swin_part1_p_CNN.yaml)"
echo "Right pane: Part 4a B1 (swin_part4a_B1_frozen_cnn.yaml)"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Attaching now..."
sleep 2
tmux attach -t "$SESSION_NAME"
