#!/bin/bash

# Script to create a tmux session for running SwinGait Part 2 class weighting configs sequentially

SESSION_NAME="swingait_part2_seq"
CONDA_ENV="myGait38"
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new detached session
tmux new-session -d -s "$SESSION_NAME"

# Rename the window
tmux rename-window -t "$SESSION_NAME:0" "part2_classweights"

# Send commands to setup environment and run the sequential script
if [ -f "$CONDA_INIT" ]; then
    tmux send-keys -t "$SESSION_NAME:part2_classweights" "source $CONDA_INIT" C-m
fi

tmux send-keys -t "$SESSION_NAME:part2_classweights" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:part2_classweights" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
tmux send-keys -t "$SESSION_NAME:part2_classweights" "./run_swingait_part2_sequential.sh" C-m

echo "=========================================="
echo "Tmux session created: $SESSION_NAME"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (while inside):"
echo "  Ctrl+b then d"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo "=========================================="
