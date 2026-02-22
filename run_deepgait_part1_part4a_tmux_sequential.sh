#!/bin/bash
# Create tmux session to run DeepGaitV2 Part 1 and Part 4a B1-B4 configs sequentially
# All configs run one after another automatically in a single window

SESSION_NAME="deepgait_part1_part4a_seq"
CONDA_ENV="myGait38"
CONDA_INIT="${HOME}/r38/miniconda3/etc/profile.d/conda.sh"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session (detached)
tmux new-session -d -s "$SESSION_NAME"

# Rename the window
tmux rename-window -t "$SESSION_NAME:0" "sequential_training"

# Initialize conda and activate environment
if [ -f "$CONDA_INIT" ]; then
    tmux send-keys -t "$SESSION_NAME:sequential_training" "source $CONDA_INIT" C-m
fi
tmux send-keys -t "$SESSION_NAME:sequential_training" "conda activate $CONDA_ENV" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "cd /cis/home/lmcdan11/Documents_Swin/OpenGait" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "echo 'DeepGaitV2 Part 1 and Part 4a B1-B4 Sequential Training'" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "echo 'All configs will run one after another automatically'" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "echo '=========================================='" C-m
tmux send-keys -t "$SESSION_NAME:sequential_training" "echo ''" C-m

# Build the sequential command
SEQUENTIAL_SCRIPT="./run_deepgait_part1_part4a_sequential.sh"

# Run the sequential script
tmux send-keys -t "$SESSION_NAME:sequential_training" "$SEQUENTIAL_SCRIPT" C-m

echo ""
echo "=========================================="
echo "Tmux session '$SESSION_NAME' created!"
echo "=========================================="
echo ""
echo "The session is running all configs sequentially."
echo ""
echo "To attach and watch progress:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (while inside):"
echo "  Ctrl+b then d"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
