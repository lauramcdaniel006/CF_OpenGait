#!/bin/bash
# Auto-start script for running all configs in tmux

cd /cis/home/lmcdan11/Documents_Swin/OpenGait

# Set up environment
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1"

# Start tmux session with the training script
tmux new-session -d -s training_all "bash run_all_configs.sh 2>&1 | tee training_log.txt"

echo "✅ Training session started in tmux!"
echo "To attach: tmux attach -t training_all"
echo "To check status: tmux list-sessions"
echo "To view output: tmux capture-pane -t training_all -p | tail -50"
