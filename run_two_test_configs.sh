#!/bin/bash

# Script to run two test configs sequentially for reproducibility testing
# Configs: swin_part4a_B3_unfrozen_cnn.yaml and swin_testb3.yaml.yaml

# Define session name and working directory
SESSION_NAME="opengait_test_configs"
WORK_DIR="/cis/home/lmcdan11/Documents_Swin/OpenGait"

# Define colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting sequential training script for test configs...${NC}"
echo -e "${BLUE}Working directory: $WORK_DIR${NC}"

# Create a temporary script to run inside tmux
cat << 'SCRIPT' > /tmp/run_test_configs.sh
#!/bin/bash

# Initialize conda and activate environment
source /cis/home/lmcdan11/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

cd /cis/home/lmcdan11/Documents_Swin/OpenGait

echo "========================================"
echo "RUN 1: swin_part4a_B3_unfrozen_cnn.yaml"
echo "========================================"
echo "Start time: $(date)"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    opengait/main.py \
    --cfgs configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml \
    --phase train \
    --log_to_file

EXIT_CODE1=$?

echo ""
echo "========================================"
if [ $EXIT_CODE1 -eq 0 ]; then
    echo "RUN 1 COMPLETED SUCCESSFULLY"
else
    echo "RUN 1 EXITED WITH CODE: $EXIT_CODE1"
fi
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Waiting 10 seconds before starting RUN 2..."
sleep 10
echo ""

echo "========================================"
echo "RUN 2: swin_testb3.yaml.yaml"
echo "========================================"
echo "Start time: $(date)"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    opengait/main.py \
    --cfgs configs/swingait/swin_testb3.yaml.yaml \
    --phase train \
    --log_to_file

EXIT_CODE2=$?

echo ""
echo "========================================"
if [ $EXIT_CODE2 -eq 0 ]; then
    echo "RUN 2 COMPLETED SUCCESSFULLY"
else
    echo "RUN 2 EXITED WITH CODE: $EXIT_CODE2"
fi
echo "End time: $(date)"
echo "========================================"
echo ""
echo "========================================"
echo "ALL TRAINING COMPLETED!"
echo "========================================"
echo "RUN 1 exit code: $EXIT_CODE1"
echo "RUN 2 exit code: $EXIT_CODE2"
echo "========================================"
SCRIPT

chmod +x /tmp/run_test_configs.sh

# Create new tmux session and run the script
echo -e "${BLUE}Creating tmux session: ${SESSION_NAME}${NC}"
tmux new-session -d -s "$SESSION_NAME" -c "$WORK_DIR"

# Run the sequential training script
tmux send-keys -t "$SESSION_NAME" "/tmp/run_test_configs.sh" C-m

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${BLUE}To attach to the session:${NC}"
echo -e "${GREEN}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo -e "${BLUE}To detach (keep running):${NC}"
echo -e "${GREEN}Press Ctrl+B, then D${NC}"
echo ""
echo -e "${BLUE}To kill the session:${NC}"
echo -e "${GREEN}tmux kill-session -t $SESSION_NAME${NC}"
echo ""
echo -e "${BLUE}To check session status:${NC}"
echo -e "${GREEN}tmux list-sessions${NC}"
echo ""
