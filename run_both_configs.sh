#!/bin/bash

# Script to run both configs sequentially in a tmux session
# Usage: ./run_both_configs.sh

# Configuration
SESSION_NAME="opengait_training"
WORK_DIR="/cis/home/lmcdan11/Documents_Swin/OpenGait"
CONFIG1="configs/swingait/swin_part1_pretrained_unfrozen.yaml"
CONFIG2="configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml"
GPUS="0,1"
NUM_GPUS=2

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenGait Sequential Training Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Session name: ${GREEN}${SESSION_NAME}${NC}"
echo -e "Config 1: ${GREEN}${CONFIG1}${NC}"
echo -e "Config 2: ${GREEN}${CONFIG2}${NC}"
echo -e "GPUs: ${GREEN}${GPUS}${NC}"
echo ""

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Session '$SESSION_NAME' already exists!${NC}"
    echo -e "${YELLOW}Do you want to kill it and create a new one? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}Old session killed.${NC}"
    else
        echo -e "${YELLOW}Exiting. Please manually kill the session or use a different name.${NC}"
        exit 1
    fi
fi

# Create a script that will run both configs sequentially
cat > /tmp/run_sequential_training.sh << 'SCRIPT'
#!/bin/bash

# Initialize conda
source /cis/home/lmcdan11/r38/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate myGait38

# Navigate to working directory
cd /cis/home/lmcdan11/Documents_Swin/OpenGait

echo "========================================"
echo "RUN 1: swin_part1_pretrained_unfrozen.yaml"
echo "========================================"
echo "Start time: $(date)"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    opengait/main.py \
    --cfgs configs/swingait/swin_part1_pretrained_unfrozen.yaml \
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
echo "RUN 2: swin_part4a_B3_unfrozen_cnn.yaml"
echo "========================================"
echo "Start time: $(date)"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    opengait/main.py \
    --cfgs configs/swingait/swin_part4a_B3_unfrozen_cnn.yaml \
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

chmod +x /tmp/run_sequential_training.sh

# Create new tmux session and run the script
echo -e "${BLUE}Creating tmux session: ${SESSION_NAME}${NC}"
tmux new-session -d -s "$SESSION_NAME" -c "$WORK_DIR"

# Run the sequential training script
tmux send-keys -t "$SESSION_NAME" "/tmp/run_sequential_training.sh" C-m

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
