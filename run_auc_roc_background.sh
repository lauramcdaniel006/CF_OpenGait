#!/bin/bash
# Run AUC-ROC evaluation in background, keeps running even if you disconnect

# Activate conda environment
source ~/r38_conda_envs/myGait38/bin/activate

# Change to OpenGait directory
cd /cis/home/lmcdan11/Documents_Swin/OpenGait

# Run script in background with nohup (no hang up)
# Output will be saved to nohup.out
nohup python reevaluate_for_auc_roc.py --device "0,1" --nproc 2 > auc_roc_evaluation.log 2>&1 &

# Get the process ID
PID=$!
echo "Script started with PID: $PID"
echo "Output is being saved to: auc_roc_evaluation.log"
echo "To check progress: tail -f auc_roc_evaluation.log"
echo "To check if still running: ps -p $PID"

