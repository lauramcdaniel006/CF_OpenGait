#!/bin/bash
cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

echo '========================================'
echo 'Starting: DeepGaitV2 Part 4a B1 (Early Frozen)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part4a_B1_partially_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: DeepGaitV2 Part 4a B2 (Early Frozen + Weights)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part4a_B2_partially_frozen_with_weights.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: DeepGaitV2 Part 4a B4 (Unfrozen + Weights)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part4a_B4_unfrozen_with_weights.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'All DeepGaitV2 Part 4a models (B1, B2, B4) completed successfully!'
echo '========================================'
