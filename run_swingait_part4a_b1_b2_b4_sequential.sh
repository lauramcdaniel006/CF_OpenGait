#!/bin/bash
cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

echo '========================================'
echo 'Starting: SwinGait Part 4a B1 (Frozen CNN)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part4a_B1_frozen_cnn.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: SwinGait Part 4a B2 (Frozen CNN + Weights)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part4a_B2_frozen_cnn_with_weights.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: SwinGait Part 4a B4 (Unfrozen CNN + Weights)'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part4a_B4_unfrozen_cnn_with_weights.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'All SwinGait Part 4a models (B1, B2, B4) completed successfully!'
echo '========================================'
