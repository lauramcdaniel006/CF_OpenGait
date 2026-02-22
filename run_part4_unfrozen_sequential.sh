#!/bin/bash
cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

echo '========================================'
echo 'Starting: SwinGait Part 4 Unfrozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part4_deepgaitv2_comparison_unfrozen_cnn.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: DeepGaitV2 Part 4 Unfrozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part4_swingait_comparison_unfrozen_cnn.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'All Part 4 Unfrozen models completed successfully!'
echo '========================================'
