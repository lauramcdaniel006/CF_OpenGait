#!/bin/bash
cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

echo '========================================'
echo 'Starting: All Trainable'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_all_trainable.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: Early Frozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_early_layers_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: First Layer Frozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_first_layer_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: First Two Frozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_first_two_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: Heavy Frozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_heavy_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: All Frozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/deepgaitv2/DeepGaitV2_part1_baseline_all_frozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'All models completed successfully!'
echo '========================================'
