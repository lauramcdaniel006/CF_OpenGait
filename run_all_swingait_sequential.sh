#!/bin/bash
cd /cis/home/lmcdan11/Documents_Swin/CF_OpenGait
source ~/r38/miniconda3/etc/profile.d/conda.sh
conda activate myGait38

echo '========================================'
echo 'Starting: Pretrained Unfrozen'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part1_pretrained_unfrozen.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: p+CNN'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part1_p_CNN.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: p+CNN+Tintro'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part1_p_CNN_Tintro.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: p+CNN+Tintro+T1'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part1_p_CNN_Tintro_T1.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'Starting: p+CNN+Tintro+T1+T2'
echo '========================================'
python run_kfold_cross_validation.py \
  --config configs/swingait/swin_part1_p_CNN_Tintro_T1_T2.yaml \
  --k 5 \
  --use-existing-partitions \
  --device 0,1 \
  --nproc 2 &&

echo '========================================'
echo 'All models completed successfully!'
echo '========================================'
