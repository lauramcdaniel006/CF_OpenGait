#!/bin/bash
# Example usage of extract_gaitset_embeddings.py

# Example 1: Extract embeddings from a pickle file
python extract_gaitset_embeddings.py \
    --config configs/gaitset/gaitset.yaml \
    --checkpoint path/to/gaitset_checkpoint.pt \
    --input /cis/net/r38/data/lmcdan11/sils/350/Prefrail/silhouettes/silhouettes.pkl \
    --output embeddings_350_prefrail.npy \
    --max_frames 50 \
    --device cuda

# Example 2: Extract embeddings from a directory of PNG files
python extract_gaitset_embeddings.py \
    --config configs/gaitset/gaitset.yaml \
    --checkpoint path/to/gaitset_checkpoint.pt \
    --input /cis/net/r38/data/lmcdan11/Output_Patients/rotatetest/go/out/29444_rotated_90 \
    --output embeddings_29444.npy \
    --max_frames 100 \
    --target_size 64 64 \
    --device cuda

# Example 3: Process without checkpoint (uses random weights - for testing only)
python extract_gaitset_embeddings.py \
    --config configs/gaitset/gaitset.yaml \
    --input /path/to/silhouettes \
    --output test_embeddings.npy \
    --device cuda

