#!/usr/bin/env python3
"""
Extract GaitSet embeddings from silhouette sequences for frailty classification.

This script follows Liu et al. (2021) pipeline:
1. Load silhouette sequences (from PNG files or pickle)
2. Extract gait features using GaitSet
3. Save embeddings for classification (AlexNet/VGG16)

Usage:
    python extract_gaitset_embeddings.py \
        --config configs/gaitset/gaitset.yaml \
        --checkpoint path/to/checkpoint.pt \
        --input /path/to/silhouettes \
        --output embeddings.npy \
        --batch_size 16
"""

import argparse
import os
import sys
import random
import logging
from datetime import timedelta
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pickle

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch MessageManager to handle missing logger without editing msg_manager.py
# We'll patch the log_info method to create logger on demand
from opengait.utils.msg_manager import MessageManager

# Get the original log_info method
import types
_original_log_info = MessageManager.log_info

# Create a new method that wraps the original
def _safe_log_info(self, *args, **kwargs):
    # Ensure logger exists before calling original method
    if not hasattr(self, 'logger') or self.logger is None:
        self.logger = logging.getLogger('opengait')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s'))
        self.logger.addHandler(handler)
    # Call original method
    return _original_log_info(self, *args, **kwargs)

# Replace the method on the class
MessageManager.log_info = _safe_log_info

from opengait.utils.common import config_loader
from opengait.utils import get_msg_mgr
from opengait.modeling.models.gaitset import GaitSet


def load_gaitset_model(config_path, checkpoint_path, device='cuda'):
    """Load GaitSet model from config and checkpoint"""
    # Initialize minimal distributed environment for single-GPU inference
    # Use unique port per process to avoid conflicts when running in parallel
    # MUST initialize distributed BEFORE calling get_msg_mgr() because get_msg_mgr() checks get_rank()
    if not torch.distributed.is_initialized():
        max_retries = 5
        for attempt in range(max_retries):
            try:
                import random
                os.environ.setdefault('MASTER_ADDR', 'localhost')
                # Use process ID + random number + attempt to ensure unique port per process
                base_port = 12356
                process_id = os.getpid()
                unique_port = base_port + (process_id % 1000) + random.randint(0, 1000) + (attempt * 100)
                os.environ['MASTER_PORT'] = str(unique_port)
                
                backend = 'nccl' if device == 'cuda' and torch.cuda.is_available() else 'gloo'
                torch.distributed.init_process_group(
                    backend=backend,
                    rank=0,
                    world_size=1,
                    init_method='env://',
                    timeout=timedelta(seconds=10)  # Shorter timeout to fail faster
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, try to continue without distributed
                    print(f"Warning: Could not initialize distributed after {max_retries} attempts: {e}")
                    print("Attempting to continue without distributed initialization...")
                    # Set a flag to handle get_msg_mgr() gracefully
                    os.environ['OPENGAIT_NO_DISTRIBUTED'] = '1'
                else:
                    # Try again with different port
                    continue
    
    # Load config
    cfg = config_loader(config_path)
    
    # Disable automatic checkpoint loading during model initialization
    # We'll load it manually after model creation
    if 'evaluator_cfg' in cfg:
        cfg['evaluator_cfg']['restore_hint'] = 0  # 0 means don't auto-load
    
    # For inference, we don't need the dataset, but the model tries to create a loader
    # Create a dummy dataset directory to satisfy the requirement
    if 'data_cfg' in cfg and 'dataset_root' in cfg['data_cfg']:
        dummy_root = '/tmp/dummy_gaitset_dataset'
        os.makedirs(dummy_root, exist_ok=True)
        # Create a dummy JSON partition file
        dummy_partition = '/tmp/dummy_gaitset_partition.json'
        import json
        with open(dummy_partition, 'w') as f:
            json.dump({"TRAIN_SET": [], "TEST_SET": []}, f)
        cfg['data_cfg']['dataset_root'] = dummy_root
        cfg['data_cfg']['dataset_partition'] = dummy_partition
        cfg['data_cfg']['cache'] = False
    
    # Initialize message manager BEFORE creating model (model needs it)
    # get_msg_mgr() requires distributed to be initialized (it calls get_rank())
    msg_mgr = get_msg_mgr()
    engine_cfg = cfg['evaluator_cfg']
    output_path = os.path.join('output/', cfg['data_cfg']['dataset_name'],
                               cfg['model_cfg']['model'], engine_cfg['save_name'])
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Set up logger directly on the singleton instance
    # This must be done before model creation since model.__init__ calls log_info
    msg_mgr.logger = logging.getLogger('opengait')
    msg_mgr.logger.setLevel(logging.INFO)
    msg_mgr.logger.propagate = False
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s]: %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    msg_mgr.logger.addHandler(console)
    
    
    # Create model
    model = GaitSet(cfg, training=False)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading checkpoint: {e}")
            print("Attempting to load with partial matching...")
            model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No checkpoint provided or file not found, using random weights")
        if checkpoint_path:
            print(f"  Checkpoint path: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, cfg


def load_silhouette_sequence(input_path, max_frames=None, target_size=(64, 64)):
    """
    Load silhouette sequence from directory of PNG files or pickle file.
    
    Args:
        input_path: Path to directory with PNG files or .pkl file
        max_frames: Maximum number of frames to load (None = all)
        target_size: Target size (width, height) for resizing
    
    Returns:
        silhouette_seq: numpy array of shape [S, H, W] or [N, S, H, W]
    """
    input_path = Path(input_path)
    
    if input_path.suffix == '.pkl':
        # Load from pickle file
        print(f"Loading silhouette sequence from pickle: {input_path}")
        with open(input_path, 'rb') as f:
            silhouette_seq = pickle.load(f)
        
        # Convert to numpy if tensor
        if torch.is_tensor(silhouette_seq):
            silhouette_seq = silhouette_seq.cpu().numpy()
        
        # Handle different shapes
        if len(silhouette_seq.shape) == 3:  # [S, H, W]
            pass  # Already correct
        elif len(silhouette_seq.shape) == 4:  # [N, S, H, W] or [S, C, H, W]
            if silhouette_seq.shape[1] > 100:  # Likely [N, S, H, W]
                silhouette_seq = silhouette_seq[0]  # Take first batch
            else:  # Likely [S, C, H, W]
                silhouette_seq = silhouette_seq[:, 0]  # Take first channel
        elif len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
            silhouette_seq = silhouette_seq[0, 0]  # Take first batch and channel
        
        # Limit frames if needed
        if max_frames and silhouette_seq.shape[0] > max_frames:
            indices = np.linspace(0, silhouette_seq.shape[0] - 1, max_frames, dtype=int)
            silhouette_seq = silhouette_seq[indices]
        
        print(f"Loaded sequence shape: {silhouette_seq.shape}")
        return silhouette_seq
    
    else:
        # Load from directory of PNG files
        print(f"Loading silhouette images from directory: {input_path}")
        image_files = sorted(input_path.glob('*.png'))
        if not image_files:
            image_files = sorted(input_path.glob('*.jpg'))
        
        if not image_files:
            raise ValueError(f"No PNG or JPG files found in {input_path}")
        
        # Limit frames if needed
        if max_frames and len(image_files) > max_frames:
            indices = np.linspace(0, len(image_files) - 1, max_frames, dtype=int)
            image_files = [image_files[i] for i in indices]
        
        # Load and resize images
        images = []
        for img_file in tqdm(image_files, desc="Loading images"):
            img = Image.open(img_file).convert('L')  # Convert to grayscale
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            images.append(np.array(img))
        
        silhouette_seq = np.stack(images, axis=0)  # [S, H, W]
        print(f"Loaded {len(images)} frames, shape: {silhouette_seq.shape}")
        return silhouette_seq


def prepare_input(silhouette_seq, device='cuda'):
    """
    Prepare silhouette sequence for GaitSet input.
    
    GaitSet expects: [n, s, h, w] where:
    - n = batch size
    - s = sequence length
    - h, w = height, width
    
    Returns:
        inputs: tuple (ipts, labs, _, _, seqL) as expected by GaitSet
    """
    # Convert to tensor
    if isinstance(silhouette_seq, np.ndarray):
        silhouette_seq = torch.from_numpy(silhouette_seq).float()
    
    # Normalize to [0, 1]
    if silhouette_seq.max() > 1:
        silhouette_seq = silhouette_seq / 255.0
    
    # Ensure shape is [S, H, W]
    if len(silhouette_seq.shape) == 4:  # [N, S, H, W]
        silhouette_seq = silhouette_seq[0]  # Take first batch
    elif len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
        silhouette_seq = silhouette_seq[0, 0]  # Take first batch and channel
    
    # Add batch dimension: [1, S, H, W]
    if len(silhouette_seq.shape) == 3:
        silhouette_seq = silhouette_seq.unsqueeze(0)
    
    silhouette_seq = silhouette_seq.to(device)
    seq_len = silhouette_seq.shape[1]
    
    # Create inputs tuple as expected by GaitSet
    # Format: (ipts, labs, _, _, seqL)
    ipts = [silhouette_seq]  # [n, s, h, w]
    labs = torch.zeros(silhouette_seq.shape[0], dtype=torch.long).to(device)  # Dummy labels
    seqL = [torch.tensor([seq_len], dtype=torch.int).to(device)]  # Sequence lengths
    
    return (ipts, labs, None, None, seqL)


def extract_embeddings(model, inputs, device='cuda'):
    """Extract embeddings from GaitSet model"""
    with torch.no_grad():
        retval = model.forward(inputs)
        embeddings = retval['inference_feat']['embeddings']
    
    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description='Extract GaitSet embeddings from silhouette sequences'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to GaitSet config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to GaitSet checkpoint file (optional, uses random weights if not provided)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input: directory with PNG files or .pkl file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for embeddings (.npy file)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (for processing multiple sequences)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (None = all)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64],
                        help='Target size for resizing silhouettes [width height] (default: 64 64)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("=" * 60)
    print("Loading GaitSet model...")
    print("=" * 60)
    model, cfg = load_gaitset_model(args.config, args.checkpoint, args.device)
    
    # Load silhouette sequence
    print("\n" + "=" * 60)
    print("Loading silhouette sequence...")
    print("=" * 60)
    silhouette_seq = load_silhouette_sequence(
        args.input,
        max_frames=args.max_frames,
        target_size=tuple(args.target_size)
    )
    
    # Prepare input
    print("\n" + "=" * 60)
    print("Preparing input...")
    print("=" * 60)
    inputs = prepare_input(silhouette_seq, args.device)
    print(f"Input shape: {inputs[0][0].shape}")
    print(f"Sequence length: {inputs[4][0].item()}")
    
    # Extract embeddings
    print("\n" + "=" * 60)
    print("Extracting embeddings...")
    print("=" * 60)
    embeddings = extract_embeddings(model, inputs, args.device)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Save embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"\n✓ Saved embeddings to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Frames processed: {silhouette_seq.shape[0]}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Output: {output_path}")
    print("\nNext step: Use embeddings with AlexNet/VGG16 for frailty classification")


if __name__ == '__main__':
    main()

