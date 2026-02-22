#!/usr/bin/env python3
"""
Grad-CAM visualization for SwinGait model
Visualizes which parts of the input silhouette sequence are important for frailty classification

IMPORTANT: Run this script from the OpenGait root directory (where opengait/ folder is located).
"""

import sys
import os

# CRITICAL: Set up path BEFORE any other imports
# Get the root directory (where this script is located)
root_dir = os.path.dirname(os.path.abspath(__file__))

# Add root directory to path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Create module aliases for OpenGait's absolute imports
# OpenGait code uses 'from utils import ...', 'from evaluation import ...', etc.
# but these are actually opengait.utils, opengait.evaluation, etc.
import types

# Create fake modules for all OpenGait top-level imports
def create_module_alias(module_name, real_module_path):
    """Create a fake module that aliases to the real module"""
    fake_module = types.ModuleType(module_name)
    sys.modules[module_name] = fake_module
    return fake_module

# Create aliases before importing anything from opengait
utils_module = create_module_alias('utils', 'opengait.utils')
evaluation_module = create_module_alias('evaluation', 'opengait.evaluation')
data_module = create_module_alias('data', 'opengait.data')
modeling_module = create_module_alias('modeling', 'opengait.modeling')

# Make data_module a proper package so 'from data import transform' works
data_module.__path__ = [os.path.join(root_dir, 'opengait', 'data')]
data_module.__package__ = 'data'

# Now import everything else
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Import real modules and populate fake modules
# This must be done carefully to avoid circular imports
import opengait.utils as opengait_utils
for attr_name in dir(opengait_utils):
    if not attr_name.startswith('_'):
        setattr(utils_module, attr_name, getattr(opengait_utils, attr_name))

# Import evaluation submodules and populate fake evaluation module
# Set up evaluator as a submodule first
evaluation_module.evaluator = types.ModuleType('evaluation.evaluator')
evaluation_module.metric = types.ModuleType('evaluation.metric')
evaluation_module.re_rank = types.ModuleType('evaluation.re_rank')

# Register in sys.modules
sys.modules['evaluation.evaluator'] = evaluation_module.evaluator
sys.modules['evaluation.metric'] = evaluation_module.metric
sys.modules['evaluation.re_rank'] = evaluation_module.re_rank

# Now import the real modules
import opengait.evaluation.metric as eval_metric
import opengait.evaluation.evaluator as eval_evaluator
import opengait.evaluation.re_rank as eval_re_rank

# Populate evaluation module with functions from metric
for attr_name in dir(eval_metric):
    if not attr_name.startswith('_'):
        setattr(evaluation_module.metric, attr_name, getattr(eval_metric, attr_name))
        setattr(evaluation_module, attr_name, getattr(eval_metric, attr_name))

# Populate evaluator submodule
for attr_name in dir(eval_evaluator):
    if not attr_name.startswith('_'):
        setattr(evaluation_module.evaluator, attr_name, getattr(eval_evaluator, attr_name))
        setattr(evaluation_module, attr_name, getattr(eval_evaluator, attr_name))

# Populate re_rank submodule
for attr_name in dir(eval_re_rank):
    if not attr_name.startswith('_'):
        setattr(evaluation_module.re_rank, attr_name, getattr(eval_re_rank, attr_name))

# Import data submodules and populate fake data module
# Need to set up the structure first to avoid circular imports
# Create submodules in data_module first
data_module.transform = types.ModuleType('data.transform')
data_module.dataset = types.ModuleType('data.dataset')
data_module.collate_fn = types.ModuleType('data.collate_fn')
data_module.sampler = types.ModuleType('data.sampler')

# Register them in sys.modules so 'from data import transform' works
sys.modules['data.transform'] = data_module.transform
sys.modules['data.dataset'] = data_module.dataset
sys.modules['data.collate_fn'] = data_module.collate_fn
sys.modules['data.sampler'] = data_module.sampler

# Now import the real modules (they can now use 'from data import transform')
import opengait.data.transform as data_transform
import opengait.data.dataset as data_dataset
import opengait.data.collate_fn as data_collate_fn
import opengait.data.sampler as data_sampler

# Populate the fake submodules with real content
for attr_name in dir(data_transform):
    if not attr_name.startswith('_'):
        setattr(data_module.transform, attr_name, getattr(data_transform, attr_name))
for attr_name in dir(data_dataset):
    if not attr_name.startswith('_'):
        setattr(data_module.dataset, attr_name, getattr(data_dataset, attr_name))
for attr_name in dir(data_collate_fn):
    if not attr_name.startswith('_'):
        setattr(data_module.collate_fn, attr_name, getattr(data_collate_fn, attr_name))
for attr_name in dir(data_sampler):
    if not attr_name.startswith('_'):
        setattr(data_module.sampler, attr_name, getattr(data_sampler, attr_name))

# Now we can import OpenGait modules
from opengait.modeling.models.swingait import SwinGait
from utils import config_loader
from einops import rearrange


class GradCAM:
    """Grad-CAM implementation for SwinGait"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: SwinGait model
            target_layer: Layer to hook for Grad-CAM (e.g., 'layer2', 'transformer')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
            return None
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                self.gradients = grad_output[0]
            return None
        
        # Get the target layer
        if self.target_layer == 'layer2':
            target_module = self.model.layer2
        elif self.target_layer == 'layer1':
            target_module = self.model.layer1
        elif self.target_layer == 'layer0':
            target_module = self.model.layer0
        elif self.target_layer == 'transformer':
            # Hook into transformer output (before temporal pooling)
            target_module = self.model.transformer
        else:
            raise ValueError(f"Unknown target layer: {self.target_layer}")
        
        # Register hooks
        self.forward_handle = target_module.register_forward_hook(forward_hook)
        self.backward_handle = target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input silhouette sequence [N, S, H, W] (4D)
            target_class: Target class index (None = use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap
            pred_class: Predicted class index
            pred_probs: Prediction probabilities
        """
        self.model.eval()
        
        # SwinGait expects input to be [N, S, H, W] (4D)
        # Ensure input is 4D
        if len(input_tensor.shape) == 3:  # [S, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # [1, S, H, W]
        elif len(input_tensor.shape) == 5:  # [N, C, S, H, W]
            # Remove channel dimension
            input_tensor = input_tensor.squeeze(1)  # [N, S, H, W]
        
        assert len(input_tensor.shape) == 4, f"Expected 4D tensor [N, S, H, W], got {input_tensor.shape}"
        
        # Forward pass
        # Prepare input in the format SwinGait expects
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)  # S dimension
        
        # Create dummy inputs for SwinGait forward
        # SwinGait expects: (ipts, labs, class_id, _, seqL)
        # where ipts is a list containing [N, S, H, W] tensor
        labs = torch.zeros(batch_size, dtype=torch.long).to(input_tensor.device)
        class_id = ['Nonfrail'] * batch_size  # Dummy class_id
        seqL = [torch.tensor([seq_len] * batch_size, dtype=torch.int, device=input_tensor.device)]
        
        # Enable gradients for input
        input_tensor = input_tensor.requires_grad_(True)
        inputs = ([input_tensor], labs, class_id, None, seqL)
        
        # Forward pass with gradient tracking
        output = self.model(inputs)
        logits = output['inference_feat']['embeddings']  # [N, C, P]
        
        # Get prediction (average over parts dimension)
        logits_avg = logits.mean(dim=-1)  # [N, C]
        pred_probs = F.softmax(logits_avg, dim=-1)
        pred_class = pred_probs.argmax(dim=-1)
        
        # Use target class if specified, otherwise use predicted
        if target_class is None:
            target_class = pred_class[0].item()
        
        # Backward pass
        self.model.zero_grad()
        
        # Get score for target class (average over parts)
        score = logits_avg[0, target_class]
        score.backward()
        
        # Generate CAM
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check hook registration.")
        
        # Get gradients and activations
        gradients = self.gradients  # [N, C, S, H, W] or similar
        activations = self.activations  # [N, C, S, H, W] or similar
        
        # Handle different tensor shapes
        if len(gradients.shape) == 5:  # [N, C, S, H, W]
            # Average over temporal dimension
            gradients = gradients.mean(dim=2)  # [N, C, H, W]
            activations = activations.mean(dim=2)  # [N, C, H, W]
        elif len(gradients.shape) == 4:  # [N, C, H, W]
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected gradient shape: {gradients.shape}")
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=False)  # [N, H, W]
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = cam[0].detach().cpu().numpy()  # [H, W] - detach before numpy conversion
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam, pred_class[0].item(), pred_probs[0].detach().cpu().numpy()
    
    def release(self):
        """Remove hooks"""
        self.forward_handle.remove()
        self.backward_handle.remove()


def visualize_gradcam(silhouette_seq, cam, output_path, class_names=['Frail', 'Prefrail', 'Nonfrail'], 
                      pred_class=None, pred_probs=None):
    """
    Visualize Grad-CAM heatmap overlaid on silhouette sequence
    
    Args:
        silhouette_seq: Silhouette sequence [N, S, H, W], [S, H, W], or [S, 1, H, W]
        cam: Grad-CAM heatmap [H, W]
        output_path: Path to save visualization
        class_names: List of class names
        pred_class: Predicted class index
        pred_probs: Prediction probabilities
    """
    # Handle input shape - convert to [S, H, W]
    if len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
        silhouette_seq = silhouette_seq.squeeze(1).squeeze(0)  # Remove C and N -> [S, H, W]
    elif len(silhouette_seq.shape) == 4:
        if silhouette_seq.shape[1] == 1:  # [N, 1, H, W] or [S, 1, H, W]
            silhouette_seq = silhouette_seq.squeeze(1)  # [N, H, W] or [S, H, W]
        if len(silhouette_seq.shape) == 3 and silhouette_seq.shape[0] == 1:  # [1, H, W]
            silhouette_seq = silhouette_seq.squeeze(0)  # [H, W] - single frame
        elif len(silhouette_seq.shape) == 4:  # [N, S, H, W]
            silhouette_seq = silhouette_seq.squeeze(0)  # Remove batch dim -> [S, H, W]
    
    # Now should be [S, H, W] or [H, W]
    if len(silhouette_seq.shape) == 2:  # [H, W] - single frame
        silhouette_seq = silhouette_seq.unsqueeze(0)  # [1, H, W] -> [S, H, W] with S=1
    
    seq_len, height, width = silhouette_seq.shape
    
    # Resize CAM to match silhouette size
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Create visualization for each frame
    num_frames = min(seq_len, 20)
    # Match DeepGaitV2 exactly - use plt.subplots with same figsize
    fig, axes = plt.subplots(2, num_frames, figsize=(18, 8))
    if seq_len == 1:
        axes = axes.reshape(2, 1)
    
    # Class names
    class_str = ""
    if pred_class is not None and pred_probs is not None:
        class_str = f"Predicted: {class_names[pred_class]} ({pred_probs[pred_class]:.2%})"
        for i, (name, prob) in enumerate(zip(class_names, pred_probs)):
            class_str += f" | {name}: {prob:.2%}"
    
    fig.suptitle(f"Grad-CAM Visualization\n{class_str}", fontsize=14, fontweight='bold')
    
    # Show frames
    for i in range(num_frames):
        frame_idx = i * (seq_len // num_frames) if seq_len > num_frames else i
        
        # Original silhouette
        sil = silhouette_seq[frame_idx]
        if torch.is_tensor(sil):
            sil = sil.detach().cpu().numpy()
        
        # Display image with correct orientation
        # Set box aspect to square AFTER displaying image (matching DeepGaitV2 behavior)
        axes[0, i].imshow(sil, cmap='gray', aspect='equal', origin='upper')
        axes[0, i].set_aspect('equal', adjustable='box')
        try:
            axes[0, i].set_box_aspect(1.0)  # Make subplot box square
        except AttributeError:
            pass  # set_box_aspect not available in older matplotlib versions
        axes[0, i].set_title(f'Frame {frame_idx}', fontsize=10)
        axes[0, i].axis('off')
        
        # Grad-CAM overlay
        sil_3ch = np.stack([sil, sil, sil], axis=-1)  # Convert to RGB
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]  # Convert to RGB
        
        # Overlay
        alpha = 0.5
        overlay = (1 - alpha) * sil_3ch + alpha * cam_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, i].imshow(overlay, aspect='equal', origin='upper')
        axes[1, i].set_aspect('equal', adjustable='box')
        try:
            axes[1, i].set_box_aspect(1.0)  # Make subplot box square
        except AttributeError:
            pass  # set_box_aspect not available in older matplotlib versions
        axes[1, i].set_title(f'Grad-CAM {frame_idx}', fontsize=10)
        axes[1, i].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, axes.shape[1]):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Grad-CAM visualization to: {output_path}")


def load_swingait_model(config_path, checkpoint_path, device='cuda'):
    """Load SwinGait model from checkpoint"""
    # Initialize minimal distributed environment for single-GPU inference
    # SwinGait's BaseModel expects distributed to be initialized (for msg_mgr)
    if not torch.distributed.is_initialized():
        try:
            # Set environment variables
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '12355')
            
            # Choose backend based on device
            if device == 'cuda' and torch.cuda.is_available():
                backend = 'nccl'
            else:
                backend = 'gloo'
            
            # Initialize with single process for inference
            torch.distributed.init_process_group(
                backend=backend,
                rank=0,
                world_size=1,
                init_method='env://'
            )
        except Exception as e:
            # If initialization fails, try to continue anyway
            print(f"Warning: Could not initialize distributed training: {e}")
            print("Attempting to continue...")
    
    # Load config
    cfg = config_loader(config_path)
    
    # Initialize message manager (required before creating model)
    # Use utils module alias we set up earlier
    from utils import get_msg_mgr
    msg_mgr = get_msg_mgr()
    engine_cfg = cfg['evaluator_cfg']
    output_path = os.path.join('output/', cfg['data_cfg']['dataset_name'],
                               cfg['model_cfg']['model'], engine_cfg['save_name'])
    # Initialize logger for inference (not full manager)
    msg_mgr.init_logger(output_path, log_to_file=False)
    
    # Create model (SwinGait expects full cfg dict, not just model_cfg)
    model = SwinGait(cfg, training=False)
    
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
                # Assume the whole dict is the state_dict
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
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM visualization for SwinGait',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Run this script from the OpenGait root directory (where opengait/ folder is located).

Example:
    cd /path/to/OpenGait
    python gradcam_swingait.py --config configs/swingait/swin_baseline.yaml ...
        """
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to SwinGait config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SwinGait checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input silhouette sequence (pickle file or directory)')
    parser.add_argument('--output', type=str, default='gradcam_output',
                        help='Output directory or filename for visualization. If directory, images will be saved there. If filename, a folder will be created based on the filename.')
    parser.add_argument('--target_layer', type=str, default='layer2',
                        choices=['layer0', 'layer1', 'layer2', 'transformer', 'all'],
                        help='Target layer for Grad-CAM. Use "all" to generate for all layers.')
    parser.add_argument('--target_class', type=int, default=None,
                        help='Target class index (None = use predicted class). 0=Frail, 1=Prefrail, 2=Nonfrail')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_frames', type=int, default=15,
                        help='Maximum number of frames to process (default: 15 for Grad-CAM, set to 0 for all frames)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Stride for frame sampling (default: 1, use every frame)')
    parser.add_argument('--frame_indices', type=str, default=None,
                        help='Comma-separated list of specific frame indices to process (e.g., "0,5,10,15,20"). Overrides max_frames and frame_stride if provided.')
    
    args = parser.parse_args()
    
    # Verify we're in the right directory
    if not os.path.exists('opengait') and not os.path.exists('utils'):
        print("WARNING: 'opengait' or 'utils' not found in current directory.")
        print("Please run this script from the OpenGait root directory.")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Clear CUDA cache before loading model
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Load model
    print("Loading SwinGait model...")
    model = load_swingait_model(args.config, args.checkpoint, args.device)
    
    # Clear cache after model loading
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory after loading model: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Load input
    print(f"Loading input from: {args.input}")
    if args.input.endswith('.pkl'):
        import pickle
        with open(args.input, 'rb') as f:
            silhouette_seq = pickle.load(f)
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(silhouette_seq):
            silhouette_seq = silhouette_seq.cpu().numpy()
        
        # Handle frame selection for pickle files
        if args.frame_indices:
            try:
                frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
                
                # Determine sequence dimension
                if len(silhouette_seq.shape) == 3:  # [S, H, W]
                    seq_dim = 0
                    total_frames = silhouette_seq.shape[0]
                elif len(silhouette_seq.shape) == 4:  # [N, S, H, W] or [S, C, H, W]
                    # Assume first dim is batch if S > 100, otherwise it's sequence
                    if silhouette_seq.shape[1] > 100:  # Likely [N, S, H, W]
                        seq_dim = 1
                        total_frames = silhouette_seq.shape[1]
                    else:  # Likely [S, C, H, W]
                        seq_dim = 0
                        total_frames = silhouette_seq.shape[0]
                elif len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
                    seq_dim = 2
                    total_frames = silhouette_seq.shape[2]
                else:
                    raise ValueError(f"Unexpected pickle file shape: {silhouette_seq.shape}")
                
                # Validate indices
                valid_indices = [i for i in frame_indices if 0 <= i < total_frames]
                if len(valid_indices) != len(frame_indices):
                    invalid = [i for i in frame_indices if i not in valid_indices]
                    print(f"Warning: Invalid frame indices {invalid} (valid range: 0-{total_frames-1}). Ignoring them.")
                if not valid_indices:
                    raise ValueError("No valid frame indices provided")
                
                # Select frames
                if len(silhouette_seq.shape) == 3:  # [S, H, W]
                    silhouette_seq = silhouette_seq[valid_indices]
                elif len(silhouette_seq.shape) == 4:
                    if seq_dim == 1:  # [N, S, H, W]
                        silhouette_seq = silhouette_seq[:, valid_indices]
                    else:  # [S, C, H, W]
                        silhouette_seq = silhouette_seq[valid_indices]
                elif len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
                    silhouette_seq = silhouette_seq[:, :, valid_indices]
                
                print(f"Selected {len(valid_indices)} frames from pickle file: {valid_indices}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing frame_indices '{args.frame_indices}': {e}")
                print("Expected format: comma-separated integers (e.g., '0,5,10,15,20')")
                sys.exit(1)
    else:
        # Assume directory with images
        from PIL import Image
        import glob
        image_files = sorted(glob.glob(os.path.join(args.input, '*.png')) + 
                            glob.glob(os.path.join(args.input, '*.jpg')))
        
        print(f"Found {len(image_files)} images")
        
        # Handle specific frame indices if provided
        if args.frame_indices:
            try:
                frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
                # Validate indices
                valid_indices = [i for i in frame_indices if 0 <= i < len(image_files)]
                if len(valid_indices) != len(frame_indices):
                    invalid = [i for i in frame_indices if i not in valid_indices]
                    print(f"Warning: Invalid frame indices {invalid} (valid range: 0-{len(image_files)-1}). Ignoring them.")
                if not valid_indices:
                    raise ValueError("No valid frame indices provided")
                image_files = [image_files[i] for i in valid_indices]
                print(f"Processing {len(image_files)} specific frames: {valid_indices}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing frame_indices '{args.frame_indices}': {e}")
                print("Expected format: comma-separated integers (e.g., '0,5,10,15,20')")
                sys.exit(1)
        else:
            # Apply frame limiting and stride
            if args.max_frames > 0 and len(image_files) > args.max_frames:
                # Sample frames with stride
                if args.frame_stride > 1:
                    image_files = image_files[::args.frame_stride]
                # Limit to max_frames
                if len(image_files) > args.max_frames:
                    # Uniformly sample max_frames from available frames
                    indices = np.linspace(0, len(image_files) - 1, args.max_frames, dtype=int)
                    image_files = [image_files[i] for i in indices]
                print(f"Processing {len(image_files)} frames (max_frames={args.max_frames}, stride={args.frame_stride})")
        
        # Load and resize images to 64x64 for memory efficiency
        # Gait models typically use 64x64 silhouettes
        target_size = (64, 64)  # (width, height) - standard gait silhouette size
        images = []
        for f in image_files:
            img = Image.open(f).convert('L')
            # Resize to 64x64
            img_resized = img.resize(target_size, Image.LANCZOS)
            images.append(np.array(img_resized))
        
        print(f"Resized images from original size to {target_size[1]}x{target_size[0]} (HxW)")
        silhouette_seq = np.stack(images, axis=0)
    
    # Convert to tensor
    if isinstance(silhouette_seq, np.ndarray):
        silhouette_seq = torch.from_numpy(silhouette_seq).float()
    
    # Normalize to [0, 1]
    if silhouette_seq.max() > 1:
        silhouette_seq = silhouette_seq / 255.0
    
    # SwinGait expects input to be [N, S, H, W] (4D), not [N, C, S, H, W]
    # The model will add the channel dimension itself with unsqueeze(1)
    if len(silhouette_seq.shape) == 3:  # [S, H, W]
        silhouette_seq = silhouette_seq.unsqueeze(0)  # [1, S, H, W]
    elif len(silhouette_seq.shape) == 5:  # [N, C, S, H, W]
        # Remove channel dimension
        silhouette_seq = silhouette_seq.squeeze(1)  # [N, S, H, W]
    # If already 4D, assume it's [N, S, H, W]
    
    # Ensure it's [N, S, H, W]
    assert len(silhouette_seq.shape) == 4, f"Expected 4D tensor [N, S, H, W], got {silhouette_seq.shape}"
    
    print(f"Input shape: {silhouette_seq.shape}")
    print(f"Input size: {silhouette_seq.numel() * 4 / 1024**2:.2f} MB (float32)")
    
    silhouette_seq = silhouette_seq.to(args.device)
    
    # Clear cache before Grad-CAM
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Determine which layers to process
    if args.target_layer == 'all':
        layers_to_process = ['layer0', 'layer1', 'layer2', 'transformer']
    else:
        layers_to_process = [args.target_layer]
    
    # Determine output directory
    # If output is a directory, use it directly
    # If output is a filename, create a folder based on it
    if os.path.splitext(args.output)[1] in ['.png', '.jpg', '.jpeg']:
        # It's a filename, create a folder based on it
        output_dir = os.path.splitext(args.output)[0]
        if not output_dir.endswith('_gradcam'):
            output_dir = output_dir + '_gradcam'
    else:
        # It's already a directory name
        output_dir = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract participant ID and class from input path if possible
    input_basename = os.path.basename(args.input.rstrip('/'))
    if input_basename.endswith('.pkl'):
        input_basename = os.path.splitext(input_basename)[0]
    
    # Try to extract participant ID from path
    participant_id = None
    frailty_class = None
    input_path_parts = args.input.split('/')
    for i, part in enumerate(input_path_parts):
        if part.isdigit() and len(part) >= 3:  # Likely participant ID
            participant_id = part
            # Check if next part is a frailty class
            if i + 1 < len(input_path_parts):
                next_part = input_path_parts[i + 1]
                if next_part in ['Frail', 'Prefrail', 'Nonfrail']:
                    frailty_class = next_part
            break
    
    # Create base filename
    if participant_id and frailty_class:
        base_filename = f"gradcam_{participant_id}_{frailty_class}"
    elif participant_id:
        base_filename = f"gradcam_{participant_id}"
    else:
        base_filename = input_basename if input_basename else "gradcam_output"
    
    # Process each layer
    output_files = []
    for layer_name in layers_to_process:
        print(f"\n{'='*70}")
        print(f"Creating Grad-CAM for layer: {layer_name}")
        print(f"{'='*70}")
        
        # Create Grad-CAM for this layer
        gradcam = GradCAM(model, layer_name)
        
        # Generate CAM
        print("Generating Grad-CAM heatmap...")
        cam, pred_class, pred_probs = gradcam.generate_cam(silhouette_seq, args.target_class)
        
        # Create output filename for this layer
        output_filename = f"{base_filename}_{layer_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Visualize
        print(f"Creating visualization for {layer_name}...")
        visualize_gradcam(silhouette_seq, cam, output_path, 
                         pred_class=pred_class, pred_probs=pred_probs)
        
        # Cleanup
        gradcam.release()
        
        output_files.append(output_path)
        print(f"✓ Saved {layer_name} visualization to: {output_path}")
    
    print(f"\n{'='*70}")
    print("Done! Generated Grad-CAM for all requested layers.")
    print(f"All images saved to: {output_dir}")
    print(f"Output files ({len(output_files)}):")
    for output_file in output_files:
        print(f"  - {os.path.basename(output_file)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

