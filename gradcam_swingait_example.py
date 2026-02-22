#!/usr/bin/env python3
"""
Example script showing how to use Grad-CAM with SwinGait
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gradcam_swingait import GradCAM, visualize_gradcam, load_swingait_model

def example_usage():
    """Example of using Grad-CAM with SwinGait"""
    
    # Configuration
    config_path = "configs/swingait/swin_baseline.yaml"  # Your SwinGait config
    checkpoint_path = "path/to/your/checkpoint.pth.tar"  # Your trained checkpoint
    input_pkl = "path/to/silhouette_sequence.pkl"  # Input silhouette sequence
    output_path = "gradcam_visualization.png"
    target_layer = "layer2"  # Options: 'layer0', 'layer1', 'layer2', 'transformer'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print("Loading SwinGait model...")
    model = load_swingait_model(config_path, checkpoint_path, device)
    
    # Load silhouette sequence
    # Expected format: [S, H, W] numpy array or torch tensor
    # S = sequence length (number of frames)
    # H, W = height, width of each silhouette
    import pickle
    with open(input_pkl, 'rb') as f:
        silhouette_seq = pickle.load(f)
    
    # Convert to tensor and normalize
    if isinstance(silhouette_seq, np.ndarray):
        silhouette_seq = torch.from_numpy(silhouette_seq).float()
    
    # Normalize to [0, 1]
    if silhouette_seq.max() > 1:
        silhouette_seq = silhouette_seq / 255.0
    
    # Add batch and channel dimensions: [1, 1, S, H, W]
    if len(silhouette_seq.shape) == 3:
        silhouette_seq = silhouette_seq.unsqueeze(0).unsqueeze(0)
    
    silhouette_seq = silhouette_seq.to(device)
    
    # Create Grad-CAM
    print(f"Creating Grad-CAM for layer: {target_layer}")
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM (target_class=None uses predicted class)
    print("Generating Grad-CAM heatmap...")
    cam, pred_class, pred_probs = gradcam.generate_cam(silhouette_seq, target_class=None)
    
    # Visualize
    print("Creating visualization...")
    class_names = ['Frail', 'Prefrail', 'Nonfrail']
    visualize_gradcam(silhouette_seq, cam, output_path, 
                     class_names=class_names,
                     pred_class=pred_class, 
                     pred_probs=pred_probs)
    
    # Cleanup
    gradcam.release()
    
    print(f"Grad-CAM visualization saved to: {output_path}")
    print(f"Predicted class: {class_names[pred_class]} ({pred_probs[pred_class]:.2%})")


if __name__ == '__main__':
    example_usage()

