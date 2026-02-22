#!/usr/bin/env python3
"""
Reshape GaitSet embeddings to 64x64 format for AlexNet/VGG16 input
Processes existing embeddings and saves reshaped versions
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm


def reshape_embedding_to_64x64(embedding):
    """
    Reshape GaitSet embedding from (1, 256, 62) to (1, 64, 64)
    
    Args:
        embedding: numpy array of shape (1, 256, 62) or (256, 62)
    
    Returns:
        Reshaped embedding of shape (1, 64, 64)
    """
    # Handle different input shapes
    if len(embedding.shape) == 3:
        # (1, 256, 62)
        emb_2d = embedding[0]  # (256, 62)
    elif len(embedding.shape) == 2:
        # (256, 62)
        emb_2d = embedding
    else:
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}, expected (1, 256, 62) or (256, 62)")
    
    # Convert to torch tensor for interpolation
    emb_tensor = torch.FloatTensor(emb_2d).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 62)
    
    # Interpolate to 64x64 using bilinear interpolation
    emb_resized = torch.nn.functional.interpolate(
        emb_tensor, size=(64, 64), mode='bilinear', align_corners=False
    )
    
    # Return as numpy array: (1, 64, 64)
    return emb_resized.squeeze(0).numpy()  # (1, 64, 64)


def process_single_embedding(input_path, output_path):
    """Process a single embedding file"""
    try:
        # Load embedding
        embedding = np.load(input_path)
        
        # Reshape
        reshaped = reshape_embedding_to_64x64(embedding)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, reshaped)
        
        return True, None
    except Exception as e:
        return False, str(e)


def process_directory(input_dir, output_dir, preserve_structure=True, skip_existing=True):
    """
    Process all embeddings in a directory structure
    
    Expected structure:
    input_dir/
      participant_id/
        frailty_label/
          embeddings.npy
    
    Output structure (if preserve_structure=True):
    output_dir/
      participant_id/
        frailty_label/
          embeddings_64x64.npy
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    label_map = {'Frail': 0, 'Prefrail': 1, 'Nonfrail': 2}
    
    # Find all embedding files
    embedding_files = []
    for participant_dir in sorted(input_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        
        participant_id = participant_dir.name
        
        for frailty_dir in sorted(participant_dir.iterdir()):
            if not frailty_dir.is_dir():
                continue
            
            frailty_label = frailty_dir.name
            if frailty_label not in label_map:
                continue
            
            embeddings_file = frailty_dir / 'embeddings.npy'
            if not embeddings_file.exists():
                continue
            
            embedding_files.append({
                'participant_id': participant_id,
                'frailty_label': frailty_label,
                'input_path': embeddings_file,
                'output_path': output_dir / participant_id / frailty_label / 'embeddings_64x64.npy' if preserve_structure else output_dir / f'embeddings_64x64_{participant_id}_{frailty_label}.npy'
            })
    
    print(f"Found {len(embedding_files)} embedding files to process")
    
    # Process each file
    success_count = 0
    skipped_count = 0
    failed_count = 0
    errors = []
    
    for file_info in tqdm(embedding_files, desc="Reshaping embeddings"):
        input_path = file_info['input_path']
        output_path = file_info['output_path']
        
        # Skip if exists
        if skip_existing and output_path.exists():
            skipped_count += 1
            continue
        
        success, error = process_single_embedding(input_path, output_path)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            errors.append(f"{input_path}: {error}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESHAPING SUMMARY")
    print("="*70)
    print(f"Total files: {len(embedding_files)}")
    print(f"Successfully reshaped: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print("="*70)
    
    return success_count, skipped_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description='Reshape GaitSet embeddings to 64x64 format for AlexNet/VGG16'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing embeddings (or single .npy file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (or single .npy file path)')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                        help='Preserve directory structure in output (default: True)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already exist in output (default: True)')
    parser.add_argument('--single_file', action='store_true',
                        help='Process a single file instead of directory')
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        print(f"Reshaping: {input_path} -> {output_path}")
        success, error = process_single_embedding(input_path, output_path)
        
        if success:
            print(f"✓ Successfully reshaped and saved to: {output_path}")
            
            # Verify output
            reshaped = np.load(output_path)
            print(f"  Original shape: {np.load(input_path).shape}")
            print(f"  Reshaped shape: {reshaped.shape}")
        else:
            print(f"✗ Error: {error}")
    else:
        # Process directory
        process_directory(
            args.input,
            args.output,
            preserve_structure=args.preserve_structure,
            skip_existing=args.skip_existing
        )


if __name__ == '__main__':
    main()

