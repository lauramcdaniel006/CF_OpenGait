#!/usr/bin/env python3
"""
Batch extract GaitSet embeddings from entire dataset.

Processes all participants in your dataset structure:
  /path/to/sils/<participant_id>/<frailty_label>/silhouettes/silhouettes.pkl

Usage:
    python batch_extract_gaitset_embeddings.py \
        --config configs/gaitset/gaitset.yaml \
        --checkpoint path/to/checkpoint.pt \
        --dataset_root /cis/net/r38/data/lmcdan11/sils \
        --output_dir /path/to/output/embeddings \
        --participant_start 300 \
        --participant_end 368
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import subprocess
from multiprocessing import Pool

# Add OpenGait to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Get the correct Python executable path and conda environment
# Use the current Python executable (should be from conda env)
PYTHON_EXECUTABLE = sys.executable

# Get conda environment name from environment variable or infer from path
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', '')
# If CONDA_DEFAULT_ENV is a path, extract just the name
if CONDA_ENV and os.sep in CONDA_ENV:
    # Extract name from path like /path/to/envs/myGait38 or /path/to/r38_conda_envs/myGait38
    CONDA_ENV = os.path.basename(CONDA_ENV)

if not CONDA_ENV:
    # Try to infer from Python path
    # Path format: /path/to/envs/myGait38/bin/python or /path/to/r38_conda_envs/myGait38/bin/python
    parts = PYTHON_EXECUTABLE.split(os.sep)
    # Look for 'bin' directory - the env name is the directory before it
    try:
        bin_idx = parts.index('bin')
        if bin_idx > 0:
            CONDA_ENV = parts[bin_idx - 1]
    except ValueError:
        # Fallback: look for 'envs' or 'conda_envs' in path
        for i, part in enumerate(parts):
            if 'envs' in part.lower() and i + 1 < len(parts):
                CONDA_ENV = parts[i + 1]
                break

# Use conda run if we have an env name, otherwise use python directly
USE_CONDA_RUN = bool(CONDA_ENV)

# Debug: print detected environment (can be removed later)
if USE_CONDA_RUN:
    print(f"Detected conda environment: {CONDA_ENV}")
    print(f"Using conda run for subprocesses")
else:
    print(f"Warning: No conda environment detected, using Python directly: {PYTHON_EXECUTABLE}")


def find_participant_pkl_files(dataset_root, participant_id):
    """Find all silhouette.pkl files for a participant"""
    participant_dir = Path(dataset_root) / str(participant_id)
    if not participant_dir.exists():
        return []
    
    pkl_files = []
    # Look for silhouettes.pkl in subdirectories
    for frailty_dir in participant_dir.iterdir():
        if frailty_dir.is_dir():
            pkl_path = frailty_dir / "silhouettes" / "silhouettes.pkl"
            if pkl_path.exists():
                pkl_files.append({
                    'path': str(pkl_path),
                    'participant_id': participant_id,
                    'frailty_label': frailty_dir.name
                })
    
    return pkl_files


def process_file(file_info, script_path, config, checkpoint, output_dir, max_frames, device, preserve_structure, skip_existing, stats_dict=None):
    """Process a single file on the specified GPU"""
    participant_id = file_info['participant_id']
    frailty_label = file_info['frailty_label']
    pkl_path = file_info['path']
    
    # Create output path
    if preserve_structure:
        participant_output_dir = output_dir / str(participant_id) / frailty_label
        participant_output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = "embeddings.npy"
        output_path = participant_output_dir / output_filename
    else:
        output_filename = f"embeddings_{participant_id}_{frailty_label}.npy"
        output_path = output_dir / output_filename
    
    # Skip if exists
    if skip_existing and output_path.exists():
        return {'status': 'skipped', 'participant_id': participant_id, 'frailty_label': frailty_label}
    
    # Build command - use Python executable directly (it's already from conda env)
    # Using conda run causes library path issues, so use Python directly
    cmd = [
        PYTHON_EXECUTABLE,
        str(script_path),
        '--config', config,
        '--input', pkl_path,
        '--output', str(output_path),
        '--device', device
    ]
    
    if checkpoint:
        cmd.extend(['--checkpoint', checkpoint])
    if max_frames:
        cmd.extend(['--max_frames', str(max_frames)])
    
    # Run extraction with environment variables and correct working directory
    # Set working directory to OpenGait root so imports work correctly
    opengait_root = Path(__file__).parent
    opengait_dir = opengait_root / 'opengait'
    # Set PYTHONPATH to include both root (for opengait package) and opengait/ (for utils imports)
    # OpenGait code uses "from utils import ..." which expects opengait/ to be in path
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    new_paths = [str(opengait_root), str(opengait_dir)]
    if pythonpath:
        env['PYTHONPATH'] = f"{':'.join(new_paths)}:{pythonpath}"
    else:
        env['PYTHONPATH'] = ':'.join(new_paths)
    
    # Set LD_LIBRARY_PATH to include conda environment lib directory for correct library loading
    # This fixes OpenCV/libstdc++ compatibility issues
    conda_env_lib = str(Path(PYTHON_EXECUTABLE).parent.parent / 'lib')
    ld_library_path = env.get('LD_LIBRARY_PATH', '')
    if ld_library_path:
        env['LD_LIBRARY_PATH'] = f"{conda_env_lib}:{ld_library_path}"
    else:
        env['LD_LIBRARY_PATH'] = conda_env_lib
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per sequence
            cwd=str(opengait_root),  # Run from OpenGait directory
            env=env  # Pass environment variables with PYTHONPATH
        )
        
        if result.returncode == 0:
            return {'status': 'success', 'participant_id': participant_id, 'frailty_label': frailty_label}
        else:
            return {
                'status': 'failed',
                'participant_id': participant_id,
                'frailty_label': frailty_label,
                'error': result.stderr[:1000]  # Increased to see full error
            }
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'participant_id': participant_id, 'frailty_label': frailty_label}
    except Exception as e:
        return {'status': 'error', 'participant_id': participant_id, 'frailty_label': frailty_label, 'error': str(e)}


def process_file_wrapper(args_tuple):
    """Wrapper function for multiprocessing (must be at module level to be picklable)"""
    file_info, device_name, script_path, config, checkpoint, output_dir, max_frames, preserve_structure, skip_existing = args_tuple
    return process_file(
        file_info,
        script_path=script_path,
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        max_frames=max_frames,
        device=device_name,
        preserve_structure=preserve_structure,
        skip_existing=skip_existing,
        stats_dict=None
    )


def main():
    parser = argparse.ArgumentParser(
        description='Batch extract GaitSet embeddings from entire dataset'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to GaitSet config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to GaitSet checkpoint file')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory containing participant folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embeddings')
    parser.add_argument('--participant_start', type=int, default=300,
                        help='Starting participant ID')
    parser.add_argument('--participant_end', type=int, default=368,
                        help='Ending participant ID')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames per sequence')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, cuda:0, cuda:1, or cpu). Use "multi" to use both GPU 0 and GPU 1')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip participants if embeddings already exist')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                        help='Preserve directory structure in output (default: True)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of parallel workers (default: 2 for multi-GPU)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all participants
    print("=" * 70)
    print("Scanning dataset for participants...")
    print("=" * 70)
    
    all_files = []
    for participant_id in range(args.participant_start, args.participant_end + 1):
        pkl_files = find_participant_pkl_files(args.dataset_root, participant_id)
        all_files.extend(pkl_files)
    
    print(f"Found {len(all_files)} silhouette sequences")
    
    # Determine GPU assignment strategy
    use_multi_gpu = args.device.lower() == 'multi'
    
    if use_multi_gpu:
        # Split files between GPU 0 and GPU 1
        print("\n" + "=" * 70)
        print(f"Using MULTI-GPU mode: GPU 0 and GPU 1")
        print("=" * 70)
        
        # Assign files to GPUs (round-robin)
        gpu0_files = [f for i, f in enumerate(all_files) if i % 2 == 0]
        gpu1_files = [f for i, f in enumerate(all_files) if i % 2 == 1]
        
        print(f"GPU 0: {len(gpu0_files)} sequences")
        print(f"GPU 1: {len(gpu1_files)} sequences")
        
        script_path = Path(__file__).parent / "extract_gaitset_embeddings.py"
        
        # Process in parallel using multiprocessing
        print("\n" + "=" * 70)
        print("Processing sequences in parallel on both GPUs...")
        print("=" * 70)
        
        # Create tasks with GPU assignments and all necessary arguments
        tasks_gpu0 = [
            (f, 'cuda:0', script_path, args.config, args.checkpoint, output_dir, args.max_frames, args.preserve_structure, args.skip_existing)
            for f in gpu0_files
        ]
        tasks_gpu1 = [
            (f, 'cuda:1', script_path, args.config, args.checkpoint, output_dir, args.max_frames, args.preserve_structure, args.skip_existing)
            for f in gpu1_files
        ]
        all_tasks = tasks_gpu0 + tasks_gpu1
        
        # Process all tasks in parallel
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.map(process_file_wrapper, all_tasks),
                total=len(all_tasks),
                desc="Processing"
            ))
        
        # Count results
        successful = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        failed = sum(1 for r in results if r['status'] in ['failed', 'timeout', 'error'])
        
        # Print any errors
        for result in results:
            if result['status'] == 'failed':
                print(f"\n❌ Failed: {result['participant_id']}/{result['frailty_label']}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            elif result['status'] == 'timeout':
                print(f"\n⏱️  Timeout: {result['participant_id']}/{result['frailty_label']}")
    
    else:
        # Single GPU mode (original behavior)
        print("\n" + "=" * 70)
        print(f"Processing sequences on {args.device}...")
        print("=" * 70)
        
        script_path = Path(__file__).parent / "extract_gaitset_embeddings.py"
        
        successful = 0
        skipped = 0
        failed = 0
        
        for file_info in tqdm(all_files, desc="Processing"):
            participant_id = file_info['participant_id']
            frailty_label = file_info['frailty_label']
            pkl_path = file_info['path']
            
            # Create output path
            if args.preserve_structure:
                participant_output_dir = output_dir / str(participant_id) / frailty_label
                participant_output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = "embeddings.npy"
                output_path = participant_output_dir / output_filename
            else:
                output_filename = f"embeddings_{participant_id}_{frailty_label}.npy"
                output_path = output_dir / output_filename
            
            # Skip if exists
            if args.skip_existing and output_path.exists():
                skipped += 1
                continue
            
            # Build command - use Python executable directly (it's already from conda env)
            # Using conda run causes library path issues, so use Python directly
            cmd = [
                PYTHON_EXECUTABLE,
                str(script_path),
                '--config', args.config,
                '--input', pkl_path,
                '--output', str(output_path),
                '--device', args.device
            ]
            
            if args.checkpoint:
                cmd.extend(['--checkpoint', args.checkpoint])
            if args.max_frames:
                cmd.extend(['--max_frames', str(args.max_frames)])
            
            # Run extraction with environment variables and correct working directory
            # Set working directory to OpenGait root so imports work correctly
            opengait_root = Path(__file__).parent
            opengait_dir = opengait_root / 'opengait'
            # Set PYTHONPATH to include both root (for opengait package) and opengait/ (for utils imports)
            # OpenGait code uses "from utils import ..." which expects opengait/ to be in path
            env = os.environ.copy()
            pythonpath = env.get('PYTHONPATH', '')
            new_paths = [str(opengait_root), str(opengait_dir)]
            if pythonpath:
                env['PYTHONPATH'] = f"{':'.join(new_paths)}:{pythonpath}"
            else:
                env['PYTHONPATH'] = ':'.join(new_paths)
            
            # Set LD_LIBRARY_PATH to include conda environment lib directory for correct library loading
            # This fixes OpenCV/libstdc++ compatibility issues
            conda_env_lib = str(Path(PYTHON_EXECUTABLE).parent.parent / 'lib')
            ld_library_path = env.get('LD_LIBRARY_PATH', '')
            if ld_library_path:
                env['LD_LIBRARY_PATH'] = f"{conda_env_lib}:{ld_library_path}"
            else:
                env['LD_LIBRARY_PATH'] = conda_env_lib
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per sequence
                    cwd=str(opengait_root),  # Run from OpenGait directory
                    env=env  # Pass environment variables with PYTHONPATH
                )
                
                if result.returncode == 0:
                    successful += 1
                else:
                    failed += 1
                    print(f"\n❌ Failed: {participant_id}/{frailty_label}")
                    print(f"   Error: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                failed += 1
                print(f"\n⏱️  Timeout: {participant_id}/{frailty_label}")
            except Exception as e:
                failed += 1
                print(f"\n❌ Error: {participant_id}/{frailty_label} - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total sequences: {len(all_files)}")
    print(f"✓ Successful: {successful}")
    print(f"⊘ Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

