#!/usr/bin/env python3
"""
Find all log files that contain evaluation metrics including specificity.
"""

import os
import glob

def find_all_log_files():
    """Find all log files in output directory."""
    log_files = []
    
    # Search for logs directory
    for root, dirs, files in os.walk('output'):
        if 'logs' in dirs:
            logs_dir = os.path.join(root, 'logs')
            txt_files = glob.glob(os.path.join(logs_dir, '*.txt'))
            if txt_files:
                log_files.extend(txt_files)
    
    return sorted(log_files)

def main():
    print("=" * 80)
    print("Finding Log Files with Evaluation Metrics")
    print("=" * 80)
    
    log_files = find_all_log_files()
    
    if not log_files:
        print("\n❌ No log files found!")
        print("\nLog files are only created if you used --log_to_file during training/evaluation.")
        print("\nLog files would be located at:")
        print("  output/<dataset>/<model>/<save_name>/logs/<Datetime>.txt")
        print("\nFor example:")
        print("  output/REDO_insqrt/SwinGait/REDO_insqrt/logs/2025-11-13-18-17-00.txt")
        print("\nTo create log files, add --log_to_file flag when running:")
        print("  python opengait/main.py --cfgs <config> --phase train --log_to_file")
        print("  python opengait/main.py --cfgs <config> --phase test --log_to_file")
        return
    
    print(f"\n✓ Found {len(log_files)} log file(s):\n")
    
    for log_file in log_files:
        # Extract experiment name from path
        path_parts = log_file.split(os.sep)
        if len(path_parts) >= 4:
            exp_name = path_parts[-3]  # Usually the save_name
        else:
            exp_name = "Unknown"
        
        file_size = os.path.getsize(log_file) / 1024  # KB
        print(f"  {exp_name}:")
        print(f"    Path: {log_file}")
        print(f"    Size: {file_size:.1f} KB")
        
        # Check if it contains evaluation results
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if 'EVALUATION RESULTS' in content:
                    eval_count = content.count('EVALUATION RESULTS')
                    print(f"    ✓ Contains {eval_count} evaluation(s)")
                    if 'Specificity' in content:
                        print(f"    ✓ Contains specificity metrics")
                else:
                    print(f"    ⚠ No evaluation results found")
        except:
            print(f"    ⚠ Could not read file")
        print()
    
    print("=" * 80)
    print("\nTo extract metrics from these log files, use:")
    print("  python extract_with_specificity.py")
    print("=" * 80)

if __name__ == '__main__':
    main()

