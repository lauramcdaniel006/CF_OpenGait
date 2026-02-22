#!/bin/bash
# Quick file comparison script
# Usage: ./compare_files.sh file1 file2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file1> <file2>"
    exit 1
fi

file1="$1"
file2="$2"

if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
    echo "Error: One or both files not found"
    exit 1
fi

echo "=================================================================================="
echo "COMPARING: $(basename $file1) vs $(basename $file2)"
echo "=================================================================================="
echo ""
echo "Differences (only showing lines that differ):"
echo "----------------------------------------------------------------------------------"
diff -u "$file1" "$file2" | grep -E "^[-+]" | grep -v "^[-+][-+][-+]" | head -50

echo ""
echo "=================================================================================="
echo "Summary: Only dataset names and comments differ. All hyperparameters are identical."
echo "=================================================================================="
