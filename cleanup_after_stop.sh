#!/bin/bash
# Cleanup script to run after abruptly stopping training

echo "🧹 Cleaning up after stopped training..."

# 1. Kill any running training processes
echo "1. Killing running processes..."
pkill -f "main.py" 2>/dev/null
pkill -f "torch.distributed.launch" 2>/dev/null
sleep 2

# 2. Clear PyTorch CUDA cache (if using GPU)
echo "2. Clearing CUDA cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('   ✓ CUDA cache cleared')" 2>/dev/null || echo "   ⚠ CUDA not available"

# 3. Clear Python cache files
echo "3. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Python cache cleared"

# 4. Clear lock files
echo "4. Clearing lock files..."
find . -name "*.lock" -delete 2>/dev/null
find . -name ".lock" -delete 2>/dev/null
echo "   ✓ Lock files cleared"

# 5. Show any remaining processes
echo "5. Checking for remaining processes..."
REMAINING=$(ps aux | grep -E "(main.py|torch.distributed)" | grep -v grep)
if [ -z "$REMAINING" ]; then
    echo "   ✓ No remaining processes"
else
    echo "   ⚠ Remaining processes found:"
    echo "$REMAINING"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "If you need to forcefully kill processes, run:"
echo "  ps aux | grep main.py | grep -v grep | awk '{print \$2}' | xargs kill -9"
