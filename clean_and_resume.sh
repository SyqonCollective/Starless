#!/bin/bash
# SCRIPT PER PULIRE MEMORIA GPU E RIPRENDERE TRAINING SU RUNPOD

echo "🧹 Cleaning GPU memory and resuming training..."

# Kill any remaining Python processes
echo "🛑 Killing remaining Python processes..."
pkill -f python || echo "No Python processes found"
pkill -f train_ || echo "No training processes found"

# Clear GPU memory
echo "⚡ Clearing GPU memory..."
nvidia-smi --gpu-reset || echo "Could not reset GPU"

# Wait a bit
sleep 2

# Check GPU memory
echo "📊 GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🔄 Resuming training with clean memory..."
python resume_training.py