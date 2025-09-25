#!/bin/bash
# SCRIPT PER AGGIORNARE E RIPRENDERE TRAINING SU RUNPOD
# Da eseguire su RunPod dopo aver pushato le modifiche

echo "🌌 Starless - Update and Resume Training"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "train_professional.py" ]; then
    echo "❌ train_professional.py not found. Please run from Starless directory."
    exit 1
fi

# Stop any running training (optional)
echo "🛑 Checking for running training processes..."
pkill -f "train_professional.py" 2>/dev/null || echo "No training processes found"

# Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

if [ $? -ne 0 ]; then
    echo "❌ Git pull failed. Please check your connection and repository status."
    exit 1
fi

echo "✅ Updates pulled successfully"

# Check for existing checkpoints
echo "📂 Checking for existing checkpoints..."
if ls checkpoints/checkpoint_epoch_*.pth 1> /dev/null 2>&1; then
    latest_checkpoint=$(ls -t checkpoints/checkpoint_epoch_*.pth | head -n1)
    echo "🔄 Found checkpoint: $latest_checkpoint"
    echo "🚀 Resuming training from checkpoint..."
    python resume_training.py
else
    echo "📝 No checkpoints found. Starting fresh training..."
    python train_professional.py
fi