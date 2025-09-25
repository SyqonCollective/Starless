#!/bin/bash

echo "🚨 MODALITÀ EMERGENZA - ZERO MIXED PRECISION"
echo "============================================================"
echo "🔧 FP32 solamente - massima stabilità"
echo "📊 Modello minimo - width=16, 1 blocco"
echo "🎯 SOLO L1 Loss - zero complessità"
echo "⚡ Mixed precision: DISABILITATO"
echo "============================================================"

# Clear GPU memory completamente
echo "🧹 Pulizia completa memoria GPU..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print('✅ GPU memory cleared completely')
"

# Verifica dataset
echo ""
echo "📁 Quick dataset check..."
TRAIN_COUNT=$(find /workspace/train_tiles/input -name "*.png" | head -10 | wc -l)
echo "📊 Found $TRAIN_COUNT sample images (showing first 10)"

# Test emergenza modello
echo ""
echo "🧪 Test modello emergenza..."
python -c "
import torch
from model import create_astro_model

# Force FP32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

model = create_astro_model(
    size='small',
    img_channel=3,
    width=16,
    middle_blk_num=1,
    enc_blk_nums=[1, 1, 1, 1],
    dec_blk_nums=[1, 1, 1, 1]
).cuda()

# Test in FP32
x = torch.randn(1, 3, 512, 512, device='cuda', dtype=torch.float32)
with torch.no_grad():
    y = model(x)

print(f'✅ Emergency model OK: {x.shape} -> {y.shape}')
print(f'📊 Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'💾 Input dtype: {x.dtype}')
print(f'💾 Output dtype: {y.dtype}')
"

if [ $? -ne 0 ]; then
    echo "❌ Emergency model test failed!"
    exit 1
fi

echo ""
echo "🚨 AVVIO TRAINING EMERGENZA..."
echo "============================================================"
echo "📝 Log: logs/emergency/training.log"
echo "💾 Checkpoints: checkpoints/emergency/"
echo "🚨 MODALITÀ: FP32 only, no mixed precision"
echo "============================================================"

# Create directories
mkdir -p logs/emergency
mkdir -p checkpoints/emergency

# Launch emergency training
python train_emergency.py 2>&1 | tee logs/emergency/training.log

echo ""
echo "============================================================"
echo "🚨 EMERGENCY TRAINING COMPLETED"
echo "============================================================"