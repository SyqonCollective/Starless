#!/bin/bash

echo "🚀 AVVIO TRAINING ULTRA-STABILE - VERSIONE DEFINITIVA"
echo "============================================================"
echo "🔧 Configurazione ottimizzata per eliminare NaN gradients"
echo "🎯 Loss function minima: L1 + Charbonnier"
echo "⚡ Batch size ridotto: 4 (massima stabilità)"
echo "🛡️  Gradient clipping conservativo: 0.5"
echo "============================================================"

# Controllo CUDA
echo "🔍 Controllo configurazione GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Test tensor creation
    x = torch.randn(2, 3, 512, 512, device='cuda', dtype=torch.float16)
    print(f'Test tensor created successfully: {x.shape}')
    del x
    torch.cuda.empty_cache()
    print('✅ GPU test passed')
else:
    print('❌ CUDA not available!')
    exit 1
"

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed!"
    exit 1
fi

# Verifica dataset
echo ""
echo "📁 Controllo dataset..."
if [ ! -d "./train_tiles/input" ]; then
    echo "❌ Training dataset not found at ./train_tiles/input"
    exit 1
fi

if [ ! -d "./train_tiles/target" ]; then
    echo "❌ Training targets not found at ./train_tiles/target"
    exit 1
fi

TRAIN_INPUT_COUNT=$(find ./train_tiles/input -name "*.png" | wc -l)
TRAIN_TARGET_COUNT=$(find ./train_tiles/target -name "*.png" | wc -l)

echo "📊 Training images: $TRAIN_INPUT_COUNT input, $TRAIN_TARGET_COUNT target"

if [ $TRAIN_INPUT_COUNT -eq 0 ] || [ $TRAIN_TARGET_COUNT -eq 0 ]; then
    echo "❌ No training data found!"
    exit 1
fi

if [ $TRAIN_INPUT_COUNT -ne $TRAIN_TARGET_COUNT ]; then
    echo "⚠️  Warning: Input/target count mismatch ($TRAIN_INPUT_COUNT vs $TRAIN_TARGET_COUNT)"
fi

# Verifica val dataset (opzionale)
if [ -d "./val_tiles/input" ] && [ -d "./val_tiles/target" ]; then
    VAL_INPUT_COUNT=$(find ./val_tiles/input -name "*.png" | wc -l)
    VAL_TARGET_COUNT=$(find ./val_tiles/target -name "*.png" | wc -l)
    echo "📊 Validation images: $VAL_INPUT_COUNT input, $VAL_TARGET_COUNT target"
else
    echo "ℹ️  Validation dataset not found - will split training set"
fi

# Crea directories necessarie
echo ""
echo "📁 Creazione directories..."
mkdir -p logs/ultra_stable
mkdir -p checkpoints/ultra_stable

# Backup di eventuali log precedenti
if [ -f "logs/ultra_stable/training.log" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv "logs/ultra_stable/training.log" "logs/ultra_stable/training_backup_$TIMESTAMP.log"
    echo "📄 Previous log backed up to training_backup_$TIMESTAMP.log"
fi

# Clear GPU memory
echo ""
echo "🧹 Pulizia memoria GPU..."
python -c "
import torch
torch.cuda.empty_cache()
print('✅ GPU cache cleared')
"

# Test rapido del modello
echo ""
echo "🧪 Test rapido architettura modello..."
python -c "
import torch
from model import create_astro_model

try:
    model = create_astro_model(
        in_channels=3,
        out_channels=3, 
        width=32,
        middle_block_num=2,
        encoder_block_nums=[1, 1, 1, 28],
        decoder_block_nums=[1, 1, 1, 1]
    )
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    
    print(f'✅ Model test passed: {x.shape} -> {y.shape}')
    print(f'📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
except Exception as e:
    print(f'❌ Model test failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Model test failed!"
    exit 1
fi

echo ""
echo "🚀 AVVIO TRAINING ULTRA-STABILE..."
echo "============================================================"
echo "📝 Log: logs/ultra_stable/training.log"
echo "💾 Checkpoints: checkpoints/ultra_stable/"
echo "📊 Tensorboard: tensorboard --logdir=logs/ultra_stable"
echo "============================================================"
echo ""

# Avvio training con timeout di sicurezza
timeout 4h python train_stable.py 2>&1 | tee logs/ultra_stable/training.log

# Controllo exit code
EXIT_CODE=$?

echo ""
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETATO CON SUCCESSO!"
    echo "📁 Checkpoints salvati in: checkpoints/ultra_stable/"
    echo "📊 Per visualizzare i risultati: tensorboard --logdir=logs/ultra_stable"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏰ TRAINING INTERROTTO PER TIMEOUT (4 ore)"
    echo "💾 Controlla i checkpoints salvati in: checkpoints/ultra_stable/"
else
    echo "❌ TRAINING TERMINATO CON ERRORE (exit code: $EXIT_CODE)"
    echo "📝 Controlla il log: logs/ultra_stable/training.log"
    echo ""
    echo "🔍 Ultime righe del log:"
    tail -20 logs/ultra_stable/training.log
fi

echo "============================================================"