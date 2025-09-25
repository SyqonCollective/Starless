#!/bin/bash

# Script di avvio rapido per training NAFNet su RunPod con RTX 5090
# Uso: ./start_training.sh [OPTIONS]

set -e  # Exit on error

echo "=== NAFNet Training per Astrofotografia - RTX 5090 ==="
echo "Data: $(date)"
echo "Host: $(hostname)"
echo

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzioni helper
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default parameters
BATCH_SIZE=20
EPOCHS=300
LR=2e-4
MODEL_SIZE="base"
CONFIG_FILE="config.yaml"
RESUME_CHECKPOINT=""
DATA_ROOT="."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LR="$2"
            shift 2
            ;;
        -m|--model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        -d|--data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Uso: $0 [OPTIONS]"
            echo "OPTIONS:"
            echo "  -b, --batch-size SIZE     Batch size (default: 16)"
            echo "  -e, --epochs NUM          Numero epoche (default: 300)"
            echo "  -lr, --learning-rate LR   Learning rate (default: 2e-4)"
            echo "  -m, --model-size SIZE     Model size: small/base/large (default: base)"
            echo "  -c, --config FILE         Config file (default: config.yaml)"
            echo "  -r, --resume PATH         Resume da checkpoint"
            echo "  -d, --data-root PATH      Data root directory (default: .)"
            echo "  -h, --help               Mostra questo help"
            exit 0
            ;;
        *)
            log_error "Opzione sconosciuta: $1"
            exit 1
            ;;
    esac
done

log_info "Configurazione training:"
echo "  Batch size: $BATCH_SIZE"
echo "  Epoche: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Model size: $MODEL_SIZE"
echo "  Config file: $CONFIG_FILE"
echo "  Data root: $DATA_ROOT"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "  Resume da: $RESUME_CHECKPOINT"
fi
echo

# Check GPU
log_info "Controllo GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo
else
    log_warning "nvidia-smi non trovato. Controllo CUDA..."
fi

# Check CUDA
if python -c "import torch; print(f'CUDA disponibile: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'Dispositivi CUDA: {torch.cuda.device_count()}')" 2>/dev/null
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        python -c "import torch; print(f'GPU corrente: {torch.cuda.get_device_name()}')" 2>/dev/null
    fi
else
    log_error "PyTorch non disponibile o problemi con CUDA"
    exit 1
fi
echo

# Check dataset structure
log_info "Controllo struttura dataset..."
python -c "
from utils import validate_dataset_structure, check_image_pair_consistency
import sys
try:
    validate_dataset_structure('$DATA_ROOT')
    check_image_pair_consistency('$DATA_ROOT')
    print('✓ Struttura dataset OK')
except Exception as e:
    print(f'✗ Errore dataset: {e}')
    sys.exit(1)
"

if [[ $? -ne 0 ]]; then
    log_error "Problemi con la struttura del dataset"
    exit 1
fi
echo

# Create directories
log_info "Creazione directory..."
mkdir -p checkpoints logs
log_success "Directory create"

# Install/check dependencies
log_info "Controllo dipendenze..."
if ! pip list | grep -q torch; then
    log_warning "Installazione dipendenze..."
    pip install -r requirements.txt
else
    log_success "Dipendenze OK"
fi
echo

# Set environment variables for optimal RTX 5090 performance
log_info "Configurazione ottimizzazioni RTX 5090..."
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Enable TF32 for Ampere/Ada GPUs
python -c "
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print('✓ Ottimizzazioni RTX 5090 abilitate')
"

# Build training command
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --lr $LR"
TRAIN_CMD="$TRAIN_CMD --data_root $DATA_ROOT"

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    if [[ -f "$RESUME_CHECKPOINT" ]]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
        log_info "Resume training da: $RESUME_CHECKPOINT"
    else
        log_error "Checkpoint non trovato: $RESUME_CHECKPOINT"
        exit 1
    fi
fi

echo
log_info "Comando di training:"
echo "  $TRAIN_CMD"
echo

# Ask for confirmation
read -p "Avviare il training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Training annullato"
    exit 0
fi

# Start training with proper error handling
log_success "Avvio training..."
echo "==============================================="
echo

# Set up signal handlers for graceful shutdown
trap 'echo -e "\n${YELLOW}[SIGNAL]${NC} Training interrotto dall utente"; exit 130' INT TERM

# Run training with real-time output
if eval $TRAIN_CMD; then
    echo
    echo "==============================================="
    log_success "Training completato con successo!"
    
    # Show final results
    if [[ -f "checkpoints/best_model.pth" ]]; then
        log_info "Miglior modello salvato in: checkpoints/best_model.pth"
    fi
    
    if [[ -d "logs" ]]; then
        log_info "Log disponibili in: logs/"
        log_info "Visualizza con: tensorboard --logdir logs"
    fi
    
else
    echo
    echo "==============================================="
    log_error "Training fallito!"
    exit 1
fi

# Optional: run quick inference test
if [[ -f "checkpoints/best_model.pth" ]] && [[ -f "train_tiles/input/$(ls train_tiles/input/ | head -1)" ]]; then
    echo
    read -p "Testare il modello su un'immagine di esempio? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TEST_IMAGE=$(ls train_tiles/input/ | head -1)
        log_info "Test su immagine: $TEST_IMAGE"
        
        python inference.py \
            --model checkpoints/best_model.pth \
            --input "train_tiles/input/$TEST_IMAGE" \
            --output "test_result.png" \
            --comparison \
            --model_size $MODEL_SIZE
        
        if [[ $? -eq 0 ]]; then
            log_success "Test completato! Risultati:"
            echo "  - Output: test_result.png"
            echo "  - Confronto: test_result_comparison.png"
        fi
    fi
fi

log_success "Script completato!"