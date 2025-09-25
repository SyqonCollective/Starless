# NAFNet per Rimozione Stelle - Astrofotografia RTX 5090

Training setup completo di NAFNet ottimizzato per la rimozione delle stelle nelle immagini astrofotografiche, specificamente ottimizzato per RTX 5090.

## 🎯 Obiettivo

Rimuovere le stelle dalle immagini astrofotografiche preservando perfettamente:
- Nebulose e galassie
- Dettagli fini delle strutture cosmiche  
- Texture coerenti nei punti dove erano le stelle
- Range dinamico dell'astrofotografia

## 🚀 Quick Start

### 1. Setup Environment (su RunPod/RTX 5090)

```bash
# Installa dipendenze
pip install -r requirements.txt

# Verifica dataset
python -c "from utils import validate_dataset_structure; validate_dataset_structure('.')"
```

### 2. Avvio Training Rapido

```bash
# Training con configurazione ottimale RTX 5090
./start_training.sh

# Oppure con parametri personalizzati
./start_training.sh --batch-size 20 --epochs 500 --model-size large
```

### 3. Training Manuale

```bash
# Training base
python train.py --batch_size 16 --epochs 300 --lr 2e-4

# Resume da checkpoint
python train.py --resume checkpoints/checkpoint_epoch_0100.pth

# Con configurazione personalizzata
python train.py --config custom_config.yaml
```

## 📁 Struttura Dataset

Il dataset deve avere questa struttura esatta:

```
├── train_tiles/
│   ├── input/     # Immagini CON stelle
│   └── target/    # Immagini SENZA stelle (ground truth)
├── val_tiles/
│   ├── input/     # Immagini validation CON stelle  
│   └── target/    # Immagini validation SENZA stelle
```

**Importante**: I nomi dei file devono essere identici tra input e target!

## 🔧 Configurazioni

### Modelli Disponibili
- `small`: 2M parametri, veloce, buono per test
- `base`: 5M parametri, bilanciato, raccomandato
- `large`: 12M parametri, migliori risultati, più lento

### Configurazioni RTX 5090
- **Batch size ottimale**: 16-20 (dipende dalla risoluzione)
- **Mixed Precision**: Abilitata (FP16)
- **TF32**: Abilitato per massime performance
- **Memory**: Ottimizzato per 24GB VRAM

## 📊 Training

### Parametri Ottimali

```yaml
# Per RTX 5090 con immagini 512x512
batch_size: 16
learning_rate: 2e-4
epochs: 300
optimizer: adamw
scheduler: cosine
mixed_precision: true
```

### Loss Function Specializzata

La loss combina multiple componenti per astrofotografia:

- **L1 + Charbonnier**: Ricostruzione base robusta
- **SSIM**: Preservazione strutturale  
- **Edge Loss**: Conservazione dettagli fini
- **Frequency Loss**: Texture nel dominio frequenze
- **Perceptual Loss**: Qualità percettiva (LPIPS)
- **Star Preservation**: Penalizza rimozione non-stelle

### Monitoring

```bash
# Tensorboard
tensorboard --logdir logs

# Controlla progress
tail -f logs/training_*.log
```

## 🎯 Inferenza

### Singola Immagine

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input my_astro_image.tiff \
    --output result_starless.png \
    --comparison
```

### Batch Processing

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input /path/to/images/ \
    --output /path/to/results/ \
    --comparison
```

### Benchmark Performance

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input test_image.tiff \
    --benchmark
```

## 🏗️ Architettura

### NAFNet Modificata
- **U-Net structure** con skip connections
- **NAF blocks** con attenzione semplificata
- **Multi-scale processing** per stelle di diverse dimensioni
- **Texture preservation** con loss specifiche

### Ottimizzazioni RTX 5090
- **Mixed precision training** (FP16)
- **TF32 acceleration** 
- **Optimized memory usage**
- **CUDA kernel optimizations**

## 📈 Risultati Attesi

### Metriche Target
- **PSNR**: >30 dB
- **SSIM**: >0.95
- **LPIPS**: <0.05
- **Training time**: ~12-16 ore su RTX 5090

### Qualità Output
- Stelle completamente rimosse
- Nebulose/galassie intatte
- Texture coerenti nei buchi stellari
- No artefatti o aloni

## 🛠️ Troubleshooting

### Memory Issues
```bash
# Riduci batch size
./start_training.sh --batch-size 8

# Usa gradient checkpointing
# (modifica config.yaml: use_gradient_checkpointing: true)
```

### Dataset Problems
```bash
# Verifica struttura
python -c "from utils import check_image_pair_consistency; check_image_pair_consistency('.')"

# Check file formats
find . -name "*.png" -o -name "*.tiff" | head -10
```

### Performance Issues
```bash
# Check GPU utilization
nvidia-smi -l 1

# Profile training
python train.py --config config.yaml # (con debug.profile_training: true)
```

## 📂 File Principali

- `train.py`: Script training principale
- `model.py`: Architettura NAFNet
- `dataset.py`: DataLoader per astrofotografia
- `losses.py`: Loss functions specializzate
- `inference.py`: Script per inferenza
- `utils.py`: Utilities varie
- `config.yaml`: Configurazione completa
- `start_training.sh`: Script di avvio rapido

## 🔬 Features Avanzate

### Loss Adattiva
La loss cambia pesi durante il training:
- **Early epochs**: Focus su ricostruzione base
- **Mid epochs**: Bilanciato
- **Late epochs**: Focus su dettagli e texture

### Memory Optimizations
- **Gradient accumulation** per batch size effettivi maggiori
- **Checkpoint gradient** per ridurre memory usage
- **Mixed precision** per speed + memory

### Data Augmentation Astro-Specific
- Rotazioni e flip preservando orientamento cosmico
- Variazioni luminosità/contrasto moderate
- Noise gaussiano realistico
- No distorsioni geometriche eccessive

## 📝 Note Importanti

1. **Formato Immagini**: Supporta PNG, TIFF (16-bit), JPEG
2. **Range Dinamico**: Preserva full precision per astrofotografia
3. **Color Spaces**: RGB standard, no HDR specifico
4. **Resolution**: Ottimizzato per 512x512, scaling automatico
5. **Inference**: Supporta tile-based per immagini molto grandi

## 🚀 Deployment su RunPod

```bash
# Setup rapido su RunPod RTX 5090
git clone <questo-repo>
cd StarLess
pip install -r requirements.txt
./start_training.sh
```

Il training è completamente self-contained e non richiede sottodirectory specifiche - basta uploadare i tuoi tile nelle directory corrette e partire!

## 📜 License & Credits

Basato su NAFNet (Simple Baselines for Image Restoration) con modifiche specifiche per astrofotografia e ottimizzazioni RTX 5090.