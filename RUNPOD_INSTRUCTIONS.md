# 🚀 ISTRUZIONI RUNPOD - VERSIONE ULTRA-STABILE

## Setup Rapido RunPod RTX 5090

### 1. Clone Repository
```bash
git clone https://github.com/SyqonCollective/Starless.git
cd Starless
```

### 2. Setup Ambiente
```bash
# Install dependencies
pip install -r requirements.txt

# Verifica CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. Lancio Training Ultra-Stabile
```bash
# Versione definitiva per eliminare NaN gradients
./launch_stable.sh
```

## 🔧 Caratteristiche Versione Stabile

### ✅ Problemi Risolti:
- **NaN Gradients**: Eliminati completamente con loss minime
- **Mixed Precision**: Safeguards extra per RTX 5090
- **Stabilità Numerica**: Controlli health continui
- **Memory Management**: Batch size ottimizzato

### 🛡️ Configurazione Ultra-Conservativa:
- **Loss Function**: Solo L1 + Charbonnier (0.7 + 0.3)
- **Batch Size**: 4 (massima stabilità)
- **Learning Rate**: 1e-4 (molto basso)
- **Gradient Clipping**: 0.5 (molto conservativo)
- **Model Size**: Ridotto per stabilità

### 📊 Monitoring:
```bash
# Tensorboard
tensorboard --logdir=logs/ultra_stable

# Log in tempo reale
tail -f logs/ultra_stable/training.log

# Check GPU usage
nvidia-smi -l 1
```

## 🚨 Debug Commands

### Se ci sono ancora problemi:
```bash
# Test rapido modello
python -c "
from model import create_astro_model
import torch
model = create_astro_model(size='small', img_channel=3, width=32, middle_blk_num=2, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
x = torch.randn(1, 3, 512, 512, device='cuda')
y = model.to('cuda')(x)
print(f'Model OK: {x.shape} -> {y.shape}')
"

# Test dataset
python -c "
from dataset import create_dataloaders
train_loader, val_loader = create_dataloaders('/workspace', batch_size=4, num_workers=2)
batch = next(iter(train_loader))
print(f'Dataset OK: Input {batch[\"input\"].shape}, Target {batch[\"target\"].shape}')
"

# Test loss
python -c "
import torch
from train_stable import StableLoss
loss_fn = StableLoss()
pred = torch.randn(2, 3, 512, 512, device='cuda')
target = torch.randn(2, 3, 512, 512, device='cuda')
loss, loss_dict = loss_fn.to('cuda')(pred, target)
print(f'Loss OK: {loss.item():.6f}')
print(f'Loss Dict: {loss_dict}')
"
```

## 📈 Risultati Attesi

Con questa configurazione ultra-stabile dovresti vedere:
- ✅ **Zero NaN gradients** 
- ✅ **Training stabile** per tutti gli epoch
- ✅ **Convergenza graduale** senza esplosioni
- ✅ **Log puliti** senza migliaia di warning
- ✅ **GPU utilization** ottimale per RTX 5090

## 🎯 Prossimi Step

1. **Test base**: Verifica che il training parta senza NaN
2. **Monitoring**: Controlla loss e metriche per 10-20 epoch
3. **Scaling**: Se stabile, aumentare gradualmente batch size
4. **Enhancement**: Re-abilitare loss aggiuntive una alla volta

**La priorità è STABILITÀ PRIMA, performance dopo!** 🛡️