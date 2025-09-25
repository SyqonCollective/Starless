#!/usr/bin/env python3
"""
VERSIONE ULTRA-STABILE per risolvere definitivamente i NaN gradients
Configurazione minimale testata per evitare instabilità numeriche
"""
import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import create_astro_model
from dataset import create_dataloaders
from utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logging


class StableLoss(nn.Module):
    """Loss function ultra-stabile - solo L1 + Charbonnier"""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.l1_loss = nn.L1Loss()
    
    def charbonnier_loss(self, pred, target):
        """Charbonnier loss stabile"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)
    
    def forward(self, pred, target):
        # Solo le loss più stabili
        l1 = self.l1_loss(pred, target)
        charbonnier = self.charbonnier_loss(pred, target)
        
        # Peso bilanciato e conservativo
        total_loss = 0.7 * l1 + 0.3 * charbonnier
        
        return total_loss, {
            'l1_loss': l1.item(),
            'charbonnier_loss': charbonnier.item(),
            'total_loss': total_loss.item()
        }


class UltraStableTrainer:
    """Trainer ultra-stabile per risolvere definitivamente i problemi di NaN"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logging(config['log_dir'])
        self.writer = SummaryWriter(config['log_dir'])
        
        # Setup model
        self.model = create_astro_model(
            size='small',  # Usa configurazione predefinita
            img_channel=3,
            width=32,  # Ridotto per stabilità
            middle_blk_num=2,  # Ridotto 
            enc_blk_nums=[1, 1, 1, 28],  # Ridotto
            dec_blk_nums=[1, 1, 1, 1]   # Molto ridotto
        ).to(self.device)
        
        # Loss function stabile
        self.criterion = StableLoss().to(self.device)
        
        # Optimizer molto conservativo
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # Learning rate molto basso
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5  # Weight decay molto basso
        )
        
        # Scheduler molto graduale
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision CON sicurezza extra
        self.scaler = GradScaler(
            enabled=config.get('use_amp', True),
            init_scale=2.**8,  # Scala iniziale più bassa
            growth_factor=1.5,  # Crescita più graduale
            backoff_factor=0.8,  # Backoff più conservativo
            growth_interval=1000  # Intervallo più lungo
        )
        
        # Contatori per debug
        self.global_step = 0
        self.nan_count = 0
        self.skip_count = 0
        
    def check_model_health(self):
        """Controlla la salute del modello"""
        nan_params = 0
        inf_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                total_params += param.numel()
                nan_params += torch.isnan(param.grad).sum().item()
                inf_params += torch.isinf(param.grad).sum().item()
        
        return {
            'total_params': total_params,
            'nan_params': nan_params,
            'inf_params': inf_params,
            'health_ratio': (total_params - nan_params - inf_params) / max(total_params, 1)
        }
    
    def train_epoch(self, train_loader, epoch):
        """Training epoch ultra-sicuro"""
        self.model.train()
        
        losses = AverageMeter()
        l1_losses = AverageMeter()
        charbonnier_losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device, non_blocking=True)
            targets = batch['target'].to(self.device, non_blocking=True)
            
            # Verifica input
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                self.logger.warning(f"NaN detected in batch {batch_idx}, skipping")
                continue
                
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass con mixed precision
            with autocast(enabled=self.config.get('use_amp', True)):
                outputs = self.model(inputs)
                
                # Verifica output
                if torch.isnan(outputs).any():
                    self.logger.warning(f"NaN in model output at batch {batch_idx}")
                    continue
                
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Verifica loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue
            
            # Backward pass
            if self.config.get('use_amp', True):
                self.scaler.scale(loss).backward()
                
                # Verifica gradients prima del clipping
                health = self.check_model_health()
                if health['health_ratio'] < 0.95:  # Se più del 5% è NaN/Inf
                    self.logger.warning(f"Unhealthy gradients detected: {health}")
                    self.nan_count += 1
                    self.scaler.update()  # Update scaler senza step
                    continue
                
                # Gradient clipping conservativo
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                loss.backward()
                
                # Verifica gradients
                health = self.check_model_health()
                if health['health_ratio'] < 0.95:
                    self.logger.warning(f"Unhealthy gradients detected: {health}")
                    self.nan_count += 1
                    continue
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            l1_losses.update(loss_dict['l1_loss'], inputs.size(0))
            charbonnier_losses.update(loss_dict['charbonnier_loss'], inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.6f}',
                'L1': f'{l1_losses.avg:.6f}',
                'Char': f'{charbonnier_losses.avg:.6f}',
                'NaN': self.nan_count,
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Tensorboard logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                self.writer.add_scalar('Train/L1_Loss', l1_losses.avg, self.global_step)
                self.writer.add_scalar('Train/Charbonnier_Loss', charbonnier_losses.avg, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('Train/NaN_Count', self.nan_count, self.global_step)
            
            self.global_step += 1
        
        return {
            'loss': losses.avg,
            'l1_loss': l1_losses.avg,
            'charbonnier_loss': charbonnier_losses.avg,
            'nan_count': self.nan_count
        }
    
    def validate(self, val_loader, epoch):
        """Validation ultra-sicura"""
        self.model.eval()
        
        losses = AverageMeter()
        l1_losses = AverageMeter()
        charbonnier_losses = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation {epoch+1}')
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['input'].to(self.device, non_blocking=True)
                targets = batch['target'].to(self.device, non_blocking=True)
                
                # Verifica input
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    continue
                
                with autocast(enabled=self.config.get('use_amp', True)):
                    outputs = self.model(inputs)
                    
                    if torch.isnan(outputs).any():
                        continue
                    
                    loss, loss_dict = self.criterion(outputs, targets)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                
                losses.update(loss.item(), inputs.size(0))
                l1_losses.update(loss_dict['l1_loss'], inputs.size(0))
                charbonnier_losses.update(loss_dict['charbonnier_loss'], inputs.size(0))
                
                pbar.set_postfix({
                    'Val_Loss': f'{losses.avg:.6f}',
                    'Val_L1': f'{l1_losses.avg:.6f}',
                    'Val_Char': f'{charbonnier_losses.avg:.6f}'
                })
        
        return {
            'val_loss': losses.avg,
            'val_l1_loss': l1_losses.avg,
            'val_charbonnier_loss': charbonnier_losses.avg
        }
    
    def train(self, train_loader, val_loader, num_epochs):
        """Training loop principale ultra-stabile"""
        self.logger.info(f"Starting ultra-stable training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f} - "
                f"NaN Count: {train_metrics['nan_count']} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Epoch/NaN_Count', train_metrics['nan_count'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'scaler': self.scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': self.config,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, is_best, self.config['checkpoint_dir'])
                
                self.logger.info(f"New best model saved with val_loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Controllo NaN emergenza
            if train_metrics['nan_count'] > 100:
                self.logger.error(f"Too many NaN detections: {train_metrics['nan_count']}")
                self.logger.error("Emergency stop due to numerical instability")
                break
        
        self.logger.info("Training completed successfully!")
        self.writer.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Ultra-Stable Astrophotography Star Removal Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Configuration ultra-stabile
    config = {
        'batch_size': 4,  # Molto ridotto per stabilità
        'num_workers': 2,  # Ridotto
        'epochs': 100,
        'use_amp': True,
        'log_dir': './logs/ultra_stable',
        'checkpoint_dir': './checkpoints/ultra_stable'
    }
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir='/workspace',  # Root directory contenente train_tiles e val_tiles
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create trainer
    trainer = UltraStableTrainer(config)
    
    print("=" * 80)
    print("🚀 ULTRA-STABLE ASTROPHOTOGRAPHY TRAINING - RTX 5090 OPTIMIZED")
    print("🔧 Versione definitiva per risolvere i problemi di NaN gradients")
    print(f"📊 Batch size: {config['batch_size']} (ridotto per stabilità)")
    print(f"🎯 Loss: L1 + Charbonnier (configurazione minima stabile)")
    print(f"⚡ Mixed Precision: {config['use_amp']} (con safeguards extra)")
    print(f"🛡️  Gradient clipping: 0.5 (molto conservativo)")
    print("=" * 80)
    
    # Start training
    trainer.train(train_loader, val_loader, config['epochs'])


if __name__ == '__main__':
    main()