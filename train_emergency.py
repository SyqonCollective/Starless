#!/usr/bin/env python3
"""
VERSIONE EMERGENZA - ZERO MIXED PRECISION
Disabilita completamente mixed precision e usa solo FP32
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import create_astro_model
from dataset import create_dataloaders
from utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logging


class EmergencyLoss(nn.Module):
    """Loss function di emergenza - SOLO L1"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        # SOLO L1 Loss - la più stabile possibile
        l1 = self.l1_loss(pred, target)
        
        return l1, {
            'l1_loss': l1.item(),
            'total_loss': l1.item()
        }


class EmergencyTrainer:
    """Trainer di emergenza - ZERO mixed precision, solo FP32"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Force FP32 - NO mixed precision
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        # Setup logging
        self.logger = setup_logging(config['log_dir'])
        self.writer = SummaryWriter(config['log_dir'])
        
        # Setup model - configurazione MINIMA
        self.model = create_astro_model(
            size='small',
            img_channel=3,
            width=16,  # DRASTICAMENTE ridotto
            middle_blk_num=1,  # MINIMO
            enc_blk_nums=[1, 1, 1, 1],  # MINIMO
            dec_blk_nums=[1, 1, 1, 1]   # MINIMO
        ).to(self.device)
        
        # Loss function MINIMA
        self.criterion = EmergencyLoss().to(self.device)
        
        # Optimizer ULTRA conservativo
        self.optimizer = optim.Adam(  # Adam semplice, non AdamW
            self.model.parameters(),
            lr=5e-5,  # Learning rate MOLTO più basso
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0  # ZERO weight decay
        )
        
        # Scheduler MOLTO graduale
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,
            gamma=0.5
        )
        
        # ZERO mixed precision
        self.use_amp = False
        
        # Contatori per debug
        self.global_step = 0
        self.nan_count = 0
        
    def check_tensors(self, *tensors, names=None):
        """Controllo rigoroso di tutti i tensori"""
        if names is None:
            names = [f"tensor_{i}" for i in range(len(tensors))]
        
        for tensor, name in zip(tensors, names):
            if torch.isnan(tensor).any():
                self.logger.error(f"NaN detected in {name}")
                return False
            if torch.isinf(tensor).any():
                self.logger.error(f"Inf detected in {name}")
                return False
            if (tensor.abs() > 1000).any():
                self.logger.warning(f"Very large values in {name}: max={tensor.abs().max()}")
                
        return True
    
    def train_epoch(self, train_loader, epoch):
        """Training epoch di emergenza - solo FP32"""
        self.model.train()
        
        losses = AverageMeter()
        l1_losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Emergency Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device, non_blocking=True, dtype=torch.float32)
            targets = batch['target'].to(self.device, non_blocking=True, dtype=torch.float32)
            
            # Controllo rigoroso degli input
            if not self.check_tensors(inputs, targets, names=['inputs', 'targets']):
                self.logger.error(f"Invalid inputs at batch {batch_idx}")
                continue
                
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass - SOLO FP32
            outputs = self.model(inputs)
            
            # Controllo output
            if not self.check_tensors(outputs, names=['outputs']):
                self.logger.error(f"Invalid outputs at batch {batch_idx}")
                continue
            
            # Loss computation
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Controllo loss
            if not self.check_tensors(loss, names=['loss']):
                self.logger.error(f"Invalid loss at batch {batch_idx}")
                continue
            
            # Backward pass - NO mixed precision
            loss.backward()
            
            # Controllo gradients
            grad_norm = 0
            nan_grads = 0
            total_params = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    total_params += param.numel()
                    if torch.isnan(param.grad).any():
                        nan_grads += torch.isnan(param.grad).sum().item()
                    if torch.isinf(param.grad).any():
                        nan_grads += torch.isinf(param.grad).sum().item()
                    grad_norm += param.grad.data.norm(2).item() ** 2
            
            grad_norm = grad_norm ** 0.5
            
            # Skip se troppi NaN nei gradienti
            if nan_grads > 0:
                self.logger.warning(f"NaN/Inf gradients detected: {nan_grads}/{total_params} at batch {batch_idx}")
                self.nan_count += 1
                continue
            
            # Skip se gradient norm troppo grande
            if grad_norm > 10.0:
                self.logger.warning(f"Large gradient norm: {grad_norm:.2f} at batch {batch_idx}")
                continue
            
            # Gradient clipping MOLTO conservativo
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            l1_losses.update(loss_dict['l1_loss'], inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.6f}',
                'L1': f'{l1_losses.avg:.6f}',
                'GradNorm': f'{grad_norm:.4f}',
                'NaN': self.nan_count,
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Early logging per debug
            if batch_idx % 5 == 0:
                self.writer.add_scalar('Train/Loss', losses.avg, self.global_step)
                self.writer.add_scalar('Train/GradNorm', grad_norm, self.global_step)
                self.writer.add_scalar('Train/NaN_Count', self.nan_count, self.global_step)
            
            self.global_step += 1
            
            # Emergency stop se troppi NaN
            if self.nan_count > 20:
                self.logger.error(f"Too many NaN detections: {self.nan_count}")
                break
        
        return {
            'loss': losses.avg,
            'l1_loss': l1_losses.avg,
            'nan_count': self.nan_count
        }
    
    def validate(self, val_loader, epoch):
        """Validation di emergenza"""
        self.model.eval()
        
        losses = AverageMeter()
        l1_losses = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Emergency Val {epoch+1}')
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['input'].to(self.device, non_blocking=True, dtype=torch.float32)
                targets = batch['target'].to(self.device, non_blocking=True, dtype=torch.float32)
                
                # Controllo input
                if not self.check_tensors(inputs, targets, names=['val_inputs', 'val_targets']):
                    continue
                
                outputs = self.model(inputs)
                
                # Controllo output
                if not self.check_tensors(outputs, names=['val_outputs']):
                    continue
                
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Controllo loss
                if not self.check_tensors(loss, names=['val_loss']):
                    continue
                
                losses.update(loss.item(), inputs.size(0))
                l1_losses.update(loss_dict['l1_loss'], inputs.size(0))
                
                pbar.set_postfix({
                    'Val_Loss': f'{losses.avg:.6f}',
                    'Val_L1': f'{l1_losses.avg:.6f}'
                })
        
        return {
            'val_loss': losses.avg,
            'val_l1_loss': l1_losses.avg
        }
    
    def train(self, train_loader, val_loader, num_epochs):
        """Training loop di emergenza"""
        self.logger.info("🚨 EMERGENCY TRAINING MODE - FP32 ONLY")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Ridotta per test rapidi
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Emergency stop se troppi NaN
            if train_metrics['nan_count'] > 20:
                self.logger.error("Emergency stop due to excessive NaN detections")
                break
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            self.logger.info(
                f"Emergency Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f} - "
                f"Val Loss: {val_metrics['val_loss']:.6f} - "
                f"NaN Count: {train_metrics['nan_count']} - "
                f"Time: {epoch_time:.2f}s"
            )
            
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
                    'best_val_loss': best_val_loss,
                    'config': self.config,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, is_best, self.config['checkpoint_dir'])
                
                self.logger.info(f"🎯 Best model saved: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping after {patience} epochs")
                break
        
        self.logger.info("🚨 Emergency training completed")
        self.writer.close()


def main():
    """Main emergency training function"""
    config = {
        'batch_size': 2,  # MOLTO ridotto
        'num_workers': 1,  # Ridotto al minimo
        'epochs': 20,     # Pochi epoch per test
        'use_amp': False, # DISABILITATO
        'log_dir': './logs/emergency',
        'checkpoint_dir': './checkpoints/emergency'
    }
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir='/workspace',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=False  # Disabilitato per sicurezza
    )
    
    # Create trainer
    trainer = EmergencyTrainer(config)
    
    print("=" * 80)
    print("🚨 EMERGENCY TRAINING MODE - ZERO MIXED PRECISION")
    print("🔧 FP32 only, minimal model, ultra-conservative")
    print(f"📊 Batch size: {config['batch_size']} (emergency)")
    print(f"🎯 Loss: SOLO L1 (minima complessità)")
    print(f"⚡ Mixed Precision: DISABLED")
    print(f"🛡️  Gradient clipping: 0.1 (emergency)")
    print("=" * 80)
    
    # Start training
    trainer.train(train_loader, val_loader, config['epochs'])


if __name__ == '__main__':
    main()