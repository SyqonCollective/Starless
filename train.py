#!/usr/bin/env python3
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from model import create_astro_model
from losses import create_loss_function, AdaptiveLoss
from dataset import create_dataloaders
from utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logging


class AstroTrainer:
    """Trainer ottimizzato per RTX 5090 - rimozione stelle astrofotografia"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rank = 0
        self.world_size = 1
        
        # Setup logging
        self.logger = setup_logging(config['log_dir'])
        
        # Setup tensorboard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Initialize model, loss, optimizer
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        self.setup_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=config['use_amp'])
        
        # Data loaders
        self.setup_data()
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.global_step = 0
        
        self.logger.info(f"Trainer inizializzato. Device: {self.device}")
        self.logger.info(f"Modello: {sum(p.numel() for p in self.model.parameters()):,} parametri")

    def setup_model(self):
        """Setup del modello NAFNet"""
        self.model = create_astro_model(
            size=self.config['model_size'],
            model_type=self.config['model_type'],
            img_channel=3,
            drop_path_rate=self.config['drop_path_rate'],
            drop_out_rate=self.config['drop_out_rate']
        )
        
        self.model = self.model.to(self.device)
        
        # Load pretrained weights if available
        if self.config['pretrained_path'] and os.path.exists(self.config['pretrained_path']):
            self.logger.info(f"Caricamento pesi pre-addestrati: {self.config['pretrained_path']}")
            checkpoint = torch.load(self.config['pretrained_path'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def setup_loss(self):
        """Setup loss function"""
        loss_config = {
            'l1_weight': self.config['l1_weight'],
            'charbonnier_weight': self.config['charbonnier_weight'],
            'ssim_weight': self.config['ssim_weight'],
            'edge_weight': self.config['edge_weight'],
            'freq_weight': self.config['freq_weight'],
            'perceptual_weight': self.config['perceptual_weight'],
            'star_preservation_weight': self.config['star_preservation_weight'],
        }
        
        base_loss = create_loss_function(loss_config, self.device)
        
        if self.config['adaptive_loss']:
            self.criterion = AdaptiveLoss(base_loss)
        else:
            self.criterion = base_loss

    def setup_optimizer(self):
        """Setup optimizer ottimizzato per RTX 5090"""
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=self.config['weight_decay'],
                eps=1e-8
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=self.config['weight_decay'],
                eps=1e-8
            )
        else:
            raise ValueError(f"Optimizer non supportato: {self.config['optimizer']}")

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        elif self.config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

    def setup_data(self):
        """Setup data loaders"""
        self.train_loader, self.val_loader = create_dataloaders(
            root_dir=self.config['data_root'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.logger.info(f"Dataset caricato: {len(self.train_loader)} batch di training, "
                        f"{len(self.val_loader)} batch di validation")

    def train_epoch(self, epoch):
        """Training di una singola epoca"""
        self.model.train()
        
        # Update adaptive loss weights
        if hasattr(self.criterion, 'update_weights'):
            self.criterion.update_weights(epoch, self.config['num_epochs'])
        
        losses = AverageMeter()
        psnr_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoca {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            input_imgs = batch['input'].to(self.device, non_blocking=True)
            target_imgs = batch['target'].to(self.device, non_blocking=True)
            
            # Forward pass con mixed precision
            with autocast(enabled=self.config['use_amp']):
                pred_imgs = self.model(input_imgs)
                loss_dict = self.criterion(pred_imgs, target_imgs, input_imgs)
                loss = loss_dict['total']
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Calcola PSNR
            with torch.no_grad():
                psnr = self.calculate_psnr(pred_imgs, target_imgs)
                psnr_meter.update(psnr.item(), input_imgs.size(0))
            
            losses.update(loss.item(), input_imgs.size(0))
            
            # Logging
            if batch_idx % self.config['log_freq'] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                
                # Log to tensorboard
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/PSNR', psnr.item(), step)
                self.writer.add_scalar('Train/LR', lr, step)
                
                # Log individual loss components
                for key, value in loss_dict.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Train/Loss_{key}', value.item(), step)
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{psnr.item():.2f}',
                    'LR': f'{lr:.2e}'
                })
            
            self.global_step += 1
        
        return losses.avg, psnr_meter.avg

    def validate(self, epoch):
        """Validation"""
        self.model.eval()
        
        losses = AverageMeter()
        psnr_meter = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                input_imgs = batch['input'].to(self.device, non_blocking=True)
                target_imgs = batch['target'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config['use_amp']):
                    pred_imgs = self.model(input_imgs)
                    loss_dict = self.criterion(pred_imgs, target_imgs, input_imgs)
                    loss = loss_dict['total']
                
                psnr = self.calculate_psnr(pred_imgs, target_imgs)
                
                losses.update(loss.item(), input_imgs.size(0))
                psnr_meter.update(psnr.item(), input_imgs.size(0))
                
                pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'PSNR': f'{psnr.item():.2f}'
                })
                
                # Save sample images
                if batch_idx == 0 and epoch % self.config['save_img_freq'] == 0:
                    self.save_sample_images(input_imgs, pred_imgs, target_imgs, epoch)
        
        # Log validation metrics
        self.writer.add_scalar('Val/Loss', losses.avg, epoch)
        self.writer.add_scalar('Val/PSNR', psnr_meter.avg, epoch)
        
        return losses.avg, psnr_meter.avg

    def calculate_psnr(self, pred, target, max_val=2.0):
        """Calcola PSNR per immagini in range [-1, 1]"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    def save_sample_images(self, input_imgs, pred_imgs, target_imgs, epoch):
        """Salva immagini di esempio"""
        import torchvision.utils as vutils
        
        # Converti da [-1, 1] a [0, 1]
        input_imgs = (input_imgs + 1) / 2
        pred_imgs = (pred_imgs + 1) / 2
        target_imgs = (target_imgs + 1) / 2
        
        # Prendi solo i primi 4 campioni
        n_samples = min(4, input_imgs.size(0))
        
        # Crea grid di immagini
        comparison = torch.cat([
            input_imgs[:n_samples],
            pred_imgs[:n_samples],
            target_imgs[:n_samples]
        ], dim=0)
        
        grid = vutils.make_grid(comparison, nrow=n_samples, normalize=False, padding=2)
        
        # Salva immagine
        sample_dir = os.path.join(self.config['checkpoint_dir'], 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        vutils.save_image(grid, os.path.join(sample_dir, f'epoch_{epoch:04d}.png'))
        
        # Log to tensorboard
        self.writer.add_image('Samples/Input_Pred_Target', grid, epoch)

    def train(self):
        """Loop di training principale"""
        self.logger.info("Inizio training...")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_psnr = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_psnr = self.validate(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_psnr)
                else:
                    self.scheduler.step()
            
            # Logging
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (self.config['num_epochs'] - epoch - 1)
            
            self.logger.info(
                f'Epoca {epoch+1}/{self.config["num_epochs"]} - '
                f'Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.2f} - '
                f'Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f} - '
                f'ETA: {eta/3600:.1f}h'
            )
            
            # Save checkpoint
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            
            if (epoch + 1) % self.config['save_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info(f"Training completato! Miglior PSNR: {self.best_psnr:.2f}")
        self.writer.close()

    def save_checkpoint(self, epoch, is_best=False):
        """Salva checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        # Salva checkpoint corrente
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1:04d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Salva miglior modello
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Nuovo miglior modello salvato: PSNR {self.best_psnr:.2f}")
        
        # Mantieni solo gli ultimi N checkpoint
        self.cleanup_checkpoints()

    def cleanup_checkpoints(self, keep_last=5):
        """Mantieni solo gli ultimi N checkpoint"""
        import glob
        
        checkpoints = glob.glob(os.path.join(self.config['checkpoint_dir'], 'checkpoint_epoch_*.pth'))
        checkpoints.sort()
        
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                os.remove(old_checkpoint)


def get_default_config():
    """Configurazione di default ottimizzata per RTX 5090"""
    return {
        # Modello
        'model_size': 'base',  # 'small', 'base', 'large'
        'model_type': 'nafnet',
        'drop_path_rate': 0.1,
        'drop_out_rate': 0.0,
        
        # Training
        'num_epochs': 300,
        'batch_size': 8,  # Ottimizzato per RTX 5090 con mixed precision completo
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'step_size': 100,
        'gamma': 0.5,
        'grad_clip': 1.0,
        'use_amp': True,  # Mixed precision per RTX 5090
        
        # Loss weights
        'l1_weight': 1.0,
        'charbonnier_weight': 1.0,
        'ssim_weight': 0.5,
        'edge_weight': 0.3,
        'freq_weight': 0.2,
        'perceptual_weight': 0.5,
        'star_preservation_weight': 2.0,
        'adaptive_loss': True,
        
        # Dataset
        'data_root': '.',
        'image_size': 512,
        'num_workers': 8,  # Ottimo per RTX 5090
        
        # Logging e salvataggio
        'log_freq': 50,
        'save_freq': 10,
        'save_img_freq': 5,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'pretrained_path': None,
    }


def main():
    parser = argparse.ArgumentParser(description='NAFNet Training per Astrofotografia')
    parser.add_argument('--config', type=str, help='Path al file di configurazione')
    parser.add_argument('--resume', type=str, help='Path al checkpoint da riprendere')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Numero di epoche')
    parser.add_argument('--data_root', type=str, help='Path ai dati')
    
    args = parser.parse_args()
    
    # Carica configurazione
    config = get_default_config()
    
    # Override con argomenti command line
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.data_root:
        config['data_root'] = args.data_root
    
    # Crea directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Inizializza trainer
    trainer = AstroTrainer(config)
    
    # Resume training se specificato
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_psnr = checkpoint['best_psnr']
        print(f"Training ripreso dall'epoca {trainer.current_epoch}")
    
    # Avvia training
    trainer.train()


if __name__ == '__main__':
    main()