#!/usr/bin/env python3
"""
PROFESSIONAL DATASET LOADER - ROBUST IMAGE LOADING
Fix definitivo per i NaN nel dataset
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from typing import Tuple, Optional
import logging
from pathlib import Path


class RobustAstroDataset(Dataset):
    """
    Dataset robusto per astrofotografia con handling professionale dei NaN
    """
    
    def __init__(
        self, 
        root_dir: str, 
        mode: str = 'train',
        image_size: int = 512,
        normalize: bool = True,
        augment: bool = True
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment and mode == 'train'
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Percorsi delle directory
        self.input_dir = os.path.join(root_dir, f'{mode}_tiles', 'input')
        self.target_dir = os.path.join(root_dir, f'{mode}_tiles', 'target')
        
        # Lista dei file immagine
        self.image_files = self._get_image_files()
        
        # Trasformazioni per astrofotografia
        self.setup_transforms()
        
        # Statistiche per debug
        self.nan_count = 0
        self.corrupted_files = []
        
        print(f"Loaded ROBUST dataset {mode}: {len(self.image_files)} tiles")
        
    def _get_image_files(self):
        """Ottiene lista file immagine con validazione"""
        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.png')])
        target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.png')])
        
        # Verifica che ci siano coppie corrispondenti
        valid_files = []
        for f in input_files:
            if f in target_files:
                valid_files.append(f)
            else:
                print(f"⚠️  Missing target for {f}")
        
        print(f"Found {len(valid_files)} valid image pairs")
        return valid_files
        
    def setup_transforms(self):
        """Setup trasformazioni robuste"""
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Rimuovo trasformazioni che possono introdurre NaN
                # A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
                # A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            ], additional_targets={'target': 'image'})
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        try:
            # Caricamento robusto delle immagini
            input_img = self._load_image_robust(input_path)
            target_img = self._load_image_robust(target_path)
            
            # Verifica che non ci siano NaN
            if np.isnan(input_img).any():
                print(f"⚠️  NaN detected in input: {filename}")
                input_img = self._fix_nan_values(input_img)
                
            if np.isnan(target_img).any():
                print(f"⚠️  NaN detected in target: {filename}")
                target_img = self._fix_nan_values(target_img)
            
            # Applica trasformazioni
            if self.transform:
                transformed = self.transform(image=input_img, target=target_img)
                input_img = transformed['image']
                target_img = transformed['target']
            
            # Verifica dimensioni
            if input_img.shape[:2] != (self.image_size, self.image_size):
                input_img = cv2.resize(input_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            if target_img.shape[:2] != (self.image_size, self.image_size):
                target_img = cv2.resize(target_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Converti a tensori
            input_tensor = self._to_tensor_safe(input_img)
            target_tensor = self._to_tensor_safe(target_img)
            
            # Normalizza se richiesto
            if self.normalize:
                input_tensor = self._normalize_tensor_safe(input_tensor)
                target_tensor = self._normalize_tensor_safe(target_tensor)
            
            return {
                'input': input_tensor,
                'target': target_tensor,
                'filename': filename
            }
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            self.corrupted_files.append(filename)
            
            # Ritorna tensori zero come fallback
            return {
                'input': torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32),
                'target': torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32),
                'filename': filename
            }
    
    def _load_image_robust(self, path: str) -> np.ndarray:
        """Caricamento robusto delle immagini con multiple strategie"""
        
        # Strategia 1: OpenCV
        try:
            if path.lower().endswith(('.tiff', '.tif')):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("OpenCV failed to load image")
                    
                if img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                else:
                    img = img.astype(np.float32) / 255.0
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("OpenCV failed to load image")
                img = img.astype(np.float32) / 255.0
            
            # Converti da BGR a RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            return img
            
        except Exception as e1:
            print(f"OpenCV failed for {path}: {e1}")
            
            # Strategia 2: PIL
            try:
                with Image.open(path) as pil_img:
                    img = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
                return img
                
            except Exception as e2:
                print(f"PIL failed for {path}: {e2}")
                
                # Strategia 3: Fallback - immagine nera
                print(f"🚨 Using black fallback for {path}")
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
    
    def _fix_nan_values(self, img: np.ndarray) -> np.ndarray:
        """Fix dei valori NaN in modo professionale"""
        # Sostituisci NaN con 0
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clamp ai valori validi
        img = np.clip(img, 0.0, 1.0)
        
        self.nan_count += 1
        
        return img
    
    def _to_tensor_safe(self, img: np.ndarray) -> torch.Tensor:
        """Conversione sicura a tensor"""
        # Assicurati che sia float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Fix eventuali NaN rimanenti
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Assicurati che sia nel range corretto
        img = np.clip(img, 0.0, 1.0)
        
        if len(img.shape) == 3:
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
        
        tensor = torch.from_numpy(img.copy()).float()
        
        # Verifica finale
        if torch.isnan(tensor).any():
            print("🚨 NaN still present after conversion, forcing to zero")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        return tensor
    
    def _normalize_tensor_safe(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizzazione sicura"""
        # Normalizzazione standard per immagini [0,1] -> [-1,1]
        normalized = (tensor - 0.5) / 0.5
        
        # Fix eventuali NaN
        normalized = torch.nan_to_num(normalized, nan=0.0)
        
        return normalized
    
    def get_stats(self):
        """Ottieni statistiche del dataset"""
        return {
            'total_files': len(self.image_files),
            'nan_fixes': self.nan_count,
            'corrupted_files': len(self.corrupted_files),
            'corrupted_list': self.corrupted_files
        }


def create_robust_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoader robusti per training e validation
    """
    
    print("🔧 Creating ROBUST dataloaders...")
    
    # Dataset robusti
    train_dataset = RobustAstroDataset(
        root_dir=root_dir,
        mode='train',
        image_size=image_size,
        normalize=True,
        augment=True
    )
    
    val_dataset = RobustAstroDataset(
        root_dir=root_dir,
        mode='val',
        image_size=image_size,
        normalize=True,
        augment=False
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Print statistiche
    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()
    
    print(f"📊 Train dataset: {train_stats['total_files']} files, {train_stats['nan_fixes']} NaN fixes, {train_stats['corrupted_files']} corrupted")
    print(f"📊 Val dataset: {val_stats['total_files']} files, {val_stats['nan_fixes']} NaN fixes, {val_stats['corrupted_files']} corrupted")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test del dataset robusto
    print("🧪 Testing robust dataset...")
    
    train_loader, val_loader = create_robust_dataloaders(
        root_dir='/workspace',
        batch_size=4,
        num_workers=1
    )
    
    print("🔍 Testing first batch...")
    batch = next(iter(train_loader))
    
    inputs = batch['input']
    targets = batch['target']
    
    print(f"✅ Batch loaded successfully:")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Target shape: {targets.shape}")
    print(f"   Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"   Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    print(f"   Input NaN count: {torch.isnan(inputs).sum()}")
    print(f"   Target NaN count: {torch.isnan(targets).sum()}")