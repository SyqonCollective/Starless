import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from PIL import Image
from typing import Tuple, Optional


class AstroDataset(Dataset):
    """
    Dataset personalizzato per astrofotografia - rimozione stelle
    Carica immagini da input (con stelle) e target (senza stelle)
    """
    
    def __init__(
        self, 
        root_dir: str, 
        mode: str = 'train',
        image_size: int = 512,
        normalize: bool = True,
        augment: bool = True
    ):
        """
        Args:
            root_dir: percorso alla directory principale
            mode: 'train' o 'val'
            image_size: dimensione delle immagini (quadrate)
            normalize: se normalizzare le immagini
            augment: se applicare data augmentation
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment and mode == 'train'
        
        # Percorsi delle directory
        self.input_dir = os.path.join(root_dir, f'{mode}_tiles', 'input')
        self.target_dir = os.path.join(root_dir, f'{mode}_tiles', 'target')
        
        # Lista dei file immagine
        self.image_files = self._get_image_files()
        
        # Trasformazioni per astrofotografia
        self.setup_transforms()
        
        print(f"Caricato dataset {mode}: {len(self.image_files)} tile 512x512")
        if len(self.image_files) == 0:
            raise ValueError(f"Nessuna immagine trovata in {self.input_dir} o {self.target_dir}")
    
    def _get_image_files(self):
        """Ottiene la lista dei file immagine disponibili"""
        input_files = set(os.listdir(self.input_dir))
        target_files = set(os.listdir(self.target_dir))
        
        # Solo file che esistono in entrambe le directory
        common_files = input_files.intersection(target_files)
        common_files = [f for f in common_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        return sorted(list(common_files))
    
    def setup_transforms(self):
        """Setup trasformazioni per astrofotografia - tile già 512x512"""
        if self.augment:
            # Augmentazioni specifiche per astrofotografia - NO RESIZE
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Rotazioni piccole per preservare orientamento stellare
                A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT),
                # Variazioni di luminosità e contrasto molto moderate per astrofotografia
                A.RandomBrightnessContrast(
                    brightness_limit=0.08, 
                    contrast_limit=0.08, 
                    p=0.3
                ),
                # Rumore gaussiano leggero che simula noise del sensore
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.25),
                # Sfocatura gaussiana molto leggera per simulare seeing atmosferico
                A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=0.1),
            ], additional_targets={'target': 'image'})
        else:
            # Nessuna trasformazione per validation - immagini già 512x512
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Carica una coppia input-target"""
        filename = self.image_files[idx]
        
        # Carica immagini
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)
        
        # Leggi immagini come float32 per preservare precisione
        input_img = self._load_image(input_path)
        target_img = self._load_image(target_path)
        
        # Applica trasformazioni (solo augmentations, no resize)
        if self.transform:
            transformed = self.transform(image=input_img, target=target_img)
            input_img = transformed['image']
            target_img = transformed['target']
        
        # Verifica che le dimensioni siano corrette (dovrebbero essere già 512x512)
        expected_size = (self.image_size, self.image_size)
        if input_img.shape[:2] != expected_size:
            # Solo se necessario, resize (non dovrebbe mai accadere con tile pre-processati)
            input_img = cv2.resize(input_img, expected_size, interpolation=cv2.INTER_LANCZOS4)
            target_img = cv2.resize(target_img, expected_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Converti a tensori PyTorch
        input_tensor = self._to_tensor(input_img)
        target_tensor = self._to_tensor(target_img)
        
        # Normalizza se richiesto
        if self.normalize:
            input_tensor = self._normalize_tensor(input_tensor)
            target_tensor = self._normalize_tensor(target_tensor)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'filename': filename
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Carica immagine preservando il range dinamico per astrofotografia"""
        # Prova a caricare come 16-bit se possibile
        if path.lower().endswith(('.tiff', '.tif')):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32) / 255.0
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = img.astype(np.float32) / 255.0
        
        # Converti da BGR a RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Converte numpy array a tensor PyTorch"""
        if len(img.shape) == 3:
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img.copy()).float()
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizza tensor per astrofotografia (preserva range dinamico)"""
        # Normalizzazione più soft per astrofotografia
        # Mantiene i dettagli nelle zone scure
        return tensor * 2.0 - 1.0  # Da [0,1] a [-1,1]


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea i DataLoader per training e validation
    
    Args:
        root_dir: percorso alla directory principale
        batch_size: dimensione del batch
        image_size: dimensione delle immagini
        num_workers: numero di worker per il caricamento
        pin_memory: usa pin memory per GPU
    
    Returns:
        Tuple di (train_loader, val_loader)
    """
    
    # Dataset di training
    train_dataset = AstroDataset(
        root_dir=root_dir,
        mode='train',
        image_size=image_size,
        normalize=True,
        augment=True
    )
    
    # Dataset di validation
    val_dataset = AstroDataset(
        root_dir=root_dir,
        mode='val',
        image_size=image_size,
        normalize=True,
        augment=False
    )
    
    # DataLoader di training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # DataLoader di validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test del dataset
    root_dir = "."
    
    try:
        train_loader, val_loader = create_dataloaders(
            root_dir=root_dir,
            batch_size=2,
            image_size=256,
            num_workers=0  # 0 per testing
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test caricamento batch
        batch = next(iter(train_loader))
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Input range: [{batch['input'].min():.3f}, {batch['input'].max():.3f}]")
        print(f"Target range: [{batch['target'].min():.3f}, {batch['target'].max():.3f}]")
        
    except Exception as e:
        print(f"Errore durante il test del dataset: {e}")