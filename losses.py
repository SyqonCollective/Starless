import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM
import lpips
import numpy as np
from typing import Dict, Optional


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss - robusto agli outlier, ideale per astrofotografia"""
    
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        return torch.mean(loss)


class EdgeLoss(nn.Module):
    """Loss per preservare i bordi e dettagli fini nelle nebulose"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel kernels per edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # Converti a grayscale se necessario
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        # Assicurati che i kernel Sobel siano dello stesso tipo dei tensori input (fix mixed precision)
        sobel_x = self.sobel_x.to(pred_gray.dtype).to(pred_gray.device)
        sobel_y = self.sobel_y.to(pred_gray.dtype).to(pred_gray.device)
        
        # Calcola gradienti
        pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        target_edge_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        
        # L1 loss sui gradienti
        return F.l1_loss(pred_edge, target_edge)


class FrequencyLoss(nn.Module):
    """Loss nel dominio delle frequenze per preservare texture dettagliate"""
    
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, pred, target):
        # FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Loss sulla magnitudine dello spettro
        return F.l1_loss(pred_mag, target_mag)


class StarPreservationLoss(nn.Module):
    """
    Loss che penalizza la rimozione di dettagli non stellari
    Usa una maschera per identificare regioni stellari vs non stellari
    """
    
    def __init__(self, star_threshold=0.8):
        super(StarPreservationLoss, self).__init__()
        self.star_threshold = star_threshold

    def create_star_mask(self, input_img, target_img):
        """
        Crea una maschera per identificare le stelle
        Stelle = regioni dove input > target significativamente
        """
        diff = input_img - target_img
        # Normalizza per canale
        diff_norm = diff / (input_img.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
        
        # Maschera stellare: dove la differenza è significativa
        star_mask = (diff_norm > self.star_threshold).float()
        
        # Maschera non stellare
        non_star_mask = 1.0 - star_mask
        
        return star_mask, non_star_mask

    def forward(self, pred, input_img, target_img):
        star_mask, non_star_mask = self.create_star_mask(input_img, target_img)
        
        # Loss pesata: più peso alle regioni non stellari che devono essere preservate
        diff = pred - target_img
        
        # Penalizza errori nelle regioni non stellari (galassie, nebulose)
        non_star_loss = F.l1_loss(pred * non_star_mask, target_img * non_star_mask)
        
        # Loss minore nelle regioni stellari (dove è OK rimuovere contenuto)
        star_loss = F.l1_loss(pred * star_mask, target_img * star_mask) * 0.5
        
        return non_star_loss + star_loss


class AstroLoss(nn.Module):
    """
    Loss combinata per astrofotografia - rimozione stelle
    Combina multiple loss function per risultati ottimali
    """
    
    def __init__(
        self,
        l1_weight=1.0,
        charbonnier_weight=1.0,
        ssim_weight=0.5,
        edge_weight=0.3,
        freq_weight=0.2,
        perceptual_weight=0.5,
        star_preservation_weight=2.0,
        device='cuda'
    ):
        super(AstroLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.perceptual_weight = perceptual_weight
        self.star_preservation_weight = star_preservation_weight
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.ssim_loss = SSIM(data_range=2.0, size_average=True, channel=3)  # per [-1,1]
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyLoss()
        self.star_preservation_loss = StarPreservationLoss()
        
        # Perceptual loss (LPIPS)
        if perceptual_weight > 0:
            self.perceptual_loss = lpips.LPIPS(net='vgg').to(device)
            # Freeze perceptual network
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
        else:
            self.perceptual_loss = None

    def forward(self, pred, target, input_img=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: predicted output
            target: ground truth target
            input_img: original input (con stelle) - necessario per star preservation
        """
        losses = {}
        total_loss = 0
        
        # L1 Loss
        if self.l1_weight > 0:
            l1 = self.l1_loss(pred, target)
            losses['l1'] = l1
            total_loss += self.l1_weight * l1
        
        # Charbonnier Loss (robusto)
        if self.charbonnier_weight > 0:
            charbonnier = self.charbonnier_loss(pred, target)
            losses['charbonnier'] = charbonnier
            total_loss += self.charbonnier_weight * charbonnier
        
        # SSIM Loss
        if self.ssim_weight > 0:
            # SSIM lavora meglio con valori [0,1], convertiamo da [-1,1]
            pred_01 = (pred + 1) / 2
            target_01 = (target + 1) / 2
            ssim_value = self.ssim_loss(pred_01, target_01)
            ssim_loss = 1 - ssim_value
            losses['ssim'] = ssim_loss
            total_loss += self.ssim_weight * ssim_loss
        
        # Edge Loss
        if self.edge_weight > 0:
            edge = self.edge_loss(pred, target)
            losses['edge'] = edge
            total_loss += self.edge_weight * edge
        
        # Frequency Loss
        if self.freq_weight > 0:
            freq = self.freq_loss(pred, target)
            losses['freq'] = freq
            total_loss += self.freq_weight * freq
        
        # Perceptual Loss
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            # LPIPS expects [-1,1] range
            perceptual = self.perceptual_loss(pred, target).mean()
            losses['perceptual'] = perceptual
            total_loss += self.perceptual_weight * perceptual
        
        # Star Preservation Loss
        if self.star_preservation_weight > 0 and input_img is not None:
            star_pres = self.star_preservation_loss(pred, input_img, target)
            losses['star_preservation'] = star_pres
            total_loss += self.star_preservation_weight * star_pres
        
        # Check for NaN/Inf in total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Return a safe fallback loss
            total_loss = torch.tensor(1.0, device=pred.device, requires_grad=True)
            
        losses['total'] = total_loss
        return losses


class AdaptiveLoss(nn.Module):
    """
    Loss adattiva che cambia i pesi durante il training
    Enfatizza diversi aspetti nelle diverse fasi del training
    """
    
    def __init__(self, base_loss: AstroLoss):
        super(AdaptiveLoss, self).__init__()
        self.base_loss = base_loss
        self.initial_weights = {
            'l1': base_loss.l1_weight,
            'charbonnier': base_loss.charbonnier_weight,
            'ssim': base_loss.ssim_weight,
            'edge': base_loss.edge_weight,
            'freq': base_loss.freq_weight,
            'perceptual': base_loss.perceptual_weight,
            'star_preservation': base_loss.star_preservation_weight,
        }

    def update_weights(self, epoch, total_epochs):
        """Aggiorna i pesi delle loss in base all'epoca corrente"""
        progress = epoch / total_epochs
        
        # Nelle prime epoche: focus su ricostruzione base
        if progress < 0.3:
            self.base_loss.l1_weight = self.initial_weights['l1'] * 2.0
            self.base_loss.charbonnier_weight = self.initial_weights['charbonnier'] * 2.0
            self.base_loss.ssim_weight = self.initial_weights['ssim'] * 0.5
            self.base_loss.perceptual_weight = self.initial_weights['perceptual'] * 0.5
        
        # Nelle epoche medie: bilanciato
        elif progress < 0.7:
            for key, weight in self.initial_weights.items():
                setattr(self.base_loss, f'{key}_weight', weight)
        
        # Nelle ultime epoche: focus su dettagli e texture
        else:
            self.base_loss.edge_weight = self.initial_weights['edge'] * 1.5
            self.base_loss.freq_weight = self.initial_weights['freq'] * 1.5
            self.base_loss.perceptual_weight = self.initial_weights['perceptual'] * 1.5
            self.base_loss.star_preservation_weight = self.initial_weights['star_preservation'] * 1.2

    def forward(self, pred, target, input_img=None):
        return self.base_loss(pred, target, input_img)


def create_loss_function(loss_config: Optional[Dict] = None, device='cuda'):
    """Factory function per creare loss function"""
    
        # Configurazione loss molto più conservativa per stabilità
    default_config = {
        'l1_weight': 1.0,
        'charbonnier_weight': 0.5,  # Ridotto per stabilità
        'ssim_weight': 0.1,         # Molto ridotto
        'edge_weight': 0.0,         # Disabilitato - problematico con mixed precision
        'freq_weight': 0.0,         # Disabilitato - può causare instabilità  
        'perceptual_weight': 0.0,   # Disabilitato - VGG può causare NaN
        'star_preservation_weight': 0.5,  # Ridotto drasticamente
    }
    
    if loss_config:
        default_config.update(loss_config)
    
    return AstroLoss(device=device, **default_config)


if __name__ == "__main__":
    # Test delle loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    input_img = torch.randn(2, 3, 256, 256).to(device)
    
    # Normalizza in range [-1, 1]
    pred = torch.tanh(pred)
    target = torch.tanh(target)
    input_img = torch.tanh(input_img)
    
    # Test loss
    loss_fn = create_loss_function(device=device)
    
    losses = loss_fn(pred, target, input_img)
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.6f}")
    
    print(f"\nTotal loss: {losses['total'].item():.6f}")