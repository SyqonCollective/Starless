import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class LayerNormFunction(torch.autograd.Function):
    """Implementazione efficiente di LayerNorm"""
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """LayerNorm per immagini 2D"""
    
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Gating mechanism semplificato per NAF"""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Blocco base di NAFNet con miglioramenti per astrofotografia"""
    
    def __init__(
        self, 
        c, 
        DW_Expand=2, 
        FFN_Expand=2, 
        drop_out_rate=0.0,
        drop_path_rate=0.0
    ):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # DropPath per regularizzazione
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        x = self.drop_path(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        x = self.drop_path(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet ottimizzato per rimozione stelle in astrofotografia
    Architettura U-Net con blocchi NAF per preservare texture
    """
    
    def __init__(
        self, 
        img_channel=3, 
        width=32, 
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], 
        dec_blk_nums=[2, 2, 2, 2],
        drop_path_rate=0.0,
        drop_out_rate=0.0
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        
        # Build DropPath rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums))]
        idx = 0
        
        # Encoder
        for num in enc_blk_nums:
            encoder_blocks = nn.Sequential(
                *[NAFBlock(chan, drop_out_rate=drop_out_rate, drop_path_rate=dpr[idx + i]) for i in range(num)]
            )
            self.encoders.append(encoder_blocks)
            idx += num
            
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # Middle
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, drop_out_rate=drop_out_rate, drop_path_rate=dpr[idx + i]) for i in range(middle_blk_num)]
        )
        idx += middle_blk_num

        # Decoder
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            
            decoder_blocks = nn.Sequential(
                *[NAFBlock(chan, drop_out_rate=drop_out_rate, drop_path_rate=dpr[idx + i]) for i in range(num)]
            )
            self.decoders.append(decoder_blocks)
            idx += num

        self.padder_size = 2 ** len(self.encoders)

        # Inizializzazione dei pesi
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Inizializzazione personalizzata per astrofotografia"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Inizializzazione Xavier per preservare il range dinamico
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle
        x = self.middle_blks(x)

        # Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp  # Skip connection globale per preservare texture

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Padding riflessivo per garantire dimensioni multiple di padder_size"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # Usa padding riflessivo invece di zero per evitare bordi artificiosi
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x


class NAFNetLocal(NAFNet):
    """
    Versione locale di NAFNet con blocchi adattivi per stelle di diverse grandezze
    """
    
    def __init__(
        self, 
        img_channel=3, 
        width=32, 
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8], 
        dec_blk_nums=[2, 2, 2, 2],
        drop_path_rate=0.0,
        drop_out_rate=0.0,
        train_size=(1, 3, 256, 256)
    ):
        super().__init__(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            drop_path_rate=drop_path_rate,
            drop_out_rate=drop_out_rate
        )
        
        self.train_size = train_size

    def forward_base(self, inp):
        """Forward base senza chunking - per uso interno"""
        return super().forward(inp)
    
    def forward(self, inp):
        # Durante il training usa l'implementazione standard
        if self.training:
            return self.forward_base(inp)
        
        # Durante l'inferenza può gestire immagini più grandi
        return self.forward_chop(inp)

    def forward_chop(self, inp, shave=64, min_size=160000):
        """
        Forward con chunking MIGLIORATO - overlap maggiore per evitare quadrettini
        shave=64 per copertura adeguata del receptive field
        """
        scale = 1
        n_GPUs = 1
        b, c, h, w = inp.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        
        lr_list = [
            inp[:, :, 0:h_size, 0:w_size],
            inp[:, :, 0:h_size, (w - w_size):w],
            inp[:, :, (h - h_size):h, 0:w_size],
            inp[:, :, (h - h_size):h, (w - w_size):w]
        ]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.forward_base(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size)
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        # BLENDING nelle zone di overlap invece di hard stitching
        output = inp.new_zeros(b, c, h, w)
        
        # Quadrante 1 (top-left) - copia completo
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        
        # Quadrante 2 (top-right) - blend orizzontale
        overlap_w = w_size - w + w_half
        for i in range(overlap_w):
            alpha = i / overlap_w
            output[:, :, 0:h_half, w_half + i] = (
                (1 - alpha) * sr_list[0][:, :, 0:h_half, w_half + i] +
                alpha * sr_list[1][:, :, 0:h_half, (w_size - w + w_half) + i]
            )
        output[:, :, 0:h_half, w_half + overlap_w:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half) + overlap_w:w_size]
        
        # Quadrante 3 (bottom-left) - blend verticale  
        overlap_h = h_size - h + h_half
        for i in range(overlap_h):
            alpha = i / overlap_h
            output[:, :, h_half + i, 0:w_half] = (
                (1 - alpha) * sr_list[0][:, :, h_half + i, 0:w_half] +
                alpha * sr_list[2][:, :, (h_size - h + h_half) + i, 0:w_half]
            )
        output[:, :, h_half + overlap_h:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half) + overlap_h:h_size, 0:w_half]
        
        # Quadrante 4 (bottom-right) - blend sia orizzontale che verticale
        for i in range(overlap_h):
            for j in range(overlap_w):
                alpha_h = i / overlap_h
                alpha_w = j / overlap_w
                
                # Interpolazione bilineare dei 4 valori
                v1 = sr_list[0][:, :, h_half + i, w_half + j]  # top-left
                v2 = sr_list[1][:, :, h_half + i, (w_size - w + w_half) + j]  # top-right  
                v3 = sr_list[2][:, :, (h_size - h + h_half) + i, w_half + j]  # bottom-left
                v4 = sr_list[3][:, :, (h_size - h + h_half) + i, (w_size - w + w_half) + j]  # bottom-right
                
                output[:, :, h_half + i, w_half + j] = (
                    (1 - alpha_h) * (1 - alpha_w) * v1 +
                    (1 - alpha_h) * alpha_w * v2 +
                    alpha_h * (1 - alpha_w) * v3 +
                    alpha_h * alpha_w * v4
                )

        return output


def create_model(model_type='nafnet', **kwargs):
    """Factory function per creare modelli NAFNet"""
    
    if model_type == 'nafnet':
        return NAFNet(**kwargs)
    elif model_type == 'nafnet_local':
        return NAFNetLocal(**kwargs)
    else:
        raise ValueError(f"Modello non supportato: {model_type}")


# Configurazioni predefinite per astrofotografia
ASTRO_CONFIGS = {
    'small': {
        'width': 32,
        'middle_blk_num': 8,
        'enc_blk_nums': [2, 2, 4, 6],
        'dec_blk_nums': [2, 2, 2, 2],
    },
    'base': {
        'width': 32,
        'middle_blk_num': 12,
        'enc_blk_nums': [2, 2, 4, 8],
        'dec_blk_nums': [2, 2, 2, 2],
    },
    'large': {
        'width': 48,
        'middle_blk_num': 16,
        'enc_blk_nums': [2, 2, 6, 12],
        'dec_blk_nums': [2, 2, 2, 2],
    }
}


def create_astro_model(size='base', model_type='nafnet', **kwargs):
    """Crea modello NAFNet ottimizzato per astrofotografia"""
    config = ASTRO_CONFIGS[size].copy()
    config.update(kwargs)
    return create_model(model_type=model_type, **config)


if __name__ == "__main__":
    # Test del modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_astro_model(size='base')
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Parametri trainabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")