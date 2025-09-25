#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import create_astro_model
from utils import tensor_to_numpy, save_image_comparison


class StarRemover:
    """Classe per rimuovere stelle dalle immagini astrofotografiche"""
    
    def __init__(self, model_path, device='cuda', model_size='base'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size
        
        # Carica il modello
        self.model = self.load_model(model_path)
        self.model.eval()
        
        print(f"Modello caricato su {self.device}")
        print(f"Parametri: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_model(self, model_path):
        """Carica il modello addestrato"""
        
        # Crea il modello
        model = create_astro_model(
            size=self.model_size,
            model_type='nafnet_local'  # Usa versione locale per immagini grandi
        )
        
        # Carica i pesi
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Gestisci diversi formati di checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Pesi caricati da: {model_path}")
            
            if 'epoch' in checkpoint:
                print(f"Epoca del modello: {checkpoint['epoch']}")
            if 'best_psnr' in checkpoint:
                print(f"Miglior PSNR: {checkpoint['best_psnr']:.2f}")
        else:
            print(f"Attenzione: file modello non trovato {model_path}")
            print("Utilizzando pesi casuali")
        
        return model.to(self.device)

    def preprocess_image(self, image_path):
        """Preprocessa immagine per l'inferenza"""
        
        # Carica immagine
        if image_path.lower().endswith(('.tiff', '.tif')):
            # Gestione immagini 16-bit
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32) / 255.0
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = img.astype(np.float32) / 255.0
        
        # BGR to RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Converti a tensor
        img_tensor = torch.from_numpy(img).float()
        
        # HWC -> CHW
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        
        # Aggiungi batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        # Normalizza a [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        
        return img_tensor.to(self.device)

    def postprocess_image(self, tensor):
        """Postprocessa tensor output"""
        
        # Da [-1, 1] a [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Remove batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # CHW -> HWC
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # To numpy
        img_np = tensor.detach().cpu().numpy()
        
        return img_np

    def remove_stars(self, image_path, output_path=None, save_comparison=False):
        """Rimuove stelle da una singola immagine"""
        
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_starless.png"
        
        # Preprocessa
        input_tensor = self.preprocess_image(image_path)
        original_size = input_tensor.shape[-2:]
        
        print(f"Processando: {image_path}")
        print(f"Dimensioni: {original_size}")
        
        # Inferenza
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Postprocessa
        output_img = self.postprocess_image(output_tensor)
        
        # Salva risultato
        output_img_pil = Image.fromarray((output_img * 255).astype(np.uint8))
        output_img_pil.save(output_path)
        
        print(f"Risultato salvato: {output_path}")
        
        # Salva confronto se richiesto
        if save_comparison:
            comparison_path = output_path.replace('.png', '_comparison.png')
            
            # Preprocessa immagine originale per confronto
            original_tensor = (input_tensor + 1.0) / 2.0  # Da [-1, 1] a [0, 1]
            original_img = self.postprocess_image(original_tensor)
            
            # Crea confronto side-by-side
            h, w = original_img.shape[:2]
            comparison = np.zeros((h, w * 2, 3))
            comparison[:, :w] = original_img
            comparison[:, w:] = output_img
            
            comparison_pil = Image.fromarray((comparison * 255).astype(np.uint8))
            comparison_pil.save(comparison_path)
            
            print(f"Confronto salvato: {comparison_path}")
        
        return output_path

    def process_directory(self, input_dir, output_dir, save_comparison=False):
        """Processa tutte le immagini in una directory"""
        
        # Crea directory output
        os.makedirs(output_dir, exist_ok=True)
        
        # Trova immagini
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"Nessuna immagine trovata in {input_dir}")
            return
        
        print(f"Trovate {len(image_files)} immagini da processare")
        
        # Processa ogni immagine
        for filename in tqdm(image_files, desc="Processando immagini"):
            input_path = os.path.join(input_dir, filename)
            
            # Nome output
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_starless.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                self.remove_stars(input_path, output_path, save_comparison)
            except Exception as e:
                print(f"Errore processando {filename}: {e}")
        
        print(f"Processamento completato. Risultati in: {output_dir}")

    def benchmark_speed(self, image_path, num_runs=10):
        """Benchmark velocità del modello"""
        
        input_tensor = self.preprocess_image(image_path)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        print(f"Tempo medio per immagine: {avg_time:.3f}s")
        print(f"FPS: {fps:.2f}")
        print(f"Dimensioni input: {input_tensor.shape}")
        
        return avg_time, fps


def main():
    parser = argparse.ArgumentParser(description='Rimozione stelle da immagini astrofotografiche')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path al modello addestrato (.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path immagine singola o directory di immagini')
    parser.add_argument('--output', type=str,
                       help='Path output (file o directory)')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base', 'large'],
                       help='Dimensione del modello')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device per inferenza')
    parser.add_argument('--comparison', action='store_true',
                       help='Salva anche immagini di confronto')
    parser.add_argument('--benchmark', action='store_true',
                       help='Esegui benchmark velocità')
    
    args = parser.parse_args()
    
    # Verifica che il modello esista
    if not os.path.exists(args.model):
        print(f"Errore: modello non trovato {args.model}")
        return
    
    # Inizializza StarRemover
    star_remover = StarRemover(
        model_path=args.model,
        device=args.device,
        model_size=args.model_size
    )
    
    # Determina se input è file o directory
    if os.path.isfile(args.input):
        # Singola immagine
        print("Modalità: singola immagine")
        
        if args.benchmark:
            star_remover.benchmark_speed(args.input)
        
        output_path = args.output if args.output else None
        star_remover.remove_stars(args.input, output_path, args.comparison)
        
    elif os.path.isdir(args.input):
        # Directory di immagini
        print("Modalità: directory")
        
        output_dir = args.output if args.output else f"{args.input}_starless"
        star_remover.process_directory(args.input, output_dir, args.comparison)
        
    else:
        print(f"Errore: input non valido {args.input}")


if __name__ == '__main__':
    main()