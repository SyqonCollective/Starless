#!/usr/bin/env python3
"""
PROFESSIONAL INFERENCE GUI
GUI per inferenza con checkpoint loading, tile processing e overlap
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
from threading import Thread
import time
from model import create_astro_model


class StarRemovalGUI:
    """GUI professionale per rimozione stelle con tile processing"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌌 Professional Star Removal - Tile Inference")
        self.root.geometry("1200x800")
        
        # Variabili
        self.model = None
        # Metal acceleration for M1 Pro
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.checkpoint_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.tile_size = tk.IntVar(value=512)
        self.overlap = tk.IntVar(value=96)
        self.processing = False
        
        self.setup_gui()
        self.show_device_info()
        
    def show_device_info(self):
        """Show device information at startup"""
        device_name = "Metal (M1 Pro)" if self.device.type == 'mps' else str(self.device).upper()
        print(f"🚀 Using device: {device_name}")
        
    def setup_gui(self):
        """Setup interfaccia GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🌌 Professional Star Removal", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Checkpoint selection
        checkpoint_frame = ttk.LabelFrame(main_frame, text="Model Checkpoint", padding="10")
        checkpoint_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(checkpoint_frame, text="Checkpoint:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(checkpoint_frame, textvariable=self.checkpoint_path, width=60).grid(row=0, column=1, padx=(10, 10))
        ttk.Button(checkpoint_frame, text="Browse...", command=self.browse_checkpoint).grid(row=0, column=2)
        ttk.Button(checkpoint_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=(10, 0))
        
        # Model status
        self.model_status = ttk.Label(checkpoint_frame, text="❌ No model loaded", foreground="red")
        self.model_status.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        # Image selection
        image_frame = ttk.LabelFrame(main_frame, text="Image Processing", padding="10")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(image_frame, text="Image:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(image_frame, textvariable=self.image_path, width=60).grid(row=0, column=1, padx=(10, 10))
        ttk.Button(image_frame, text="Browse...", command=self.browse_image).grid(row=0, column=2)
        
        # Processing parameters
        params_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(params_frame, text="Tile Size:").grid(row=0, column=0, sticky=tk.W)
        tile_combo = ttk.Combobox(params_frame, textvariable=self.tile_size, values=[256, 512, 1024], width=10)
        tile_combo.grid(row=0, column=1, padx=(10, 20))
        
        ttk.Label(params_frame, text="Overlap:").grid(row=0, column=2, sticky=tk.W)
        overlap_combo = ttk.Combobox(params_frame, textvariable=self.overlap, values=[32, 64, 96, 128], width=10)
        overlap_combo.grid(row=0, column=3, padx=(10, 20))
        overlap_combo.set(96)  # Default overlap più alto per astrofotografia
        
        # Process button
        self.process_btn = ttk.Button(params_frame, text="🚀 Process Image", command=self.process_image)
        self.process_btn.grid(row=0, column=4, padx=(20, 0))
        
        # Progress bars
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(progress_frame, text="Overall Progress:").grid(row=0, column=0, sticky=tk.W)
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Label(progress_frame, text="Tiles Progress:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.tile_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.tile_progress.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        
        progress_frame.columnconfigure(1, weight=1)
        
        # Status labels
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.tile_status_label = ttk.Label(status_frame, text="", foreground="blue")
        self.tile_status_label.grid(row=0, column=1, sticky=tk.E)
        
        status_frame.columnconfigure(1, weight=1)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Canvas for image preview
        self.canvas = tk.Canvas(preview_frame, width=800, height=400, bg='black')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
    def browse_checkpoint(self):
        """Browse per checkpoint"""
        filename = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.checkpoint_path.set(filename)
    
    def browse_image(self):
        """Browse per immagine"""
        filename = filedialog.askopenfilename(
            title="Select Image to Process",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.image_path.set(filename)
            self.preview_image(filename)
    
    def preview_image(self, image_path):
        """Preview dell'immagine"""
        try:
            # Carica immagine
            img = Image.open(image_path)
            
            # Resize per preview
            img.thumbnail((800, 400), Image.Resampling.LANCZOS)
            
            # Converti per tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Aggiorna canvas
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot preview image: {e}")
    
    def load_model(self):
        """Carica il modello dal checkpoint"""
        if not self.checkpoint_path.get():
            messagebox.showerror("Error", "Please select a checkpoint file")
            return
        
        try:
            self.status_label.config(text="Loading model...", foreground="orange")
            self.root.update()
            
            # Crea modello
            self.model = create_astro_model(
                size='small',
                img_channel=3,
                width=32,
                middle_blk_num=2,
                enc_blk_nums=[2, 2, 4, 6],
                dec_blk_nums=[2, 2, 2, 2]
            ).to(self.device)
            
            # Carica checkpoint
            checkpoint = torch.load(self.checkpoint_path.get(), map_location=self.device)
            
            # Carica state dict
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            # Update status
            params = sum(p.numel() for p in self.model.parameters())
            device_name = "Metal (M1 Pro)" if self.device.type == 'mps' else str(self.device).upper()
            self.model_status.config(text=f"✅ Model loaded: {params:,} parameters on {device_name}", foreground="green")
            self.status_label.config(text="Model loaded successfully", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load model: {e}")
            self.model_status.config(text="❌ Failed to load model", foreground="red")
            self.status_label.config(text="Ready", foreground="green")
    
    def process_image(self):
        """Processa l'immagine con tile overlap"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
        
        # Start processing in thread
        thread = Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Thread per processing dell'immagine"""
        try:
            self.processing = True
            self.process_btn.config(state='disabled')
            self.progress.config(mode='determinate')
            self.tile_progress.config(mode='determinate')
            
            # Update status
            self.status_label.config(text="Processing image...", foreground="orange")
            self.root.update()
            
            # Carica immagine
            image_path = self.image_path.get()
            original_img = self.load_image_for_inference(image_path)
            
            self.status_label.config(text="Processing tiles...", foreground="orange")
            self.root.update()
            
            # Processa con tile overlap
            processed_img = self.process_with_tiles(original_img)
            
            self.status_label.config(text="Saving result...", foreground="orange")
            self.root.update()
            
            # Salva risultato
            output_path = self.save_result(image_path, processed_img)
            
            # Update status
            self.status_label.config(text=f"✅ Completed! Saved to: {output_path}", foreground="green")
            
            # Preview result
            self.preview_image(output_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")
            self.status_label.config(text="❌ Processing failed", foreground="red")
        
        finally:
            self.processing = False
            self.process_btn.config(state='normal')
            self.progress.config(value=0)
            self.tile_progress.config(value=0)
            self.tile_status_label.config(text="")
    
    def load_image_for_inference(self, image_path):
        """Carica immagine per inferenza"""
        # Prova con OpenCV prima
        try:
            if image_path.lower().endswith(('.tiff', '.tif')):
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                else:
                    img = img.astype(np.float32) / 255.0
            else:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = img.astype(np.float32) / 255.0
            
            # BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
            
        except:
            # Fallback con PIL
            with Image.open(image_path) as pil_img:
                img = np.array(pil_img.convert('RGB')).astype(np.float32) / 255.0
            return img
    
    def process_with_tiles(self, image):
        """Processa immagine con tile overlap professionale"""
        tile_size = self.tile_size.get()
        overlap = self.overlap.get()
        
        h, w, c = image.shape
        stride = tile_size - overlap
        
        # Calcola numero di tile
        n_tiles_h = (h - overlap + stride - 1) // stride
        n_tiles_w = (w - overlap + stride - 1) // stride
        
        # Pad image se necessario
        pad_h = n_tiles_h * stride + overlap - h
        pad_w = n_tiles_w * stride + overlap - w
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode='reflect')
        
        # Result image
        result = np.zeros_like(image)
        weight_map = np.zeros(image.shape[:2])
        
        total_tiles = n_tiles_h * n_tiles_w
        current_tile = 0
        
        # Setup progress bars
        self.progress.config(maximum=100)
        self.tile_progress.config(maximum=total_tiles)
        
        with torch.no_grad():
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    current_tile += 1
                    
                    # Update progress bars
                    tile_progress = (current_tile / total_tiles) * 100
                    self.progress.config(value=tile_progress)
                    self.tile_progress.config(value=current_tile)
                    
                    # Update status labels
                    self.status_label.config(text=f"Processing tiles... {tile_progress:.1f}%", foreground="orange")
                    self.tile_status_label.config(text=f"Tile {current_tile}/{total_tiles}")
                    self.root.update()
                    
                    # Extract tile
                    y1 = i * stride
                    y2 = y1 + tile_size
                    x1 = j * stride
                    x2 = x1 + tile_size
                    
                    tile = image[y1:y2, x1:x2]
                    
                    # Preprocess tile
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    tile_tensor = (tile_tensor - 0.5) / 0.5  # Normalize [-1, 1]
                    
                    # Inference with proper device acceleration
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            output = self.model(tile_tensor)
                    else:
                        # For MPS (Metal) and CPU
                        output = self.model(tile_tensor)
                    
                    # Postprocess
                    output = (output * 0.5 + 0.5).clamp(0, 1)  # Denormalize
                    processed_tile = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    # Advanced weight for smooth blending (Gaussian-like)
                    weight = np.ones((tile_size, tile_size))
                    if overlap > 0:
                        # Create smooth Gaussian-like weight mask
                        fade = overlap
                        
                        # Create 1D weight profile (smooth falloff)
                        x = np.linspace(0, 1, fade)
                        smooth_profile = 0.5 * (1 + np.cos(np.pi * x))  # Cosine falloff
                        
                        # Apply to edges
                        if fade > 0:
                            # Top edge
                            for k in range(fade):
                                if k < len(smooth_profile):
                                    weight[k, :] *= smooth_profile[k]
                            
                            # Bottom edge
                            for k in range(fade):
                                if k < len(smooth_profile):
                                    weight[-(k+1), :] *= smooth_profile[k]
                            
                            # Left edge  
                            for k in range(fade):
                                if k < len(smooth_profile):
                                    weight[:, k] *= smooth_profile[k]
                            
                            # Right edge
                            for k in range(fade):
                                if k < len(smooth_profile):
                                    weight[:, -(k+1)] *= smooth_profile[k]
                    
                    # Add to result
                    result[y1:y2, x1:x2] += processed_tile * weight[:, :, np.newaxis]
                    weight_map[y1:y2, x1:x2] += weight
        
        # Normalize by weight
        weight_map[weight_map == 0] = 1
        result = result / weight_map[:, :, np.newaxis]
        
        # Crop to original size
        result = result[:h, :w]
        
        # Post-processing per rimuovere artifacts
        result = self.post_process_result(result, image)
        
        return result
    
    def post_process_result(self, result, original):
        """Post-processing per correggere artifacts e migliorare texture"""
        try:
            # Ensure same shape
            if result.shape != original.shape:
                return result
            
            # 1. Correzione valori estremi (quadrettini neri)
            # Identifica pixel troppo scuri rispetto al vicinato
            from scipy import ndimage
            
            # Convert to grayscale for analysis
            if len(result.shape) == 3:
                gray_result = np.mean(result, axis=2)
                gray_original = np.mean(original, axis=2)
            else:
                gray_result = result
                gray_original = original
            
            # Trova regioni troppo scure (possibili artifacts)
            local_mean = ndimage.uniform_filter(gray_result, size=5)
            dark_mask = (gray_result < local_mean * 0.3) & (gray_result < 0.1)
            
            # Dilata leggermente la mask per catturare bordi
            dark_mask = ndimage.binary_dilation(dark_mask, iterations=1)
            
            # Per ogni canale, correggi le regioni scure
            corrected = result.copy()
            for c in range(result.shape[2] if len(result.shape) == 3 else 1):
                if len(result.shape) == 3:
                    channel = result[:, :, c]
                    orig_channel = original[:, :, c]
                else:
                    channel = result
                    orig_channel = original
                
                # Smooth interpolation nelle regioni scure
                smoothed = ndimage.gaussian_filter(channel, sigma=2.0)
                
                # Blend con l'originale nelle regioni problematiche
                blend_factor = 0.3  # Mantieni un po' dell'effetto originale
                corrected_channel = np.where(
                    dark_mask,
                    blend_factor * orig_channel + (1 - blend_factor) * smoothed,
                    channel
                )
                
                if len(result.shape) == 3:
                    corrected[:, :, c] = corrected_channel
                else:
                    corrected = corrected_channel
            
            # 2. Leggero sharpening per recuperare dettagli
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            
            if len(corrected.shape) == 3:
                for c in range(corrected.shape[2]):
                    corrected[:, :, c] = ndimage.convolve(corrected[:, :, c], kernel)
            else:
                corrected = ndimage.convolve(corrected, kernel)
            
            # Clamp values
            corrected = np.clip(corrected, 0, 1)
            
            return corrected
            
        except Exception as e:
            print(f"Warning: Post-processing failed: {e}")
            return result
    
    def save_result(self, original_path, processed_img):
        """Salva il risultato"""
        original_path = Path(original_path)
        output_path = original_path.parent / f"{original_path.stem}_starless{original_path.suffix}"
        
        # Convert to uint8 or uint16 based on original
        if original_path.suffix.lower() in ['.tiff', '.tif']:
            # Save as 16-bit TIFF
            processed_img = (processed_img * 65535).astype(np.uint16)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), processed_img)
        else:
            # Save as 8-bit
            processed_img = (processed_img * 255).astype(np.uint8)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), processed_img)
        
        return str(output_path)
    
    def run(self):
        """Avvia la GUI"""
        self.root.mainloop()


def main():
    """Main function"""
    app = StarRemovalGUI()
    app.run()


if __name__ == '__main__':
    main()