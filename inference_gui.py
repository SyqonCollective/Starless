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
        self.overlap = tk.IntVar(value=64)
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
        overlap_combo = ttk.Combobox(params_frame, textvariable=self.overlap, values=[32, 64, 128], width=10)
        overlap_combo.grid(row=0, column=3, padx=(10, 20))
        
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
            
            # Validazione canali
            if len(original_img.shape) != 3 or original_img.shape[2] != 3:
                raise Exception(f"Expected RGB image [H,W,3], got shape: {original_img.shape}")
            
            print(f"📐 Image loaded: {original_img.shape[1]}x{original_img.shape[0]} RGB")
            
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
        """Carica immagine per inferenza con gestione canali"""
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
            
            # Gestione canali
            if len(img.shape) == 3:
                if img.shape[2] == 4:  # RGBA -> RGB
                    img = img[:, :, :3]  # Rimuovi canale alpha
                    print("📷 Converted RGBA to RGB")
                elif img.shape[2] == 3:  # BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:  # Grayscale -> RGB
                img = np.stack([img, img, img], axis=2)
                print("📷 Converted Grayscale to RGB")
            
            return img
            
        except Exception as e:
            print(f"⚠️ OpenCV failed: {e}, trying PIL...")
            # Fallback con PIL
            try:
                with Image.open(image_path) as pil_img:
                    # Forza conversione a RGB (gestisce RGBA, P, L, etc.)
                    rgb_img = pil_img.convert('RGB')
                    img = np.array(rgb_img).astype(np.float32) / 255.0
                    print(f"📷 PIL: Converted {pil_img.mode} to RGB")
                return img
            except Exception as e2:
                raise Exception(f"Both OpenCV and PIL failed: {e}, {e2}")
    
    def process_with_tiles(self, image):
        """USA IL FORWARD_CHOP FIXATO DEL MODELLO invece del tiling manuale"""
        h, w, c = image.shape
        print(f"🔧 Using MODEL forward_chop (FIXED) for {h}x{w} image")
        
        # Setup progress indeterminata per il processing del modello
        self.progress.config(mode='indeterminate')
        self.tile_progress.config(mode='indeterminate')
        self.progress.start(10)
        self.tile_progress.start(15)
        
        self.status_label.config(text="Processing with fixed model inference...", foreground="orange")
        self.tile_status_label.config(text="Using forward_chop with bilinear blending")
        self.root.update()
        
        with torch.no_grad():
            # Converti a tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Normalizza correttamente
            image_tensor = (image_tensor - 0.5) / 0.5  # [-1, 1]
            
            print(f"📐 Input tensor: {image_tensor.shape}, range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            
            # Usa il forward_chop FIXATO del modello
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(image_tensor)
            else:
                output = self.model(image_tensor)
            
            print(f"📐 Output tensor: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Denormalizza correttamente
            output = (output * 0.5 + 0.5).clamp(0, 1)
            
            print(f"📐 Denormalized: range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Converti a numpy
            result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Stop progress bars
        self.progress.stop()
        self.tile_progress.stop()
        self.progress.config(mode='determinate', value=100)
        self.tile_progress.config(mode='determinate', value=1, maximum=1)
        
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