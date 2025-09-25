#!/usr/bin/env python3
"""
TEST SCRIPT - Verifica fix quadrettini neri
Testa il nuovo forward_chop con blending Hann
"""
import torch
import numpy as np
from PIL import Image
import cv2
from model import create_astro_model

def test_tiling_fix():
    """Test del fix per i quadrettini neri"""
    print("🧪 Testing tiling fix for black squares...")
    
    # Crea modello
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = create_astro_model(
        size='small',
        img_channel=3,
        width=32,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 4, 6],
        dec_blk_nums=[2, 2, 2, 2]
    ).to(device)
    
    model.eval()
    
    # Crea immagine di test con stelle simulate
    print("🌟 Creating test image with simulated stars...")
    H, W = 1024, 1024
    test_img = np.zeros((H, W, 3), dtype=np.float32)
    
    # Aggiungi stelle simulate
    np.random.seed(42)
    for _ in range(50):
        x = np.random.randint(100, W-100)
        y = np.random.randint(100, H-100)
        intensity = np.random.uniform(0.7, 1.0)
        
        # Stella gaussiana
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= y+dy < H and 0 <= x+dx < W:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= 5:
                        val = intensity * np.exp(-dist*dist / 4.0)
                        test_img[y+dy, x+dx] = val
    
    # Aggiungi noise e sfondo
    test_img += np.random.normal(0, 0.05, test_img.shape)
    test_img = np.clip(test_img, 0, 1)
    
    print(f"📐 Test image: {H}x{W}")
    
    # Test con modello
    with torch.no_grad():
        # Converti a tensor
        input_tensor = torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor = (input_tensor - 0.5) / 0.5  # Normalize [-1, 1]
        
        print("🔄 Processing with new tiling method...")
        print(f"   - Using device: {device}")
        print(f"   - Tile size: 512")
        print(f"   - Overlap: 64 (with Hann blending)")
        
        # Forward pass (userà automaticamente forward_chop per immagini grandi)
        output = model(input_tensor)
        
        # Denormalize e converti
        output = (output * 0.5 + 0.5).clamp(0, 1)
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Salva risultati
    print("💾 Saving test results...")
    
    # Input
    input_img = (test_img * 255).astype(np.uint8)
    cv2.imwrite('test_input_stars.png', cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    
    # Output
    output_img = (result * 255).astype(np.uint8)
    cv2.imwrite('test_output_starless.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    
    # Difference map
    diff = np.abs(test_img - result)
    diff_img = (diff * 255 / diff.max()).astype(np.uint8)
    cv2.imwrite('test_difference_map.png', cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))
    
    print("✅ Test completed!")
    print("📁 Files saved:")
    print("   - test_input_stars.png (original with stars)")
    print("   - test_output_starless.png (processed)")
    print("   - test_difference_map.png (difference)")
    print("\n🔍 Check test_output_starless.png for:")
    print("   ❌ NO black squares where stars were")
    print("   ❌ NO grid artifacts")
    print("   ✅ Smooth texture replacement")
    print("   ✅ Natural looking background")

if __name__ == '__main__':
    test_tiling_fix()