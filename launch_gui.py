#!/usr/bin/env python3
"""
LAUNCH SCRIPT - Inference GUI
"""
import os
import sys
import subprocess

def launch_inference_gui():
    """Launch inference GUI"""
    print("🌌 Launching Professional Star Removal GUI...")
    
    # Check if in correct directory
    if not os.path.exists('inference_gui.py'):
        print("❌ inference_gui.py not found. Run from StarLess directory.")
        return
    
    # Launch GUI
    try:
        subprocess.run([sys.executable, 'inference_gui.py'], check=True)
    except KeyboardInterrupt:
        print("\n✅ GUI closed by user")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

if __name__ == '__main__':
    launch_inference_gui()