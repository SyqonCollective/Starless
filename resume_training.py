#!/usr/bin/env python3
"""
RESUME TRAINING SCRIPT
Riprende il training dal checkpoint più recente
"""
import os
import sys
import subprocess
import glob

def find_latest_checkpoint():
    """Trova il checkpoint più recente"""
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Directory {checkpoints_dir} not found")
        return None
    
    # Cerca checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoints_dir}")
        return None
    
    # Trova il più recente
    latest = max(checkpoint_files, key=os.path.getctime)
    print(f"📂 Latest checkpoint found: {latest}")
    return latest

def resume_training():
    """Resume training from latest checkpoint"""
    print("🔄 Resuming Professional Star Removal Training...")
    
    # Trova checkpoint
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("❌ No checkpoint found. Starting fresh training instead...")
        # Fallback to fresh training
        try:
            subprocess.run([sys.executable, 'train_professional.py'], check=True)
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
        except Exception as e:
            print(f"❌ Error in training: {e}")
        return
    
    # Resume training
    try:
        env = os.environ.copy()
        env['RESUME_CHECKPOINT'] = checkpoint_path
        subprocess.run([sys.executable, 'train_professional.py'], env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")
    except Exception as e:
        print(f"❌ Error resuming training: {e}")

if __name__ == '__main__':
    resume_training()