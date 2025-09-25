import os
import logging
import torch
import numpy as np
from datetime import datetime
import json
import yaml
from typing import Dict, Any
import cv2
from PIL import Image


class AverageMeter:
    """Keeps track of average values"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(log_dir: str, log_level=logging.INFO):
    """Setup logging configuration"""
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('AstroTrainer')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(state: Dict[str, Any], filename: str, is_best: bool = False):
    """Save training checkpoint"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save checkpoint
        torch.save(state, filename)
        print(f"✅ Checkpoint saved: {filename}")
        
        if is_best:
            best_filename = filename.replace('.pth', '_best.pth')
            torch.save(state, best_filename)
            print(f"🎯 Best model saved: {best_filename}")
            
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        raise


def load_checkpoint(filename: str, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint.get('epoch', 0)})")
        return checkpoint
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return None


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """Calculate SSIM between two images"""
    from pytorch_msssim import ssim
    return ssim(img1, img2, data_range=max_val, size_average=True)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    if tensor.dim() == 3:
        # CHW -> HWC
        tensor = tensor.permute(1, 2, 0)
    
    numpy_img = tensor.detach().cpu().numpy()
    
    # Clip to valid range
    numpy_img = np.clip(numpy_img, 0, 1)
    
    return numpy_img


def save_image_comparison(input_img, pred_img, target_img, save_path):
    """Save comparison of input, prediction, and target"""
    
    # Convert tensors to numpy
    input_np = tensor_to_numpy(input_img)
    pred_np = tensor_to_numpy(pred_img)
    target_np = tensor_to_numpy(target_img)
    
    # Create comparison
    h, w = input_np.shape[:2]
    comparison = np.zeros((h, w * 3, 3))
    
    comparison[:, :w] = input_np
    comparison[:, w:2*w] = pred_np
    comparison[:, 2*w:3*w] = target_np
    
    # Convert to PIL and save
    comparison = (comparison * 255).astype(np.uint8)
    Image.fromarray(comparison).save(save_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")


def get_device_info():
    """Get information about available compute devices"""
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_cached': torch.cuda.memory_cached(i),
                'memory_allocated': torch.cuda.memory_allocated(i),
            }
            info[f'cuda_device_{i}'] = device_info
    
    return info


def print_model_summary(model, input_size=(3, 512, 512)):
    """Print model summary with parameter count and FLOPs estimation"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimation
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Model size: {param_size:.2f} MB")
    
    # Input size info
    input_mb = np.prod(input_size) * 4 / (1024 ** 2)
    print(f"Input size: {input_size}")
    print(f"Input memory: {input_mb:.2f} MB")
    
    print("=" * 60)


def create_lr_scheduler(optimizer, scheduler_type, **kwargs):
    """Create learning rate scheduler"""
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            verbose=kwargs.get('verbose', True)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def validate_dataset_structure(data_root):
    """Validate that dataset has correct structure"""
    
    required_dirs = [
        'train_tiles/input',
        'train_tiles/target',
        'val_tiles/input', 
        'val_tiles/target'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_root, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        raise FileNotFoundError(f"Missing required directories: {missing_dirs}")
    
    # Check for images in each directory
    for dir_path in required_dirs:
        full_path = os.path.join(data_root, dir_path)
        files = [f for f in os.listdir(full_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
        
        if not files:
            print(f"Warning: No images found in {dir_path}")
        else:
            print(f"Found {len(files)} images in {dir_path}")


def check_image_pair_consistency(data_root):
    """Check that input and target directories have matching files"""
    
    pairs = [
        ('train_tiles/input', 'train_tiles/target'),
        ('val_tiles/input', 'val_tiles/target')
    ]
    
    for input_dir, target_dir in pairs:
        input_path = os.path.join(data_root, input_dir)
        target_path = os.path.join(data_root, target_dir)
        
        input_files = set(os.listdir(input_path))
        target_files = set(os.listdir(target_path))
        
        # Filter only image files
        input_images = {f for f in input_files 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))}
        target_images = {f for f in target_files 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))}
        
        common_files = input_images.intersection(target_images)
        missing_input = target_images - input_images
        missing_target = input_images - target_images
        
        print(f"\n{input_dir} vs {target_dir}:")
        print(f"  Common files: {len(common_files)}")
        
        if missing_input:
            print(f"  Missing in input: {len(missing_input)} files")
            if len(missing_input) <= 5:
                print(f"    {list(missing_input)}")
        
        if missing_target:
            print(f"  Missing in target: {len(missing_target)} files")
            if len(missing_target) <= 5:
                print(f"    {list(missing_target)}")


def cleanup_old_checkpoints(checkpoint_dir, keep_last=5, keep_best=True):
    """Clean up old checkpoint files, keeping only the most recent ones"""
    
    import glob
    
    # Get all checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    checkpoints.sort(key=lambda x: os.path.getctime(x))
    
    # Keep the most recent checkpoints
    if len(checkpoints) > keep_last:
        to_remove = checkpoints[:-keep_last]
        
        for checkpoint in to_remove:
            # Don't remove best model
            if keep_best and 'best' in checkpoint:
                continue
            
            try:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {os.path.basename(checkpoint)}")
            except OSError:
                pass


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test device info
    print("\nDevice Info:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test dataset validation
    print("\nDataset Structure Validation:")
    try:
        validate_dataset_structure(".")
        check_image_pair_consistency(".")
    except Exception as e:
        print(f"Dataset validation error: {e}")
    
    print("\nUtilities test completed.")