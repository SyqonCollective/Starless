#!/usr/bin/env python3
"""
DIAGNOSTIC SUITE PROFESSIONALE
Analisi completa e sistematica per identificare la root cause dei NaN gradients
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import cv2
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from dataset import create_dataloaders
from model import create_astro_model


class ComprehensiveDiagnostics:
    """Suite diagnostica completa per identificare problemi sistemici"""
    
    def __init__(self, dataset_root='/workspace', output_dir='./diagnostics'):
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.report = defaultdict(dict)
        
        print("🔬 COMPREHENSIVE DIAGNOSTICS SUITE")
        print("=" * 60)
        
    def run_full_diagnostics(self):
        """Esegue diagnostica completa"""
        print("📊 Starting comprehensive analysis...")
        
        # 1. Dataset Analysis
        self.analyze_dataset()
        
        # 2. Model Architecture Analysis
        self.analyze_model_architecture()
        
        # 3. Forward Pass Analysis
        self.analyze_forward_pass()
        
        # 4. Loss Function Analysis
        self.analyze_loss_functions()
        
        # 5. Training Setup Analysis
        self.analyze_training_setup()
        
        # 6. Generate Report
        self.generate_report()
        
        print("✅ Comprehensive diagnostics completed!")
        print(f"📄 Report saved to: {self.output_dir}/diagnostic_report.json")
        
    def analyze_dataset(self):
        """Analisi completa del dataset"""
        print("\n🔍 DATASET ANALYSIS")
        print("-" * 40)
        
        # Load sample data
        train_loader, val_loader = create_dataloaders(
            self.dataset_root, 
            batch_size=16, 
            num_workers=2
        )
        
        # Analyze multiple batches
        input_stats = []
        target_stats = []
        
        print("📈 Analyzing data statistics...")
        for i, batch in enumerate(tqdm(train_loader, desc="Analyzing batches")):
            if i >= 10:  # Analyze first 10 batches
                break
                
            inputs = batch['input'].numpy()
            targets = batch['target'].numpy()
            
            # Check for NaN/Inf
            input_nan = np.isnan(inputs).sum()
            input_inf = np.isinf(inputs).sum()
            target_nan = np.isnan(targets).sum()
            target_inf = np.isinf(targets).sum()
            
            if input_nan > 0 or input_inf > 0:
                print(f"⚠️  FOUND NaN/Inf in inputs batch {i}: NaN={input_nan}, Inf={input_inf}")
            if target_nan > 0 or target_inf > 0:
                print(f"⚠️  FOUND NaN/Inf in targets batch {i}: NaN={target_nan}, Inf={target_inf}")
            
            # Statistics
            input_stats.extend([
                inputs.min(), inputs.max(), inputs.mean(), inputs.std()
            ])
            target_stats.extend([
                targets.min(), targets.max(), targets.mean(), targets.std()
            ])
        
        # Dataset statistics
        self.report['dataset'] = {
            'input_range': [np.min(input_stats[::4]), np.max(input_stats[1::4])],
            'input_mean': np.mean(input_stats[2::4]),
            'input_std': np.mean(input_stats[3::4]),
            'target_range': [np.min(target_stats[::4]), np.max(target_stats[1::4])],
            'target_mean': np.mean(target_stats[2::4]),
            'target_std': np.mean(target_stats[3::4]),
            'samples_analyzed': i + 1
        }
        
        print(f"📊 Input range: [{self.report['dataset']['input_range'][0]:.4f}, {self.report['dataset']['input_range'][1]:.4f}]")
        print(f"📊 Target range: [{self.report['dataset']['target_range'][0]:.4f}, {self.report['dataset']['target_range'][1]:.4f}]")
        print(f"📊 Input mean±std: {self.report['dataset']['input_mean']:.4f}±{self.report['dataset']['input_std']:.4f}")
        print(f"📊 Target mean±std: {self.report['dataset']['target_mean']:.4f}±{self.report['dataset']['target_std']:.4f}")
        
        # Sample images analysis
        sample_batch = next(iter(train_loader))
        self.analyze_sample_images(sample_batch)
        
    def analyze_sample_images(self, batch):
        """Analizza immagini campione"""
        inputs = batch['input'][:4]  # First 4 images
        targets = batch['target'][:4]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(4):
            # Input image
            inp_img = inputs[i].permute(1, 2, 0).numpy()
            inp_img = np.clip(inp_img, 0, 1)
            axes[0, i].imshow(inp_img)
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            # Target image
            tgt_img = targets[i].permute(1, 2, 0).numpy()
            tgt_img = np.clip(tgt_img, 0, 1)
            axes[1, i].imshow(tgt_img)
            axes[1, i].set_title(f'Target {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Sample images saved to: {self.output_dir}/sample_images.png")
        
    def analyze_model_architecture(self):
        """Analisi dell'architettura del modello"""
        print("\n🏗️  MODEL ARCHITECTURE ANALYSIS")
        print("-" * 40)
        
        # Test different model configurations
        configs = [
            {'size': 'small', 'width': 16, 'middle_blk_num': 1},
            {'size': 'small', 'width': 32, 'middle_blk_num': 2},
            {'size': 'base', 'width': 32, 'middle_blk_num': 2},
        ]
        
        for i, config in enumerate(configs):
            print(f"\n📋 Testing config {i+1}: {config}")
            
            try:
                model = create_astro_model(**config).to(self.device)
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Test forward pass
                x = torch.randn(1, 3, 512, 512, device=self.device)
                
                with torch.no_grad():
                    y = model(x)
                
                # Check weight initialization
                init_stats = self.analyze_weight_initialization(model)
                
                config_report = {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'output_shape': list(y.shape),
                    'forward_pass_success': True,
                    'weight_init_stats': init_stats
                }
                
                self.report['model_configs'][f'config_{i+1}'] = config_report
                
                print(f"✅ Config {i+1} successful: {total_params:,} params")
                
            except Exception as e:
                print(f"❌ Config {i+1} failed: {e}")
                self.report['model_configs'][f'config_{i+1}'] = {
                    'error': str(e),
                    'forward_pass_success': False
                }
        
    def analyze_weight_initialization(self, model):
        """Analizza l'inizializzazione dei pesi"""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy()
                stats[name] = {
                    'mean': float(param_data.mean()),
                    'std': float(param_data.std()),
                    'min': float(param_data.min()),
                    'max': float(param_data.max()),
                    'zeros': int((param_data == 0).sum()),
                    'shape': list(param_data.shape)
                }
        
        return stats
        
    def analyze_forward_pass(self):
        """Analisi dettagliata del forward pass"""
        print("\n🔄 FORWARD PASS ANALYSIS")
        print("-" * 40)
        
        model = create_astro_model(size='small', width=32, middle_blk_num=2).to(self.device)
        
        # Hook for intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'nan_count': torch.isnan(output).sum().item(),
                        'inf_count': torch.isinf(output).sum().item()
                    }
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                module.register_forward_hook(hook_fn(name))
        
        # Test forward pass
        train_loader, _ = create_dataloaders(self.dataset_root, batch_size=4, num_workers=1)
        sample_batch = next(iter(train_loader))
        
        inputs = sample_batch['input'].to(self.device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        # Check for numerical issues in activations
        problematic_layers = []
        for layer_name, stats in activations.items():
            if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                problematic_layers.append(layer_name)
                print(f"⚠️  Issues in {layer_name}: NaN={stats['nan_count']}, Inf={stats['inf_count']}")
        
        self.report['forward_pass'] = {
            'activations': activations,
            'problematic_layers': problematic_layers,
            'output_stats': {
                'mean': outputs.mean().item(),
                'std': outputs.std().item(),
                'min': outputs.min().item(),
                'max': outputs.max().item(),
                'nan_count': torch.isnan(outputs).sum().item(),
                'inf_count': torch.isinf(outputs).sum().item()
            }
        }
        
        if len(problematic_layers) == 0:
            print("✅ No numerical issues detected in forward pass")
        else:
            print(f"❌ Found issues in {len(problematic_layers)} layers")
            
    def analyze_loss_functions(self):
        """Analisi delle funzioni di loss"""
        print("\n📉 LOSS FUNCTION ANALYSIS")
        print("-" * 40)
        
        # Test different loss functions
        loss_functions = {
            'L1': nn.L1Loss(),
            'L2': nn.MSELoss(),
            'Smooth_L1': nn.SmoothL1Loss(),
            'Huber': nn.HuberLoss()
        }
        
        # Generate test data
        pred = torch.randn(4, 3, 512, 512, device=self.device)
        target = torch.randn(4, 3, 512, 512, device=self.device)
        
        loss_results = {}
        
        for name, loss_fn in loss_functions.items():
            try:
                loss = loss_fn(pred, target)
                
                # Test gradient computation
                loss.backward(retain_graph=True)
                
                loss_results[name] = {
                    'value': loss.item(),
                    'gradient_success': True,
                    'finite': torch.isfinite(loss).item()
                }
                
                print(f"✅ {name} Loss: {loss.item():.6f}")
                
            except Exception as e:
                loss_results[name] = {
                    'error': str(e),
                    'gradient_success': False
                }
                print(f"❌ {name} Loss failed: {e}")
        
        self.report['loss_functions'] = loss_results
        
    def analyze_training_setup(self):
        """Analisi del setup di training"""
        print("\n⚙️  TRAINING SETUP ANALYSIS")
        print("-" * 40)
        
        model = create_astro_model(size='small', width=32, middle_blk_num=2).to(self.device)
        
        # Test different optimizers
        optimizers = {
            'Adam': torch.optim.Adam(model.parameters(), lr=1e-4),
            'AdamW': torch.optim.AdamW(model.parameters(), lr=1e-4),
            'SGD': torch.optim.SGD(model.parameters(), lr=1e-4),
        }
        
        optimizer_results = {}
        
        for name, optimizer in optimizers.items():
            try:
                # Simple training step
                inputs = torch.randn(2, 3, 512, 512, device=self.device)
                targets = torch.randn(2, 3, 512, 512, device=self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.L1Loss()(outputs, targets)
                loss.backward()
                
                # Check gradients
                grad_norm = 0
                nan_grads = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        nan_grads += torch.isnan(param.grad).sum().item()
                
                grad_norm = grad_norm ** 0.5
                
                optimizer.step()
                
                optimizer_results[name] = {
                    'gradient_norm': grad_norm,
                    'nan_gradients': nan_grads,
                    'step_success': True
                }
                
                print(f"✅ {name}: grad_norm={grad_norm:.4f}, nan_grads={nan_grads}")
                
            except Exception as e:
                optimizer_results[name] = {
                    'error': str(e),
                    'step_success': False
                }
                print(f"❌ {name} failed: {e}")
        
        self.report['training_setup'] = optimizer_results
        
    def generate_report(self):
        """Genera report completo"""
        self.report['timestamp'] = datetime.now().isoformat()
        self.report['device'] = str(self.device)
        
        # Save detailed report
        with open(self.output_dir / 'diagnostic_report.json', 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Generate summary
        summary = self.generate_summary()
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write(summary)
        
        print("\n" + "=" * 60)
        print("📄 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(summary)
        
    def generate_summary(self):
        """Genera summary dei problemi trovati"""
        issues = []
        recommendations = []
        
        # Check dataset issues
        dataset = self.report.get('dataset', {})
        if dataset.get('input_range', [0, 1])[1] > 10:
            issues.append("Dataset values outside expected range [0,1]")
            recommendations.append("Add proper normalization to dataset")
        
        # Check model issues
        forward_pass = self.report.get('forward_pass', {})
        if forward_pass.get('problematic_layers'):
            issues.append(f"Numerical issues in layers: {forward_pass['problematic_layers']}")
            recommendations.append("Review weight initialization and activation functions")
        
        # Check training issues
        training = self.report.get('training_setup', {})
        for opt_name, opt_result in training.items():
            if opt_result.get('nan_gradients', 0) > 0:
                issues.append(f"NaN gradients with {opt_name} optimizer")
                recommendations.append("Use gradient clipping and lower learning rate")
        
        summary = "🔍 ISSUES FOUND:\n"
        if issues:
            for issue in issues:
                summary += f"  ❌ {issue}\n"
        else:
            summary += "  ✅ No major issues detected\n"
        
        summary += "\n💡 RECOMMENDATIONS:\n"
        if recommendations:
            for rec in recommendations:
                summary += f"  🔧 {rec}\n"
        else:
            summary += "  ✅ System appears stable\n"
        
        return summary


def main():
    """Run comprehensive diagnostics"""
    diagnostics = ComprehensiveDiagnostics()
    diagnostics.run_full_diagnostics()


if __name__ == '__main__':
    main()