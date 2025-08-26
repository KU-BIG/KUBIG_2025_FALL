"""
Training optimization utilities
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict
import numpy as np


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_val.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GradientClipping:
    """Advanced gradient clipping strategies"""
    
    def __init__(self, max_norm=1.0, norm_type=2, adaptive=False):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.grad_history = []
        self.history_size = 100
    
    def clip(self, model):
        """Clip gradients with optional adaptive scaling"""
        # Get all gradients
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
        
        if not grads:
            return 0.0
        
        # Compute total norm
        total_norm = torch.norm(torch.cat(grads), self.norm_type).item()
        
        # Adaptive clipping based on gradient history
        if self.adaptive:
            self.grad_history.append(total_norm)
            if len(self.grad_history) > self.history_size:
                self.grad_history.pop(0)
            
            if len(self.grad_history) >= 10:
                # Adjust max_norm based on gradient statistics
                mean_norm = np.mean(self.grad_history)
                std_norm = np.std(self.grad_history)
                adaptive_max = mean_norm + 2 * std_norm
                clip_value = min(self.max_norm, adaptive_max)
            else:
                clip_value = self.max_norm
        else:
            clip_value = self.max_norm
        
        # Clip gradients
        clip_coef = clip_value / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm


class WarmupCosineScheduler:
    """Custom warmup + cosine annealing scheduler"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        max_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        if max_lr is None:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            self.base_lrs = [max_lr for _ in optimizer.param_groups]
        
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        # Update learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.min_lr + (base_lr - self.min_lr) * scale
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        
        # Update LR to match loaded state
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.min_lr + (base_lr - self.min_lr) * scale
            param_group['lr'] = lr


class LossBalancer:
    """Balance multiple loss components dynamically"""
    
    def __init__(self, loss_names, initial_weights=None, balance_period=100):
        self.loss_names = loss_names
        self.balance_period = balance_period
        self.step_count = 0
        
        if initial_weights is None:
            self.weights = {name: 1.0 for name in loss_names}
        else:
            self.weights = initial_weights.copy()
        
        self.loss_history = {name: [] for name in loss_names}
        self.history_size = balance_period
    
    def update(self, losses: Dict[str, float]):
        """Update loss history and rebalance weights"""
        # Update history
        for name, value in losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(value)
                if len(self.loss_history[name]) > self.history_size:
                    self.loss_history[name].pop(0)
        
        self.step_count += 1
        
        # Rebalance periodically
        if self.step_count % self.balance_period == 0:
            self._rebalance()
    
    def _rebalance(self):
        """Rebalance loss weights based on gradients"""
        # Compute average magnitudes
        avg_losses = {}
        for name in self.loss_names:
            if self.loss_history[name]:
                avg_losses[name] = np.mean(self.loss_history[name])
            else:
                avg_losses[name] = 1.0
        
        # Normalize weights to balance gradients
        if avg_losses:
            max_loss = max(avg_losses.values())
            for name in self.loss_names:
                if avg_losses[name] > 0:
                    self.weights[name] = max_loss / avg_losses[name]
    
    def get_weighted_loss(self, losses: Dict[str, torch.Tensor]):
        """Compute weighted total loss"""
        total_loss = 0
        for name, loss in losses.items():
            if name in self.weights:
                total_loss = total_loss + self.weights[name] * loss
            else:
                total_loss = total_loss + loss
        return total_loss


class MemoryEfficientCheckpointing:
    """Memory-efficient model checkpointing"""
    
    @staticmethod
    def save_checkpoint(model, optimizer, scheduler, epoch, step, path, **kwargs):
        """Save checkpoint with CPU offloading"""
        # Move to CPU before saving to reduce memory
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state_dict': {
                'state': {k: {kk: vv.cpu() if torch.is_tensor(vv) else vv 
                             for kk, vv in v.items()} 
                         for k, v in optimizer.state.items()},
                'param_groups': optimizer.param_groups
            }
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add any additional info
        checkpoint.update(kwargs)
        
        # Save with torch.save
        torch.save(checkpoint, path)
        
        print(f"Checkpoint saved to {path}")
    
    @staticmethod
    def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
        """Load checkpoint with proper device placement"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('step', 0)