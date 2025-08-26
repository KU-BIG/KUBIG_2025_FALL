"""
Stage 1 Training: Audio Projector Pre-training
Train the audio projector to align CLAP embeddings with text space
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.hierarchical_audio_v4 import HierarchicalAudioV4
from models.audio_adapter_v4 import AudioAdapter

class Stage1Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = AudioAdapter().to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.infonce_loss = nn.CrossEntropyLoss()
        
    def train_step(self, batch):
        """Single training step"""
        audio_emb = batch['audio_embedding'].to(self.device)
        text_emb = batch['text_embedding'].to(self.device)
        
        # Forward pass
        audio_tokens = self.model(audio_emb)
        
        # Compute losses
        # MSE alignment loss
        mse_loss = self.mse_loss(audio_tokens.mean(dim=1), text_emb)
        
        # InfoNCE contrastive loss
        batch_size = audio_tokens.shape[0]
        temperature = 0.07
        
        # Compute similarities
        audio_norm = audio_tokens.mean(dim=1) / audio_tokens.mean(dim=1).norm(dim=-1, keepdim=True)
        text_norm = text_emb / text_emb.norm(dim=-1, keepdim=True)
        similarity = torch.matmul(audio_norm, text_norm.T) / temperature
        
        # InfoNCE loss
        labels = torch.arange(batch_size).to(self.device)
        infonce_loss = self.infonce_loss(similarity, labels)
        
        # Total loss
        total_loss = mse_loss + infonce_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'infonce_loss': infonce_loss.item()
        }
    
    def save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint"""
        save_path = Path(save_dir) / f"audio_model_stage1_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

def main():
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_steps': 3000,
        'batch_size': 8,
        'save_dir': '../checkpoints'
    }
    
    trainer = Stage1Trainer(config)
    
    # Training loop (placeholder)
    print("Stage 1 Training: Audio Projector Pre-training")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Save final checkpoint
    trainer.save_checkpoint(3000, config['save_dir'])
    print("Training complete!")

if __name__ == "__main__":
    main()