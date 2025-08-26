#!/usr/bin/env python3
"""
AudioCaps dataset with precomputed latents for memory-efficient training
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, List, Optional, Tuple
import random

class AudioCapsLatentDataset(Dataset):
    """
    AudioCaps dataset that loads precomputed VAE latents instead of images
    This significantly reduces memory usage during training
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        audio_duration: float = 10.0,
        sample_rate: int = 48000,
        composition_strategy: str = 'matching',
        composition_shift: int = 0,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing audiocaps data
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            audio_duration: Duration of audio clips in seconds
            sample_rate: Audio sample rate
            composition_strategy: How to pair audio with images
            composition_shift: Offset for composition pairing
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.split = split
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.composition_strategy = composition_strategy
        self.composition_shift = composition_shift
        
        # Set paths
        self.audio_dir = self.data_root / 'audio'
        self.latents_dir = self.data_root / 'latents'
        self.metadata_path = self.data_root / 'metadata_unified.json'
        
        # Validate directories
        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {self.latents_dir}")
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get samples for this split
        all_samples = metadata.get('samples', [])
        
        # First try to use the 'split' field in each sample
        samples_with_split = [s for s in all_samples if s.get('split') == split]
        
        if samples_with_split:
            # Use the split field from metadata
            self.samples = samples_with_split
            print(f"Using metadata split field: {len(self.samples)} samples for {split}")
        else:
            # Fallback: create split based on indices
            print(f"Warning: No samples with split='{split}', using fallback split")
            np.random.seed(seed)
            indices = np.random.permutation(len(all_samples))
            
            train_size = int(0.8 * len(all_samples))
            val_size = int(0.1 * len(all_samples))
            
            if split == 'train':
                split_indices = indices[:train_size]
            elif split == 'val':
                split_indices = indices[train_size:train_size + val_size]
            else:  # test
                split_indices = indices[train_size + val_size:]
            
            self.samples = [all_samples[i] for i in split_indices]
        
        # Filter samples with existing latents
        valid_samples = []
        for sample in self.samples:
            latent_path = self.latents_dir / f"{sample['id']}.pt"
            audio_path = self.audio_dir / f"{sample['id']}.wav"
            
            if latent_path.exists() and audio_path.exists():
                valid_samples.append(sample)
        
        self.samples = valid_samples
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Create composition pairs
        self.pairs = self._create_composition_pairs()
        print(f"Composition strategy: {composition_strategy}")
        print(f"Created {len(self.pairs)} composition pairs")
        
        # Precompute audio length
        self.target_length = int(self.sample_rate * self.audio_duration)
    
    def _create_composition_pairs(self) -> List[Tuple[int, int]]:
        """Create audio-image pairs based on composition strategy"""
        n = len(self.samples)
        
        if self.composition_strategy == 'matching':
            # Each audio with its corresponding image
            return [(i, i) for i in range(n)]
        
        elif self.composition_strategy == 'shifted':
            # Shift image indices by composition_shift
            return [(i, (i + self.composition_shift) % n) for i in range(n)]
        
        elif self.composition_strategy == 'random':
            # Random pairing
            np.random.seed(42)
            image_indices = np.random.permutation(n)
            return [(i, image_indices[i]) for i in range(n)]
        
        else:
            raise ValueError(f"Unknown composition strategy: {self.composition_strategy}")
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Ensure mono (1D for CLAP)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=False)
            else:
                waveform = waveform.squeeze(0)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Pad or trim to target length
            waveform = self._process_audio_length(waveform)
            
            return waveform
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence as fallback
            return torch.zeros(self.target_length)
    
    def _process_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio to target length"""
        current_length = audio.shape[0]  # 1D audio
        
        if current_length > self.target_length:
            # Trim
            start = (current_length - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        elif current_length < self.target_length:
            # Pad
            pad_left = (self.target_length - current_length) // 2
            pad_right = self.target_length - current_length - pad_left
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right))
        
        return audio
    
    def _load_latent(self, latent_path: Path) -> torch.Tensor:
        """Load precomputed VAE latent"""
        try:
            latent = torch.load(str(latent_path), map_location='cpu', weights_only=True)
            # Validate shape
            if latent.shape != (4, 64, 64):
                print(f"Warning: Unexpected latent shape {latent.shape} for {latent_path}")
                return torch.zeros(4, 64, 64)
            return latent
        except Exception as e:
            print(f"Error loading latent {latent_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(4, 64, 64)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with precomputed latent"""
        audio_idx, image_idx = self.pairs[idx]
        
        audio_sample = self.samples[audio_idx]
        image_sample = self.samples[image_idx]
        
        # Load audio
        audio_path = self.audio_dir / f"{audio_sample['id']}.wav"
        audio = self._load_audio(audio_path)
        
        # Load precomputed latent instead of image
        latent_path = self.latents_dir / f"{image_sample['id']}.pt"
        latent = self._load_latent(latent_path)
        
        # Get caption
        caption = image_sample.get('caption', '')
        
        return {
            'audio': audio,
            'latent': latent,  # Return latent instead of image
            'caption': caption,
            'audio_id': audio_sample['id'],
            'image_id': image_sample['id']
        }