"""
AudioCaps Hierarchical Dataset for CLAP2Diffusion V4
Loads AudioCaps data with hierarchical caption parsing and composition classification
"""

import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from PIL import Image
import random
from pathlib import Path

# Import our caption parser
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.caption_parser import AudioCaptionParser


class AudioCapsHierarchicalDataset(Dataset):
    """
    AudioCaps dataset with hierarchical caption parsing
    Supports compositional learning with text-audio relationship classification
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        split: str = "train",
        sample_rate: int = 44100,
        audio_duration: float = 10.0,
        image_size: int = 512,
        use_augmentation: bool = True,
        composition_strategy: str = "balanced",  # balanced, creative, matching
        max_samples: Optional[int] = None,
        load_images: bool = True,  # Stage 1 can set this to False to save I/O
        parse_captions: bool = True  # Enable hierarchical caption parsing
    ):
        """
        Args:
            data_root: Root directory containing audio/, frames/ subdirectories
            metadata_path: Path to metadata JSON file
            split: Dataset split (train/val/test)
            sample_rate: Audio sample rate
            audio_duration: Fixed audio duration in seconds
            image_size: Image size for resizing
            use_augmentation: Whether to use data augmentation
            composition_strategy: Strategy for text-audio pairing
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_root = Path(data_root)
        self.audio_dir = self.data_root / "audio"
        self.frames_dir = self.data_root / "frames"
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.audio_length = int(sample_rate * audio_duration)
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.load_images = load_images
        self.composition_strategy = composition_strategy
        
        # Lazy initialization of caption parser (only when needed)
        self.caption_parser = None
        self._parse_captions = parse_captions
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get samples from metadata
        if isinstance(metadata, dict) and 'samples' in metadata:
            all_samples = metadata['samples']
        else:
            all_samples = metadata
        
        # Filter by split
        self.samples = [
            item for item in all_samples 
            if item.get('split', 'train') == split
        ]
        
        # Limit samples if specified
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        # Parse all captions once for efficiency (only if needed)
        self.parsed_captions = {}
        if self._parse_captions:
            # Initialize parser only when actually needed
            if self.caption_parser is None:
                try:
                    # Ensure NLTK data is downloaded
                    import nltk
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    try:
                        nltk.data.find('taggers/averaged_perceptron_tagger')
                    except LookupError:
                        nltk.download('averaged_perceptron_tagger', quiet=True)
                    
                    self.caption_parser = AudioCaptionParser()
                except Exception as e:
                    print(f"Warning: Caption parser initialization failed: {e}")
                    print("Continuing without caption parsing...")
                    self._parse_captions = False
            
            if self.caption_parser is not None:
                for sample in self.samples:
                    caption = sample.get('caption', '')
                    self.parsed_captions[sample['id']] = self.caption_parser.parse_caption(caption)
        else:
            # Use simple placeholders without parsing
            for sample in self.samples:
                self.parsed_captions[sample['id']] = {
                    'primary': sample.get('caption', ''),
                    'secondary': '',
                    'environment': ''
                }
        
        # Create composition pairs based on strategy
        self.composition_pairs = self._create_composition_pairs()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Composition strategy: {composition_strategy}")
        print(f"Created {len(self.composition_pairs)} composition pairs")
    
    def _create_composition_pairs(self) -> List[Dict]:
        """
        Create text-audio pairs based on composition strategy
        """
        pairs = []
        
        for i, sample in enumerate(self.samples):
            base_pair = {
                'audio_id': sample['id'],
                'image_id': sample['id'],
                'caption': sample['caption'],
                'parsed': self.parsed_captions[sample['id']],
                'composition_type': 'matching'  # Default
            }
            
            if self.composition_strategy == "balanced":
                # Create different types of pairs
                pairs.append(base_pair)  # Matching pair
                
                # Create complementary pair (different but related)
                if i + 1 < len(self.samples):
                    comp_pair = base_pair.copy()
                    comp_pair['image_id'] = self.samples[i + 1]['id']
                    comp_pair['composition_type'] = 'complementary'
                    pairs.append(comp_pair)
                
                # Create creative pair (random pairing)
                if len(self.samples) > 10:
                    creative_idx = random.randint(0, len(self.samples) - 1)
                    if creative_idx != i:
                        creative_pair = base_pair.copy()
                        creative_pair['image_id'] = self.samples[creative_idx]['id']
                        creative_pair['composition_type'] = 'creative'
                        pairs.append(creative_pair)
            
            elif self.composition_strategy == "creative":
                # Mostly creative pairings
                for _ in range(3):
                    idx = random.randint(0, len(self.samples) - 1)
                    pair = base_pair.copy()
                    pair['image_id'] = self.samples[idx]['id']
                    pair['composition_type'] = 'creative' if idx != i else 'matching'
                    pairs.append(pair)
            
            else:  # matching
                # Only matching pairs
                pairs.append(base_pair)
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.composition_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with hierarchical audio features
        
        Returns:
            Dictionary containing:
            - audio: Raw audio waveform [audio_length]
            - image: Image tensor [3, H, W]
            - caption: Original caption string
            - hierarchy: Dict with foreground, background, ambience labels
            - composition_type: Type of text-audio relationship
            - metadata: Additional metadata
        """
        pair = self.composition_pairs[idx]
        
        # Load audio
        audio_path = self.audio_dir / f"{pair['audio_id']}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary (use functional for better performance)
        if sr != self.sample_rate:
            import torchaudio.functional as F
            audio = F.resample(audio, sr, self.sample_rate)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Ensure audio is 1D for CLAP (remove channel dimension)
        audio = audio.squeeze(0)
        
        # Pad or trim to fixed length
        audio = self._process_audio_length(audio)
        
        # Audio augmentation
        if self.use_augmentation and self.composition_strategy != "matching":
            audio = self._augment_audio(audio)
        
        # Load image (skip if load_images is False for Stage 1)
        if self.load_images:
            image_path = self.frames_dir / f"{pair['image_id']}.jpg"
            if not image_path.exists():
                image_path = self.frames_dir / f"{pair['image_id']}.png"
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {pair['image_id']}.jpg or .png in {self.frames_dir}")
            
            image = Image.open(image_path).convert('RGB')
            image = self._process_image(image)
        else:
            # Return dummy tensor for compatibility (Stage 1 doesn't use it)
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Get hierarchical labels from parsed caption
        if self.caption_parser is not None:
            hierarchy_labels = self.caption_parser.get_hierarchy_labels(pair['parsed'])
        else:
            # Use simple fallback when parser is not initialized
            hierarchy_labels = {
                'foreground': pair['caption'],
                'background': '',
                'ambience': ''
            }
        
        # Prepare output
        output = {
            'audio': audio,  # Keep [1, audio_length] for consistency
            'image': image,
            'caption': pair['caption'],
            'hierarchy': hierarchy_labels,
            'composition_type': pair['composition_type'],
            'metadata': {
                'audio_id': pair['audio_id'],
                'image_id': pair['image_id'],
                'complexity': pair['parsed'].get('complexity', 'medium'),
                'categories': pair['parsed'].get('categories', []),
                'relationships': pair['parsed'].get('relationships', [])
            }
        }
        
        return output
    
    def _process_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Pad or trim audio to fixed length (now expects 1D audio)
        """
        current_length = audio.shape[0]  # Changed from shape[1] to shape[0] for 1D
        
        if current_length > self.audio_length:
            # Trim from random position
            if self.use_augmentation:
                start = random.randint(0, current_length - self.audio_length)
                audio = audio[start:start + self.audio_length]  # Removed [:, ...] indexing
            else:
                # Center crop
                start = (current_length - self.audio_length) // 2
                audio = audio[start:start + self.audio_length]  # Removed [:, ...] indexing
        elif current_length < self.audio_length:
            # Pad with zeros
            padding = self.audio_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio
    
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply audio augmentation
        """
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            audio = audio * gain
        
        # Random noise
        if random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.005
            audio = audio + noise
        
        # Clip to valid range
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process image to tensor
        """
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        
        # Channels first
        image = image.permute(2, 0, 1)
        
        # Image augmentation
        if self.use_augmentation:
            # Random horizontal flip
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
            
            # Random brightness
            if random.random() < 0.3:
                brightness = random.uniform(0.9, 1.1)
                image = image * brightness
                image = torch.clamp(image, -1.0, 1.0)
        
        return image
    
    def get_composition_statistics(self) -> Dict[str, int]:
        """
        Get statistics about composition types in the dataset
        """
        stats = {}
        for pair in self.composition_pairs:
            comp_type = pair['composition_type']
            stats[comp_type] = stats.get(comp_type, 0) + 1
        return stats


def seed_worker(worker_id):
    """
    Worker initialization function to ensure different random seeds per worker
    """
    import random
    import numpy as np
    
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class AudioCapsHierarchicalDataLoader:
    """
    DataLoader wrapper with additional functionality for hierarchical AudioCaps
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        load_images: bool = True,  # Stage 1 optimization
        **dataset_kwargs
    ):
        """
        Create data loaders for train, val, and test splits
        """
        self.data_root = data_root
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create datasets
        self.train_dataset = AudioCapsHierarchicalDataset(
            data_root=data_root,
            metadata_path=metadata_path,
            split="train",
            use_augmentation=True,
            load_images=load_images,  # Pass through load_images flag
            **dataset_kwargs
        )
        
        self.val_dataset = AudioCapsHierarchicalDataset(
            data_root=data_root,
            metadata_path=metadata_path,
            split="val",
            use_augmentation=False,
            load_images=load_images,  # Pass through load_images flag
            **dataset_kwargs
        )
        
        self.test_dataset = AudioCapsHierarchicalDataset(
            data_root=data_root,
            metadata_path=metadata_path,
            split="test",
            use_augmentation=False,
            load_images=load_images,  # Pass through load_images flag
            **dataset_kwargs
        )
        
        # Create data loaders
        # Fix: Use custom collate_fn and optimize DataLoader settings
        # Set up generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(42)  # Use a fixed seed for reproducibility
        
        # Build DataLoader kwargs conditionally to avoid prefetch_factor=None error
        train_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'collate_fn': collate_fn,  # Critical: Use custom collate function
            'pin_memory': True,
            'drop_last': True,
            'persistent_workers': (num_workers > 0),  # Optimize worker reuse
            'worker_init_fn': seed_worker,  # Ensure different seeds per worker
            'generator': generator  # Control randomness
        }
        if num_workers > 0:
            train_kwargs['prefetch_factor'] = 2  # Only set if workers > 0
        
        self.train_loader = DataLoader(self.train_dataset, **train_kwargs)
        
        val_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'collate_fn': collate_fn,  # Use custom collate
            'pin_memory': True,
            'persistent_workers': (num_workers > 0),
            'worker_init_fn': seed_worker
        }
        if num_workers > 0:
            val_kwargs['prefetch_factor'] = 2
        
        self.val_loader = DataLoader(self.val_dataset, **val_kwargs)
        
        test_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'collate_fn': collate_fn,  # Use custom collate
            'pin_memory': True,
            'persistent_workers': (num_workers > 0),
            'worker_init_fn': seed_worker
        }
        if num_workers > 0:
            test_kwargs['prefetch_factor'] = 2
        
        self.test_loader = DataLoader(self.test_dataset, **test_kwargs)
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all three data loaders"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("AudioCaps Hierarchical Dataset Statistics")
        print("="*50)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        
        print(f"\nBatch configuration:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Workers: {self.num_workers}")
        
        print(f"\nComposition statistics (Train):")
        stats = self.train_dataset.get_composition_statistics()
        for comp_type, count in stats.items():
            percentage = 100 * count / len(self.train_dataset)
            print(f"  {comp_type}: {count} ({percentage:.1f}%)")
        
        print("="*50)


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for batching
    """
    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    
    # Collect other data
    captions = [item['caption'] for item in batch]
    hierarchies = [item['hierarchy'] for item in batch]
    composition_types = [item['composition_type'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'audio': audio,
        'image': images,  # Fixed: singular for consistency
        'caption': captions,  # Fixed: singular for consistency  
        'hierarchy': hierarchies,  # Fixed: singular for consistency
        'composition_type': composition_types,  # Fixed: singular for consistency
        'metadata': metadata
    }


if __name__ == "__main__":
    # Test the dataset
    print("Testing AudioCaps Hierarchical Dataset...")
    
    # Use relative paths or environment variables
    data_root = os.environ.get("AUDIOCAPS_ROOT", "./data/audiocaps")
    metadata_path = os.environ.get("AUDIOCAPS_METADATA", "./data/audiocaps/metadata.json")
    
    # Create sample metadata if it doesn't exist
    if not os.path.exists(metadata_path):
        print("Creating sample metadata...")
        sample_metadata = [
            {
                "id": "sample_001",
                "caption": "A dog barks while birds chirp in the background",
                "split": "train"
            },
            {
                "id": "sample_002",
                "caption": "Cars passing by as rain falls on the street",
                "split": "train"
            },
            {
                "id": "sample_003",
                "caption": "A woman speaks in a quiet room",
                "split": "val"
            }
        ]
        
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        print(f"Created sample metadata at {metadata_path}")
    
    # Test dataset loading
    try:
        dataset = AudioCapsHierarchicalDataset(
            data_root=data_root,
            metadata_path=metadata_path,
            split="train",
            composition_strategy="balanced",
            max_samples=10  # Limit for testing
        )
        
        print(f"\nDataset created with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: tensor of shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value).__name__}")
            
            print(f"\nHierarchy labels:")
            for level, label in sample['hierarchy'].items():
                print(f"  {level}: {label}")
        
        # Test DataLoader
        loader = AudioCapsHierarchicalDataLoader(
            data_root=data_root,
            metadata_path=metadata_path,
            batch_size=2,
            num_workers=0,  # 0 for testing
            composition_strategy="balanced"
        )
        
        loader.print_statistics()
        
    except Exception as e:
        print(f"Note: Full testing requires actual AudioCaps data")
        print(f"Error: {e}")
    
    print("\nAudioCaps Hierarchical Dataset implementation complete!")