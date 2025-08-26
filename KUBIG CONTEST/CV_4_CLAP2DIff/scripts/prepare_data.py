"""
Data Preparation Script for AudioCaps Dataset
Prepare audio files and metadata for training
"""

import os
import json
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

def process_audio_file(audio_path, output_path, target_sr=44100, duration=10):
    """Process single audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=duration)
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Save processed audio
        sf.write(output_path, audio, target_sr)
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

def create_metadata(audio_dir, output_path):
    """Create metadata JSON for audio files"""
    audio_dir = Path(audio_dir)
    metadata = []
    
    for audio_file in audio_dir.glob("*.wav"):
        metadata.append({
            "id": audio_file.stem,
            "audio": audio_file.name,
            "duration": 10,  # Default duration
            "sample_rate": 44100
        })
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def split_dataset(metadata, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test"""
    np.random.seed(42)
    np.random.shuffle(metadata)
    
    n = len(metadata)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = metadata[:train_size]
    val_data = metadata[train_size:train_size + val_size]
    test_data = metadata[train_size + val_size:]
    
    return train_data, val_data, test_data

def prepare_audiocaps(raw_dir, output_dir, num_samples=None):
    """Prepare AudioCaps dataset"""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'audio').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'frames').mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing AudioCaps dataset from {raw_dir}")
    
    # Load AudioCaps CSV
    csv_path = raw_dir / "audiocaps.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if num_samples:
            df = df.head(num_samples)
    else:
        # Create dummy data for demonstration
        df = pd.DataFrame({
            'youtube_id': [f'video_{i}' for i in range(100)],
            'caption': [f'Audio description {i}' for i in range(100)],
            'start_time': [0] * 100
        })
    
    print(f"Processing {len(df)} audio samples...")
    
    all_metadata = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_id = row['youtube_id']
        caption = row['caption']
        
        # Process audio (placeholder - would download/process real audio)
        audio_path = raw_dir / 'audio' / f"{audio_id}.wav"
        
        if audio_path.exists():
            # Determine split
            rand = np.random.random()
            if rand < 0.8:
                split = 'train'
            elif rand < 0.9:
                split = 'val'
            else:
                split = 'test'
            
            # Process and save audio
            output_audio = output_dir / split / 'audio' / f"{audio_id}.wav"
            if process_audio_file(audio_path, output_audio):
                all_metadata.append({
                    'id': audio_id,
                    'audio': f"{audio_id}.wav",
                    'caption': caption,
                    'split': split
                })
    
    # Save metadata for each split
    for split in ['train', 'val', 'test']:
        split_metadata = [m for m in all_metadata if m['split'] == split]
        metadata_path = output_dir / split / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        print(f"{split}: {len(split_metadata)} samples")
    
    return all_metadata

def create_sample_data(output_dir):
    """Create sample data for testing"""
    output_dir = Path(output_dir)
    
    print("Creating sample data for testing...")
    
    # Create sample audio files
    sample_captions = [
        "Thunder and heavy rain",
        "Birds chirping in the morning",
        "Ocean waves crashing on beach",
        "Cat meowing loudly",
        "Dog barking in distance",
        "Traffic noise in city",
        "Wind blowing through trees",
        "Children playing in playground"
    ]
    
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        audio_dir = split_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of samples per split
        if split == 'train':
            n_samples = 5
        elif split == 'val':
            n_samples = 2
        else:
            n_samples = 1
        
        metadata = []
        for i in range(n_samples):
            # Create dummy audio file (silent)
            audio = np.zeros(44100 * 10)  # 10 seconds of silence
            audio_path = audio_dir / f"sample_{split}_{i:03d}.wav"
            sf.write(audio_path, audio, 44100)
            
            metadata.append({
                'id': f"sample_{split}_{i:03d}",
                'audio': audio_path.name,
                'caption': sample_captions[i % len(sample_captions)],
                'split': split
            })
        
        # Save metadata
        metadata_path = split_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print("Sample data created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Prepare AudioCaps dataset")
    parser.add_argument("--raw_dir", type=str, help="Raw AudioCaps directory")
    parser.add_argument("--output_dir", type=str, default="../data/audiocaps",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to process (for testing)")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample data for testing")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AudioCaps Data Preparation")
    print("="*60)
    
    if args.create_sample:
        create_sample_data(args.output_dir)
    elif args.raw_dir:
        metadata = prepare_audiocaps(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
        print(f"\n✓ Processed {len(metadata)} audio samples")
    else:
        print("Please provide --raw_dir or use --create_sample")

if __name__ == "__main__":
    main()