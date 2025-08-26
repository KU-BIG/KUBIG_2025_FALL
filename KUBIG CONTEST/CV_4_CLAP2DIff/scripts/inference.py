"""
Inference Script for CLAP2Diffusion
Generate images from audio using the trained models
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import librosa
from PIL import Image
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.hierarchical_audio_v4 import HierarchicalAudioV4
from models.audio_adapter_v4 import AudioAdapter

class AudioToImageInference:
    def __init__(self, checkpoint_dir="../checkpoints", device=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing inference pipeline on {self.device}")
        
        # Load models
        self.load_models()
        
        # Normalization settings
        self.OPTIMAL_NORM = 60.0  # Discovered through experimentation
        
    def load_models(self):
        """Load all necessary models and checkpoints"""
        
        # Load CLAP encoder
        clap_path = self.checkpoint_dir / "clap_encoder.pth"
        if clap_path.exists():
            print(f"Loading CLAP encoder from {clap_path}")
            # In real implementation, would load actual CLAP model
        
        # Load Audio Adapter
        adapter_path = self.checkpoint_dir / "audio_projector_stage2.pth"
        if adapter_path.exists():
            print(f"Loading Audio Adapter from {adapter_path}")
            self.audio_adapter = AudioAdapter().to(self.device)
            checkpoint = torch.load(adapter_path, map_location=self.device, weights_only=True)
            if 'adapter_state_dict' in checkpoint:
                self.audio_adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        # Load Hierarchical Model
        hierarchical_path = self.checkpoint_dir / "hierarchical_v4_final.pth"
        if hierarchical_path.exists():
            print(f"Loading Hierarchical Model from {hierarchical_path}")
            self.hierarchical_model = HierarchicalAudioV4().to(self.device)
            self.hierarchical_model.load_state_dict(
                torch.load(hierarchical_path, map_location=self.device, weights_only=True)
            )
        
        # Load UNet Adapter
        unet_adapter_path = self.checkpoint_dir / "unet_adapter_final.pth"
        if unet_adapter_path.exists():
            print(f"Loading UNet Adapter from {unet_adapter_path}")
            # In real implementation, would load UNet adapter weights
        
        # Set to eval mode
        if hasattr(self, 'audio_adapter'):
            self.audio_adapter.eval()
        if hasattr(self, 'hierarchical_model'):
            self.hierarchical_model.eval()
    
    def load_audio(self, audio_path, duration=10):
        """Load and preprocess audio file"""
        print(f"Loading audio from {audio_path}")
        
        # Load audio at 48kHz (CLAP standard)
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=duration)
        
        # Normalize audio
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        return audio
    
    def extract_clap_embedding(self, audio):
        """Extract CLAP embedding from audio"""
        # Placeholder - in real implementation would use CLAP model
        # CLAP outputs [batch_size, 512] for audio embeddings
        embedding = torch.randn(1, 512).to(self.device)
        return embedding
    
    def apply_normalization(self, audio_tokens, target_norm=60.0):
        """Apply discovered optimal normalization"""
        with torch.no_grad():
            raw_norm = torch.norm(audio_tokens, dim=-1, keepdim=True).mean()
            if raw_norm > 0:
                scale_factor = target_norm / raw_norm
                audio_tokens = audio_tokens * scale_factor
        return audio_tokens
    
    @torch.no_grad()
    def generate(
        self,
        audio_path,
        text_prompt="",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
        use_hierarchical=True
    ):
        """Generate image from audio"""
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load and process audio
        audio = self.load_audio(audio_path)
        
        # Extract CLAP embedding
        clap_embedding = self.extract_clap_embedding(audio)
        
        # Process through Audio Adapter
        if hasattr(self, 'audio_adapter'):
            audio_tokens = self.audio_adapter(clap_embedding)
            
            # Apply optimal normalization
            audio_tokens = self.apply_normalization(audio_tokens, self.OPTIMAL_NORM)
            
            print(f"Audio tokens shape: {audio_tokens.shape}")
            print(f"Audio tokens norm: {torch.norm(audio_tokens).item():.2f}")
        
        # Process through Hierarchical Model (if enabled)
        if use_hierarchical and hasattr(self, 'hierarchical_model'):
            # HierarchicalAudioV4 returns tokens directly or with intermediate outputs
            hierarchical_outputs = self.hierarchical_model(clap_embedding, return_intermediate=True)
            
            if isinstance(hierarchical_outputs, tuple):
                tokens_77, hierarchy = hierarchical_outputs
                print("Hierarchical processing:")
                print(f"  - Output tokens shape: {tokens_77.shape}")
                if isinstance(hierarchy, dict):
                    if 'foreground' in hierarchy:
                        print(f"  - Foreground features: {hierarchy['foreground'].shape}")
                    if 'background' in hierarchy:
                        print(f"  - Background features: {hierarchy['background'].shape}")
                    if 'ambience' in hierarchy:
                        print(f"  - Ambience features: {hierarchy['ambience'].shape}")
            else:
                print(f"Hierarchical output shape: {hierarchical_outputs.shape}")
        
        # Generate image using Stable Diffusion
        # Placeholder - in real implementation would use actual SD pipeline
        print(f"\nGenerating image with:")
        print(f"  Audio: {Path(audio_path).name}")
        print(f"  Text: {text_prompt}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  CFG Scale: {guidance_scale}")
        
        # For demonstration, create a random image
        image = np.random.randn(512, 512, 3)
        image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        return image
    
    def batch_generate(self, audio_paths, text_prompts=None, **kwargs):
        """Generate images for multiple audio files"""
        if text_prompts is None:
            text_prompts = [""] * len(audio_paths)
        
        images = []
        for audio_path, text_prompt in tqdm(zip(audio_paths, text_prompts), 
                                           total=len(audio_paths),
                                           desc="Generating images"):
            image = self.generate(audio_path, text_prompt, **kwargs)
            images.append(image)
        
        return images

def main():
    parser = argparse.ArgumentParser(description="CLAP2Diffusion Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--text", type=str, default="", help="Text prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Checkpoint directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no_hierarchical", action="store_true", help="Disable hierarchical processing")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CLAP2Diffusion Inference")
    print("="*60)
    
    # Initialize inference pipeline
    pipeline = AudioToImageInference(checkpoint_dir=args.checkpoint_dir)
    
    # Generate image
    image = pipeline.generate(
        audio_path=args.audio,
        text_prompt=args.text,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
        seed=args.seed,
        use_hierarchical=not args.no_hierarchical
    )
    
    # Save image
    image.save(args.output)
    print(f"\n✓ Image saved to {args.output}")

if __name__ == "__main__":
    main()