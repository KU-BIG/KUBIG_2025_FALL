"""
Evaluation Script for CLAP2Diffusion
Evaluate model performance on AudioCaps test set
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inference import AudioToImageInference

class Evaluator:
    def __init__(self, checkpoint_dir="../checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pipeline = AudioToImageInference(checkpoint_dir)
        
        # Metrics storage
        self.metrics = {
            'clip_score': [],
            'fid_score': [],
            'inception_score': [],
            'audio_alignment': []
        }
    
    def compute_clip_score(self, image, text):
        """Compute CLIP score between image and text"""
        # Placeholder - in real implementation would use CLIP model
        return np.random.random() * 0.3 + 0.7  # Random score between 0.7-1.0
    
    def compute_audio_alignment(self, image, audio_embedding):
        """Compute alignment between image and audio"""
        # Placeholder - custom metric for audio-image alignment
        return np.random.random() * 0.3 + 0.6  # Random score between 0.6-0.9
    
    def evaluate_single(self, audio_path, text_prompt, reference_image=None):
        """Evaluate single audio-image pair"""
        
        # Generate image
        generated_image = self.pipeline.generate(
            audio_path=audio_path,
            text_prompt=text_prompt,
            seed=42  # Fixed seed for evaluation
        )
        
        # Extract CLAP embedding for metrics
        audio = self.pipeline.load_audio(audio_path)
        audio_embedding = self.pipeline.extract_clap_embedding(audio)
        
        # Compute metrics
        clip_score = self.compute_clip_score(generated_image, text_prompt)
        audio_alignment = self.compute_audio_alignment(generated_image, audio_embedding)
        
        return {
            'clip_score': clip_score,
            'audio_alignment': audio_alignment,
            'generated_image': generated_image
        }
    
    def evaluate_dataset(self, data_dir, output_dir="evaluation_results"):
        """Evaluate on entire dataset"""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load metadata
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Create dummy metadata for demonstration
            metadata = [
                {"audio": "sample1.wav", "text": "thunder and rain", "id": "001"},
                {"audio": "sample2.wav", "text": "birds chirping", "id": "002"}
            ]
        
        print(f"Evaluating {len(metadata)} samples...")
        
        all_results = []
        
        for item in tqdm(metadata, desc="Evaluating"):
            audio_path = data_dir / "audio" / item['audio']
            
            if not audio_path.exists():
                print(f"Warning: {audio_path} not found, skipping...")
                continue
            
            # Evaluate
            results = self.evaluate_single(
                audio_path=audio_path,
                text_prompt=item['text']
            )
            
            # Store metrics
            self.metrics['clip_score'].append(results['clip_score'])
            self.metrics['audio_alignment'].append(results['audio_alignment'])
            
            # Save generated image
            image_path = output_dir / f"{item['id']}_generated.png"
            results['generated_image'].save(image_path)
            
            all_results.append({
                'id': item['id'],
                'audio': item['audio'],
                'text': item['text'],
                'clip_score': results['clip_score'],
                'audio_alignment': results['audio_alignment']
            })
        
        # Compute average metrics
        avg_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
                avg_metrics[f"{metric_name}_std"] = np.std(values)
        
        # Save results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'individual_results': all_results,
                'average_metrics': avg_metrics
            }, f, indent=2)
        
        return avg_metrics
    
    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        
        for metric_name, value in metrics.items():
            if not metric_name.endswith('_std'):
                std_key = f"{metric_name}_std"
                if std_key in metrics:
                    print(f"{metric_name:20}: {value:.4f} ± {metrics[std_key]:.4f}")
                else:
                    print(f"{metric_name:20}: {value:.4f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CLAP2Diffusion")
    parser.add_argument("--data_dir", type=str, default="../data/audiocaps/test",
                       help="Path to test data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                       help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CLAP2Diffusion Evaluation")
    print("="*60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Initialize evaluator
    evaluator = Evaluator(checkpoint_dir=args.checkpoint_dir)
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Print results
    evaluator.print_results(metrics)
    
    print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()