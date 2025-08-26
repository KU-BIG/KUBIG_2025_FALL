"""
CLAP2Diffusion Gradio Application
Audio-to-Image Generation with Hierarchical Processing
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.hierarchical_audio_v4 import HierarchicalAudioV4
from models.audio_adapter_v4 import AudioAdapter
from models.audio_attention_processor import AudioAttnProcessor

class AudioToImageGenerator:
    """Main pipeline for audio-to-image generation"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load pretrained models"""
        # Initialize models
        self.audio_model = HierarchicalAudioV4().to(self.device)
        self.audio_adapter = AudioAdapter().to(self.device)
        
        # Load checkpoints if available
        audio_checkpoint = self.checkpoint_dir / "audio_projector_stage2.pth"
        adapter_checkpoint = self.checkpoint_dir / "unet_adapter_final.pth"
        
        if audio_checkpoint.exists():
            self.audio_model.load_state_dict(torch.load(audio_checkpoint, map_location=self.device, weights_only=True))
            print(f"Loaded audio model from {audio_checkpoint}")
        
        if adapter_checkpoint.exists():
            self.audio_adapter.load_state_dict(torch.load(adapter_checkpoint, map_location=self.device, weights_only=True))
            print(f"Loaded adapter model from {adapter_checkpoint}")
            
        self.audio_model.eval()
        self.audio_adapter.eval()
        
    def generate(
        self,
        audio_path,
        text_prompt,
        norm_value=60,
        num_steps=50,
        cfg_scale=7.5,
        seed=-1,
        model_type="Hierarchical"
    ):
        """Generate image from audio and text"""
        
        if seed == -1:
            seed = np.random.randint(0, 2**32-1)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Process audio (placeholder for actual implementation)
        # In real implementation, this would:
        # 1. Load audio with librosa
        # 2. Extract CLAP embeddings
        # 3. Apply hierarchical processing
        # 4. Generate image with Stable Diffusion
        
        # For demo, return a placeholder
        result_image = np.random.randn(512, 512, 3)
        result_image = ((result_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        info = f"""
Generation Complete!
Model: {model_type}
Audio: {Path(audio_path).name if audio_path else 'None'}
Text: {text_prompt}
Norm: {norm_value}
Steps: {num_steps}
CFG: {cfg_scale}
Seed: {seed}
"""
        
        return result_image, info

# Initialize generator
generator = AudioToImageGenerator()

# Create Gradio interface
with gr.Blocks(title="CLAP2Diffusion - Audio to Image Generation") as demo:
    gr.Markdown("""
    # CLAP2Diffusion: Audio-to-Image Generation
    ### Hierarchical Audio Processing with Norm Optimization
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Audio input
            audio_input = gr.Audio(
                label="Upload Audio",
                type="filepath",
                sources="upload"
            )
            
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=["Hierarchical", "SonicDiffusion", "Baseline"],
                value="Hierarchical",
                label="Model Type"
            )
            
            # Text prompt
            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter a description...",
                value="a beautiful landscape"
            )
            
            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False):
                norm_slider = gr.Slider(
                    minimum=10, maximum=200, value=60, step=5,
                    label="Audio Normalization (60 is optimal)"
                )
                
                steps_slider = gr.Slider(
                    minimum=20, maximum=100, value=50, step=5,
                    label="Inference Steps"
                )
                
                cfg_slider = gr.Slider(
                    minimum=1, maximum=20, value=7.5, step=0.5,
                    label="CFG Scale"
                )
                
                seed_input = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
            
            generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            output_info = gr.Textbox(label="Generation Info", lines=8)
    
    # Examples
    gr.Examples(
        examples=[
            ["../data/samples/thunder.wav", "Hierarchical", "stormy beach", 60, 50, 7.5, -1],
            ["../data/samples/birds.wav", "Hierarchical", "forest morning", 60, 50, 7.5, -1],
            ["../data/samples/ocean.wav", "SonicDiffusion", "sunset beach", 60, 50, 7.5, -1],
        ],
        inputs=[audio_input, model_dropdown, text_input, norm_slider, steps_slider, cfg_slider, seed_input],
        outputs=[output_image, output_info],
        fn=generator.generate
    )
    
    # Connect button
    generate_btn.click(
        fn=generator.generate,
        inputs=[audio_input, text_input, norm_slider, steps_slider, cfg_slider, seed_input, model_dropdown],
        outputs=[output_image, output_info]
    )

if __name__ == "__main__":
    import os
    # Get authentication from environment variables
    auth_user = os.getenv("GRADIO_USERNAME", "admin")
    auth_pass = os.getenv("GRADIO_PASSWORD", "clap2diffusion")
    
    # Launch with authentication and secure defaults
    demo.launch(
        share=False,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),  # Default to localhost
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        auth=(auth_user, auth_pass) if auth_pass else None,
        ssl_verify=False
    )