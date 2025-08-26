"""
CLAP Audio Encoder Module for CLAP2Diffusion
Handles audio feature extraction using CLAP model
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional, Union, List
from transformers import ClapModel, ClapProcessor
import librosa


class CLAPAudioEncoder(nn.Module):
    """
    Audio encoder using CLAP (Contrastive Language-Audio Pretraining) model.
    Extracts audio features that can be used for conditioning diffusion models.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        sample_rate: int = 48000,
        target_length: float = 10.0,
        device: str = "cuda",
        freeze: bool = False
    ):
        """
        Initialize CLAP audio encoder.
        
        Args:
            model_name: Pretrained CLAP model name
            sample_rate: Sample rate for audio processing (CLAP uses 48kHz)
            target_length: Target audio length in seconds
            device: Device to run the model on
            freeze: Whether to freeze the encoder weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.device = device
        
        # Load CLAP model and processor
        self.clap_model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        
        # Get embedding dimension - CLAP uses 512-dim embeddings by default
        # According to the CLAP documentation, the audio_projection is a ClapProjectionLayer
        # We need to check the actual weight dimensions
        try:
            # For ClapAudioModelWithProjection, the projection layer maps to the embedding space
            if hasattr(self.clap_model, 'audio_model_output'):
                # Some CLAP models have different structures
                self.embedding_dim = self.clap_model.audio_model_output.audio_projection.weight.shape[0]
            elif hasattr(self.clap_model, 'audio_projection'):
                # Check the weight matrix dimensions
                if hasattr(self.clap_model.audio_projection, 'weight'):
                    # Output dimension is the first dimension of the weight matrix
                    self.embedding_dim = self.clap_model.audio_projection.weight.shape[0]
                else:
                    # Default CLAP embedding dimension
                    self.embedding_dim = 512
            else:
                # Default CLAP embedding dimension
                self.embedding_dim = 512
        except Exception as e:
            print(f"Warning: Could not determine CLAP embedding dimension: {e}")
            self.embedding_dim = 512
        
        # Freeze encoder if specified
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze all parameters in the CLAP encoder."""
        for param in self.clap_model.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze all parameters in the CLAP encoder."""
        for param in self.clap_model.parameters():
            param.requires_grad = True
    
    def preprocess_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int
    ) -> torch.Tensor:
        """
        Preprocess audio to match CLAP requirements.
        
        Args:
            audio: Raw audio waveform
            sample_rate: Original sample rate of the audio
            
        Returns:
            Preprocessed audio tensor
        """
        # Convert to numpy if tensor (handle BFloat16)
        if isinstance(audio, torch.Tensor):
            # Convert BFloat16 to Float32 for numpy compatibility
            if audio.dtype == torch.bfloat16:
                audio = audio.float()
            audio = audio.cpu().numpy()
        
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = audio.mean(axis=-1)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=self.sample_rate
            )
        
        # Pad or truncate to target length
        target_samples = int(self.sample_rate * self.target_length)
        if len(audio) < target_samples:
            # Pad with zeros
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Truncate
            audio = audio[:target_samples]
        
        return audio
    
    def encode_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor, List],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Encode audio to CLAP embeddings.
        
        Args:
            audio: Audio input (can be raw waveform or batch)
            sample_rate: Sample rate of input audio
            
        Returns:
            Audio embeddings [batch_size, embedding_dim]
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Handle batch processing
        if isinstance(audio, list):
            processed_audio = []
            for a in audio:
                processed = self.preprocess_audio(a, sample_rate)
                processed_audio.append(processed)
            audio = np.stack(processed_audio)
        else:
            audio = self.preprocess_audio(audio, sample_rate)
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]  # Add batch dimension
        
        # Process through CLAP
        inputs = self.processor(
            audios=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Get audio features
        audio_features = self.clap_model.get_audio_features(**inputs)
        
        # Normalize features
        audio_features = audio_features / audio_features.norm(p=2, dim=-1, keepdim=True)
        
        return audio_features
    
    def forward(
        self,
        audio: Union[np.ndarray, torch.Tensor, List],
        sample_rate: int = None
    ) -> torch.Tensor:
        """
        Forward pass through the audio encoder.
        
        Args:
            audio: Audio input
            sample_rate: Sample rate of input audio
            
        Returns:
            Audio embeddings [batch_size, embedding_dim]
        """
        return self.encode_audio(audio, sample_rate)
    
    def get_audio_embeds_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Load audio from file and encode to embeddings.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio embeddings
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Encode
        return self.encode_audio(audio, sr)


class CLAPTextEncoder(nn.Module):
    """
    Text encoder using CLAP model for audio-text alignment.
    Can be used for text-based audio generation guidance.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cuda",
        freeze: bool = True
    ):
        """
        Initialize CLAP text encoder.
        
        Args:
            model_name: Pretrained CLAP model name
            device: Device to run the model on
            freeze: Whether to freeze the encoder weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        
        # Load CLAP model and processor
        self.clap_model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        
        # Get embedding dimension
        self.embedding_dim = self.clap_model.text_projection.out_features
        
        # Usually freeze text encoder
        if freeze:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze all parameters in the text encoder."""
        for param in self.clap_model.text_model.parameters():
            param.requires_grad = False
        for param in self.clap_model.text_projection.parameters():
            param.requires_grad = False
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to CLAP embeddings.
        
        Args:
            text: Text input (single string or list)
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        # Process text
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        
        # Get text features
        text_features = self.clap_model.get_text_features(**inputs)
        
        # Normalize features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Forward pass through the text encoder."""
        return self.encode_text(text)


def compute_audio_text_similarity(
    audio_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute similarity between audio and text embeddings.
    
    Args:
        audio_embeds: Audio embeddings [batch_size, embedding_dim]
        text_embeds: Text embeddings [batch_size, embedding_dim]
        temperature: Temperature for scaling similarity
        
    Returns:
        Similarity matrix [batch_size, batch_size]
    """
    # Normalize embeddings
    audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # Compute similarity
    similarity = torch.matmul(audio_embeds, text_embeds.T) / temperature
    
    return similarity