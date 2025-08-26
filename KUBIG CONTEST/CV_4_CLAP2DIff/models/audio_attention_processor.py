"""
Audio-Conditioned Attention Processor for Hierarchical UNet Injection
Implements level-aware audio token injection into specific UNet blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from diffusers.models.attention_processor import Attention, AttnProcessor


class AudioAttnProcessor(nn.Module):  # CRITICAL: Must inherit from nn.Module!
    """
    Custom attention processor that injects hierarchical audio tokens.
    Uses Add-FiLM by default for memory efficiency.
    """
    
    def __init__(
        self,
        level: str,  # 'early', 'mid', or 'late'
        audio_dim: int = 768,
        hidden_dim: int = 768,
        mode: str = "add",  # 'add' for FiLM, 'concat' for KV concatenation
        dropout: float = 0.1,
        bottleneck_dim: int = 64  # Reduce parameters with bottleneck
    ):
        super().__init__()
        self.level = level
        self.mode = mode
        
        # Bottleneck projection to reduce parameters (768*64*2 instead of 768*768*2)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, bottleneck_dim),  # 768 -> 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim)  # 64 -> 768
        )
        
        # Learnable gate for FiLM modulation
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Process attention with audio injection.
        
        Args:
            attn: The attention module
            hidden_states: Input hidden states [B, N, C]
            encoder_hidden_states: Text embeddings [B, T, C]
            cross_attention_kwargs: Should contain 'audio' dict with level keys
        """
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        # Get audio tokens for this level from cross_attention_kwargs
        audio = cross_attention_kwargs.get('audio', None)
        audio_tokens = None
        
        if audio is not None and self.level in audio:
            audio_tokens = audio[self.level]  # [B, K, D]
            # Ensure proper dtype alignment (device already handled by module placement)
            audio_tokens = audio_tokens.to(dtype=hidden_states.dtype)
        
        # Process encoder hidden states with audio injection
        if encoder_hidden_states is not None and audio_tokens is not None:
            # Project audio tokens
            audio_projected = self.audio_proj(audio_tokens)  # [B, K, hidden_dim]
            # Ensure dtype alignment
            audio_projected = audio_projected.to(dtype=encoder_hidden_states.dtype)
            
            if self.mode == "add":
                # Add-FiLM: Pool audio tokens and add to text embeddings
                audio_pooled = audio_projected.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
                # Broadcast and add with learned gate
                gate = torch.sigmoid(self.alpha).to(dtype=encoder_hidden_states.dtype)
                encoder_hidden_states = encoder_hidden_states + gate * audio_pooled
                
            elif self.mode == "concat":
                # KV-concat: Concatenate audio tokens to text sequence
                # Limit audio tokens to reduce memory
                max_audio_tokens = 4
                if audio_projected.shape[1] > max_audio_tokens:
                    # Use adaptive pooling to reduce tokens
                    audio_projected = F.adaptive_avg_pool1d(
                        audio_projected.transpose(1, 2), 
                        max_audio_tokens
                    ).transpose(1, 2)
                encoder_hidden_states = torch.cat([encoder_hidden_states, audio_projected], dim=1)
        
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        # Standard attention computation
        hidden_states = attn.to_q(hidden_states) * scale
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for attention
        query = attn.head_to_batch_dim(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Compute attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class AudioProcessorManager:
    """
    Manages the mapping of audio processors to UNet blocks.
    """
    
    def __init__(self, unet):
        self.unet = unet
        self.processors = {}
        self.level_mapping = self._create_level_mapping()
        
    def _create_level_mapping(self) -> Dict[str, list]:
        """
        Create mapping from levels to UNet block names.
        
        Returns:
            Dictionary mapping 'early', 'mid', 'late' to processor names
        """
        mapping = {"early": [], "mid": [], "late": []}
        
        # Get all attention processor names
        all_names = list(self.unet.attn_processors.keys())
        
        for name in all_names:
            # Skip if not cross-attention
            if "attn1" in name:  # self-attention
                continue
                
            # Map based on block position (corrected logic)
            # Early: shallow encoder blocks (high-level features)
            # Mid: middle blocks (mid-level features)
            # Late: deep encoder + shallow decoder (low-level features)
            if "mid_block" in name:
                mapping["mid"].append(name)
            elif "down_blocks.0" in name or "down_blocks.1" in name:
                mapping["early"].append(name)  # Shallow encoder
            elif "down_blocks.2" in name or "down_blocks.3" in name:
                mapping["late"].append(name)   # Deep encoder
            elif "up_blocks.0" in name or "up_blocks.1" in name:
                mapping["late"].append(name)   # Shallow decoder (still low-level)
            elif "up_blocks.2" in name or "up_blocks.3" in name:
                mapping["mid"].append(name)    # Deep decoder (mid-level reconstruction)
            else:
                # Default to mid for unmapped blocks
                mapping["mid"].append(name)
        
        return mapping
    
    def setup_processors(
        self,
        audio_dim: int = 768,
        hidden_dim: int = None,
        mode: str = "add",
        dropout: float = 0.1
    ):
        """
        Setup audio attention processors for each level.
        
        Args:
            audio_dim: Dimension of audio tokens
            hidden_dim: Hidden dimension of UNet (auto-detected if None)
            mode: 'add' or 'concat'
            dropout: Dropout rate
        """
        # Auto-detect hidden dimension from first cross-attention
        if hidden_dim is None:
            for name, proc in self.unet.attn_processors.items():
                if "attn2" in name:  # cross-attention
                    # Try to get dimension from the attention module
                    parts = name.split(".")
                    module = self.unet
                    for part in parts[:-1]:  # Navigate to parent module
                        if hasattr(module, part):
                            module = getattr(module, part)
                        elif part.isdigit():
                            module = module[int(part)]
                    if hasattr(module, 'to_k'):
                        hidden_dim = module.to_k.in_features
                        break
            
            if hidden_dim is None:
                hidden_dim = 768  # Default for SD v1.5
        
        # Start with existing processors to preserve optimizations
        new_processors = dict(self.unet.attn_processors)
        
        # Only replace cross-attention processors with our custom ones
        for level, names in self.level_mapping.items():
            # Create one processor per level (shared across blocks)
            processor = AudioAttnProcessor(
                level=level,
                audio_dim=audio_dim,
                hidden_dim=hidden_dim,
                mode=mode,
                dropout=dropout
            )
            
            # Assign to all blocks in this level (only cross-attention)
            for name in names:
                new_processors[name] = processor
        
        # Set the processors
        self.unet.set_attn_processor(new_processors)
        self.processors = new_processors
        
        print(f"Setup audio processors:")
        print(f"  Early blocks: {len(self.level_mapping['early'])}")
        print(f"  Mid blocks: {len(self.level_mapping['mid'])}")
        print(f"  Late blocks: {len(self.level_mapping['late'])}")
        
    def get_audio_kwargs(self, routed_tokens: Dict[str, torch.Tensor]) -> Dict:
        """
        Prepare cross_attention_kwargs with audio tokens.
        
        Args:
            routed_tokens: Dictionary with 'early', 'mid', 'late' tokens
            
        Returns:
            Dictionary to pass as cross_attention_kwargs
        """
        return {"audio": routed_tokens}