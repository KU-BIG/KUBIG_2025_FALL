"""
Audio Adapter V4 - Based on SonicDiffusion architecture
Transforms CLAP embeddings into audio tokens for cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Dict, Tuple


class AudioTokenGenerator(nn.Module):
    """
    Efficient audio token generator that transforms CLAP embeddings
    into a sequence of audio tokens for cross-attention.
    
    Uses learned queries instead of massive linear projection.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,  # CLAP embedding dimension
        hidden_dim: int = 768,  # Token dimension (same as text)
        num_tokens: int = 16,   # Number of audio tokens (16-32 recommended)
        num_layers: int = 4,     # Number of self-attention layers
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Learned query tokens for audio
        self.audio_queries = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        
        # Positional embeddings for tokens
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        
        # Project CLAP embedding to multiple keys and values (one per token)
        # Using low-rank factorization to control parameters
        self.audio_to_kv = nn.Sequential(
            nn.Linear(audio_dim, 256),  # Low-rank bottleneck
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim * 2 * num_tokens)  # Generate all K,V
        )
        
        # Self-attention layers for refining audio tokens
        self.self_attn_layers = nn.ModuleList([
            AudioSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        nn.init.xavier_uniform_(self.audio_queries)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, audio_embedding: torch.Tensor) -> torch.Tensor:
        """
        Transform CLAP embedding into audio tokens
        
        Args:
            audio_embedding: [batch_size, audio_dim] CLAP embeddings
            
        Returns:
            audio_tokens: [batch_size, num_tokens, hidden_dim]
        """
        batch_size = audio_embedding.shape[0]
        
        # Expand queries for batch and add positional embeddings
        queries = repeat(self.audio_queries, 'n d -> b n d', b=batch_size)
        pos_embed = repeat(self.pos_embed, 'n d -> b n d', b=batch_size)
        queries = queries + pos_embed
        
        # Generate keys and values from audio embedding
        kv = self.audio_to_kv(audio_embedding)  # [b, hidden_dim * 2 * num_tokens]
        kv = kv.view(batch_size, self.num_tokens, 2, self.hidden_dim)
        k, v = kv.unbind(dim=2)  # Each [b, num_tokens, hidden_dim]
        
        # Cross-attention: queries attend to audio-derived keys/values
        scale = self.hidden_dim ** -0.5
        scores = torch.einsum('bnd,bmd->bnm', queries, k) * scale
        attn_weights = F.softmax(scores, dim=-1)  # [b, n, n]
        
        # Apply attention to values
        audio_tokens = torch.einsum('bnm,bmd->bnd', attn_weights, v) + queries
        
        # Refine through self-attention layers
        for i, (self_attn, layer_norm) in enumerate(zip(self.self_attn_layers, self.layer_norms)):
            residual = audio_tokens
            audio_tokens = layer_norm(audio_tokens)
            audio_tokens = self_attn(audio_tokens) + residual
        
        # Final projection
        audio_tokens = self.output_proj(audio_tokens)
        
        return audio_tokens


class AudioSelfAttention(nn.Module):
    """
    Self-attention layer for refining audio tokens
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to audio tokens
        
        Args:
            x: [batch_size, num_tokens, hidden_dim]
            
        Returns:
            out: [batch_size, num_tokens, hidden_dim]
        """
        b, n, d = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Compute attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class AudioCrossAttention(nn.Module):
    """
    Cross-attention module for audio conditioning in UNet blocks.
    This is added as a separate layer after text cross-attention.
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = 768,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        gate_init: float = -5.0  # Initialize gate to ~0 (sigmoid(-5) ≈ 0.007)
    ):
        super().__init__()
        
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        # Norm for input
        self.norm = nn.LayerNorm(query_dim)
        
        # Q projection from hidden states
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # K, V projections from audio context
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable gate for controlling audio influence
        self.gate = nn.Parameter(torch.tensor(gate_init))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply audio cross-attention
        
        Args:
            hidden_states: [batch_size, seq_len, query_dim]
            audio_context: [batch_size, num_tokens, context_dim]
            attention_mask: Optional attention mask
            
        Returns:
            out: [batch_size, seq_len, query_dim]
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Generate Q from hidden states
        q = self.to_q(hidden_states)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        
        # Generate K, V from audio context
        k = self.to_k(audio_context)
        v = self.to_v(audio_context)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Compute attention scores
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            dots = dots.masked_fill(~attention_mask, -torch.finfo(dots.dtype).max)
        
        # Softmax
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.to_out(out)
        
        # Apply gate and add residual
        gate_value = torch.sigmoid(self.gate)
        out = residual + gate_value * out
        
        return out


class AudioAdapter(nn.Module):
    """
    Complete audio adapter module combining token generation and cross-attention.
    This module can be integrated into existing UNet blocks.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        hidden_dim: int = 768,
        num_tokens: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Audio token generator
        self.token_generator = AudioTokenGenerator(
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, audio_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate audio tokens from CLAP embedding
        
        Args:
            audio_embedding: [batch_size, audio_dim]
            
        Returns:
            audio_tokens: [batch_size, num_tokens, hidden_dim]
        """
        return self.token_generator(audio_embedding)