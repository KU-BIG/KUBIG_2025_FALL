"""
Improved Hierarchical Audio V4 with Soft Decomposition and Adaptive Weights
Implements soft token assignment with temperature annealing, adaptive hierarchy weights,
and level-to-UNet routing for CLAP2Diffusion V4.

Key improvements over original V4:
- Soft assignment instead of rigid 5-3-2 split
- Adaptive hierarchy weights (2-5K params) instead of 3 global scalars
- Temperature annealing (2.0 → 0.5) for stable learning
- Proper losses: diffusion + orthogonality + entropy + prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List, Union


class TemperatureScheduler:
    """
    Temperature scheduler for soft hierarchical decomposition.
    Implements annealing from T_max to T_min over training steps.
    """
    
    def __init__(
        self,
        decomposer: nn.Module,
        T_max: float = 2.0,
        T_min: float = 0.5,
        total_steps: int = 5000,
        warmup_steps: int = 200,
        mode: str = 'cosine'
    ):
        """
        Args:
            decomposer: SoftHierarchicalDecomposition module with temperature buffer
            T_max: Maximum (initial) temperature
            T_min: Minimum (final) temperature
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps to stay at T_max
            mode: Annealing mode ('linear' or 'cosine')
        """
        self.decomposer = decomposer
        self.T_max = T_max
        self.T_min = T_min
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.mode = mode
        
        # Initialize at T_max
        self.decomposer.set_temperature(T_max)
    
    def step(self, current_step: int):
        """Update temperature based on current training step."""
        if current_step < self.warmup_steps:
            # Stay at T_max during warmup
            temperature = self.T_max
        elif current_step >= self.total_steps or self.total_steps <= self.warmup_steps:
            # Stay at T_min after training or if no annealing period
            temperature = self.T_min
        else:
            # Anneal from T_max to T_min
            # Safe division - we've already checked total_steps > warmup_steps
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.mode == 'cosine':
                # Cosine annealing
                temperature = self.T_min + (self.T_max - self.T_min) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.mode == 'linear':
                # Linear annealing
                temperature = self.T_max - (self.T_max - self.T_min) * progress
            else:
                raise ValueError(f"Unknown annealing mode: {self.mode}")
        
        self.decomposer.set_temperature(temperature)


class SoftHierarchicalDecomposition(nn.Module):
    """
    Soft hierarchical decomposition with temperature-annealed assignment.
    Replaces rigid 5-3-2 split with learnable soft assignments.
    
    Features:
    - Soft token assignment to semantic levels (foreground/background/ambience)
    - Temperature annealing for progressive sharpening of assignments
    - Level anchors as learnable prototypes
    - Entropy regularization to prevent collapse
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_tokens: int = 10,
        num_levels: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 2.0
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_tokens = num_tokens
        self.num_levels = num_levels
        
        # Token generator with factorized design for ~0.6M params
        # Reduced bottleneck: 512 -> 512 -> 768
        self.shared_mlp = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, text_dim)  # 512*512 + 512*768 = ~0.65M params
        )
        
        # Per-token learned offsets for diversity
        self.token_offsets = nn.Parameter(torch.randn(num_tokens, text_dim) * 0.02)
        
        # Level anchors (learnable prototypes for each semantic level)
        self.level_anchors = nn.Parameter(torch.randn(num_levels, text_dim) * 0.02)
        
        # Gating head for computing assignment logits (meet 8K param budget)
        self.gating_head = nn.Sequential(
            nn.Linear(text_dim, 10),  # 768*10 + 10 = 7,690 params
            nn.GELU(),
            nn.Linear(10, num_levels)  # 10*3 + 3 = 33 params
        )  # Total: ~7.7K params (within 8K soft assignment budget)
        
        # Temperature buffer for annealing
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        
        # Prior distribution for soft 5-3-2 encouragement
        prior = torch.tensor([5.0, 3.0, 2.0]) / 10.0  # Normalized to sum to 1
        self.register_buffer('level_prior', prior)
        
        # Cross-hierarchy attention with smaller bottleneck
        self.cross_hierarchy_attn = CrossHierarchyAttention(
            dim=text_dim,
            num_heads=4,
            dropout=dropout,
            bottleneck_dim=192,  # Further reduced from 256
            mlp_ratio=1.5  # Further reduced from 2.0
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(text_dim)
    
    @torch.no_grad()
    def set_temperature(self, temperature: float):
        """Update temperature for annealing"""
        self.temperature.fill_(max(temperature, 0.1))  # Minimum temperature to avoid numerical issues
    
    def compute_assignments(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments of tokens to hierarchy levels.
        
        Args:
            tokens: [B, K, D] token embeddings
            
        Returns:
            assignments: [B, K, L] soft assignment probabilities
        """
        B, K, D = tokens.shape
        
        # Normalize tokens and anchors for stable dot product
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        anchors_norm = F.normalize(self.level_anchors, p=2, dim=-1)
        
        # Compute similarity between tokens and level anchors
        similarity = torch.einsum('bkd,ld->bkl', tokens_norm, anchors_norm)
        # Scale similarity for better gradient flow (learnable temperature)
        similarity = similarity * 10.0  # Reasonable scale for cosine similarity
        
        # Add learned gating bias
        gate_logits = self.gating_head(tokens)  # [B, K, L]
        logits = similarity + gate_logits
        
        # Apply temperature-scaled softmax
        assignments = F.softmax(logits / self.temperature, dim=-1)
        
        return assignments
    
    def forward(
        self,
        audio_features: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with soft hierarchical decomposition.
        
        Args:
            audio_features: [B, 512] CLAP audio features
            return_stats: Whether to return assignment statistics
            
        Returns:
            tokens: [B, K, D] hierarchically decomposed tokens
            info: Dictionary with assignments, temperature, and stats
        """
        B = audio_features.shape[0]
        
        # Generate K tokens from audio features using factorized design
        shared_features = self.shared_mlp(audio_features)  # [B, D]
        shared_features = shared_features.unsqueeze(1)  # [B, 1, D]
        tokens = shared_features + self.token_offsets.unsqueeze(0)  # [B, K, D]
        
        # Compute soft assignments
        assignments = self.compute_assignments(tokens)  # [B, K, L]
        
        # Apply cross-hierarchy attention
        tokens_attended = self.cross_hierarchy_attn(tokens)
        tokens_out = self.norm(tokens_attended)
        
        # Prepare output info
        info = {
            'tokens': tokens_out,
            'assignments': assignments,
            'temperature': self.temperature.item(),
            'level_anchors': self.level_anchors
        }
        
        if return_stats:
            # Compute assignment statistics for monitoring
            with torch.no_grad():
                # Average assignment per level
                avg_assignment = assignments.mean(dim=[0, 1])  # [L]
                # Entropy of assignments (higher = more uniform)
                entropy = -(assignments * (assignments + 1e-8).log()).sum(dim=-1).mean()
                # Effective number of levels used
                effective_levels = torch.exp(entropy)
                
                info['stats'] = {
                    'avg_assignment': avg_assignment,
                    'entropy': entropy.item(),
                    'effective_levels': effective_levels.item()
                }
        
        return tokens_out, info


class AdaptiveHierarchyWeights(nn.Module):
    """
    Adaptive hierarchy weight network that generates sample-specific weights.
    Replaces the 3 global scalar weights with a small network (2-5K params).
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        hidden_dim: int = 6,  # Reduced to 6 to meet ~3K param target per PLAN
        num_levels: int = 3,
        use_audio_context: bool = True
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.use_audio_context = use_audio_context
        
        if use_audio_context:
            # Compact network ~3K params: 512*6 + 6*3 = 3072 + 18 ≈ 3.1K params
            self.weight_network = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),  # 512*6 = 3072
                nn.GELU(),
                nn.LayerNorm(hidden_dim),  # 6 params
                nn.Linear(hidden_dim, num_levels)  # 6*3 = 18
            )
        else:
            # Simple learnable weights (fallback)
            self.weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Generate hierarchy weights for the given audio.
        
        Args:
            audio_features: [B, 512] CLAP features
            
        Returns:
            weights: [B, L] normalized weights for each level
        """
        if self.use_audio_context:
            # Generate weights from audio
            logits = self.weight_network(audio_features)  # [B, L]
            weights = F.softmax(logits, dim=-1)
        else:
            # Use global weights
            weights = F.softmax(self.weights, dim=0)
            weights = weights.unsqueeze(0).expand(audio_features.shape[0], -1)
        
        return weights


class LevelToUNetRouter(nn.Module):
    """
    Routes hierarchy levels to appropriate UNet scales.
    Maps semantic levels to UNet blocks for targeted conditioning.
    """
    
    def __init__(
        self,
        num_levels: int = 3,
        text_dim: int = 768
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.text_dim = text_dim
        
        # Level-specific gates for UNet blocks
        self.level_gates = nn.ParameterDict({
            'early': nn.Parameter(torch.zeros(1)),  # Ambience → early blocks
            'mid': nn.Parameter(torch.zeros(1)),    # Background → mid blocks
            'late': nn.Parameter(torch.zeros(1))    # Foreground → late blocks
        })
        
        # Routing matrix (learned mapping from levels to UNet scales)
        # Initialize with bias towards expected routing
        routing_init = torch.tensor([
            [0.1, 0.3, 0.6],  # Level 0 (foreground) → mostly late
            [0.2, 0.6, 0.2],  # Level 1 (background) → mostly mid
            [0.6, 0.3, 0.1],  # Level 2 (ambience) → mostly early
        ])
        self.routing_matrix = nn.Parameter(routing_init)
    
    def forward(
        self,
        tokens: torch.Tensor,
        assignments: torch.Tensor,
        hierarchy_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Route tokens to UNet blocks based on level assignments.
        
        Args:
            tokens: [B, K, D] token embeddings
            assignments: [B, K, L] soft assignments
            hierarchy_weights: [B, L] optional adaptive weights for levels
            
        Returns:
            routed: Dictionary with 'early', 'mid', 'late' tokens
        """
        B, K, D = tokens.shape
        
        # Apply hierarchy weights if provided
        if hierarchy_weights is not None:
            # Modulate assignments with adaptive weights
            assignments = assignments * hierarchy_weights.unsqueeze(1)  # [B, K, L]
            assignments = assignments / (assignments.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Compute routing weights: [B, K, L] @ [L, 3] -> [B, K, 3]
        routing = torch.matmul(assignments, F.softmax(self.routing_matrix, dim=1))
        
        # Split routing for each UNet scale
        routing_early = routing[:, :, 0:1]   # [B, K, 1]
        routing_mid = routing[:, :, 1:2]     # [B, K, 1]
        routing_late = routing[:, :, 2:3]    # [B, K, 1]
        
        # Apply gating and create routed tokens
        gate_early = torch.sigmoid(self.level_gates['early'])
        gate_mid = torch.sigmoid(self.level_gates['mid'])
        gate_late = torch.sigmoid(self.level_gates['late'])
        
        routed = {
            'early': tokens * routing_early * gate_early,  # Ambience-focused
            'mid': tokens * routing_mid * gate_mid,        # Background-focused
            'late': tokens * routing_late * gate_late      # Foreground-focused
        }
        
        return routed


# TemperatureScheduler removed - using the first definition above


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block for Perceiver-style decoder"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Smaller FFN with 2x expansion (not 4x)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, 77, d_model] - learned query tokens
            keys_values: [B, 10, d_model] - audio tokens to attend to
        Returns:
            Updated queries [B, 77, d_model]
        """
        # Cross-attention with pre-norm
        q_norm = self.ln_q(queries)
        kv_norm = self.ln_kv(keys_values)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        queries = queries + attn_out
        
        # FFN with residual
        queries = queries + self.ffn(queries)
        return queries


class AudioProjectionTransformer77(nn.Module):
    """
    Perceiver-style cross-attention decoder that projects 10 audio tokens to 77 CLIP tokens.
    Uses learned queries and cross-attention for parameter efficiency.
    Target: ~2.2M params (reduced from 28M)
    """

    def __init__(
        self,
        audio_dim: int = 768,
        clip_dim: int = 768,
        bottleneck_dim: int = 256,  # Reduced to meet 4.2M parameter budget
        num_heads: int = 8,
        num_layers: int = 4,  # Increased per PROJECT_PLAN_V4_FINAL.md
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.audio_dim = audio_dim
        self.clip_dim = clip_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Project audio tokens to bottleneck dimension
        self.audio_proj = nn.Linear(audio_dim, bottleneck_dim)
        
        # Learned query embeddings for 77 output tokens
        self.queries = nn.Parameter(torch.randn(77, bottleneck_dim) * 0.02)
        
        # Positional embeddings for queries
        self.query_pos = nn.Parameter(torch.zeros(77, bottleneck_dim))
        
        # Cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(bottleneck_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Project back to CLIP dimension
        self.out_proj = nn.Linear(bottleneck_dim, clip_dim)
        
        # Final layer norm for CLIP compatibility
        self.out_norm = nn.LayerNorm(clip_dim)
        
        # Optional: learnable CLIP position embeddings
        self.clip_pos_embed = nn.Parameter(torch.zeros(1, 77, clip_dim))
        nn.init.trunc_normal_(self.clip_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 10, 768] hierarchical audio tokens

        Returns:
            tokens77: [batch, 77, 768] CLIP-compatible tokens
        """
        B = x.shape[0]
        
        # Project audio to bottleneck dimension
        audio_features = self.audio_proj(x)  # [B, 10, bottleneck_dim]
        
        # Initialize queries with positional encoding
        queries = self.queries.unsqueeze(0) + self.query_pos.unsqueeze(0)  # [1, 77, bottleneck_dim]
        queries = queries.expand(B, -1, -1)  # [B, 77, bottleneck_dim]
        
        # Apply cross-attention blocks
        for block in self.blocks:
            queries = block(queries, audio_features)
        
        # Project to CLIP dimension
        output = self.out_proj(queries)  # [B, 77, clip_dim]
        
        # Add CLIP positional embeddings and normalize
        output = output + self.clip_pos_embed
        output = self.out_norm(output)
        
        return output


class CrossHierarchyAttention(nn.Module):
    """
    Efficient self-attention across hierarchy levels using bottleneck dimension.
    Reduces parameters from ~7M to ~0.5M while maintaining expressiveness.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        bottleneck_dim: int = 256,  # Reduced from full 768
        mlp_ratio: float = 2.0  # Reduced from 4.0
    ):
        super().__init__()
        
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        
        # Check divisibility for bottleneck
        if bottleneck_dim % num_heads != 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Project to bottleneck
        self.input_proj = nn.Linear(dim, bottleneck_dim)
        
        # Pre-norm for attention (in bottleneck space)
        self.norm1 = nn.LayerNorm(bottleneck_dim)
        
        # Attention components (in bottleneck space)
        self.qkv = nn.Linear(bottleneck_dim, bottleneck_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Pre-norm for MLP (in bottleneck space)
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        
        # Compact MLP in bottleneck space
        mlp_hidden_dim = int(bottleneck_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, bottleneck_dim),
            nn.Dropout(dropout)
        )
        
        # Project back to original dimension
        self.output_proj = nn.Linear(bottleneck_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with residual connections through bottleneck"""
        B, N, C = x.shape
        
        # Store original input for final residual
        x_orig = x
        
        # Project to bottleneck dimension
        x = self.input_proj(x)  # [B, N, bottleneck_dim]
        
        # Self-attention block with pre-norm and residual (in bottleneck space)
        residual = x
        x = self.norm1(x)
        
        # Generate Q, K, V in bottleneck space
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values (in bottleneck space)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.bottleneck_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Add residual connection in bottleneck space
        x = residual + x
        
        # MLP block with pre-norm and residual (in bottleneck space)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        # Project back to original dimension and add residual from input
        x = self.output_proj(x)
        x = x_orig + x  # Residual from original input
        
        return x


class ImprovedHierarchicalAudioEncoder(nn.Module):
    """
    Complete improved V4 encoder with soft decomposition and adaptive weights.
    Combines all components for the full audio encoding pipeline.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_tokens: int = 10,
        num_levels: int = 3,
        out_tokens: int = 77,
        dropout: float = 0.1,
        use_adaptive_weights: bool = True,
        use_soft_decomposition: bool = True
    ):
        super().__init__()
        
        self.use_soft_decomposition = use_soft_decomposition
        
        if use_soft_decomposition:
            # Soft hierarchical decomposition
            self.decomposer = SoftHierarchicalDecomposition(
                audio_dim=audio_dim,
                text_dim=text_dim,
                num_tokens=num_tokens,
                num_levels=num_levels,
                dropout=dropout
            )
        else:
            # Legacy rigid decomposition for backward compatibility
            self.decomposer = HierarchicalAudioDecomposition(
                audio_dim=audio_dim,
                text_dim=text_dim,
                dropout=dropout
            )
        
        # Adaptive hierarchy weights
        if use_adaptive_weights:
            self.adaptive_weights = AdaptiveHierarchyWeights(
                audio_dim=audio_dim,
                hidden_dim=6,  # Reduced to 6 to meet ~3K param target per PLAN
                num_levels=num_levels,
                use_audio_context=True
            )
        else:
            self.adaptive_weights = None
        
        # Level to UNet router
        self.router = LevelToUNetRouter(
            num_levels=num_levels,
            text_dim=text_dim
        )
        
        # Projection to 77 tokens using Perceiver-style architecture
        self.projector = AudioProjectionTransformer77(
            audio_dim=text_dim,
            clip_dim=text_dim,
            bottleneck_dim=256,  # Reduced from 320 to meet 4.2M parameter budget
            num_heads=8,
            num_layers=4  # Increased from 2 to 4 per PROJECT_PLAN_V4_FINAL.md
        )
        
        # Temperature scheduler (created externally and passed during training)
        self.temperature_scheduler = None
    
    def compute_losses(
        self,
        assignments: torch.Tensor,
        tokens: torch.Tensor,
        hierarchy_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for Stage 2 training.
        
        Args:
            assignments: [B, K, L] soft assignments
            tokens: [B, K, D] token embeddings
            hierarchy_weights: [B, L] optional adaptive weights
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Entropy loss (encourage exploration, prevent collapse)
        # Return raw loss, weight will be applied in training_step
        # FIX: Entropy should be positive (we want to maximize it, so minimize negative entropy)
        entropy = -(assignments * (assignments + 1e-8).log()).sum(dim=-1).mean()
        losses['entropy'] = entropy  # Already negative, no need for double negative!
        
        # Orthogonality loss (encourage diverse tokens)
        # Return raw loss, weight will be applied in training_step
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        gram = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B, K, K]
        I = torch.eye(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        ortho_loss = F.mse_loss(gram, I)
        losses['orthogonality'] = ortho_loss
        
        # Prior loss (soft encouragement towards 5-3-2 distribution)
        # Return raw loss, weight will be applied in training_step
        if self.use_soft_decomposition and hasattr(self.decomposer, 'level_prior'):
            avg_assignment = assignments.mean(dim=1)  # [B, L] - already probabilities
            prior = self.decomposer.level_prior.unsqueeze(0)  # [1, L]
            # FIX: Correct KL divergence orientation - KL(empirical || prior) not KL(prior || empirical)
            # PyTorch's kl_div expects input=log(Q), target=P to compute KL(P || Q)
            # We want KL(empirical || prior), so input=log(prior), target=empirical
            prior_loss = F.kl_div(
                prior.expand_as(avg_assignment).log(),  # log(prior)
                avg_assignment,  # empirical distribution
                reduction='batchmean'
            )
            losses['prior'] = prior_loss
        else:
            losses['prior'] = torch.tensor(0.0, device=tokens.device)
        
        return losses
    
    def forward(
        self,
        audio_features: torch.Tensor,
        return_all: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Complete forward pass through improved V4 encoder.
        
        Args:
            audio_features: [B, 512] CLAP features
            return_all: Whether to return all intermediate outputs
            
        Returns:
            tokens_77: [B, 77, 768] final output tokens
            info: (optional) Dictionary with all intermediate outputs
        """
        if self.use_soft_decomposition:
            # Soft hierarchical decomposition
            tokens_10, decomp_info = self.decomposer(audio_features, return_stats=True)
            assignments = decomp_info['assignments']
        else:
            # Legacy rigid decomposition
            tokens_10 = self.decomposer(audio_features)
            # Create dummy assignments for compatibility
            assignments = torch.zeros(tokens_10.shape[0], tokens_10.shape[1], 3, device=tokens_10.device)
            decomp_info = {'temperature': 1.0}
        
        # Adaptive hierarchy weights (if enabled)
        if self.adaptive_weights is not None:
            hierarchy_weights = self.adaptive_weights(audio_features)
        else:
            hierarchy_weights = None
        
        # Route to UNet scales with hierarchy weights
        routed_tokens = self.router(tokens_10, assignments, hierarchy_weights)
        
        # Project to 77 tokens
        tokens_77 = self.projector(tokens_10)
        
        if return_all:
            # Compute losses for training
            losses = self.compute_losses(
                assignments,
                tokens_10,
                hierarchy_weights
            )
            
            info = {
                'tokens_10': tokens_10,
                'tokens_77': tokens_77,
                'assignments': assignments,
                'routed': routed_tokens,
                'hierarchy_weights': hierarchy_weights,
                'losses': losses,
                'stats': decomp_info.get('stats', {}),
                'temperature': decomp_info['temperature']
            }
            return tokens_77, info
        
        return tokens_77


# Legacy classes for backward compatibility
class HierarchicalAudioDecomposition(nn.Module):
    """
    Legacy rigid hierarchical decomposition (kept for backward compatibility).
    Decomposes CLAP audio features into hierarchical semantic tokens:
    - Foreground: Primary sounds/activities (5 tokens)
    - Background: Secondary/environmental sounds (3 tokens)  
    - Ambience: Overall atmosphere/mood (2 tokens)
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_foreground: int = 5,
        num_background: int = 3,
        num_ambience: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_foreground = num_foreground
        self.num_background = num_background
        self.num_ambience = num_ambience
        self.total_tokens = num_foreground + num_background + num_ambience
        
        # Projection layers for each hierarchy level
        self.foreground_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim * num_foreground)
        )
        
        self.background_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim * num_background)
        )
        
        self.ambience_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, text_dim * num_ambience)
        )
        
        # Learnable hierarchy weights
        self.hierarchy_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(text_dim)
        
        # Cross-hierarchy attention with consistent bottleneck
        self.cross_hierarchy_attn = CrossHierarchyAttention(
            text_dim, num_heads=4, dropout=dropout,
            bottleneck_dim=192  # Match soft decomposition for consistency
        )
        
    def forward(
        self, 
        audio_features: torch.Tensor,
        return_hierarchy: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Forward pass to decompose audio into hierarchical tokens"""
        batch_size = audio_features.shape[0]
        
        # Project to each hierarchy level
        foreground = self.foreground_proj(audio_features)
        background = self.background_proj(audio_features)
        ambience = self.ambience_proj(audio_features)
        
        # Reshape to token sequences
        foreground = foreground.view(batch_size, self.num_foreground, self.text_dim)
        background = background.view(batch_size, self.num_background, self.text_dim)
        ambience = ambience.view(batch_size, self.num_ambience, self.text_dim)
        
        # Apply hierarchy weights
        weights = F.softmax(self.hierarchy_weights, dim=0)
        foreground = foreground * weights[0]
        background = background * weights[1]
        ambience = ambience * weights[2]
        
        # Concatenate all hierarchy levels
        audio_tokens = torch.cat([foreground, background, ambience], dim=1)
        
        # Apply cross-hierarchy attention
        audio_tokens = self.cross_hierarchy_attn(audio_tokens)
        
        # Layer normalization
        audio_tokens = self.layer_norm(audio_tokens)
        
        if return_hierarchy:
            hierarchy_dict = {
                'foreground': foreground,
                'background': background,
                'ambience': ambience,
                'weights': weights,
                'combined': audio_tokens
            }
            return audio_tokens, hierarchy_dict
        
        return audio_tokens


class HierarchicalAudioV4(nn.Module):
    """
    Legacy Stage 1 encoder (kept for backward compatibility).
    Uses rigid decomposition + projection to 77 tokens.
    """

    def __init__(
        self,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_foreground: int = 5,
        num_background: int = 3,
        num_ambience: int = 2,
        out_tokens: int = 77,
        projector_layers: int = 4,
        projector_heads: int = 8,
        projector_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.decomposer = HierarchicalAudioDecomposition(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_foreground=num_foreground,
            num_background=num_background,
            num_ambience=num_ambience,
            dropout=dropout,
        )

        self.projector = AudioProjectionTransformer77(
            audio_dim=text_dim,
            clip_dim=text_dim,
            bottleneck_dim=256,
            num_heads=projector_heads,
            num_layers=projector_layers,
            dropout=dropout,
        )

    def forward(self, clap_features: torch.Tensor, return_intermediate: bool = False):
        """Forward pass through encoder"""
        tokens10, hierarchy = self.decomposer(clap_features, return_hierarchy=True)
        tokens77 = self.projector(tokens10)
        if return_intermediate:
            hierarchy = dict(hierarchy)
            hierarchy['tokens10'] = tokens10
            return tokens77, hierarchy
        return tokens77


if __name__ == "__main__":
    print("Testing Improved Hierarchical Audio V4...")
    
    # Create improved encoder
    encoder = ImprovedHierarchicalAudioEncoder(
        use_adaptive_weights=True,
        use_soft_decomposition=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    audio_features = torch.randn(batch_size, 512)
    
    # Simple forward
    tokens_77 = encoder(audio_features)
    print(f"Output shape: {tokens_77.shape}")  # [4, 77, 768]
    
    # Forward with all outputs
    tokens_77, info = encoder(audio_features, return_all=True)
    print(f"Tokens 10 shape: {info['tokens_10'].shape}")  # [4, 10, 768]
    print(f"Assignments shape: {info['assignments'].shape}")  # [4, 10, 3]
    print(f"Temperature: {info['temperature']}")
    print(f"Losses: {list(info['losses'].keys())}")
    print(f"Stats: {info['stats']}")
    
    # Test temperature scheduler
    scheduler = TemperatureScheduler(
        encoder.decomposer,
        T_max=2.0,
        T_min=0.5,
        total_steps=2000
    )
    
    # Simulate training steps
    for step in [0, 100, 500, 1000, 1500, 2000]:
        T = scheduler.step(step)
        print(f"Step {step}: Temperature = {T:.3f}")
    
    print("\nImproved Hierarchical Audio V4 test passed!")