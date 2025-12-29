import torch
import torch.nn as nn
from torch import log

from torch import Tensor
from einops import reduce
from einops import einsum
from einops import rearrange
from einops import pack, unpack

from torch.nn.functional import mse_loss

from typing import Tuple

from genie.utils import default

# Removed unused entropy function - entropy is now computed directly using log-probabilities
# in the LookupFreeQuantization.forward() method for better numerical stability

# Simplified version of the lucidrains implementation at:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py#L49
class LookupFreeQuantization(nn.Module):
    '''
    Lookup-Free Quantization module as originally introduced
    in the paper "Language Model Beats Diffusion: Tokenizer
    is key to visual generation" Yu et al. (2024).
    '''
    
    def __init__(
        self,
        codebook_dim : int,
        num_codebook : int = 1,
        input_dim : int | None = None,
        use_bias : bool = True,
        frac_sample : float = 1.,
        commit_weight : float = 0.25,
        entropy_weight : float = 0.1,
        diversity_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        codebook_size = (2 ** codebook_dim) * num_codebook
        input_dim = default(input_dim, codebook_size)
        
        project = input_dim != codebook_dim * num_codebook
        
        self.proj_inp = nn.Linear(input_dim, codebook_dim * num_codebook, bias=use_bias) if project else nn.Identity()
        self.proj_out = nn.Linear(codebook_dim * num_codebook, input_dim, bias=use_bias) if project else nn.Identity()
        
        self.frac_sample = frac_sample
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebook
        self.codebook_size = codebook_size
        self.commit_weight = commit_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        
        # * Initialize the codebook
        # Use the bit_mask to generate the bit-codes for all the codebook entries
        # and then convert them to the actual codebook values {-1, 1}. Resulting
        # codebook will have shape (codebook_size, d_codebook).
        self.register_buffer('bit_mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        
        codes = torch.arange(codebook_size, dtype=int)[:, None] & self.bit_mask
        self.register_buffer('codebook', 2 * (codes != 0).float() - 1, persistent=False)
        
    def forward(
        self,
        inp : Tensor,
        beta : float = 100.,
        transpose : bool = False
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor | None]:
        
        # Standardize the input tensor to have shape (batch_size, seq_len, inp_dim)
        inp = rearrange(inp, 'b d ... -> b ... d') if transpose else inp
        inp, ps = pack([inp], 'b * d')
        
        inp = self.proj_inp(inp)
        
        # Split into n_codebook parts
        inp = rearrange(inp, 'b n (c d) -> b n c d', c=self.num_codebooks)
        
        # Quantize by simply assigning {-1, 1} to the input tensor depending on the sign
        # of the input tensor values. This is the lookup-free quantization step.
        # See Eq. (3) in the original paper. To obtain the quantized-code indices
        # we simply sum the bit-codes representation of the quantized values.
        quant = inp.sign()
        idxs = reduce((inp > 0).int() * self.bit_mask.int(), 'b n c d -> b n c', 'sum')
        
        # Use straight-through estimator to back-propagate through the quantization step
        code = (inp + (quant - inp).detach()) if self.training else quant
        code = rearrange(code, 'b n c d -> b n (c d)')
        
        # Reconstruct the input tensor from the quantized values
        out = self.proj_out(code)
        out = unpack(out, ps, 'b * d')[0]
        out = rearrange(out, 'b ... d -> b d ...') if transpose else out
        
        # NOTE: Squeeze to remove the n_codebook dimension
        idxs = unpack(idxs, ps, 'b * d')[0].squeeze()
        
        # No need to compute the loss if we are not training
        if not self.training: return (out, idxs), None
        
        # Compute the entropy loss using numerically stable log-softmax
        # This prevents overflow in softmax and NaN in entropy calculation
        inp_scores = 2 * einsum(inp, self.codebook, '... i d, j d -> ... i j')
        
        # Normalize scores to prevent overflow (subtract max for numerical stability)
        # This is the log-softmax trick: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        inp_scores_max = inp_scores.max(dim=-1, keepdim=True)[0].detach()
        inp_scores_normalized = inp_scores - inp_scores_max
        
        # Apply beta scaling after normalization to maintain gradient flow
        beta_safe = min(beta, 100.0)  # Reasonable upper bound
        logits = inp_scores_normalized * beta_safe
        
        # Use log-softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = rearrange(log_probs, 'b n ... -> (b n) ...')
        
        # Convert to probabilities for average calculation
        # exp(log_probs) is safe because log_probs is bounded (from log_softmax)
        inp_prob = log_probs.exp()
        
        # Compute average probability across batch
        avg_prob = reduce(inp_prob, '... c d -> c d', 'mean')
        
        # Compute entropy using log-probabilities directly
        # H(p) = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
        # inp_prob shape: (b*n, c, codebook_size)
        # inp_ent: average entropy per sample per codebook (already normalized by codebook_size)
        inp_ent = -(inp_prob * log_probs).sum(dim=-1).mean()  # Mean over (b*n, c) -> scalar
        
        # For average entropy, compute entropy of the average distribution
        # avg_prob shape: (c, codebook_size) - average probability distribution across batch
        # avg_ent: entropy of average distribution per codebook (already normalized by codebook_size)
        avg_log_prob = torch.log(avg_prob.clamp(min=1e-10, max=1.0))
        avg_ent = -(avg_prob * avg_log_prob).sum(dim=-1).mean()  # Mean over c -> scalar
        
        # Both inp_ent and avg_ent are already normalized (entropy is sum over codebook_size, then mean)
        # They should be in range [0, log(codebook_size)] ≈ [0, 6.93] for codebook_size=1024
        # No need for additional clamping as the values are already bounded
        entropy_loss = inp_ent + self.diversity_weight * avg_ent
        
        # Compute commitment loss
        commit_loss = mse_loss(inp, quant.detach(), reduction = 'mean')
        
        # Compute the complete final loss
        loss = entropy_loss * self.entropy_weight + commit_loss * self.commit_weight
        
        return (out, idxs), loss