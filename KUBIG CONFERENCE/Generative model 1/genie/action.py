from typing import Tuple
from torch import Tensor
import torch.nn as nn

from math import prod
from torch.nn.functional import mse_loss

from einops.layers.torch import Rearrange

from genie.module import parse_blueprint
from genie.module.quantization import LookupFreeQuantization
from genie.module.video import CausalConv3d, Downsample, Upsample
from genie.utils import Blueprint

from pdb import set_trace as st

class LazyLinear(nn.Module):
    """Linear layer that lazily initialises its weight on first forward.

    This avoids hard-coding the `in_features` value and makes the module
    robust to different encoder output shapes coming from blueprints.
    """
    def __init__(self, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.linear: nn.Linear | None = None

    def forward(self, x: Tensor) -> Tensor:
        # x expected shape (..., in_features)
        in_features = x.shape[-1]
        if self.linear is None:
            # Create the linear layer and move it to the same device/dtype as the input
            lin = nn.Linear(in_features, self.out_features, bias=self.bias)
            try:
                lin = lin.to(device=x.device, dtype=x.dtype)
            except Exception:
                # Fallback: move parameters to device only
                lin = lin.to(device=x.device)
            self.linear = lin
        return self.linear(x)

REPR_ACT_ENC = (
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 64,
        'd_inp': 512
    }),
)

REPR_ACT_DEC = (
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 8,
        'd_inp': 64,
    }),
)

class LatentAction(nn.Module):
    '''Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    '''
    
    def __init__(
        self,
        enc_desc: Blueprint,
        dec_desc: Blueprint,
        d_codebook: int,
        inp_channels: int = 3,
        inp_shape : int | Tuple[int, int] = (64, 64),
        ker_size : int | Tuple[int, int] = 3,
        n_embd: int = 256,
        n_codebook: int = 1,
        lfq_bias : bool = True,
        lfq_frac_sample : float = 1.,
        lfq_commit_weight : float = 0.25,
        lfq_entropy_weight : float = 0.1,
        lfq_diversity_weight : float = 1.,
        quant_loss_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        if isinstance(inp_shape, int): inp_shape = (inp_shape, inp_shape)
        
        self.proj_in = CausalConv3d(
            inp_channels,
            out_channels=n_embd,
            kernel_size=ker_size
        )
        
        self.proj_out = CausalConv3d(
            n_embd,
            out_channels=inp_channels,
            kernel_size=ker_size
        )
        
        # Build the encoder and decoder based on the blueprint
        self.enc_layers, self.enc_ext = parse_blueprint(enc_desc)
        self.dec_layers, self.dec_ext = parse_blueprint(dec_desc)

        # Infer the expected input channels for the first encoder layer so
        # that proj_in produces a tensor with matching channel count.
        expected_enc_in = None
        if len(self.enc_layers) > 0:
            first = self.enc_layers[0]
            # Many downsample modules wrap a conv in `go_down`; prefer that
            inner = getattr(first, 'go_down', None)
            if inner is not None and hasattr(inner, 'in_channels'):
                expected_enc_in = int(getattr(inner, 'in_channels'))
            else:
                # Try common attribute names on the layer itself
                for attr in ('in_channels', 'inp_channels', 'inp_dim', 'out_channels'):
                    if hasattr(first, attr) and getattr(first, attr) is not None:
                        expected_enc_in = int(getattr(first, attr))
                        break

        # Fallback to n_embd if we couldn't infer
        expected_enc_in = expected_enc_in or n_embd

        # If proj_in currently produces a different number of channels, recreate it
        # so that its output matches what the encoder expects.
        try:
            current_out = int(getattr(self.proj_in, 'out_channels', getattr(self.proj_in, 'out_dim', n_embd)))
        except Exception:
            current_out = n_embd

        if current_out != expected_enc_in:
            self.proj_in = CausalConv3d(
                inp_channels,
                out_channels=expected_enc_in,
                kernel_size=ker_size,
            )
            self.proj_out = CausalConv3d(
                expected_enc_in,
                out_channels=inp_channels,
                kernel_size=ker_size,
            )
        
        # Keep track of space-time up/down factors
        enc_fact = prod(enc.factor for enc in self.enc_layers if isinstance(enc, (Downsample, Upsample)))
        dec_fact = prod(dec.factor for dec in self.dec_layers if isinstance(dec, (Downsample, Upsample)))
        
        assert enc_fact * dec_fact == 1, 'The product of the space-time up/down factors must be 1.'
        
        # Add the projections to the action space.
        # Use a lazy linear so we don't depend on fragile hard-coded
        # in_features (which depend on blueprint up/downsampling).
        # The linear should output `d_codebook * n_codebook` features so
        # the subsequent quantizer receives the expected flattened dim.
        act_feat_dim = int(d_codebook * n_codebook)
        self.to_act = nn.Sequential(
            Rearrange('b c t ... -> b t (c ...)'),
            LazyLinear(act_feat_dim, bias=False),
        )
        
        # Build the quantization module
        # Pass `input_dim` equal to the to_act output size so the quantizer's
        # internal projection has matching dimensions. This avoids the
        # default behavior of using 2**codebook_dim which can be huge.
        self.quant = LookupFreeQuantization(
            codebook_dim       = d_codebook,
            num_codebook       = n_codebook,
            input_dim          = act_feat_dim,
            use_bias           = lfq_bias,
            frac_sample        = lfq_frac_sample,
            commit_weight      = lfq_commit_weight,
            entropy_weight     = lfq_entropy_weight,
            diversity_weight   = lfq_diversity_weight,
        )
        
        self.d_codebook = d_codebook
        self.n_codebook = n_codebook
        self.quant_loss_weight = quant_loss_weight
        
    def sample(self, idxs : Tensor) -> Tensor:
        '''Sample the action codebook values based on the indices.'''
        return self.quant.codebook[idxs]
        
    def encode(
        self,
        video: Tensor,
        mask : Tensor | None = None,
        transpose : bool = False,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        # Debug: log shapes so we can trace channel mismatches
        try:
            pass
            # print(f"[LatentAction.encode] incoming video.shape={tuple(video.shape)}")
        except Exception:
            pass

        video = self.proj_in(video)
        try:
            pass
            # print(f"[LatentAction.encode] after proj_in -> video.shape={tuple(video.shape)}")
        except Exception:
            pass

        # If we have encoder layers, try to print what the first expects
        if len(self.enc_layers) > 0:
            first = self.enc_layers[0]
            info = {
                'type': type(first).__name__,
                'in_channels': getattr(first, 'in_channels', None),
                'inp_channels': getattr(first, 'inp_channels', None),
                'inp_dim': getattr(first, 'inp_dim', None),
                'out_channels': getattr(first, 'out_channels', None),
            }
            try:
                pass
                # print(f"[LatentAction.encode] first_enc={info}")
            except Exception:
                pass
        # video = [1, 64, 16, 64, 64]
        # Encode the video frames into latent actions
        for enc in self.enc_layers:
            video = enc(video, mask=mask)
        
        # video = [1, 512, 16, 16, 16]
        # Project to latent action space
        act : Tensor = self.to_act(video)
        # act = [1, 16, 10]
        # st()

        # Quantize the latent actions
        (act, idxs), q_loss = self.quant(act, transpose=transpose)
                
        return (act, idxs, video), q_loss
    
    def decode(
        self,
        video : Tensor,
        q_act : Tensor,
    ) -> Tensor:        
        # Decode the video frames based on past history and
        # the quantized latent actions
        for dec, has_ext in zip(self.dec_layers, self.dec_ext):
            video = dec(
                video,
                cond=(
                    None, # No space condition
                    q_act if has_ext else None,
                )
            )
            
        recon = self.proj_out(video)
        
        return recon
        
    def forward(
        self,
        video: Tensor,
        mask : Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        
        # Encode the video frames into latent actions
        (act, idxs, enc_video), q_loss = self.encode(video, mask=mask)
        # act = [1, 16, 10]
        # enc_video = [1, 512, 16, 16, 16]
        # st()
        
        # Decode the last video frame based on all the previous
        # frames and the quantized latent actions
        recon = self.decode(enc_video, act)  # [1,3,16,64,64]
        
        # Compute the reconstruction loss
        # Reconstruction loss
        rec_loss = mse_loss(recon, video)
        
        # Compute the total loss by combining the individual
        # losses, weighted by the corresponding loss weights
        loss = rec_loss\
            + q_loss * self.quant_loss_weight
        
        return idxs, loss, (
            rec_loss,
            q_loss,
        )