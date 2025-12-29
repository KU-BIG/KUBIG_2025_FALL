import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.nn.functional import mse_loss

from typing import Any, Tuple, Dict, Callable, Iterable, List
from itertools import zip_longest

from lightning import LightningModule

from genie.module.loss import GANLoss
from genie.module.loss import PerceptualLoss
from genie.module.quantization import LookupFreeQuantization
from genie.utils import default, exists

from genie.module.attention import SpaceTimeAttention

from pdb import set_trace as st

OptimizerCallable = Callable[[Iterable], Optimizer]

# Representation-tokenizer encoder/decoder blueprints used by tests
REPR_TOK_ENC = (
    ('spacetime_downsample', {
        'in_channels' : 3,
        'kernel_size' : 3,
        'out_channels' : 512,
        'time_factor' : 1,
        'space_factor' : 4,
    }),
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 64,
        'transpose' : True,
    }),
)

REPR_TOK_DEC = (
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 64,
        'transpose' : True,
    }),
    ('depth2spacetime_upsample', {
        'in_channels' : 512,
        'kernel_size' : 3,
        'out_channels' : 3,
        'time_factor' : 1,
        'space_factor' : 4,
    })
)


def _build_from_blueprint(desc: Iterable[Tuple[str, Dict[str, Any]]], initial_in: int | None = None) -> nn.Sequential:
    """Very small blueprint interpreter used by unit tests to build
    a deterministic encoder/decoder pipeline. This keeps the tokenizer
    implementation simple so shapes are easy to follow.
    Supported ops: 'spacetime_downsample', 'space-time_attn', 'depth2spacetime_upsample'
    """
    modules: List[nn.Module] = []
    last_out = initial_in
    for name, cfg in desc:
        if name == 'spacetime_downsample':
            k = cfg.get('kernel_size', 3)
            t = cfg.get('time_factor', 1)
            s = cfg.get('space_factor', 1)
            in_ch = cfg.get('in_channels', last_out)
            out_ch = cfg.get('out_channels')
            # If previous module produced a different number of channels,
            # insert a 1x1 projection to match expected input channels.
            if last_out is not None and in_ch is not None and in_ch != last_out:
                modules.append(nn.Conv3d(last_out, in_ch, kernel_size=1))
            # Conv3d expects (C, T, H, W) ordering when input is (B, C, T, H, W)
            modules.append(
                nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(k, k, k),
                    stride=(t, s, s),
                    padding=((k - 1) // 2, (k - 1) // 2, (k - 1) // 2),
                )
            )
            last_out = out_ch
        elif name == 'space-time_attn':
            # Use SpaceTimeAttention; feed d_inp=last_out to keep LayerNorm dims consistent
            n_head = cfg.get('n_head', cfg.get('n_rep', 8))
            d_head = cfg.get('d_head', 64)
            transpose = cfg.get('transpose', False)
            # If last_out is None the attention will default its input dim
            modules.append(SpaceTimeAttention(n_head=n_head, d_head=d_head, d_inp=last_out, transpose=transpose))
            # SpaceTimeAttention advertises out_channels on instances
            last_out = modules[-1].out_channels
        elif name == 'depth2spacetime_upsample':
            k = cfg.get('kernel_size', 3)
            t = cfg.get('time_factor', 1)
            s = cfg.get('space_factor', 1)
            in_ch = cfg.get('in_channels', last_out)
            out_ch = cfg.get('out_channels')
            # Insert projection if previous output channels don't match expected in_ch
            if last_out is not None and in_ch is not None and in_ch != last_out:
                modules.append(nn.Conv3d(last_out, in_ch, kernel_size=1))
            # To exactly invert a previous stride=(t,s,s) downsample we set
            # output_padding = (t-1, s-1, s-1). With kernel k and padding=(k-1)//2
            # this yields output spatial size = input_size * stride.
            modules.append(
                nn.ConvTranspose3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(k, k, k),
                    stride=(t, s, s),
                    padding=((k - 1) // 2, (k - 1) // 2, (k - 1) // 2),
                    output_padding=(max(0, t - 1), max(0, s - 1), max(0, s - 1)),
                )
            )
            last_out = out_ch
        else:
            raise ValueError(f'Unsupported blueprint op: {name}')

    return nn.Sequential(*modules)


class VideoTokenizer(LightningModule):
    """Cleaner, blueprint-driven VideoTokenizer simplified for unit tests.

    This implementation avoids the project-specific `parse_blueprint`
    machinery and instead directly interprets a small set of ops.
    It's intentionally small so dimension flow is transparent during debugging.
    """

    def __init__(
        self,
        enc_desc: Iterable[Tuple[str, Dict[str, Any]]],
        dec_desc: Iterable[Tuple[str, Dict[str, Any]]],
        disc_kwargs: Dict[str, Any] = {},
        d_codebook: int = 10,
        n_codebook: int = 1,
        lfq_bias: bool = True,
        lfq_frac_sample: float = 1.,
        lfq_commit_weight: float = 0.25,
        lfq_entropy_weight: float = 0.1,
        lfq_diversity_weight: float = 1.,
        lfq_beta: float = 10.,
        optimizer: OptimizerCallable = AdamW,
        perceptual_model: str = 'vgg16',
        perc_feat_layers: Iterable[str] = ('features.6', 'features.13', 'features.18', 'features.25'),
        gan_discriminate: str = 'frames',
        gan_frames_per_batch: int = 4,
        gan_loss_weight: float = 1.,
        perc_loss_weight: float = 1.,
        quant_loss_weight: float = 1.,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer

        # Build encoder directly from the small blueprint language
        self.enc_layers = _build_from_blueprint(enc_desc, initial_in=None)

        # last encoder dim (channel count)
        last_enc_dim = [m.out_channels for m in self.enc_layers.modules() if hasattr(m, 'out_channels')][-1]

        # build decoder, letting the builder know encoder's output channels
        self.dec_layers = _build_from_blueprint(dec_desc, initial_in=last_enc_dim)

        # quant module
        self.quant = LookupFreeQuantization(
            codebook_dim=d_codebook,
            num_codebook=n_codebook,
            input_dim=last_enc_dim,
            use_bias=lfq_bias,
            frac_sample=lfq_frac_sample,
            commit_weight=lfq_commit_weight,
            entropy_weight=lfq_entropy_weight,
            diversity_weight=lfq_diversity_weight,
        )

        # expose hid channels for tests
        self.hid_channels = int(last_enc_dim)

        # losses
        self.perc_crit = PerceptualLoss(model_name=perceptual_model, feat_layers=perc_feat_layers, num_frames=gan_frames_per_batch) if perc_feat_layers else nn.Identity()
        self.gan_crit = GANLoss(discriminate=gan_discriminate, num_frames=gan_frames_per_batch, **disc_kwargs) if gan_loss_weight > 0 else nn.Identity()

        self.gan_loss_weight = gan_loss_weight
        self.perc_loss_weight = perc_loss_weight
        self.quant_loss_weight = quant_loss_weight
        self.lfq_beta = lfq_beta
        self.save_hyperparameters()
        # st()

    # Provide simple accessors used by `Genie` to obtain the
    # canonical encoder/decoder blueprints by name.
    def get_enc(self, name: str):
        if name == 'repr_tok':
            return REPR_TOK_ENC
        raise ValueError(f'Unknown encoder: {name}')

    def get_dec(self, name: str):
        if name == 'repr_tok':
            return REPR_TOK_DEC
        raise ValueError(f'Unknown decoder: {name}')

    def encode(self, video: Tensor, cond: Tensor | None = None) -> Tensor:
        # enc_layers is an nn.Sequential built from the small blueprint
        return self.enc_layers(video)

    def decode(self, quant: Tensor, cond: Tensor | None = None) -> Tensor:
        # dec_layers is an nn.Sequential built from the small blueprint
        decoded = self.dec_layers(quant)
        # Clamp instead of sigmoid to avoid gradient vanishing
        # Sigmoid can cause gradient vanishing when values are extreme
        # Clamp preserves gradients better while still ensuring valid range
        return torch.clamp(decoded, 0.0, 1.0)

    @torch.no_grad()
    def tokenize(self, video: Tensor, beta: float = 100., transpose: bool = True) -> Tuple[Tensor, Tensor]:
        self.eval()
        enc = self.encode(video)
        (quant_video, idxs), _ = self.quant(enc, beta=beta, transpose=transpose)
        self.train()
        return quant_video, idxs

    def forward(
        self,
        video : Tensor,
        beta : float | None = None,
        transpose : bool = True,
        train_discriminator : bool = True,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        if beta is None:
            beta = self.lfq_beta
        
        # Clamp beta to prevent numerical instability in quantization
        # Beta values > 100 can cause overflow in softmax
        beta = min(beta, 100.0)
        
        enc_video = self.encode(video)  # [1, 512 ,64, 32, 32] = [b, c, t, h, w]
        
        # Use fp32 for quantization to prevent NaN in mixed precision
        with torch.cuda.amp.autocast(enabled=False):
            (quant_video, idxs), quant_loss = self.quant(enc_video.float(), beta=beta, transpose=transpose)  # [1, 512 ,64, 32, 32] // idxs = [64, 32, 32]
            quant_video = quant_video.to(enc_video.dtype)
        
        rec_video = self.decode(quant_video)  # [1, 3, 16, 64, 64]
        
        # * Compute the tokenizer loss
        # Reconstruction loss
        rec_loss = mse_loss(rec_video, video)
        
        # GAN loss (if enabled)
        # CRITICAL: In GAN training, we must freeze one network while training the other
        # - When training Generator: Discriminator must be frozen (no gradients)
        # - When training Discriminator: Generator output must be detached (already handled in GANLoss)
        if self.gan_loss_weight > 0:
            if train_discriminator:
                # Training Discriminator: Generator output is already detached in GANLoss.forward(train_gen=False)
                dis_loss = self.gan_crit(rec_video, video, train_gen=False)
                # Generator loss is not computed during discriminator training
                gen_loss = video.new_tensor(0.0)
            else:
                # Training Generator: Discriminator must be frozen (no gradients)
                # Temporarily disable gradients for discriminator to prevent it from receiving gradients
                for param in self.gan_crit.disc.parameters():
                    param.requires_grad = False
                
                # Compute generator loss (Discriminator is frozen, so no gradients flow to it)
                gen_loss = self.gan_crit(rec_video, video, train_gen=True)
                
                # Re-enable gradients for discriminator (for next discriminator training step)
                for param in self.gan_crit.disc.parameters():
                    param.requires_grad = True
                
                # Discriminator loss is not computed during generator training
                dis_loss = video.new_tensor(0.0)
        else:
            gen_loss = video.new_tensor(0.0)
            dis_loss = video.new_tensor(0.0)

        # Perceptual loss (if enabled)
        if self.perc_loss_weight > 0:
            # rec_video is already in [0, 1] range due to clamp in decode()
            # Clamp video input to ensure valid range for perceptual model
            video_clamped = torch.clamp(video, 0, 1)
            
            # Compute perceptual loss (uses fp32 internally and handles NaN/Inf)
            # PerceptualLoss.forward() handles NaN/Inf internally
            perc_loss = self.perc_crit(rec_video, video_clamped)
        else:
            perc_loss = video.new_tensor(0.0)
        
        # Compute the total loss by combining the individual
        # losses, weighted by the corresponding loss weights
        quant_scaled = 0.0
        if exists(quant_loss):
            quant_scaled = quant_loss * self.quant_loss_weight
        
        # GAN loss: only include the loss for the network being trained
        # - When training generator: only gen_loss
        # - When training discriminator: only dis_loss
        gan_loss_weighted = (gen_loss if not train_discriminator else dis_loss) * self.gan_loss_weight
        
        loss = rec_loss\
            + gan_loss_weighted\
            + perc_loss  * self.perc_loss_weight\
            + quant_scaled\
        
        return loss, (
            rec_loss,
            gen_loss if self.gan_loss_weight > 0 else 0,
            dis_loss if self.gan_loss_weight > 0 else 0,
            perc_loss if self.perc_loss_weight > 0 else 0,
            quant_loss if exists(quant_loss) and self.quant_loss_weight > 0 else 0,
        )
    
    # * Lightning core functions
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Balance Generator and Discriminator training frequency
        # Strategy: Give generator a warm-up period, then train discriminator very rarely
        # - First 5 epochs: Only train generator (discriminator frozen)
        # - After 5 epochs: Train discriminator every 50 steps (2% frequency)
        # This allows generator to learn basic reconstruction before facing discriminator
        train_discriminator = False
        if self.gan_loss_weight > 0:
            current_epoch = self.trainer.current_epoch if hasattr(self.trainer, 'current_epoch') else 0
            
            # Warm-up period: First 5 epochs, only train generator
            if current_epoch < 5:
                train_discriminator = False
            else:
                # After warm-up: Train discriminator more frequently to maintain balance
                # Increased from 50 to 20 steps (5% frequency) to prevent discriminator from becoming too strong
                # This gives generator 19 times more training steps, which is still significant
                train_discriminator = (batch_idx % 20 == 0)
        
        # Compute the training loss
        loss, aux_losses = self(batch, train_discriminator=train_discriminator)
        
        # Log all metrics to logger
        self.log_dict(
            {
                'train_loss': loss,
                'train_rec_loss'  : aux_losses[0],
                'train_gen_loss'  : aux_losses[1],
                'train_dis_loss'  : aux_losses[2],
                'train_perc_loss' : aux_losses[3],
                'train_quant_loss': aux_losses[4],
            },
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        # Log key metrics to progress bar
        self.log('loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log('rec', aux_losses[0], prog_bar=True, on_step=True, sync_dist=True)
        if self.gan_loss_weight > 0:
            # Display generator loss (negative means generator is improving)
            # When gen_loss is negative and large, it means discriminator thinks fake is real (good for generator)
            # When gen_loss is positive and large, it means discriminator is too strong
            gen_loss_scaled = aux_losses[1] * self.gan_loss_weight
            self.log('gan', gen_loss_scaled, prog_bar=True, on_step=True, sync_dist=True)
            # Log discriminator training frequency for monitoring
            self.log('dis_train', float(train_discriminator), prog_bar=False, on_step=True, sync_dist=True)
        if self.perc_loss_weight > 0 and aux_losses[3] != 0:
            self.log('perc', aux_losses[3], prog_bar=True, on_step=True, sync_dist=True)
        if aux_losses[4] != 0:
            self.log('quant', aux_losses[4], prog_bar=True, on_step=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the validation loss
        # In validation, we don't train discriminator (train_discriminator=False)
        # This ensures consistent validation metrics
        loss, aux_losses = self(batch, train_discriminator=False)
        
        # Log the validation loss
        self.log_dict(
            {
                'val_loss': loss,
                'val_rec_loss'  : aux_losses[0],
                'val_gen_loss'  : aux_losses[1],
                'val_dis_loss'  : aux_losses[2],
                'val_perc_loss' : aux_losses[3],
                'val_quant_loss': aux_losses[4],
            },
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True
        )
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        outs = self.trainer.logged_metrics
        epoch = self.trainer.current_epoch
        train_loss = outs.get('train_loss', float('nan'))
        train_rec = outs.get('train_rec_loss', float('nan'))
        train_quant = outs.get('train_quant_loss', float('nan'))
        
        train_perc = outs.get('train_perc_loss', float('nan'))
        
        # GAN loss 출력 (GAN loss weight가 0보다 클 때만)
        if self.gan_loss_weight > 0:
            train_gen = outs.get('train_gen_loss', float('nan'))
            train_dis = outs.get('train_dis_loss', float('nan'))
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                  f"Rec: {train_rec:.4f} | "
                  f"Gen: {train_gen:.4f} | "
                  f"Dis: {train_dis:.4f} | "
                  f"Perc: {train_perc:.4f} | "
                  f"Quant: {train_quant:.4f}")
        else:
            print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
                  f"Rec: {train_rec:.4f} | "
                  f"Perc: {train_perc:.4f} | "
                  f"Quant: {train_quant:.4f}")

    def on_validation_epoch_end(self) -> None:
        outs = self.trainer.logged_metrics
        epoch = self.trainer.current_epoch
        val_loss = outs.get('val_loss', float('nan'))
        val_rec = outs.get('val_rec_loss', float('nan'))
        val_quant = outs.get('val_quant_loss', float('nan'))
        
        val_perc = outs.get('val_perc_loss', float('nan'))
        
        # GAN loss 출력 (GAN loss weight가 0보다 클 때만)
        if self.gan_loss_weight > 0:
            val_gen = outs.get('val_gen_loss', float('nan'))
            val_dis = outs.get('val_dis_loss', float('nan'))
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | "
                  f"Rec: {val_rec:.4f} | "
                  f"Gen: {val_gen:.4f} | "
                  f"Dis: {val_dis:.4f} | "
                  f"Perc: {val_perc:.4f} | "
                  f"Quant: {val_quant:.4f}\n")
        else:
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | "
                  f"Rec: {val_rec:.4f} | "
                  f"Perc: {val_perc:.4f} | "
                  f"Quant: {val_quant:.4f}\n")

    def on_validation_end(self) -> None:
        # Maybe put here example of video reconstructions?
        pass
    
    def configure_optimizers(self):
        """Configure optimizers with separate learning rates for generator and discriminator.
        
        This addresses the fundamental issue of GAN training instability by:
        1. Using lower learning rate for discriminator to prevent it from learning too fast
        2. Balancing the training speed between generator and discriminator
        
        This is the ROOT CAUSE solution, not just clipping values.
        """
        if self.gan_loss_weight > 0 and hasattr(self, 'gan_crit') and not isinstance(self.gan_crit, nn.Identity):
            # Separate parameters for generator (encoder/decoder/quant) and discriminator
            gen_params = []
            dis_params = []
            
            for name, param in self.named_parameters():
                if 'gan_crit' in name or 'disc' in name:
                    dis_params.append(param)
                else:
                    gen_params.append(param)
            
            # Get learning rate from hyperparameters
            # Current generator learning rate: 1e-3 (from config/tokenize.yaml)
            # If generator is not learning well, consider increasing to 2e-3 or 5e-3
            gen_lr = 1e-3  # Default learning rate
            if hasattr(self, 'hparams') and self.hparams is not None:
                if 'optimizer' in self.hparams:
                    opt_config = self.hparams['optimizer']
                    if isinstance(opt_config, dict) and 'init_args' in opt_config:
                        gen_lr = opt_config['init_args'].get('lr', 1e-3)
            
            # Use lower learning rate for discriminator to prevent it from learning too fast
            # This is a fundamental solution to GAN training instability
            # Discriminator learning 10x slower prevents it from overpowering the generator
            # Increased from 0.02 (50x) to 0.1 (10x) to allow discriminator to learn enough
            # to provide useful gradients to generator, while still being slower than generator
            dis_lr = gen_lr * 0.1  # Discriminator uses 10x lower learning rate
            
            # Get other optimizer arguments (weight_decay, etc.)
            opt_kwargs = {}
            if hasattr(self, 'hparams') and self.hparams is not None:
                if 'optimizer' in self.hparams:
                    opt_config = self.hparams['optimizer']
                    if isinstance(opt_config, dict) and 'init_args' in opt_config:
                        opt_kwargs = opt_config['init_args'].copy()
                        opt_kwargs.pop('lr', None)  # Remove lr as we set it per group
            
            # Create optimizer with parameter groups
            # This allows different learning rates for generator and discriminator
            if gen_params and dis_params:
                param_groups = [
                    {'params': gen_params, 'lr': gen_lr, **opt_kwargs},
                    {'params': dis_params, 'lr': dis_lr, **opt_kwargs},
                ]
                optim = self.optimizer(param_groups)
            else:
                # Fallback: use standard optimizer if separation failed
                optim = self.optimizer(self.parameters())
        else:
            # If GAN is not used, use standard optimizer
            optim = self.optimizer(self.parameters())
        
        # Add gradient clipping to prevent exploding gradients in GAN training
        # This is especially important when using mixed precision
        return {
            'optimizer': optim,
            'gradient_clip_val': 1.0,  # Clip gradients to max norm of 1.0
            'gradient_clip_algorithm': 'norm',
        }