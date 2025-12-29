from einops import rearrange
from lightning import LightningModule
from torch import Tensor
import torch
from torch.optim import AdamW
from torch.optim import Optimizer

from genie.action import LatentAction
from genie.dynamics import DynamicsModel
from genie.tokenizer import VideoTokenizer, REPR_TOK_ENC, REPR_TOK_DEC

from typing import Callable, Iterable

from genie.utils import default
import inspect

from pdb import set_trace as st

OptimizerCallable = Callable[[Iterable], Optimizer]

TEST_DESC = (
    ('space-time_attn', {
        'n_rep': 8,
        'n_head': 8,
        'd_head': 64,
        'd_inp': 64,
        'd_out': 512,
        'transpose': True,
    }),
    ('depth2spacetime_upsample', {
        'in_channels': 512,
        'kernel_size': 3,
        'out_channels': 3,
        'time_factor': 1,
        'space_factor': 4,
    })
)

class Genie(LightningModule):
    '''
    Generative Interactive Environment model from Bruce et al. (2024).
    The model is composed of:
    - A (pre-trained) video tokenizer based on the MaskVit-2 architecture.
    - A Latent Action model that build a (quantized) dictionary of latent actions
    - A Dynamics Model that predicts the next frame given the current frame and the latent action.
    '''
    def __init__(
        self,
        tokenizer : VideoTokenizer | None = None,
        tokenizer_checkpoint: str | None = None,
        optimizer : OptimizerCallable = AdamW,
        img_prompt : Tensor | None = None,
    ):
        super().__init__()
        
        # Pre-trained video tokenizer
        # Accept either a VideoTokenizer instance or a path to a checkpoint
        # provided via `tokenizer_checkpoint` (preferred) or `tokenizer`.
        # If a checkpoint path is provided, load the tokenizer using
        # the helper implemented on this class so loading behaviour is
        # centralized (and not delegated to the tokenizer class).
        if tokenizer is None and tokenizer_checkpoint is not None:
            tokenizer = self.__class__.load_tokenizer_from_checkpoint(tokenizer_checkpoint)

        # backward-compatible: if tokenizer is a string treat it as a path
        if isinstance(tokenizer, str):
            tokenizer = self.__class__.load_tokenizer_from_checkpoint(tokenizer)

        if tokenizer is None:
            raise ValueError('A pretrained tokenizer instance or tokenizer_checkpoint path must be provided')

        self.tokenizer = tokenizer
        
        # Model dimensions
        self.embed_dim = 64
        # Determine tokenizer/dynamics vocab sizes from tokenizer when available
        # tokenizer.quant.codebook_size holds the tokenizer token vocabulary size
        if hasattr(tokenizer, 'quant') and hasattr(tokenizer.quant, 'codebook_size'):
            self.tok_codebook = int(tokenizer.quant.codebook_size)
        else:
            self.tok_codebook = 8
        # action codebook size: use d_codebook (inferred from tokenizer) as default
        self.act_codebook = None
        
        self.enc_desc = tokenizer.get_enc('repr_tok')
        self.dec_desc = tokenizer.get_dec('repr_tok')
        # Robustly obtain d_codebook from tokenizer (support different tokenizer implementations)
        if hasattr(tokenizer, 'd_codebook'):
            self.d_codebook = int(tokenizer.d_codebook)
        else:
            q = getattr(tokenizer, 'quant', None)
            if q is not None and hasattr(q, 'codebook_dim'):
                self.d_codebook = int(q.codebook_dim)
            elif hasattr(tokenizer, 'hid_channels'):
                # fallback: some tokenizers expose hidden channel dim
                self.d_codebook = int(tokenizer.hid_channels)
            else:
                # final fallback to a sane default
                self.d_codebook = 8
        self.inp_channels = 3
        self.inp_shape = (64, 64)
        self.ker_size = 3
        self.n_embd = 64
        self.n_codebook = 1
        self.lfq_bias = True
        self.lfq_frac_sample = 1.
        self.lfq_commit_weight = 0.25
        self.lfq_entropy_weight = 0.1
        self.lfq_diversity_weight = 1.
        
        self.latent_action = LatentAction(
            self.enc_desc,
            self.dec_desc,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            inp_shape=self.inp_shape,
            ker_size=self.ker_size,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            lfq_bias=self.lfq_bias,
            lfq_frac_sample=self.lfq_frac_sample,
            lfq_commit_weight=self.lfq_commit_weight,
            lfq_entropy_weight=self.lfq_entropy_weight,
            lfq_diversity_weight=self.lfq_diversity_weight,
        )
        
        # After constructing latent_action we can set the action codebook size
        # The quant module exposes the full codebook size (number of entries)
        # which is what the dynamics model's action vocabulary should use.
        qmod = getattr(self.latent_action, 'quant', None)
        if qmod is not None and hasattr(qmod, 'codebook_size'):
            self.act_codebook = int(qmod.codebook_size)
        else:
            # fallback: use d_codebook (bits) as a conservative default
            self.act_codebook = int(self.d_codebook)

        # Build dynamics model with the correct token vocabulary size
        self.dynamics_model = DynamicsModel(
            desc=TEST_DESC,
            tok_vocab=self.tok_codebook,
            act_vocab=self.act_codebook,
            embed_dim=self.embed_dim,
        )
        
        self.optimizer = optimizer
        self.img_prompt = img_prompt
        
        self.save_hyperparameters()

    @torch.no_grad()
    def forward(
        self,
        prompt : Tensor,
        actions : Tensor,
        num_frames : int | None = None,
        steps_per_frame : int = 25,
    ) -> Tensor:
        '''
        Inference mode for the model. Generate videos from an initial
        image prompt and a sequence of latent actions.
        '''
        num_frames = default(num_frames, actions.shape[1])
        
        # Make sure prompt has correct shape for video
        match prompt.dim():
            case 3: pattern = 'b h w -> b 1 1 h w'
            case 4: pattern = 'b c h w -> b c 1 h w'
            case 5: pattern = 'b c t h w -> b c t h w'
            case _: raise ValueError('Prompt must have 3, 4 or 5 dimensions')
        
        # st()
        prompt = rearrange(prompt, pattern)
        
        # Tokenize the input prompt
        tokens = self.tokenizer.tokenize(prompt)
        
        for t in range(num_frames):
            # Predict the next frame based on the previous frame and the action
            new_tok = self.dynamics_model.generate(
                tokens,
                actions[:, :t],
                steps=steps_per_frame,
            )
            
            # Add the new frame to the video
            tokens = torch.stack((tokens, new_tok), dim=2)
            
        # Return the generated video
        video = self.tokenizer.decode(tokens)
        
        return video
    
    def compute_loss(self, video : Tensor) -> Tensor:
        # Tokenize the input video (returns quantized features and token indices)
        quant_video, token_idxs = self.tokenizer.tokenize(video)
        # quant_video: [1, 512 ,16, 32, 32] <- 인코더와 양자화만 거치는게 tokenize 이기 때문에 차원 맞음
        # token_idxs: [16, 32, 32]

        # st()
        # Normalize token indices to shape (b, t, h, w)
        if isinstance(token_idxs, (tuple, list)):
            token_idxs = token_idxs[0]
        token_idxs = torch.as_tensor(token_idxs)  # [16, 32, 32]
        if token_idxs.ndim == 3:
            # (t, h, w) -> (1, t, h, w)
            token_idxs = token_idxs.unsqueeze(0)  # 너가 실행됨 -> [1, 16, 32, 32] (batch size 추가)
        elif token_idxs.ndim == 5 and token_idxs.shape[1] == 1:  # skip
            # (b, 1, t, h, w) -> (b, t, h, w)
            token_idxs = token_idxs.squeeze(1)

        # Decode quantized features back to RGB video and feed that to LatentAction.
        # LatentAction expects a video tensor with channel dim (e.g., (b, 3, t, H, W)).
        try:
            rec_video = self.tokenizer.decode(quant_video)  # [1,3,16,64,64] <- 디코더 거친 후의 차원
        except Exception:
            # If decode fails, fall back to original video
            rec_video = video

        # st()
        # Extract latent actions from the reconstructed video
        act_id, act_loss, (act_rec_loss, act_q_loss) = self.latent_action(rec_video)  # act_id = [16]
        # st()

        # Ensure token indices and act_id are on the same device as the model
        device = getattr(self, 'device', None)
        if device is None:
            # fallback: infer device from model parameters
            try:
                device = next(self.parameters()).device
            except Exception:
                device = token_idxs.device if hasattr(token_idxs, 'device') else torch.device('cpu')

        token_idxs = token_idxs.to(device)  # [1, 16, 32, 32] (from 206번째 코드)
        act_id = act_id.to(device)  # 16

        # Validate token / action indices before sending to GPU ops to avoid
        # device-side asserts. Do checks on CPU to provide clear errors.
        try:
            tmin = int(token_idxs.min().item())  # 실행 -> 1007
            tmax = int(token_idxs.max().item())  # 실행 -> 1007
        except Exception:
            tmin, tmax = None, None
        try:
            amin = int(act_id.min().item())  # 실행 -> 275
            amax = int(act_id.max().item())  # 실행 -> 1014
        except Exception:
            amin, amax = None, None

        # Ensure integer dtype
        if not token_idxs.dtype in (torch.int64, torch.int32):
            token_idxs = token_idxs.long()  # skip
        if not act_id.dtype in (torch.int64, torch.int32):
            act_id = act_id.long()  # skip

        # Check ranges against dynamics model vocab sizes (if available)
        tok_vocab = getattr(self.dynamics_model, 'tok_vocab', None)  # 1024 (tok_emb의 맨 마지막 = [1024, 64])
        act_vocab = getattr(self.dynamics_model, 'act_vocab', None)  # 1024  (ack_emb의 맨 마지막 = [1, 1024, 1, 1, 64])
        if tok_vocab is not None and tmax is not None and tmax >= tok_vocab:  # skip
            # Move small diagnostics to CPU and raise helpful error
            raise RuntimeError(
                f"token_idxs contains out-of-range index (max={tmax}) >= tok_vocab ({tok_vocab})."
                f" token_idxs.min={tmin}, token_idxs.shape={tuple(token_idxs.shape)}"
            )
        if act_vocab is not None and amax is not None and amax >= act_vocab:  # skip
            raise RuntimeError(
                f"act_id contains out-of-range index (max={amax}) >= act_vocab ({act_vocab})."
                f" act_id.min={amin}, act_id.shape={tuple(act_id.shape)}"
            )

        # Build a mask on the same device to avoid device-mismatch inside DynamicsModel
        b, t, h, w = token_idxs.shape  # [1, 16, 32, 32] <- 206번째 줄에서 정한 것과 바뀌지 않음
        rate = torch.empty(1).uniform_(0.5, 1.0).item()
        mask = torch.distributions.Bernoulli(rate).sample((b, t, h, w)).bool().to(device)  # [1,17,32,32]
    
        # st()
        # Compute the next-frame prediction loss via the dynamics model using token indices
        # st()
        dyn_loss = self.dynamics_model.compute_loss(token_idxs, act_id, mask=mask)
        
        # Combine both latent action and dynamics model losses
        loss = act_loss + dyn_loss
        
        return loss, (
            ('act_loss', act_loss),
            ('dyn_loss', dyn_loss),
            ('act_rec_loss', act_rec_loss),
            ('act_q_loss', act_q_loss),
        )

    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the training loss
        loss, aux_losses = self.compute_loss(batch)
        
        # Log the training loss
        self.log_dict(
            {**{'train_loss' : loss}, **{f'train/{k}': v for k, v in aux_losses}},
            logger=True,
            on_step=True,
            sync_dist=True,
        )
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the validation loss
        loss, aux_losses = self.compute_loss(batch)
        
        # Log the training loss
        self.log_dict(
            {**{'val_loss' : loss}, **{f'val/{k}': v for k, v in aux_losses}},
            logger=True,
            on_step=True,
            sync_dist=True,
        )
        
        return loss
    
    def on_validation_end(self) -> None:
        '''Generate sample videos at the end of the validation loop'''
        
        # Generate a sample video from a given image prompt and random actions
        num_frames = 16
        prompt = default(self.img_prompt, torch.randn(1, 3, 64, 64))
        actions = torch.randint(0, self.latent_action.d_codebook, size=(num_frames,))
        
        video = self(
            prompt,
            actions,
            num_frames=num_frames,
            steps_per_frame=25
        )
        
        self.logger.experiment.add_video(
            f'Generated Video #1',
            video,
            global_step=self.global_step,
        )

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )
        
        return optim

    @classmethod
    def load_from_checkpoint(
        cls,
        tokenizer_checkpoint: str,
        tokenizer_kwargs: dict | None = None,
        map_location: str | None = None,
        **genie_kwargs,
    ):
        """Utility to create a `Genie` instance from a pretrained
        `VideoTokenizer` checkpoint file. This method will attempt to
        reconstruct the tokenizer from the checkpoint's saved
        hyper-parameters and state-dict. If the checkpoint does not
        contain hyper-parameters, `tokenizer_kwargs` can be provided.

        Args:
            tokenizer_checkpoint: path to the tokenizer .ckpt file
            tokenizer_kwargs: dict overrides for tokenizer constructor
            map_location: passed to torch.load
            genie_kwargs: additional args passed to `Genie` constructor
        """
        from genie.tokenizer import VideoTokenizer

        tokenizer_kwargs = dict(tokenizer_kwargs or {})

        # Load checkpoint dict
        ckpt = torch.load(tokenizer_checkpoint, map_location=map_location or 'cpu')

        # Try to recover tokenizer init kwargs from checkpoint hyper-parameters
        hparams = {}
        for key in ('hyper_parameters', 'hyper_parameters_saved', 'hparams', 'hparams_name'):
            if key in ckpt and isinstance(ckpt[key], dict):
                hparams.update(ckpt[key])
                break

        # Merge hparams with explicit overrides (explicit wins)
        hparams.update(tokenizer_kwargs)

        # Sanitize checkpoint hparams: remove private keys (like jsonargparse internals)
        hparams = {k: v for k, v in hparams.items() if not str(k).startswith('_')}

        # Filter hparams to only those accepted by VideoTokenizer.__init__
        try:
            sig_params = set(inspect.signature(VideoTokenizer.__init__).parameters.keys()) - {'self'}
            filtered_hparams = {k: v for k, v in hparams.items() if k in sig_params}
        except Exception:
            filtered_hparams = hparams

        # Instantiate tokenizer (best-effort). If that fails, fall back to safe defaults.
        try:
            tokenizer = VideoTokenizer(**filtered_hparams)
        except Exception:
            safe_d = filtered_hparams.get('d_codebook', 8)
            try:
                safe_d = int(safe_d)
            except Exception:
                safe_d = 8
            # clamp to a reasonable number of bits to avoid enormous codebooks
            safe_d = max(1, min(safe_d, 12))
            tokenizer = VideoTokenizer(
                enc_desc=VideoTokenizer.REPR_TOK_ENC if hasattr(VideoTokenizer, 'REPR_TOK_ENC') else None,
                dec_desc=VideoTokenizer.REPR_TOK_DEC if hasattr(VideoTokenizer, 'REPR_TOK_DEC') else None,
                d_codebook=safe_d,
            )

        # Load state dict if present
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', None))
        if state_dict is None:
            # Some checkpoints store parameters at top-level; try using ckpt itself
            state_dict = {k: v for k, v in ckpt.items() if k.startswith('enc_layers') or k.startswith('dec_layers') or k.endswith('.weight')}

        if state_dict:
            try:
                tokenizer.load_state_dict(state_dict, strict=False)
            except Exception:
                # Try mapping keys: strip "model." prefix if present
                new_sd = {k.replace('model.', ''): v for k, v in state_dict.items()}
                tokenizer.load_state_dict(new_sd, strict=False)

        # Finally, construct Genie with the loaded tokenizer
        return cls(tokenizer=tokenizer, **genie_kwargs)

    @classmethod
    def load_tokenizer_from_checkpoint(
        cls,
        tokenizer_checkpoint: str,
        tokenizer_kwargs: dict | None = None,
        map_location: str | None = None,
    ):
        """Load and return a VideoTokenizer instance from a checkpoint.
        This is a helper used by `__init__` so the loading logic is
        centralized in `Genie`.
        """
        from genie.tokenizer import VideoTokenizer

        tokenizer_kwargs = dict(tokenizer_kwargs or {})

        ckpt = torch.load(tokenizer_checkpoint, map_location=map_location or 'cpu')

        hparams = {}
        for key in ('hyper_parameters', 'hyper_parameters_saved', 'hparams', 'hparams_name'):
            if key in ckpt and isinstance(ckpt[key], dict):
                hparams.update(ckpt[key])
                break

        hparams.update(tokenizer_kwargs)

        # Sanitize and filter hparams before instantiating
        hparams = {k: v for k, v in hparams.items() if not str(k).startswith('_')}
        try:
            sig_params = set(inspect.signature(VideoTokenizer.__init__).parameters.keys()) - {'self'}
            filtered_hparams = {k: v for k, v in hparams.items() if k in sig_params}
        except Exception:
            filtered_hparams = hparams

        try:
            tokenizer = VideoTokenizer(**filtered_hparams)
        except Exception:
            safe_d = filtered_hparams.get('d_codebook', 8)
            try:
                safe_d = int(safe_d)
            except Exception:
                safe_d = 8
            safe_d = max(1, min(safe_d, 12))
            tokenizer = VideoTokenizer(enc_desc=REPR_TOK_ENC, dec_desc=REPR_TOK_DEC, d_codebook=safe_d)

        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', None))
        if state_dict is None:
            state_dict = {k: v for k, v in ckpt.items() if k.startswith('enc_layers') or k.startswith('dec_layers') or k.endswith('.weight')}

        if state_dict:
            try:
                tokenizer.load_state_dict(state_dict, strict=False)
            except Exception:
                new_sd = {k.replace('model.', ''): v for k, v in state_dict.items()}
                tokenizer.load_state_dict(new_sd, strict=False)

        return tokenizer