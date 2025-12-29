import torch
import torch.nn as nn
from typing import Literal
from math import inf, pi, prod
from torch import Tensor, softmax
from torch.nn.functional import cross_entropy

from einops import pack, rearrange, unpack
from einops.layers.torch import Rearrange

from genie.utils import Blueprint, default
from genie.module import parse_blueprint

from pdb import set_trace as st

class DynamicsModel(nn.Module):
    '''Dynamics Model (DM) used to predict future video frames
    given the history of past video frames and the latent actions.
    The DM model employs the Mask-GIT architecture as introduced
    in Chang et al. (2022).
    '''
    
    def __init__(
        self,
        desc: Blueprint,
        tok_vocab: int,
        act_vocab: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        
        self.dec_layers, self.ext_kw = parse_blueprint(desc)
        
        self.head = nn.Linear(embed_dim, tok_vocab)
        
        self.tok_emb = nn.Embedding(tok_vocab, embed_dim)
        self.act_emb = nn.Sequential(
            nn.Embedding(act_vocab, embed_dim),
            Rearrange('b t d -> b t 1 1 d'),
        )
        
        self.tok_vocab = tok_vocab
        self.act_vocab = act_vocab
        self.embed_dim = embed_dim
        
    def forward(
        self,
        tokens : Tensor,
        act_id : Tensor,
    ) -> Tensor:
        '''
        Predicts the next video token based on the previous tokens
        '''

        # Ensure token indices are integer dtype
        if not tokens.dtype in (torch.int64, torch.int32):
            tokens = tokens.long()
        if not act_id.dtype in (torch.int64, torch.int32):
            act_id = act_id.long()

        
        if act_id.dim() == 1:
            act_id = act_id.unsqueeze(0)

        # Embed tokens and actions (action embedding module wraps an
        # Embedding followed by a Rearrange). We access the raw action
        # embedding (the first module) so we can adapt its shape to the
        # token embedding shape below.
        tok_emb = self.tok_emb(tokens)  # [1, 16, 32, 32, 64]
        act_emb_raw = self.act_emb[0](act_id)  #[1, 16, 64]
        
        # Case A: spatial token input (b, t, h, w) -> tok_emb shape (b,t,h,w,d)
        if tok_emb.dim() == 5:  # 실행
            b, t, h, w, d = tok_emb.shape
            # act_emb_raw should be (b, t, d). Convert to (b, t, 1, 1, d)
            # so it broadcasts with the spatial token embedding.
            if act_emb_raw.dim() == 2:
                # (b, t) -> (b, t, d) handled above; if batch missing, it
                # was expanded earlier.
                pass
            if act_emb_raw.shape[1] != t:
                raise RuntimeError(f'act_id length ({act_emb_raw.shape[1]}) does not match token time dim ({t})')
            act_emb = rearrange(act_emb_raw, 'b t d -> b t 1 1 d')
            # Broadcast-add and keep spatial dims: (b, t, h, w, d)
            tokens = tok_emb + act_emb

        # Case B: flattened token input (b, seq) -> tok_emb shape (b, seq, d)
        elif tok_emb.dim() == 3:  # skip
            b, seq, d = tok_emb.shape
            # act_emb_raw might be (b, seq, d) already or (b, t, d) where
            # t divides seq (t * spatial = seq). Try to align them.
            if act_emb_raw.shape[1] == seq:
                tokens = tok_emb + act_emb_raw
            else:
                # Attempt to expand per-time action embedding across the
                # spatial positions if compatible
                t = act_emb_raw.shape[1]
                if seq % t == 0:
                    reps = seq // t
                    act_exp = act_emb_raw.repeat_interleave(reps, dim=1)
                    tokens = tok_emb + act_exp
                else:
                    raise RuntimeError(f'Cannot align action embeddings (len {act_emb_raw.shape[1]}) with token sequence (len {seq})')

        # Case C: sometimes embeddings come as (b, t, n, d) where n == h*w
        # (e.g. packed spatial positions). Try to infer spatial dims when
        # possible, otherwise treat as flattened sequence.
        elif tok_emb.dim() == 4:  # skip
            b, t, n, d = tok_emb.shape
            s = int(n ** 0.5)
            if s * s == n:
                # reshape to (b, t, h, w, d)
                tokens = rearrange(tok_emb, 'b t (h w) d -> b t h w d', h=s, w=s)
            else:
                # fallback: flatten spatial positions into sequence
                tokens = rearrange(tok_emb, 'b t n d -> b (t n) d')
        else:
            raise RuntimeError(f'Unexpected token embedding ndim={tok_emb.dim()}')
        
        # Predict the next video token based on previous tokens and actions
        # If we have spatial tokens in channel-last form (b, t, h, w, d),
        # convert to channel-first (b, d, t, h, w) because attention modules
        # expect channels first.
        converted_to_channel_first = False
        if tokens.dim() == 5:
            tokens = rearrange(tokens, 'b t h w d -> b d t h w')
            converted_to_channel_first = True

        # tokens examples after conversion: (b, d, t, h, w)
        for idx, (dec, has_ext) in enumerate(zip(self.dec_layers, self.ext_kw)):
            try:
                # print(f"[Dynamics.forward] before dec {idx} ({dec.__class__.__name__}) tokens.shape={tuple(tokens.shape)} dtype={tokens.dtype}")
                pass
            except Exception:
                pass

            # Pre-decode alignment: find the decoder's expected input feature
            # size. Prefer attention/LayerNorm declared dims (d_inp / norm),
            # otherwise fall back to first Conv3d/Linear encountered.
            expected_in = None
            preproj_name = None
            # Try to read declared input dim from module attributes
            # Prefer the first LayerNorm's normalized_shape found anywhere
            # inside the decoder (this usually indicates the attention
            # feature dimension). Fall back to dec.d_inp attribute.
            ln_found = None
            for m in dec.modules():
                if isinstance(m, nn.LayerNorm):
                    ln_found = m
                    break
            if ln_found is not None and hasattr(ln_found, 'normalized_shape'):
                ns = ln_found.normalized_shape
                if isinstance(ns, (tuple, list)) and len(ns) >= 1:
                    expected_in = int(ns[0])
                elif isinstance(ns, int):
                    expected_in = int(ns)
            elif hasattr(dec, 'd_inp') and dec.d_inp is not None:
                expected_in = int(dec.d_inp)

            # Channel-first 5D case (b, c, t, h, w)
            if tokens.dim() == 5:
                in_ch = tokens.shape[1]
                # If we don't have expected_in yet, look for first Conv3d
                if expected_in is None:
                    for m in dec.modules():
                        if isinstance(m, nn.Conv3d):
                            expected_in = m.in_channels
                            break
                # If still unknown, skip preproj
                if expected_in is not None and in_ch != expected_in:
                    preproj_name = f'_preproj_conv3d_{idx}'
                    if not hasattr(self, preproj_name):
                        mod = nn.Conv3d(in_ch, expected_in, kernel_size=1)
                        self.add_module(preproj_name, mod)
                        getattr(self, preproj_name).to(tokens.device)
                    try:
                        # print(f"[Dynamics.forward] applying preproj {preproj_name}: {in_ch} -> {expected_in}")
                        pass
                    except Exception:
                        pass
                    tokens = getattr(self, preproj_name)(tokens)  # [1, 512, 16, 32, 32]
                    # st()

            # Flattened/sequence 3D case (b, seq, d)
            elif tokens.dim() == 3:
                in_d = tokens.shape[-1]
                if expected_in is None:
                    for m in dec.modules():
                        if isinstance(m, nn.Linear):
                            expected_in = m.in_features
                            break
                if expected_in is not None and in_d != expected_in:
                    preproj_name = f'_preproj_linear_{idx}'
                    if not hasattr(self, preproj_name):
                        mod = nn.Linear(in_d, expected_in)
                        self.add_module(preproj_name, mod)
                        getattr(self, preproj_name).to(tokens.device)
                    try:
                        pass
                        # print(f"[Dynamics.forward] applying preproj {preproj_name}: {in_d} -> {expected_in}")
                    except Exception:
                        pass
                    tokens = getattr(self, preproj_name)(tokens)

            # st()
            tokens = dec(tokens)  # [1, 3, 16, 128, 128]
            # st()

            # If a decoder changed the feature/channel dimension, project
            # it back to `self.embed_dim`. Create and register a small
            # projection module lazily so we don't modify other modules.
            # Support both channel-first 3D conv inputs (b, c, t, h, w)
            # and flattened sequence inputs (b, seq, d).
            proj_name = None
            if tokens.dim() == 5:  # 실행
                in_ch = tokens.shape[1]
                if in_ch != self.embed_dim:  # 실행
                    proj_name = f'_proj_conv3d_{idx}'
                    if not hasattr(self, proj_name):  # 실행
                        mod = nn.Conv3d(in_ch, self.embed_dim, kernel_size=1)
                        # register and move to same device
                        self.add_module(proj_name, mod)
                        getattr(self, proj_name).to(tokens.device)
                    # apply conv projection
                    tokens = getattr(self, proj_name)(tokens)  # [1, 64, 16, 128, 128]

            elif tokens.dim() == 3:
                in_d = tokens.shape[-1]
                if in_d != self.embed_dim:
                    proj_name = f'_proj_linear_{idx}'
                    if not hasattr(self, proj_name):
                        mod = nn.Linear(in_d, self.embed_dim)
                        self.add_module(proj_name, mod)
                        getattr(self, proj_name).to(tokens.device)
                    tokens = getattr(self, proj_name)(tokens)
            try:
                pass  # 실행
                # print(f"[Dynamics.forward] after  dec {idx} ({dec.__class__.__name__}) tokens.shape={tuple(tokens.shape)} dtype={tokens.dtype}")
            except Exception:
                pass

        # st()
        # Convert back to channel-last (b, t, h, w, d) if we converted earlier
        if converted_to_channel_first and tokens.dim() == 5:
            tokens = rearrange(tokens, 'b d t h w -> b t h w d')  # [1, 64, 16, 128, 128] -> [1, 16, 128, 128, 64]

        # st()
        # Compute the next token probability
        logits = self.head(tokens)  # [1, 16, 128, 128, 1024]
        try:
            pass  # 실행
            # print(f"[Dynamics.forward] logits shape={tuple(logits.shape)} head.weight.shape={tuple(self.head.weight.shape)}")
        except Exception:
            pass
        
        return logits, logits[:, -1]
    
    def compute_loss(
        self,
        tokens : Tensor,
        act_id : Tensor,
        mask : Tensor | None = None,
        fill : float = 0.,
    ) -> Tensor:
        
        b, t, h, w = tokens.shape  # [1, 16, 32, 32]
        
        # Create Bernoulli mask if not provided
        if mask is None:
            # random rate in [0.5, 1]
            rate = torch.empty(1).uniform_(0.5, 1).item()
            mask = (torch.rand((b, t, h, w), device=tokens.device) < rate).bool()
        else:
            # move provided mask to tokens device
            mask = mask.to(tokens.device)  # 실행
        
        # Mask tokens b.ased on external mask as training signal
        tokens = torch.masked_fill(tokens, mask, fill).to(device=tokens.device)  # mask와 tokens의 shape이 [1, 16, 32, 32]로 동일하기 때문에 masked_fill 적용 가능
        
        # Compute the model prediction for the next token
        # st()
        logits, _ = self(tokens, act_id.detach())  # [1, 16, 128, 128, 1024]
        
        # Only compute loss on the tokens that were masked
        # st()
        # Align mask and token targets with logits spatial resolution.
        # logits: (b, t, H, W, d)
        # mask:   (b, t, h, w)
        bL, tL, HL, WL, _ = logits.shape
        bM, tM, h, w = mask.shape
        # st()

        if (bL, tL) != (bM, tM):
            raise RuntimeError(f'Batch/time size mismatch between logits {bL,tL} and mask {bM,tM}')

        if HL == h and WL == w:
            mask_logits = mask
            tokens_for_logits = tokens
        else:
            # Prefer integer repeat_interleave when possible
            if HL % h == 0 and WL % w == 0:
                sh = HL // h
                sw = WL // w
                mask_logits = mask.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)
                tokens_for_logits = tokens.repeat_interleave(sh, dim=2).repeat_interleave(sw, dim=3)
            else:
                # Fallback: nearest interpolation on (b*t,1,h,w)
                mf = mask.view(bM * tM, 1, h, w).float()
                mf_up = torch.nn.functional.interpolate(mf, size=(HL, WL), mode='nearest')
                mask_logits = mf_up.view(bM, tM, HL, WL).bool()

                tf = tokens.view(bM * tM, 1, h, w).float()
                tf_up = torch.nn.functional.interpolate(tf, size=(HL, WL), mode='nearest')
                tokens_for_logits = tf_up.view(bM, tM, HL, WL).to(tokens.dtype)

        logits = logits[mask_logits]
        tokens = tokens_for_logits[mask_logits]
        
        # Rearrange tokens to have shape (batch * seq_len, vocab_size)
        logits = rearrange(logits, '... d -> (...) d')
        target = rearrange(tokens, '...   -> (...)')
        
        # Compute the cross-entropy loss between the predicted and actual tokens
        loss = cross_entropy(logits, target)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens : Tensor,
        act_id : Tensor,
        steps : int = 10,
        which : Literal['linear', 'cosine', 'arccos'] = 'linear',
        temp : float = 1.,
        topk : int = 50,
        masked_tok : int = 0,
    ) -> Tensor:
        '''
        Given past token and action history, predicts the next token
        via the Mask-GIT sampling technique.
        '''
        b, t, h, w = tokens.shape
        # st()
                
        # Get the sampling schedule
        schedule = self.get_schedule(steps, shape=(h, w), which=which)
        
        # Initialize a fully active mask to signal that all the tokens
        # must receive a prediction. The mask will be updated at each
        # step based on the sampling schedule.
        mask = torch.ones(b, h, w, dtype=bool, device=tokens.device)  # (b, h, w) = (b, 32, 32)
        code = torch.full((b, h, w), masked_tok, device=tokens.device) # (b, h, w)
        mock = torch.zeros(b, dtype=int, device=tokens.device)  # (b,)
        
        tok_id, ps = pack([tokens, code], 'b * h w')   # tok_id=(b,3,h,w) , len(ps)=2
        act_id, _ = pack([act_id, mock], 'b *')  # (b, 3)
        # st()
        
        for num_tokens in schedule:
            # If no more tokens to predict, return
            if mask.sum() == 0: break
            
            # Get prediction for the next tokens
            _, logits = self(tok_id, act_id)

            # logits may come as (b, H, W, d) when t=1 was collapsed; normalize to (b, t, H, W, d)
            if logits.dim() == 4:
                logits = logits.unsqueeze(1)

            # st()
            # Downsample logits spatially to match input token resolution (h_tok, w_tok)
            # Align logits down to tokens to keep token/code/mask shapes consistent inside generate.
            H_log, W_log = logits.shape[-3], logits.shape[-2]
            h_tok, w_tok = tokens.shape[-2], tokens.shape[-1]
            if (H_log, W_log) != (h_tok, w_tok):
                log_b, log_t = logits.shape[0], logits.shape[1]
                logits_flat = rearrange(logits, 'b t H W d -> (b t) d H W')
                logits_ds = torch.nn.functional.interpolate(
                    logits_flat, size=(h_tok, w_tok), mode='nearest')
                logits = rearrange(logits_ds, '(b t) d h w -> b t h w d', b=log_b, t=log_t)  # [1, 1, 32, 32, 1024]
            # st()
            
            # Refine the mask based on the sampling schedule
            prob = softmax(logits / temp, dim=-1)  # (b, 1, h, w, 1024)
            prob, ps = pack([prob], '* d')
            pred = torch.multinomial(prob, num_samples=1)
            conf = torch.gather(prob, -1, pred)
            conf = unpack(conf, ps, '* d')[0].squeeze()
            
            # We paint the k-tokens with highest confidence, excluding the
            # already predicted tokens from the mask
            conf = conf.view_as(mask)  # (b, 32, 32)
            conf[~mask.bool()] = -inf
            # st()
            idxs = torch.topk(conf.view(b, -1), k=num_tokens, dim=-1).indices  # [b,1]
            # st()
            
            code, cps = pack([code], 'b *')
            mask, mps = pack([mask], 'b *')
            pred = pred.view(b, -1)
            
            # Fill the code with sampled tokens & update mask
            vals = torch.gather(pred, -1, idxs).to(code.dtype)
            code.scatter_(1, idxs, vals)
            mask.scatter_(1, idxs, False)
            
            code = unpack(code, cps, 'b *')[0]  # (b, 32, 32)
            mask = unpack(mask, mps, 'b *')[0]  # (b, 32, 32)
            
            pred_tok, ps = pack([tokens, code], 'b * h w')  # (b, 3, 32, 32)
            
        assert mask.sum() == 0, f'Not all tokens were predicted. {mask.sum()} tokens left.'
        return pred_tok
    
    def get_schedule(
        self,
        steps: int,
        shape: tuple[int, int],
        which: Literal['linear', 'cosine', 'arccos'] = 'linear',
    ) -> Tensor:
        n = prod(shape)
        t = torch.linspace(1, 0, steps)
        
        
        match which:
            case 'linear':
                s = 1 - t
            case 'cosine':
                s = torch.cos(t * pi * .5)
            case 'arccos':
                s = torch.acos(t) / (pi * .5)
            case _:
                raise ValueError(f'Unknown schedule type: {which}')
        
        # Fill the schedule with the ratio of tokens to predict
        schedule = (s / s.sum()) * n
        schedule = schedule.round().int().clamp(min=1)
        
        # Make sure that the total number of tokens to predict is
        # equal to the vocab size
        schedule[-1] += n - schedule.sum()
        
        return schedule