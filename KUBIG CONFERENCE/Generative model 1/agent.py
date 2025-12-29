import torch
from torch import optim
import lightning as L
from lightning.pytorch.cli import LightningCLI

from genie.tokenizer import VideoTokenizer
from genie.action import LatentAction, REPR_ACT_ENC, REPR_ACT_DEC
from genie.dynamics import DynamicsModel
from genie.genie import TEST_DESC
from genie.module.agent import WorldModelAgent
from genie.dataset import LightningPlatformer2D
from genie.utils import default

from pdb import set_trace as st

def _load_tokenizer(ckpt: str) -> VideoTokenizer:
	tok = VideoTokenizer.load_from_checkpoint(ckpt, map_location='cpu')
	tok.eval()
	for p in tok.parameters():
		p.requires_grad = False
	return tok


def _extract_prefixed_state(state_dict: dict, prefix: str) -> dict:
	"""Return a state_dict with `prefix.` removed from keys."""
	plen = len(prefix) + 1
	return {k[plen:]: v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}


def _summarize_state(prefix: str, sd: dict) -> None:
	print(f"\n[STATE SUMMARY] prefix='{prefix}' total={len(sd)}")
	# print a few keys and shapes
	shown = 0
	for k, v in sd.items():
		try:
			print(f"  - {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', 'n/a')}")
		except Exception:
			print(f"  - {k}: type={type(v)}")
		shown += 1
		if shown >= 10:
			break
	# print specific commonly mismatching params when present
	for key in (
		'proj_in.conv3d.weight',
		'proj_in.conv3d.bias',
		'proj_out.conv3d.weight',
		'proj_out.conv3d.bias',
		'dec_layers.0.space_attn.norm.weight',
		'dec_layers.0.space_attn.norm.bias',
	):
		if key in sd:
			v = sd[key]
			try:
				print(f"  * {key}: shape={tuple(v.shape)}")
			except Exception:
				pass


def _load_latent_action(ckpt: str) -> LatentAction:
	state = torch.load(ckpt, map_location='cpu')
	hparams = state.get('hyper_parameters', {})
	# fall back to defaults if missing
	enc_desc = hparams.get('action_enc_desc', hparams.get('enc_desc', REPR_ACT_ENC))
	dec_desc = hparams.get('action_dec_desc', hparams.get('dec_desc', REPR_ACT_DEC))
	d_codebook = default(hparams.get('action_d_codebook'), default(hparams.get('d_codebook'), 10))
	n_codebook = hparams.get('action_n_codebook', hparams.get('n_codebook', 1))
	latent_dim = hparams.get('action_latent_dim', hparams.get('latent_dim', None))
	model = LatentAction(
		enc_desc=enc_desc,
		dec_desc=dec_desc,
		d_codebook=d_codebook,
		inp_channels=hparams.get('inp_channels', 3),
		inp_shape=hparams.get('inp_shape', (64, 64)),
		ker_size=hparams.get('ker_size', 3),
		n_embd=hparams.get('n_embd', 256),
		n_codebook=n_codebook,
		latent_dim=latent_dim,
		lfq_bias=hparams.get('lfq_bias', True),
		lfq_frac_sample=hparams.get('lfq_frac_sample', 1.),
		lfq_commit_weight=hparams.get('lfq_commit_weight', 0.25),
		lfq_entropy_weight=hparams.get('lfq_entropy_weight', 0.1),
		lfq_diversity_weight=hparams.get('lfq_diversity_weight', 1.),
	)
	missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
	if missing or unexpected:
		print(f"[LatentAction] missing keys: {missing}, unexpected keys: {unexpected}")
	model.eval()
	for p in model.parameters():
		p.requires_grad = False
	return model


def _load_dynamics(ckpt: str, tok_vocab: int, act_vocab: int, embed_dim: int = 64) -> DynamicsModel:
	state = torch.load(ckpt, map_location='cpu')
	hparams = state.get('hyper_parameters', {})
	desc = hparams.get('desc', TEST_DESC)
	model = DynamicsModel(
		desc=desc,
		tok_vocab=tok_vocab,
		act_vocab=act_vocab,
		embed_dim=embed_dim,
	)
	missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
	if missing or unexpected:
		print(f"[Dynamics] missing keys: {missing}, unexpected keys: {unexpected}")
	model.eval()
	for p in model.parameters():
		p.requires_grad = False
	return model


def _load_latent_and_dynamics_from_world(
	world_ckpt: str,
	tokenizer: VideoTokenizer,
	default_desc=TEST_DESC,
	embed_dim: int = 64,
):
	state = torch.load(world_ckpt, map_location='cpu')
	hparams = state.get('hyper_parameters', {})
	sd = state['state_dict']
	# quick overview of world ckpt contents
	print("\n[World CKPT] keys sample:")
	for i, k in enumerate(sd.keys()):
		if i >= 10:
			break
		print(f"  - {k}")

	# Latent action setup: infer decoder dims from checkpoint and filter matching params
	la_state = _extract_prefixed_state(sd, 'latent_action')
	_summarize_state('latent_action', la_state)

	# Infer decoder feature dim and head size from checkpoint
	dec_dim = None
	dec_head = None
	num_dec_layers = 0
	ln_key = 'dec_layers.0.space_attn.norm.weight'
	freq_key = 'dec_layers.0.space_attn.embed.freq'
	if ln_key in la_state:
		try:
			dec_dim = int(la_state[ln_key].shape[0])
		except Exception:
			dec_dim = None
	if freq_key in la_state:
		try:
			# RotaryEmbedding 2d uses length d_head/2
			dec_head = int(la_state[freq_key].shape[0]) * 2
		except Exception:
			dec_head = None
	# Count decoder layers by scanning keys
	for k in la_state.keys():
		if k.startswith('dec_layers.'):
			try:
				idx = int(k.split('.')[1])
				num_dec_layers = max(num_dec_layers, idx + 1)
			except Exception:
				pass

	base_enc = hparams.get('enc_desc', REPR_ACT_ENC)
	base_dec = hparams.get('dec_desc', REPR_ACT_DEC)
	# Override decoder blueprint to align with checkpoint if we inferred dims
	if dec_dim is not None or dec_head is not None:
		# default heads: keep 8 unless otherwise specified in hparams
		n_head = 8
		if isinstance(base_dec, tuple) and len(base_dec) > 0:
			# build single block with corrected params and repetitions
			blk_name, blk_params = base_dec[0]
			blk_params = dict(blk_params)
			if dec_dim is not None:
				blk_params['d_inp'] = dec_dim
			if dec_head is not None:
				blk_params['d_head'] = dec_head
			blk_params['n_head'] = blk_params.get('n_head', n_head)
			blk_params['n_rep'] = num_dec_layers or blk_params.get('n_rep', 8)
			dec_desc = ((blk_name, blk_params),)
		else:
			dec_desc = (('space-time_attn', {
				'n_rep': num_dec_layers or 8,
				'n_head': n_head,
				'd_head': dec_head or 64,
				'd_inp': dec_dim or 512,
			}),)
	else:
		dec_desc = base_dec

	la_kwargs = dict(
		enc_desc=base_enc,
		dec_desc=dec_desc,
		d_codebook=default(hparams.get('d_codebook'), 10),
		inp_channels=hparams.get('inp_channels', 3),
		inp_shape=hparams.get('inp_shape', (64, 64)),
		ker_size=hparams.get('ker_size', 3),
		n_embd=hparams.get('n_embd', 256),
		n_codebook=hparams.get('n_codebook', 1),
		lfq_bias=hparams.get('lfq_bias', True),
		lfq_frac_sample=hparams.get('lfq_frac_sample', 1.),
		lfq_commit_weight=hparams.get('lfq_commit_weight', 0.25),
		lfq_entropy_weight=hparams.get('lfq_entropy_weight', 0.1),
		lfq_diversity_weight=hparams.get('lfq_diversity_weight', 1.),
	)
	latent_action = LatentAction(**la_kwargs)

	# Helper: filter state dict to only keys with matching shapes
	def _filter_matching_params(model, state):
		model_sd = model.state_dict()
		filtered = {}
		mismatched = []
		for k, v in state.items():
			if k in model_sd:
				try:
					if tuple(v.shape) == tuple(model_sd[k].shape):
						filtered[k] = v
					else:
						mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
				except Exception:
					mismatched.append((k, 'n/a', 'n/a'))
		if len(mismatched) > 0:
			print(f"[LatentAction/world] skipped {len(mismatched)} mismatched params (showing up to 10):")
			for i, (k, shp_ckpt, shp_model) in enumerate(mismatched[:10]):
				print(f"  - {k}: ckpt={shp_ckpt} model={shp_model}")
		return filtered

	la_state_filtered = _filter_matching_params(latent_action, la_state)
	missing, unexpected = latent_action.load_state_dict(la_state_filtered, strict=False)
	if missing or unexpected:
		print(f"[LatentAction/world] missing keys: {missing}, unexpected keys: {unexpected}")
	latent_action.eval()
	for p in latent_action.parameters():
		p.requires_grad = False

	# Dynamics setup
	tok_vocab = int(getattr(tokenizer.quant, 'codebook_size', 512))
	act_vocab = int(getattr(latent_action.quant, 'codebook_size', 256))
	desc = hparams.get('desc', default_desc)
	dyn_state = _extract_prefixed_state(sd, 'dynamics_model')
	_summarize_state('dynamics_model', dyn_state)
	# Infer embed_dim from head.weight if present
	inferred_embed = None
	if 'head.weight' in dyn_state:
		try:
			inferred_embed = int(dyn_state['head.weight'].shape[1])
		except Exception:
			inferred_embed = None
	dyn_embed_dim = inferred_embed or embed_dim
	dyn = DynamicsModel(desc=desc, tok_vocab=tok_vocab, act_vocab=act_vocab, embed_dim=dyn_embed_dim)

	# Filter matching params before loading
	def _filter_matching_params_dyn(model, state):
		model_sd = model.state_dict()
		filtered = {}
		mismatched = []
		for k, v in state.items():
			if k in model_sd:
				try:
					if tuple(v.shape) == tuple(model_sd[k].shape):
						filtered[k] = v
					else:
						mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
				except Exception:
					mismatched.append((k, 'n/a', 'n/a'))
		if len(mismatched) > 0:
			print(f"[Dynamics/world] skipped {len(mismatched)} mismatched params (showing up to 10):")
			for i, (k, shp_ckpt, shp_model) in enumerate(mismatched[:10]):
				print(f"  - {k}: ckpt={shp_ckpt} model={shp_model}")
		return filtered

	dyn_state_filtered = _filter_matching_params_dyn(dyn, dyn_state)
	missing_d, unexpected_d = dyn.load_state_dict(dyn_state_filtered, strict=False)
	if missing_d or unexpected_d:
		print(f"[Dynamics/world] missing keys: {missing_d}, unexpected keys: {unexpected_d}")
	dyn.eval()
	for p in dyn.parameters():
		p.requires_grad = False

	return latent_action, dyn


class AgentModule(L.LightningModule):
	def validation_step(self, batch, batch_idx):
		video = batch
		if video.dim() == 4:
			video = video.unsqueeze(0)
		elif video.dim() == 5:
			if video.shape[2] == 3 or video.shape[2] <= 4:
				pass
			elif video.shape[1] == 3 or video.shape[1] <= 4:
				video = video.permute(0, 2, 1, 3, 4).contiguous()
		start_clip = video[:, :2]
		start_clip = start_clip.to(self.device)
		self._ensure_world_ready(start_clip)
		start_tokens = self.agent.env.encode_tokens(start_clip)
		loss, stats = self.agent.loss(start_tokens)
		self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
		for k, v in stats.items():
			self.log(f'val_{k}', v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
		return loss
	def __init__(
		self,
		tokenizer_ckpt: str,
		world_ckpt: str | None = None,
		latent_action_ckpt: str | None = None,
		dynamics_ckpt: str | None = None,
		horizon: int = 4,
		gamma: float = 0.99,
		entropy_coef: float = 0.01,
		lr: float = 3e-4,
	) -> None:
		super().__init__()

		if world_ckpt is None and (latent_action_ckpt is None or dynamics_ckpt is None):
			raise RuntimeError('Provide either world_ckpt (combined) or both latent_action_ckpt and dynamics_ckpt')

		tokenizer = _load_tokenizer(tokenizer_ckpt)

		self.lr = lr
		self.save_hyperparameters()

		# Load world or separate checkpoints and initialize agent
		latent_action, dynamics = _load_latent_and_dynamics_from_world(world_ckpt, tokenizer)
		
		self.agent = WorldModelAgent(
			tokenizer=tokenizer,
			latent_action=latent_action,
			dynamics=dynamics,
			horizon=horizon,
			gamma=gamma,
			entropy_coef=entropy_coef,
		)
  
		# Flag to ensure we move frozen world components to the correct device/dtype lazily
		self._world_ready = False

	def _ensure_world_ready(self, ref_tensor):
		"""Move frozen world components to the ref tensor's device once."""
		if getattr(self, '_world_ready', False):
			return
		env = self.agent.env
		# Only move device, not dtype (frozen models stay in their trained precision)
		for m in (env.tokenizer, env.latent_action, env.dynamics):
			m.to(device=ref_tensor.device)
		self.agent.policy.to(ref_tensor.device)
		self._world_ready = True

	def training_step(self, batch, batch_idx):
		# Dataset returns (T, C, H, W) with output_format='t c h w'
		video = batch
		# st()
		
		# Normalize to (B, T, C, H, W) format
		if video.dim() == 4:
			# (T, C, H, W) -> (1, T, C, H, W)
			video = video.unsqueeze(0)
		elif video.dim() == 5:
			# Already batched, check if it's (B, T, C, H, W) or (B, C, T, H, W)
			if video.shape[2] == 3 or video.shape[2] <= 4:
				# (B, T, C, H, W) - correct format
				pass
			elif video.shape[1] == 3 or video.shape[1] <= 4:
				# (B, C, T, H, W) -> (B, T, C, H, W)
				video = video.permute(0, 2, 1, 3, 4).contiguous()
		
		# Use the first two frames
		start_clip = video[:, :2]  # (B, 2, C, H, W)
		start_clip = start_clip.to(self.device)
  
		# Ensure tokenizer/latent_action/dynamics live on the same device/dtype as inputs
		self._ensure_world_ready(start_clip)
		start_tokens = self.agent.env.encode_tokens(start_clip)  # (b, t>=2, h, w)
		# rollout expects (b, t, h, w)
		loss, stats = self.agent.loss(start_tokens)
		self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
		for k, v in stats.items():
			self.log(f'train_{k}', v, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
		return loss

	def configure_optimizers(self):
		return optim.Adam(self.agent.parameters(), lr=self.lr)


def cli_main():
	LightningCLI(AgentModule, LightningPlatformer2D)


if __name__ == '__main__':
	cli_main()