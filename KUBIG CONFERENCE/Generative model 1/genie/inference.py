"""
Inference module for Genie model.
Loads trained models and generates videos from initial frames using the agent.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from einops import rearrange
import numpy as np
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES, cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB

from genie.tokenizer import VideoTokenizer
from genie.action import LatentAction
from genie.dynamics import DynamicsModel
from genie.module.agent import WorldModelAgent, WorldModelEnv
from agent import _load_tokenizer, _load_latent_and_dynamics_from_world


def load_video_frames(video_path: str, num_frames: int = 2, start_frame: int = 0) -> Tensor:
    """Load specified number of frames from a video file.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to load
        start_frame: Starting frame index
        
    Returns:
        Tensor of shape (num_frames, C, H, W) in range [0, 1]
    """
    cap = VideoCapture(video_path)
    total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    
    num_frames = min(num_frames, total_frames - start_frame)
    cap.set(CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cvtColor(frame, COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).float() / 255.0
            # Convert to (C, H, W)
            frame = rearrange(frame, 'h w c -> c h w')
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not load any frames from {video_path}")
    
    # Stack frames: (num_frames, C, H, W)
    video = torch.stack(frames)
    return video


def save_video(video: Tensor, output_path: str, fps: int = 10):
    """Save video tensor to file.
    
    Args:
        video: Tensor of shape (T, C, H, W) or (B, T, C, H, W) in range [0, 1]
        output_path: Output video file path
        fps: Frames per second
    """
    # Normalize to (T, C, H, W)
    if video.dim() == 5:
        video = video[0]  # Take first batch
    elif video.dim() == 4:
        pass  # Already (T, C, H, W)
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")
    
    # Convert to numpy and scale to [0, 255]
    video_np = (video.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Convert to (T, H, W, C) for OpenCV
    video_np = rearrange(video_np, 't c h w -> t h w c')
    
    # Get video dimensions
    T, H, W, C = video_np.shape
    
    # Create video writer
    fourcc = VideoWriter_fourcc(*'mp4v')
    out = VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in video_np:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cvtColor(frame, COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to {output_path}")


def extract_ground_truth_action(
    video: Tensor,
    tokenizer: VideoTokenizer,
    latent_action: LatentAction,
    device: torch.device
) -> Tensor:
    """Extract ground truth action from video using latent action model.
    
    Args:
        video: Tensor of shape (T, C, H, W) or (B, T, C, H, W)
        tokenizer: VideoTokenizer instance
        latent_action: LatentAction instance
        device: Device to run on
        
    Returns:
        Action indices of shape (T-1,) or (B, T-1)
    """
    # Normalize to (B, T, C, H, W)
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    # Convert to (B, C, T, H, W) for tokenizer
    video = rearrange(video, 'b t c h w -> b c t h w').to(device)
    
    # Tokenize video
    with torch.no_grad():
        quant_video, token_idxs = tokenizer.tokenize(video, beta=10.0, transpose=True)
        
        # Decode quantized features back to RGB video
        rec_video = tokenizer.decode(quant_video)
        
        # Extract latent actions using encode method directly (no loss calculation needed)
        # LatentAction.encode returns ((act, idxs, enc_video), q_loss)
        # We only need idxs (action indices)
        (act, idxs, enc_video), q_loss = latent_action.encode(rec_video, mask=None, transpose=False)
    
    # idxs shape can be (T,), (B, T), or (B, T, num_codebooks) depending on quantizer
    # Normalize to (T,) - action indices per frame
    if idxs.dim() == 1:
        # Already (T,)
        pass
    elif idxs.dim() == 2:
        # (B, T) -> take first batch
        idxs = idxs[0]
    elif idxs.dim() == 3:
        # (B, T, num_codebooks) -> take first batch and first codebook
        idxs = idxs[0, :, 0]
    else:
        # Flatten and take first sequence
        idxs = idxs.flatten()[:rec_video.shape[2]]  # Take first T elements
    
    return idxs.cpu()


class GenieInference:
    """Inference class for Genie model with agent."""
    
    def __init__(
        self,
        tokenizer_ckpt: str,
        world_ckpt: str,
        agent_ckpt: str,
        device: str = 'cuda',
        horizon: int = 4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        """Initialize inference with trained models.
        
        Args:
            tokenizer_ckpt: Path to tokenizer checkpoint
            world_ckpt: Path to world model (action + dynamics) checkpoint
            agent_ckpt: Path to agent checkpoint
            device: Device to run on ('cuda' or 'cpu')
            horizon: Agent horizon
            gamma: Agent discount factor
            entropy_coef: Agent entropy coefficient
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = _load_tokenizer(tokenizer_ckpt)
        self.tokenizer = self.tokenizer.to(self.device)
        
        # Load latent action and dynamics from world checkpoint
        print("Loading world model (latent action + dynamics)...")
        self.latent_action, self.dynamics = _load_latent_and_dynamics_from_world(
            world_ckpt,
            self.tokenizer,
        )
        self.latent_action = self.latent_action.to(self.device)
        self.dynamics = self.dynamics.to(self.device)
        
        # Load agent
        print("Loading agent...")
        self.agent = WorldModelAgent(
            tokenizer=self.tokenizer,
            latent_action=self.latent_action,
            dynamics=self.dynamics,
            horizon=horizon,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        
        # Load agent checkpoint
        agent_state = torch.load(agent_ckpt, map_location=self.device)
        # Extract policy state dict (may be under 'agent.policy' or 'policy')
        state_dict = agent_state.get('state_dict', agent_state)
        policy_state = {}
        for k, v in state_dict.items():
            if 'policy' in k:
                # Remove 'agent.' prefix if present
                new_k = k.replace('agent.', '') if 'agent.' in k else k
                policy_state[new_k] = v
        
        if policy_state:
            self.agent.policy.load_state_dict(policy_state, strict=False)
        
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        
        print("All models loaded successfully!")
    
    @torch.no_grad()
    def generate_with_agent(
        self,
        initial_frames: Tensor,
        num_frames: int = 14,
        steps_per_frame: int = 10,
        temperature: float = 1.0,
        topk: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """Generate video using agent actions.
        
        Args:
            initial_frames: Initial frames of shape (2, C, H, W) or (B, 2, C, H, W)
            num_frames: Number of frames to generate
            steps_per_frame: Mask-GIT steps per frame
            
        Returns:
            generated_video: Generated video tensor (T, C, H, W)
            agent_actions: Actions taken by agent (num_frames,)
        """
        # Normalize to (B, T, C, H, W)
        if initial_frames.dim() == 4:
            initial_frames = initial_frames.unsqueeze(0)
        
        # Convert to (B, C, T, H, W) for tokenizer
        initial_frames = rearrange(initial_frames, 'b t c h w -> b c t h w').to(self.device)
        
        # Encode initial frames to tokens
        env = WorldModelEnv(
            tokenizer=self.tokenizer,
            latent_action=self.latent_action,
            dynamics=self.dynamics,
            steps_per_frame=steps_per_frame,
        )
        # Store generation parameters for custom step function
        env._gen_temp = temperature
        env._gen_topk = topk
        
        # Override step_tokens to use temperature and topk
        original_step = env.step_tokens
        def step_with_params(tokens, action_idx):
            device = next(self.dynamics.parameters()).device
            tokens = tokens.to(device)
            action_idx = action_idx.to(device)
            if not tokens.dtype in (torch.int64, torch.int32):
                tokens = tokens.long()
            if not action_idx.dtype in (torch.int64, torch.int32):
                action_idx = action_idx.long()
            with torch.cuda.amp.autocast(enabled=False):
                next_tokens = self.dynamics.generate(
                    tokens, action_idx, 
                    steps=steps_per_frame,
                    temp=temperature,
                    topk=topk
                )
            if next_tokens.dim() == 3:
                next_tokens = next_tokens.unsqueeze(1)
            if next_tokens.shape[1] != tokens.shape[1]:
                next_tokens_for_reward = next_tokens[:, -tokens.shape[1]:]
            else:
                next_tokens_for_reward = next_tokens
            if env.initial_tokens is not None:
                initial_tokens = env.initial_tokens.to(device)
                reward, done = env._ninja_reward(tokens, next_tokens_for_reward, initial_tokens)
            else:
                reward = env._default_reward(tokens, next_tokens_for_reward)
                done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
            return next_tokens.detach(), reward.detach(), done.detach()
        env.step_tokens = step_with_params
        env.tokenizer = env.tokenizer.to(self.device)
        env.latent_action = env.latent_action.to(self.device)
        env.dynamics = env.dynamics.to(self.device)
        
        # Reset environment
        start_tokens = env.encode_tokens(initial_frames)
        env.reset(start_tokens)
        
        # Store tokens and actions
        all_tokens = [start_tokens]
        agent_actions = []
        
        # Generate frames using agent
        tokens = start_tokens
        for _ in range(num_frames):
            # Agent selects action
            logits, _ = self.agent.policy(tokens)
            action = torch.argmax(logits, dim=-1)  # Greedy action
            
            # Build action sequence aligned with token time dimension
            t_len = tokens.shape[1]
            action_idx = action.new_zeros((tokens.shape[0], t_len))
            action_idx[:, -1] = action[0]  # Apply action at last time position
            
            # Step in token space
            next_tokens, _, _ = env.step_tokens(tokens, action_idx)
            
            all_tokens.append(next_tokens[:, -1:])  # Take last frame
            agent_actions.append(action[0].item())
            
            # Update tokens for next step
            tokens = next_tokens
        
        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens, dim=1)  # (B, T, H, W)
        
        # Decode tokens to video
        # Convert token indices to quantized features using codebook
        # The quantizer has proj_inp and proj_out to map between input_dim and codebook_dim
        if hasattr(self.tokenizer.quant, 'codebook') and hasattr(self.tokenizer.quant, 'proj_out'):
            codebook = self.tokenizer.quant.codebook  # (codebook_size, codebook_dim)
            
            # Flatten spatial dimensions for lookup
            B, T, H, W = all_tokens.shape
            tokens_flat = all_tokens.view(B, T, H * W).long()  # (B, T, H*W)
            
            # Clamp tokens to valid range
            tokens_flat = torch.clamp(tokens_flat, 0, codebook.shape[0] - 1)
            
            # Lookup in codebook: (B, T, H*W) -> (B, T, H*W, codebook_dim)
            codebook_values = codebook[tokens_flat]  # (B, T, H*W, codebook_dim)
            
            # Flatten to (B, T*H*W, codebook_dim) for projection
            codebook_flat = codebook_values.view(B, T * H * W, -1)
            
            # Project through proj_out to get quantized features in input_dim space
            # proj_out: (codebook_dim * num_codebook) -> input_dim
            quant_flat = self.tokenizer.quant.proj_out(codebook_flat)  # (B, T*H*W, input_dim)
            
            # Reshape to (B, input_dim, T, H, W) = (B, hid_channels, T, H, W)
            quant_features = rearrange(quant_flat, 'b (t h w) d -> b d t h w', t=T, h=H, w=W)
            
            # Decode to video
            generated_video = self.tokenizer.decode(quant_features)
            
            # Convert to (T, C, H, W)
            # Remove batch dimension first, then rearrange
            if generated_video.dim() == 5:
                generated_video = generated_video.squeeze(0)  # Remove batch dimension: (B, C, T, H, W) -> (C, T, H, W)
            # Rearrange: (C, T, H, W) -> (T, C, H, W)
            generated_video = rearrange(generated_video, 'c t h w -> t c h w')
            
        else:
            # Fallback: return placeholder
            print("Warning: Could not decode tokens to video. Returning tokens only.")
            generated_video = torch.zeros(num_frames + 2, 3, 64, 64)
        
        # Convert agent_actions to tensor and move to CPU
        agent_actions = torch.tensor(agent_actions, device='cpu')
        
        return generated_video.cpu(), agent_actions
    
    @torch.no_grad()
    def generate_with_ground_truth_actions(
        self,
        initial_frames: Tensor,
        ground_truth_video: Tensor,
        num_frames: int = 14,
        steps_per_frame: int = 10,
        temperature: float = 1.0,
        topk: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """Generate video using ground truth actions from the original video.
        
        Args:
            initial_frames: Initial frames of shape (2, C, H, W)
            ground_truth_video: Full ground truth video of shape (T, C, H, W)
            num_frames: Number of frames to generate
            steps_per_frame: Mask-GIT steps per frame
            
        Returns:
            generated_video: Generated video using GT actions (T, C, H, W)
            gt_actions: Ground truth actions (num_frames,)
        """
        # Extract ground truth actions
        gt_actions = extract_ground_truth_action(
            ground_truth_video,
            self.tokenizer,
            self.latent_action,
            self.device
        )
        
        # Use first num_frames actions
        gt_actions = gt_actions[:num_frames]
        
        # Normalize initial frames
        if initial_frames.dim() == 4:
            initial_frames = initial_frames.unsqueeze(0)
        initial_frames = rearrange(initial_frames, 'b t c h w -> b c t h w').to(self.device)
        
        # Encode initial frames
        env = WorldModelEnv(
            tokenizer=self.tokenizer,
            latent_action=self.latent_action,
            dynamics=self.dynamics,
            steps_per_frame=steps_per_frame,
        )
        env.tokenizer = env.tokenizer.to(self.device)
        env.latent_action = env.latent_action.to(self.device)
        env.dynamics = env.dynamics.to(self.device)
        
        # Override step_tokens to use temperature and topk
        original_step = env.step_tokens
        def step_with_params(tokens, action_idx):
            device = next(self.dynamics.parameters()).device
            tokens = tokens.to(device)
            action_idx = action_idx.to(device)
            if not tokens.dtype in (torch.int64, torch.int32):
                tokens = tokens.long()
            if not action_idx.dtype in (torch.int64, torch.int32):
                action_idx = action_idx.long()
            with torch.cuda.amp.autocast(enabled=False):
                next_tokens = self.dynamics.generate(
                    tokens, action_idx, 
                    steps=steps_per_frame,
                    temp=temperature,
                    topk=topk
                )
            if next_tokens.dim() == 3:
                next_tokens = next_tokens.unsqueeze(1)
            if next_tokens.shape[1] != tokens.shape[1]:
                next_tokens_for_reward = next_tokens[:, -tokens.shape[1]:]
            else:
                next_tokens_for_reward = next_tokens
            if env.initial_tokens is not None:
                initial_tokens = env.initial_tokens.to(device)
                reward, done = env._ninja_reward(tokens, next_tokens_for_reward, initial_tokens)
            else:
                reward = env._default_reward(tokens, next_tokens_for_reward)
                done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
            return next_tokens.detach(), reward.detach(), done.detach()
        env.step_tokens = step_with_params
        
        start_tokens = env.encode_tokens(initial_frames)
        env.reset(start_tokens)
        
        # Generate using ground truth actions
        all_tokens = [start_tokens]
        tokens = start_tokens
        
        # Convert gt_actions to tensor if needed
        if not isinstance(gt_actions, torch.Tensor):
            gt_actions = torch.tensor(gt_actions)
        gt_actions = gt_actions.to(self.device)
        
        for action in gt_actions:
            # Build action sequence
            t_len = tokens.shape[1]
            action_idx = torch.full((tokens.shape[0], t_len), action.item(), dtype=torch.long, device=self.device)
            
            # Step in token space
            next_tokens, _, _ = env.step_tokens(tokens, action_idx)
            
            all_tokens.append(next_tokens[:, -1:])
            tokens = next_tokens
        
        # Concatenate tokens
        all_tokens = torch.cat(all_tokens, dim=1)
        
        # Decode to video (same as in generate_with_agent)
        if hasattr(self.tokenizer.quant, 'codebook') and hasattr(self.tokenizer.quant, 'proj_out'):
            codebook = self.tokenizer.quant.codebook
            B, T, H, W = all_tokens.shape
            tokens_flat = all_tokens.view(B, T, H * W).long()
            tokens_flat = torch.clamp(tokens_flat, 0, codebook.shape[0] - 1)
            
            # Lookup in codebook
            codebook_values = codebook[tokens_flat]  # (B, T, H*W, codebook_dim)
            
            # Flatten for projection
            codebook_flat = codebook_values.view(B, T * H * W, -1)
            
            # Project through proj_out
            quant_flat = self.tokenizer.quant.proj_out(codebook_flat)  # (B, T*H*W, input_dim)
            
            # Reshape to (B, input_dim, T, H, W)
            quant_features = rearrange(quant_flat, 'b (t h w) d -> b d t h w', t=T, h=H, w=W)
            
            generated_video = self.tokenizer.decode(quant_features)
            # Remove batch dimension first, then rearrange
            if generated_video.dim() == 5:
                generated_video = generated_video.squeeze(0)  # Remove batch dimension: (B, C, T, H, W) -> (C, T, H, W)
            # Rearrange: (C, T, H, W) -> (T, C, H, W)
            generated_video = rearrange(generated_video, 'c t h w -> t c h w')
        else:
            generated_video = torch.zeros(num_frames + 2, 3, 64, 64)
        
        # Move gt_actions to CPU if it's a tensor, otherwise convert to tensor on CPU
        if isinstance(gt_actions, torch.Tensor):
            gt_actions = gt_actions.cpu()
        else:
            gt_actions = torch.tensor(gt_actions, device='cpu')
        
        return generated_video.cpu(), gt_actions

