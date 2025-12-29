import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Callable, Tuple
from einops import rearrange

from genie.tokenizer import VideoTokenizer
from genie.action import LatentAction
from genie.dynamics import DynamicsModel
from genie.utils import default

from pdb import set_trace as st

class WorldModelEnv:
    """Lightweight wrapper that treats the frozen world model as a simulator."""

    def __init__(
        self,
        tokenizer: VideoTokenizer,
        latent_action: LatentAction,
        dynamics: DynamicsModel,
        steps_per_frame: int = 10,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.tokenizer = tokenizer.eval()  # freeze
        self.latent_action = latent_action.eval()  # freeze
        self.dynamics = dynamics.eval()  # freeze
        for m in (self.tokenizer, self.latent_action, self.dynamics):
            for p in m.parameters():
                p.requires_grad = False
        self.steps_per_frame = steps_per_frame
        self.reward_fn = reward_fn  # Will use _ninja_reward by default
        
        # Store initial tokens for episode reset detection (Ninja game specific)
        self.initial_tokens: torch.Tensor | None = None
    
    def reset(self, initial_tokens: torch.Tensor) -> None:
        """Reset environment with initial tokens for episode tracking."""
        self.initial_tokens = initial_tokens.clone()

    @torch.no_grad()
    def encode_tokens(self, frame: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Encode a frame into discrete tokens using the frozen tokenizer."""
        # Normalize input to (B, C, T, H, W) - following genie.py pattern
        x = frame
        # st()
        if x.dim() == 3:
            # (C, H, W) -> (B=1, C, T=1, H, W)
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 4:
            # (B, C, H, W) -> (B, C, T=1, H, W)
            x = x.unsqueeze(2)
        elif x.dim() == 5:
            # Either (B, C, T, H, W) or (B, T, C, H, W). Detect and permute if needed.
            b, a1, a2, h, w = x.shape
            # Heuristic: channels is usually <=4, time usually > 4
            if a1 <= 4 and a2 > 4:
                # already (B, C, T, H, W)
                pass
            elif a2 <= 4 and a1 > 4:
                # (B, T, C, H, W) -> (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            else:
                # Fallback: if one of them equals 3, treat that as channels
                if a1 == 3 and a2 != 3:
                    pass
                elif a2 == 3 and a1 != 3:
                    x = x.permute(0, 2, 1, 3, 4).contiguous()  # 여기
                else:
                    # default to (B, C, T, H, W) assumption
                    pass
        else:
            raise RuntimeError(f"Unsupported input dim for tokenizer: {x.dim()}")

        # Use tokenizer.tokenize() like in genie.py - returns (quant, token_idxs)
        # 최종적인 x 차원 = (b, c, t, h, w)
        result = self.tokenizer.tokenize(x, beta=beta, transpose=True)
        
        # Handle tuple return (quant_video, token_idxs)
        if isinstance(result, (tuple, list)):
            _, token_idxs = result
        else:
            token_idxs = result
        
        # Normalize token indices to shape (b, t, h, w) - following genie.py line 197-204
        if isinstance(token_idxs, (tuple, list)):
            token_idxs = token_idxs[0]
        token_idxs = torch.as_tensor(token_idxs)
        
        if token_idxs.ndim == 3:
            # (t, h, w) -> (1, t, h, w)
            token_idxs = token_idxs.unsqueeze(0)
        elif token_idxs.ndim == 5 and token_idxs.shape[1] == 1:
            # (b, 1, t, h, w) -> (b, t, h, w)
            token_idxs = token_idxs.squeeze(1)
        elif token_idxs.ndim == 2:
            # (h, w) -> (1, 1, h, w)
            token_idxs = token_idxs.unsqueeze(0).unsqueeze(0)
        
        # Ensure long dtype
        if not token_idxs.dtype in (torch.int64, torch.int32):
            token_idxs = token_idxs.long()
        
        # Check token value range BEFORE upsampling
        if hasattr(self.tokenizer, 'quant') and hasattr(self.tokenizer.quant, 'codebook_size'):
            vocab_size = int(self.tokenizer.quant.codebook_size)  # 1024
            min_val = token_idxs.min().item()  # 256
            max_val = token_idxs.max().item()  # 883
            if min_val < 0 or max_val >= vocab_size:
                print(f"WARNING: token_idxs out of range! min={min_val}, max={max_val}, vocab_size={vocab_size}")
                token_idxs = torch.clamp(token_idxs, 0, vocab_size - 1)
        
        # Return tokens at their natural resolution from tokenizer
        return token_idxs  # 3[1, 2, 32, 32]

    @torch.no_grad()
    def step_tokens(self, tokens: torch.Tensor, action_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one step in token space given an action index.
        
        Returns:
            next_tokens: (b, t, h, w) next token states
            reward: (b,) reward for this step
            done: (b,) boolean tensor indicating episode termination
        """
        # Move inputs to same device as dynamics
        device = next(self.dynamics.parameters()).device
        tokens = tokens.to(device)  # (b, t, h, w)
        action_idx = action_idx.to(device)  # (b,t)
        
        # Ensure proper dtypes
        if not tokens.dtype in (torch.int64, torch.int32):
            tokens = tokens.long()
        if not action_idx.dtype in (torch.int64, torch.int32):
            action_idx = action_idx.long()
        
        # Generate next tokens using dynamics model; disable autocast to avoid fp16 overflow
        # Lightning may run under mixed precision; keep dynamics in full precision for stability
        with torch.cuda.amp.autocast(enabled=False):
            next_tokens = self.dynamics.generate(tokens, action_idx, steps=self.steps_per_frame)
        if next_tokens.dim() == 3:
            # (b, h, w) -> (b, 1, h, w)
            next_tokens = next_tokens.unsqueeze(1)

        # Align time dimensions for reward: dynamics may return t+1 frames
        if next_tokens.shape[1] != tokens.shape[1]:
            next_tokens_for_reward = next_tokens[:, -tokens.shape[1]:]
        else:
            next_tokens_for_reward = next_tokens
        
        # Use Ninja-specific reward function with initial tokens for death/clear detection
        if self.initial_tokens is not None:
            initial_tokens = self.initial_tokens.to(device)
            reward, done = self._ninja_reward(tokens, next_tokens_for_reward, initial_tokens)
        else:
            # Fallback if initial_tokens not set
            reward = self._default_reward(tokens, next_tokens_for_reward)
            done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
        
        return next_tokens.detach(), reward.detach(), done.detach()

    def _ninja_reward(
        self,
        prev_tokens: torch.Tensor, 
        next_tokens: torch.Tensor,
        initial_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ninja game-specific reward function.
        
        Game mechanics: Character fixed at center, background scrolls left when moving right.
        
        Rewards:
        - Time penalty: -1 per frame (encourage fast completion)
        - Right movement: +0.5 (detected via scroll pattern)
        - Bomb death: -100, done=True (big change + similar to initial frame = reset)
        - Mushroom collection: +100, done=True (big change + different from initial = level clear)
        
        Returns:
            reward: (b,) tensor of rewards
            done: (b,) boolean tensor indicating episode termination
        """
        device = prev_tokens.device
        b = prev_tokens.shape[0]
        
        # Initialize rewards and done flags
        reward = torch.full((b,), -1.0, device=device)  # Time penalty per frame
        done = torch.zeros(b, dtype=torch.bool, device=device)
        
        # Calculate overall change ratio
        diff = (next_tokens != prev_tokens).float()  # (b, t, h, w)
        total_pixels = diff.shape[1] * diff.shape[2] * diff.shape[3]
        change_ratio = diff.sum(dim=(1, 2, 3)) / total_pixels  # (b,)
        
        # Detect big events (death or level clear) - threshold: >50% token change
        big_change_mask = change_ratio > 0.5
        
        if big_change_mask.any():
            # Calculate similarity to initial frame
            # Align time dimensions: next_tokens might have different t than initial_tokens
            next_t_for_initial = next_tokens
            if next_tokens.shape[1] != initial_tokens.shape[1]:
                # Take the last t frames from next_tokens to match initial_tokens time dimension
                t_initial = initial_tokens.shape[1]
                next_t_for_initial = next_tokens[:, -t_initial:]
            
            initial_diff = (next_t_for_initial != initial_tokens).float()
            total_pixels_initial = initial_diff.shape[1] * initial_diff.shape[2] * initial_diff.shape[3]
            similarity_to_initial = 1.0 - (initial_diff.sum(dim=(1, 2, 3)) / total_pixels_initial)
            
            # High similarity to initial (>70%) = death (reset to start)
            death_mask = big_change_mask & (similarity_to_initial > 0.7)
            # Low similarity to initial = level clear (mushroom collected)
            clear_mask = big_change_mask & (similarity_to_initial <= 0.7)
            
            # Apply bomb death penalty
            reward = torch.where(death_mask, torch.tensor(-100.0, device=device), reward)
            done = done | death_mask
            
            # Apply mushroom collection reward
            reward = torch.where(clear_mask, torch.tensor(100.0, device=device), reward)
            done = done | clear_mask
        
        # Detect right movement via scroll pattern (only if not dead/cleared)
        # When moving right: background scrolls left = left edge changes more
        # The left columns should shift, creating a distinctive pattern
        if not done.all():
            active_mask = ~done
            
            # Compare left-most columns (scroll detection)
            left_cols = 4  # Check first 4 columns
            left_diff = (next_tokens[:, :, :, :left_cols] != prev_tokens[:, :, :, :left_cols]).float()
            left_change = left_diff.mean(dim=(1, 2, 3))  # (b,)
            
            # Compare center region (character region should be relatively stable)
            _, _, h, w = diff.shape
            center_start = w // 3
            center_end = 2 * w // 3
            center_diff = diff[:, :, :, center_start:center_end]
            center_change = center_diff.mean(dim=(1, 2, 3))  # (b,)
            
            # Scroll pattern: left edge changing more than center = moving right
            # Also require minimum overall movement to avoid false positives
            min_movement = 0.02
            scroll_detected = (
                (left_change > center_change * 1.2) &  # Left changes more than center
                (left_change > min_movement) &  # Minimum movement threshold
                active_mask
            )
            
            # Add right movement reward
            reward = torch.where(
                scroll_detected, 
                reward + 0.5, 
                reward
            )
        
        return reward, done
    
    @staticmethod
    def _default_reward(prev_tokens: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
        """Legacy default reward function (not used for Ninja)."""
        diff = (next_tokens != prev_tokens).float()
        movement_reward = diff.mean(dim=(1, 2, 3)) * 0.1
        return movement_reward


class PolicyValueNet(nn.Module):
    """Simple policy/value network over token observations."""

    def __init__(self, tok_vocab: int, act_vocab: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(tok_vocab, hidden_dim)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_vocab),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (b, t, h, w)
        if tokens.dim() == 3:
            tokens = tokens.unsqueeze(0)
        
        # Clamp tokens to valid embedding range
        tokens = tokens.long()
        tokens = torch.clamp(tokens, 0, self.token_emb.num_embeddings - 1)
        
        emb = self.token_emb(tokens)  # (b, t, h, w, d)
        emb = emb.mean(dim=(1, 2, 3))  # (b, d)
        logits = self.policy(emb)
        value = self.value(emb).squeeze(-1)
        return logits, value


class WorldModelAgent(torch.nn.Module):
    """A lightweight agent that learns a policy over latent actions using the frozen world model."""

    def __init__(
        self,
        tokenizer: VideoTokenizer,
        latent_action: LatentAction,
        dynamics: DynamicsModel,
        horizon: int = 4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.env = WorldModelEnv(tokenizer, latent_action, dynamics)
        self.horizon = horizon
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # vocab sizes
        if hasattr(tokenizer, 'quant') and hasattr(tokenizer.quant, 'codebook_size'):
            tok_vocab = int(tokenizer.quant.codebook_size)
        else:
            tok_vocab = 512
        if hasattr(latent_action, 'quant') and hasattr(latent_action.quant, 'codebook_size'):
            act_vocab = int(latent_action.quant.codebook_size)
        else:
            act_vocab = 256

        self.policy = PolicyValueNet(tok_vocab=tok_vocab, act_vocab=act_vocab)

    def rollout(self, start_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a rollout in the world model environment.
        
        Returns:
            logps: (b, horizon) log probabilities
            values: (b, horizon) value estimates
            rewards: (b, horizon) rewards
            actions: (b, horizon) actions taken
            all_logits: (b, horizon, act_vocab) policy logits
            dones: (b, horizon) episode termination flags
        """
        tokens = start_tokens  # (b, t, h, w)
        device = tokens.device
        bsz = tokens.shape[0]
        
        # Reset environment with initial tokens for death/clear detection
        self.env.reset(start_tokens)
        
        logps, values, rewards, actions, all_logits, dones = [], [], [], [], [], []
        episode_done = torch.zeros(bsz, dtype=torch.bool, device=device)
        
        for _ in range(self.horizon):
            # Policy expects (b, t, h, w)
            logits, value = self.policy(tokens)  # logits=(b, act_vocab) , value=(b)
            dist = Categorical(logits=logits)
            action = dist.sample()  # (b,)
            logp = dist.log_prob(action)  # (b,)
            
            # Build action sequence aligned with token time dimension
            t_len = tokens.shape[1]
            action_idx = action.new_zeros((bsz, t_len))  # (b, t_len)
            action_idx[:, -1] = action  # apply current action at last time position

            next_tokens, reward, done = self.env.step_tokens(tokens, action_idx)
            
            # Mask rewards for already-done episodes (no more rewards after termination)
            reward = torch.where(episode_done, torch.zeros_like(reward), reward)
            done = done & ~episode_done  # Only mark done for newly done episodes
            
            logps.append(logp)
            values.append(value)
            rewards.append(reward)
            actions.append(action)
            all_logits.append(logits)
            dones.append(done)
            
            # Update episode_done for next iteration
            episode_done = episode_done | done
            
            # Keep tokens for next step (even if done, needed for value calculation)
            tokens = next_tokens.detach()

        logps = torch.stack(logps, dim=1)
        values = torch.stack(values, dim=1)
        rewards = torch.stack(rewards, dim=1)
        actions = torch.stack(actions, dim=1)
        all_logits = torch.stack(all_logits, dim=1)  # (b, horizon, act_vocab)
        dones = torch.stack(dones, dim=1)  # (b, horizon)
        return logps, values, rewards, actions, all_logits, dones

    def loss(self, start_tokens: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute PPO-style loss for the agent.
        
        Returns:
            total_loss: Combined policy, value, and entropy loss
            stats: Dictionary of training statistics
        """
        # Ensure rollout runs on same device as policy/dynamics
        device = next(self.policy.parameters()).device
        start_tokens = start_tokens.to(device)  # (b, t, h, w)

        logps, values, rewards, _, all_logits, dones = self.rollout(start_tokens)
        
        # Compute discounted returns with proper episode termination handling
        # When done=True, future rewards should not be discounted back
        returns = []
        g = torch.zeros_like(rewards[:, 0])
        rewards_list = torch.unbind(rewards, dim=1)
        dones_list = torch.unbind(dones, dim=1)
        
        for r, d in zip(reversed(rewards_list), reversed(dones_list)):
            # If episode terminates, reset cumulative return to just the current reward
            g = torch.where(d, r, r + self.gamma * g)
            returns.append(g.clone())  # clone to avoid sharing
        returns = torch.stack(list(reversed(returns)), dim=1)

        advantages = returns - values.detach()
        policy_loss = -(logps * advantages).mean()
        value_loss = F.mse_loss(values, returns.detach())  # detach returns for value loss
        
        # Correct entropy calculation using logits
        entropy = Categorical(logits=all_logits.view(-1, all_logits.shape[-1])).entropy().mean()
        
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Additional stats for Ninja game
        death_count = (rewards < -50).sum().float()  # Count bomb deaths
        clear_count = (rewards > 50).sum().float()  # Count mushroom clears
        
        stats = {
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
            'entropy': entropy.detach(),
            'return_mean': returns.mean().detach(),
            'reward_mean': rewards.mean().detach(),
            'death_count': death_count.detach(),
            'clear_count': clear_count.detach(),
        }
        return total_loss, stats
