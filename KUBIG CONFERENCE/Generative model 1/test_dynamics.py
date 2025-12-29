#!/usr/bin/env python3
"""
Test script to generate video from dynamics model with manual action inputs.
Usage:
    # Using YAML config file (recommended):
    python test_dynamics_video.py --config config/test_dynamics.yaml
    
    # Or using command line arguments:
    python test_dynamics_video.py --tokenizer_ckpt <path> --dynamics_ckpt <path> --latent_action_ckpt <path> --actions "0,1,2,3,4"
"""

import torch
import cv2
import numpy as np
import argparse
import yaml
from einops import rearrange
from pathlib import Path
from scipy.ndimage import gaussian_filter

from agent import _load_tokenizer, _load_dynamics, _load_latent_action, _load_latent_and_dynamics_from_world
from genie.tokenizer import VideoTokenizer


def tokens_to_quantized_via_retokenize(tokenizer, token_idxs, initial_quant_video_shape):
    """
    Alternative method: Create a dummy video from tokens, then re-tokenize to get quant_video.
    This is more accurate but requires creating a temporary video.
    """
    # This is a placeholder - would need to implement dummy video creation
    # For now, fall back to direct method
    return tokens_to_quantized(tokenizer, token_idxs, transpose=True)


def tokens_to_quantized(tokenizer, token_idxs, transpose=True):
    """
    Convert token indices to quantized features by using quantizer's forward pass.
    
    Instead of manually reconstructing, we create a dummy input that will produce
    the same quantized values, then use quantizer.forward() to get the proper output.
    """
    if token_idxs.dim() == 3:
        token_idxs = token_idxs.unsqueeze(0)  # (1, t, h, w)
    
    b, t, h, w = token_idxs.shape
    device = token_idxs.device
    
    # Get quantizer
    quant = tokenizer.quant
    
    if not hasattr(quant, 'codebook'):
        raise RuntimeError("Quantizer doesn't have codebook attribute")
    
    codebook = quant.codebook  # (codebook_size, codebook_dim)
    codebook_size, codebook_dim = codebook.shape
    num_codebooks = quant.num_codebooks
    
    # Clamp token indices to valid range
    token_idxs = torch.clamp(token_idxs.long(), 0, codebook_size - 1)
    
    # Lookup codebook values: (b, t, h, w) -> quantized values {-1, 1}
    token_flat = token_idxs.view(-1)  # (b*t*h*w,)
    quant_values = codebook[token_flat]  # (b*t*h*w, codebook_dim) - values are {-1, 1}
    
    # Create a dummy input that, after proj_inp and split, will have sign() = quant_values
    # We need to create input where sign(input) = quant_values after proj_inp and split
    # The trick: create input with large magnitude so sign(input) = quant_values
    # But we need to account for proj_inp first...
    
    # Actually, a better approach: directly construct the code that quantizer produces
    # quantizer does: quant = inp.sign() -> code = quant -> proj_out(code)
    # We have quant_values from codebook, so we can directly use them
    
    n = b * t * h * w
    
    # Reshape quant_values to match quantizer's internal format after split
    # quantizer splits into (b, n, num_codebooks, codebook_dim)
    d_per_codebook = codebook_dim  # codebook_dim is per-codebook dimension
    quant_values = quant_values.view(n, 1, d_per_codebook)  # (n, 1, codebook_dim)
    if num_codebooks > 1:
        quant_values = quant_values.expand(n, num_codebooks, d_per_codebook)  # (n, num_codebooks, codebook_dim)
    
    # Rearrange to match quantizer: (b, n, num_codebooks, codebook_dim) -> (b, n, num_codebooks * codebook_dim)
    code = quant_values.reshape(n, num_codebooks * d_per_codebook)  # (n, num_codebooks * codebook_dim)
    
    # Create a dummy input tensor that will produce these quantized values
    # We need to create input where, after proj_inp and split, sign(input) = quant_values
    # The simplest way: create input with large magnitude and correct sign
    # But proj_inp might change things, so we need to work backwards from proj_out
    
    # Actually, let's use quantizer.forward() directly by creating a dummy input
    # that will produce the same quantized values
    
    # Method: Create dummy input with shape (b, d, t, h, w) where d is the input_dim
    # We need to find input_dim from quantizer
    input_dim = quant.proj_inp.in_features if hasattr(quant.proj_inp, 'in_features') else quant.codebook_dim * num_codebooks
    
    # Create dummy input: we want sign(proj_inp(dummy_input)) = quant_values after split
    # This is complex, so let's try a different approach:
    # Create input where proj_inp(input) has the right sign
    
    # Simpler: directly use the code and apply proj_out
    # But we need to account for the pack/unpack and rearrange operations
    
    # Reshape code to (b, n, num_codebooks * codebook_dim)
    code = code.view(b, n // b, num_codebooks * d_per_codebook)  # (b, n, num_codebooks * codebook_dim)
    
    # Apply proj_out
    quant_features_flat = quant.proj_out(code)  # (b, n, output_dim)
    
    # Unpack: (b, n, output_dim) -> (b, t, h, w, output_dim)
    output_dim = quant_features_flat.shape[-1]
    quant_features = quant_features_flat.view(b, t, h, w, output_dim)  # (b, t, h, w, output_dim)
    
    # Rearrange: (b, t, h, w, output_dim) -> (b, output_dim, t, h, w) if transpose
    if transpose:
        quant_features = rearrange(quant_features, 'b t h w d -> b d t h w')
    else:
        quant_features = rearrange(quant_features, 'b t h w d -> b d t h w')
    
    # Apply tokenizer's post_quant_proj
    quant_features = tokenizer.post_quant_proj(quant_features)
    
    return quant_features


def save_video(frames, output_path, fps=15):
    """Save frames as MP4 video."""
    if len(frames) == 0:
        print("No frames to save!")
        return
    
    print(f"Saving {len(frames)} frames to {output_path}...")
    
    # Get frame dimensions
    first_frame = frames[0]
    if first_frame.ndim == 2:
        height, width = first_frame.shape
        channels = 1
    else:
        height, width = first_frame.shape[:2]
        channels = first_frame.shape[2] if first_frame.ndim == 3 else 3
    
    print(f"Frame dimensions: {width}x{height}, channels: {channels}")
    
    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    frames_written = 0
    for i, frame in enumerate(frames):
        # Ensure frame has correct shape and type
        if frame.ndim == 2:
            # Grayscale, convert to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3:
            if frame.shape[2] == 3:
                # RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 1:
                # Single channel to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame
        else:
            print(f"Warning: Unexpected frame shape at index {i}: {frame.shape}")
            continue
        
        # Ensure frame is uint8
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        
        # Resize if needed
        if frame_bgr.shape[:2] != (height, width):
            frame_bgr = cv2.resize(frame_bgr, (width, height))
        
        out.write(frame_bgr)
        frames_written += 1
    
    out.release()
    print(f"Successfully saved {frames_written} frames to {output_path}")
    
    if frames_written == 0:
        print("Warning: No frames were written to the video file!")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Generate video from dynamics model with manual actions')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--tokenizer_ckpt', type=str, default=None, help='Path to tokenizer checkpoint')
    parser.add_argument('--world_ckpt', type=str, default=None, help='Path to world checkpoint (combined)')
    parser.add_argument('--dynamics_ckpt', type=str, default=None, help='Path to dynamics model checkpoint')
    parser.add_argument('--latent_action_ckpt', type=str, default=None, help='Path to latent action checkpoint')
    parser.add_argument('--actions', type=str, default=None, help='Comma-separated action indices (e.g., "0,1,2,3,4")')
    parser.add_argument('--initial_frame', type=str, default=None, help='Path to initial frame image (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--steps_per_frame', type=int, default=None, help='MaskGIT steps per frame')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        config = load_config(args.config)
        checkpoints = config.get('checkpoints', {})
        generation = config.get('generation', {})
        
        # Override with command line args if provided
        tokenizer_ckpt = args.tokenizer_ckpt or checkpoints.get('tokenizer')
        world_ckpt = args.world_ckpt or checkpoints.get('world_ckpt')
        dynamics_ckpt = args.dynamics_ckpt or checkpoints.get('dynamics_ckpt')
        latent_action_ckpt = args.latent_action_ckpt or checkpoints.get('latent_action_ckpt')
        
        # Handle actions: could be list or string
        actions_config = generation.get('actions', [])
        if args.actions:
            actions_str = args.actions
        elif isinstance(actions_config, list):
            actions_str = actions_config  # Keep as list
        else:
            actions_str = str(actions_config)
        
        # initial_frame can be in generation section or at root level
        initial_frame = args.initial_frame or generation.get('initial_frame') or config.get('initial_frame')
        output = args.output or generation.get('output', 'output.mp4')
        steps_per_frame = args.steps_per_frame or generation.get('steps_per_frame', 25)
        device = args.device or generation.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Use command line args only
        if not args.tokenizer_ckpt:
            parser.error("--tokenizer_ckpt is required when --config is not provided")
        if not args.world_ckpt and (not args.dynamics_ckpt or not args.latent_action_ckpt):
            parser.error("Either --world_ckpt or both --dynamics_ckpt and --latent_action_ckpt are required")
        if not args.actions:
            parser.error("--actions is required when --config is not provided")
        
        tokenizer_ckpt = args.tokenizer_ckpt
        world_ckpt = args.world_ckpt
        dynamics_ckpt = args.dynamics_ckpt
        latent_action_ckpt = args.latent_action_ckpt
        actions_str = args.actions
        initial_frame = args.initial_frame
        output = args.output or 'output.mp4'
        steps_per_frame = args.steps_per_frame or 25
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Parse actions - handle both list and string formats
    if isinstance(actions_str, list):
        action_indices = [int(a) for a in actions_str]
    elif isinstance(actions_str, str):
        # Remove brackets if present and split by comma
        actions_str = actions_str.strip()
        if actions_str.startswith('[') and actions_str.endswith(']'):
            actions_str = actions_str[1:-1]
        action_indices = [int(a.strip()) for a in actions_str.split(',') if a.strip()]
    else:
        raise ValueError(f"Invalid actions format: {actions_str}")
    print(f"Action sequence: {action_indices}")
    
    # Load models
    print("Loading tokenizer...")
    tokenizer = _load_tokenizer(tokenizer_ckpt)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    
    # Check if latent_action_ckpt and dynamics_ckpt are the same file (combined checkpoint)
    use_combined = False
    if world_ckpt:
        use_combined = True
        combined_ckpt = world_ckpt
    elif latent_action_ckpt and dynamics_ckpt and latent_action_ckpt == dynamics_ckpt:
        use_combined = True
        combined_ckpt = latent_action_ckpt
        print(f"Detected combined checkpoint: {combined_ckpt}")
    
    # Load dynamics and latent_action (either from combined checkpoint or separate checkpoints)
    if use_combined:
        print("Loading from combined checkpoint...")
        latent_action, dynamics = _load_latent_and_dynamics_from_world(combined_ckpt, tokenizer)
        latent_action = latent_action.to(device)
        dynamics = dynamics.to(device)
    else:
        print("Loading latent action...")
        latent_action = _load_latent_action(latent_action_ckpt)
        latent_action = latent_action.to(device)
        
        # Get vocab sizes
        if hasattr(tokenizer.quant, 'codebook_size'):
            tok_vocab = int(tokenizer.quant.codebook_size)
        else:
            tok_vocab = 512  # default
            print(f"Warning: Could not get tokenizer vocab size, using default {tok_vocab}")
        
        if hasattr(latent_action.quant, 'codebook_size'):
            act_vocab = int(latent_action.quant.codebook_size)
        else:
            act_vocab = 64  # default
            print(f"Warning: Could not get action vocab size, using default {act_vocab}")
        
        print(f"Tokenizer vocab size: {tok_vocab}, Action vocab size: {act_vocab}")
        
        print("Loading dynamics...")
        dynamics = _load_dynamics(dynamics_ckpt, tok_vocab, act_vocab)
        dynamics = dynamics.to(device)
        dynamics.eval()
    
    # Get action vocab size for validation
    if hasattr(latent_action.quant, 'codebook_size'):
        act_vocab = int(latent_action.quant.codebook_size)
    else:
        act_vocab = 64  # default
    
    # Validate actions
    for i, act in enumerate(action_indices):
        if act < 0 or act >= act_vocab:
            raise ValueError(f"Action {act} at position {i} is out of range [0, {act_vocab-1}]")
    
    # Create initial frame
    if initial_frame and Path(initial_frame).exists():
        initial_frame_path = Path(initial_frame)
        
        # Check if it's a video file (mp4, avi, etc.) or an image file
        if initial_frame_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Load first frame from video
            print(f"Loading first frame from video: {initial_frame}")
            cap = cv2.VideoCapture(str(initial_frame_path))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError(f"Could not read first frame from video: {initial_frame}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            frame = frame.astype(np.float32) / 255.0
            initial_frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, 64, 64)
            initial_frame_tensor = initial_frame_tensor.to(device)
            print(f"Successfully loaded first frame from video: {initial_frame}")
        else:
            # Load image
            print(f"Loading image: {initial_frame}")
            img = cv2.imread(str(initial_frame_path))
            if img is None:
                raise ValueError(f"Could not read image: {initial_frame}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            # Resize to expected size (64x64 for Ninja)
            img = cv2.resize(img, (64, 64))
            initial_frame_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, 64, 64)
            initial_frame_tensor = initial_frame_tensor.to(device)
            print(f"Successfully loaded image: {initial_frame}")
    else:
        # Create a black frame as default
        initial_frame_tensor = torch.zeros(1, 3, 1, 64, 64, device=device)
        if initial_frame:
            print(f"Warning: initial_frame path '{initial_frame}' does not exist. Using black frame.")
        else:
            print("Using black frame as initial frame (consider providing --initial_frame in YAML config)")
    
    # Tokenize initial frame
    print("Tokenizing initial frame...")
    with torch.no_grad():
        quant_video, token_idxs = tokenizer.tokenize(initial_frame_tensor, beta=10.0, transpose=True)
    
    # Store initial quant_video for later use (to decode initial frame correctly)
    initial_quant_video = quant_video.clone()
    initial_quant_video_shape = quant_video.shape
    print(f"Initial quant_video shape: {initial_quant_video_shape}")
    
    # Verify initial frame can be decoded correctly
    print("Verifying initial frame decoding...")
    with torch.no_grad():
        tokenizer.eval()
        decoded_initial = tokenizer.decode(initial_quant_video)
        tokenizer.train()
        initial_min = decoded_initial.min().item()
        initial_max = decoded_initial.max().item()
        initial_mean = decoded_initial.mean().item()
        print(f"Initial decoded frame stats: min={initial_min:.3f}, max={initial_max:.3f}, mean={initial_mean:.3f}")
    
    # Verify initial frame can be decoded correctly
    print("Verifying initial frame decoding...")
    with torch.no_grad():
        tokenizer.eval()
        decoded_initial = tokenizer.decode(initial_quant_video)
        tokenizer.train()
        initial_min = decoded_initial.min().item()
        initial_max = decoded_initial.max().item()
        initial_mean = decoded_initial.mean().item()
        print(f"Initial decoded frame stats: min={initial_min:.3f}, max={initial_max:.3f}, mean={initial_mean:.3f}")
    
    # Normalize token_idxs shape to (b, t, h, w)
    if isinstance(token_idxs, (tuple, list)):
        token_idxs = token_idxs[0]
    token_idxs = torch.as_tensor(token_idxs, device=device)
    
    print(f"Raw token_idxs shape after tokenize: {token_idxs.shape}, ndim: {token_idxs.ndim}")
    
    # Handle different input shapes
    if token_idxs.ndim == 2:
        # (h, w) -> (1, 1, h, w)
        token_idxs = token_idxs.unsqueeze(0).unsqueeze(0)
    elif token_idxs.ndim == 3:
        # (t, h, w) or (h, w, t) -> (1, t, h, w)
        # Assume (t, h, w) if first dim is small, otherwise might be (h, w, t)
        if token_idxs.shape[0] <= 16:  # likely time dimension
            token_idxs = token_idxs.unsqueeze(0)  # (1, t, h, w)
        else:
            # Might be (h, w, t), need to permute
            token_idxs = token_idxs.permute(2, 0, 1).unsqueeze(0)  # (1, t, h, w)
    elif token_idxs.ndim == 4:
        # Already (b, t, h, w) or (b, h, w, t)
        if token_idxs.shape[1] <= 16:  # likely (b, t, h, w)
            pass  # Already correct
        else:
            # Might be (b, h, w, t), permute
            token_idxs = token_idxs.permute(0, 3, 1, 2)  # (b, t, h, w)
    elif token_idxs.ndim == 5 and token_idxs.shape[1] == 1:
        # (b, 1, t, h, w) -> (b, t, h, w)
        token_idxs = token_idxs.squeeze(1)
    elif token_idxs.ndim == 5:
        # (b, c, t, h, w) -> take first channel or reshape
        if token_idxs.shape[1] == 1:
            token_idxs = token_idxs.squeeze(1)  # (b, t, h, w)
        else:
            raise ValueError(f"Unexpected 5D token shape: {token_idxs.shape}")
    
    # Final validation
    if token_idxs.ndim != 4:
        raise ValueError(f"token_idxs must be 4D (b, t, h, w) after normalization, got shape {token_idxs.shape}")
    if token_idxs.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {token_idxs.shape[0]}")
    
    print(f"Final initial tokens shape: {token_idxs.shape} (should be (1, t, h, w))")
    
    # Use original initial frame directly (don't decode from quant_video to avoid inaccuracy)
    all_frames = []
    # Convert initial_frame_tensor directly to numpy (this is the original frame from video)
    if initial_frame_tensor.ndim == 5:
        initial_frame_np = initial_frame_tensor[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()  # (h, w, c)
    elif initial_frame_tensor.ndim == 4:
        initial_frame_np = initial_frame_tensor[0, :, :, :].permute(1, 2, 0).cpu().numpy()  # (h, w, c)
    else:
        initial_frame_np = initial_frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    initial_frame_np = np.clip(initial_frame_np, 0, 1)
    initial_frame_np = (initial_frame_np * 255).astype(np.uint8)
    all_frames.append(initial_frame_np)
    print(f"Initial frame shape: {all_frames[0].shape} (using original frame directly)")
    
    # Generate frames step by step
    print("Generating frames...")
    current_tokens = token_idxs  # (b, t, h, w)
    
    # Store quant_video for each frame to maintain consistency
    # We'll use the previous frame's quant_video structure as a template
    current_quant_video = initial_quant_video.clone()  # Start with initial frame's quant_video
    
    # Ensure current_tokens has correct shape (b, t, h, w)
    if current_tokens.ndim != 4:
        raise ValueError(f"current_tokens must be 4D (b, t, h, w), got shape {current_tokens.shape}")
    
    print(f"Starting with tokens shape: {current_tokens.shape}")
    
    for step, action_idx in enumerate(action_indices):
        print(f"Step {step+1}/{len(action_indices)}: action={action_idx}, current_tokens shape: {current_tokens.shape}")
        
        # Ensure current_tokens is 4D
        if current_tokens.ndim != 4:
            raise ValueError(f"current_tokens must be 4D at step {step}, got shape {current_tokens.shape}")
        
        # Prepare action tensor: (b, t) where t matches current_tokens time dimension
        # dynamics.generate internally packs actions with a mock action for the new frame
        # So we need: actions for existing tokens + action for new frame
        # pack([act_id, mock], 'b *') means: [existing actions, new action]
        t_dim = current_tokens.shape[1]
        
        # The action should be applied to generate the NEW frame
        # dynamics.generate packs: [existing_actions, new_action]
        # So we provide actions for existing frames + the new action
        # For simplicity, use the same action for all existing frames and the new one
        action_tensor = torch.full(
            (1, t_dim), 
            action_idx, 
            device=device, 
            dtype=torch.long
        )  # (1, t) - action for each existing time step
        
        # Generate next tokens
        with torch.no_grad():
            next_tokens = dynamics.generate(
                current_tokens,
                action_tensor,
                steps=steps_per_frame,
                temp=1.0,
            )
        
        print(f"  Generated next_tokens shape: {next_tokens.shape}")
        
        # next_tokens might be (b, h, w) or (b, t, h, w)
        # dynamics.generate returns (b, t+1, h, w) where t+1 includes the new frame
        # We need to extract only the new frame (last time step)
        if next_tokens.ndim == 3:
            # (b, h, w) -> (b, 1, h, w)
            new_frame_tokens = next_tokens.unsqueeze(1)
        elif next_tokens.ndim == 4:
            # (b, t, h, w) - take only the last frame (the newly generated one)
            new_frame_tokens = next_tokens[:, -1:]  # (b, 1, h, w)
        else:
            raise ValueError(f"next_tokens must be 3D or 4D, got shape {next_tokens.shape}")
        
        # Debug: Check if new tokens are actually different
        new_tokens_min = new_frame_tokens.min().item()
        new_tokens_max = new_frame_tokens.max().item()
        new_tokens_mean = new_frame_tokens.float().mean().item()
        new_tokens_unique = len(torch.unique(new_frame_tokens))
        print(f"  New frame tokens stats: min={new_tokens_min}, max={new_tokens_max}, mean={new_tokens_mean:.2f}, unique={new_tokens_unique}")
        
        # Update current_tokens: keep history (e.g., last 2 frames) and add new frame
        # For simplicity, keep last frame and add new one
        # Ensure we maintain (b, t, h, w) shape
        if current_tokens.shape[1] >= 2:
            # Keep last frame and add new one
            current_tokens = torch.cat([current_tokens[:, -1:], new_frame_tokens], dim=1)  # (b, 2, h, w)
        else:
            current_tokens = torch.cat([current_tokens, new_frame_tokens], dim=1)  # (b, t+1, h, w)
        
        # Ensure shape is maintained
        if current_tokens.ndim != 4:
            raise ValueError(f"current_tokens lost 4D shape after update, got {current_tokens.shape}")
        
        # Convert tokens to quantized features and decode
        # IMPORTANT: We use dynamics model's generated tokens (new_frame_tokens) to create the frame
        # First frame comes from original video, rest come from dynamics model predictions
        try:
            # Method: Use tokens_to_quantized directly, decode, and apply minimal post-processing
            # We skip re-tokenization to avoid token mismatch, but this means we accept some grid artifacts
            
            # Step 1: Convert dynamics-generated tokens to quant_video using tokens_to_quantized
            quant_features = tokens_to_quantized(tokenizer, new_frame_tokens, transpose=True)
            
            # Step 2: Decode to get video frame (may have grid artifacts due to tokens_to_quantized inaccuracy)
            tokenizer.eval()
            decoded_frame = tokenizer.decode(quant_features)  # (b, c, t, h, w)
            tokenizer.train()
            
            # Step 3: Clamp to valid range
            decoded_frame = torch.clamp(decoded_frame, 0, 1)
            
            # Step 4: Minimal post-processing to reduce grid artifacts while preserving details
            # Convert to numpy for processing
            if decoded_frame.ndim == 5:
                frame_np = decoded_frame[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()  # (h, w, c)
            elif decoded_frame.ndim == 4:
                frame_np = decoded_frame[0, :, :, :].permute(1, 2, 0).cpu().numpy()  # (h, w, c)
            else:
                frame_np = decoded_frame.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # Apply very minimal Gaussian blur to reduce grid artifacts (sigma=0.2, very subtle)
            frame_np = gaussian_filter(frame_np, sigma=0.2, mode='reflect')
            
            # Apply very light bilateral filter (minimal processing)
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_np = cv2.bilateralFilter(frame_np, d=3, sigmaColor=20, sigmaSpace=20)  # Very light filter
            frame_np = frame_np.astype(np.float32) / 255.0
            
            # Temporal smoothing: blend with previous frame to maintain continuity
            if step > 0 and len(all_frames) > 0:
                prev_frame_np = all_frames[-1].astype(np.float32) / 255.0
                # Blend 95% new frame, 5% previous frame (very light temporal smoothing)
                frame_np = 0.95 * frame_np + 0.05 * prev_frame_np
            
            # Final clamp
            frame_np = np.clip(frame_np, 0, 1)
            
            # Store the post-processed frame for output
            decoded_frame_np = frame_np.copy()
            
            # For next iteration, we still need to update quant_video
            # Convert back to torch tensor for re-tokenization
            decoded_frame_for_next = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)  # (1, c, 1, h, w)
            
            # Re-tokenize the post-processed frame to update quant_video for next iteration
            with torch.no_grad():
                tokenizer.eval()
                current_quant_video, _ = tokenizer.tokenize(decoded_frame_for_next, beta=10.0, transpose=True)
                tokenizer.train()
            
            # Use the post-processed frame directly (don't decode again to preserve post-processing)
            decoded_frame = torch.from_numpy(decoded_frame_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device)  # (1, c, 1, h, w)
            
            
            # Debug: Check if decoded frame has reasonable values
            frame_min = decoded_frame.min().item()
            frame_max = decoded_frame.max().item()
            frame_mean = decoded_frame.mean().item()
            frame_std = decoded_frame.std().item()
            print(f"  Decoded frame stats: min={frame_min:.3f}, max={frame_max:.3f}, mean={frame_mean:.3f}, std={frame_std:.3f}")
            
            # Normalize decoded frame to [0, 1] range if needed
            # This is a workaround - ideally quant_video should be correct
            if frame_min < 0 or frame_max > 1:
                # Try to normalize to [0, 1] range using percentile-based normalization
                # This is more robust than min-max normalization
                frame_flat = decoded_frame.flatten()
                p1 = torch.quantile(frame_flat, 0.01).item()
                p99 = torch.quantile(frame_flat, 0.99).item()
                
                if p99 > p1:
                    # Normalize using percentiles
                    decoded_frame = (decoded_frame - p1) / (p99 - p1)
                    decoded_frame = torch.clamp(decoded_frame, 0, 1)
                    print(f"  Normalized using percentiles (p1={p1:.3f}, p99={p99:.3f})")
                else:
                    # Fallback to simple clamp
                    decoded_frame = torch.clamp(decoded_frame, 0, 1)
                    print(f"  Clamped to [0, 1] range")
            
            # decoded_frame should be (b, c, t, h, w)
            # Convert to numpy: (b, c, t, h, w) -> (h, w, c)
            if decoded_frame.ndim == 5:
                # (b, c, t, h, w) -> take first batch, all channels, first time, all spatial
                decoded_frame = decoded_frame[0, :, 0, :, :]  # (c, h, w) - take first batch and first time
                decoded_frame = decoded_frame.permute(1, 2, 0)  # (h, w, c)
            elif decoded_frame.ndim == 4:
                # (b, c, h, w) or (c, t, h, w)
                if decoded_frame.shape[0] == 1:  # batch dimension
                    decoded_frame = decoded_frame.squeeze(0)  # (c, h, w) or (c, t, h, w)
                if decoded_frame.ndim == 4:  # (c, t, h, w)
                    decoded_frame = decoded_frame[:, 0, :, :]  # (c, h, w)
                decoded_frame = decoded_frame.permute(1, 2, 0)  # (h, w, c)
            elif decoded_frame.ndim == 3:
                # (c, h, w)
                decoded_frame = decoded_frame.permute(1, 2, 0)  # (h, w, c)
            else:
                raise ValueError(f"Unexpected decoded_frame shape: {decoded_frame.shape}")
            
            decoded_frame = decoded_frame.cpu().numpy()
            decoded_frame = np.clip(decoded_frame, 0, 1)
            decoded_frame = (decoded_frame * 255).astype(np.uint8)
            
            # Validate frame
            if decoded_frame.shape[2] != 3:
                print(f"Warning: Frame has {decoded_frame.shape[2]} channels, expected 3")
                if decoded_frame.shape[2] == 1:
                    decoded_frame = np.repeat(decoded_frame, 3, axis=2)
            
            all_frames.append(decoded_frame)
            print(f"  Successfully decoded frame, shape: {decoded_frame.shape}, min: {decoded_frame.min()}, max: {decoded_frame.max()}")
        except Exception as e:
            print(f"Error decoding frame at step {step+1}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: use previous frame or create a placeholder
            if len(all_frames) > 0:
                all_frames.append(all_frames[-1].copy())  # Repeat last frame
                print(f"  Using previous frame as fallback")
            else:
                placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
                all_frames.append(placeholder)
    
    # Save video
    print(f"Saving {len(all_frames)} frames to video...")
    # Get fps and frames_per_action from config if available
    if args.config:
        config = load_config(args.config)
        fps = config.get('generation', {}).get('fps', 15)
        frames_per_action = config.get('generation', {}).get('frames_per_action', fps)  # Default: 1 second per action
    else:
        fps = 15
        frames_per_action = fps  # Default: 1 second per action
    
    # Repeat each frame multiple times so each action lasts at least 1 second
    print(f"Repeating each frame {frames_per_action} times (1 action = {frames_per_action/fps:.2f} seconds)...")
    expanded_frames = []
    for i, frame in enumerate(all_frames):
        for _ in range(frames_per_action):
            expanded_frames.append(frame)
    all_frames = expanded_frames
    print(f"Expanded to {len(all_frames)} frames")
    
    # Ensure minimum video length (at least 2 seconds)
    min_frames = max(2 * fps, 30)  # At least 2 seconds or 30 frames
    if len(all_frames) < min_frames:
        print(f"Video too short ({len(all_frames)} frames), repeating frames to reach {min_frames} frames...")
        repeat_factor = (min_frames // len(all_frames)) + 1
        all_frames = (all_frames * repeat_factor)[:min_frames]
        print(f"Expanded to {len(all_frames)} frames")
    
    save_video(all_frames, output, fps=fps)
    video_duration = len(all_frames) / fps
    print(f"Done! Video duration: {video_duration:.2f} seconds ({len(all_frames)} frames at {fps} fps)")


if __name__ == '__main__':
    main()

