"""
Main inference script for Genie model.
Loads trained models and generates videos comparing agent actions vs ground truth actions.
"""
import torch
import yaml
import argparse
from pathlib import Path
import numpy as np

from genie.inference import (
    GenieInference,
    load_video_frames,
    save_video,
    extract_ground_truth_action,
)


def main():
    parser = argparse.ArgumentParser(description='Genie Inference')
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference.yaml',
        help='Path to inference config file'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to input video (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output video (overrides config)'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.video:
        config['inference']['video_path'] = args.video
    if args.output:
        config['inference']['output_path'] = args.output
    
    # Get paths
    tokenizer_ckpt = config['checkpoints']['tokenizer']
    world_ckpt = config['checkpoints']['world']
    agent_ckpt = config['checkpoints']['agent']
    video_path = config['inference']['video_path']
    output_path = config['inference']['output_path']
    num_frames = config['inference']['num_frames_to_generate']
    steps_per_frame = config['inference']['steps_per_frame']
    temperature = config['inference'].get('temperature', 1.0)
    topk = config['inference'].get('topk', 50)
    fps = config['inference'].get('fps', 10)
    device = config['inference']['device']
    
    # Model hyperparameters
    horizon = config['model']['horizon']
    gamma = config['model']['gamma']
    entropy_coef = config['model']['entropy_coef']
    
    print("=" * 60)
    print("Genie Inference")
    print("=" * 60)
    print(f"Tokenizer checkpoint: {tokenizer_ckpt}")
    print(f"World checkpoint: {world_ckpt}")
    print(f"Agent checkpoint: {agent_ckpt}")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Number of frames to generate: {num_frames}")
    print("=" * 60)
    
    # Initialize inference
    inference = GenieInference(
        tokenizer_ckpt=tokenizer_ckpt,
        world_ckpt=world_ckpt,
        agent_ckpt=agent_ckpt,
        device=device,
        horizon=horizon,
        gamma=gamma,
        entropy_coef=entropy_coef,
    )
    
    # Load video frames
    print(f"\nLoading video from {video_path}...")
    initial_frames = load_video_frames(video_path, num_frames=2, start_frame=0)
    print(f"Loaded initial frames: {initial_frames.shape}")
    
    # Load full video for ground truth comparison
    full_video = load_video_frames(video_path, num_frames=2 + num_frames, start_frame=0)
    print(f"Loaded full video: {full_video.shape}")
    
    # Generate with agent
    print("\nGenerating video with agent actions...")
    agent_video, agent_actions = inference.generate_with_agent(
        initial_frames=initial_frames,
        num_frames=num_frames,
        steps_per_frame=steps_per_frame,
        temperature=temperature,
        topk=topk,
    )
    print(f"Generated agent video: {agent_video.shape}")
    print(f"Agent actions: {agent_actions.tolist()}")
    
    # Generate with ground truth actions
    print("\nGenerating video with ground truth actions...")
    gt_video, gt_actions = inference.generate_with_ground_truth_actions(
        initial_frames=initial_frames,
        ground_truth_video=full_video,
        num_frames=num_frames,
        steps_per_frame=steps_per_frame,
        temperature=temperature,
        topk=topk,
    )
    print(f"Generated GT video: {gt_video.shape}")
    print(f"Ground truth actions: {gt_actions.tolist()}")
    
    # Compare actions
    print("\n" + "=" * 60)
    print("Action Comparison")
    print("=" * 60)
    print(f"Agent actions:     {agent_actions.tolist()}")
    print(f"Ground truth:     {gt_actions.tolist()}")
    
    # Calculate action match rate
    # Ensure both tensors are on CPU and have same dtype
    if isinstance(agent_actions, torch.Tensor):
        agent_actions = agent_actions.cpu()
    else:
        agent_actions = torch.tensor(agent_actions, device='cpu')
    
    if isinstance(gt_actions, torch.Tensor):
        gt_actions = gt_actions.cpu()
    else:
        gt_actions = torch.tensor(gt_actions, device='cpu')
    
    if len(agent_actions) == len(gt_actions):
        matches = (agent_actions == gt_actions).sum().item()
        match_rate = matches / len(agent_actions) * 100
        print(f"Action match rate: {match_rate:.1f}% ({matches}/{len(agent_actions)})")
    else:
        print("Warning: Action sequences have different lengths")
    
    # Save videos
    print("\n" + "=" * 60)
    print("Saving videos...")
    print("=" * 60)
    
    # Save agent video
    agent_output_path = output_path.replace('.mp4', '_agent.mp4')
    save_video(agent_video, agent_output_path, fps=fps)
    print(f"Saved agent video to: {agent_output_path}")
    
    # Save ground truth video
    gt_output_path = output_path.replace('.mp4', '_ground_truth.mp4')
    save_video(gt_video, gt_output_path, fps=fps)
    print(f"Saved ground truth video to: {gt_output_path}")
    
    # Save comparison video (side by side)
    print("\nCreating side-by-side comparison video...")
    # Concatenate videos side by side
    # Pad to same length
    max_len = max(agent_video.shape[0], gt_video.shape[0])
    
    # Ensure videos are on CPU before processing
    agent_video = agent_video.cpu() if isinstance(agent_video, torch.Tensor) else agent_video
    gt_video = gt_video.cpu() if isinstance(gt_video, torch.Tensor) else gt_video
    
    # Pad agent video
    if agent_video.shape[0] < max_len:
        padding = torch.zeros(max_len - agent_video.shape[0], *agent_video.shape[1:], device='cpu')
        agent_video = torch.cat([agent_video, padding], dim=0)
    
    # Pad GT video
    if gt_video.shape[0] < max_len:
        padding = torch.zeros(max_len - gt_video.shape[0], *gt_video.shape[1:], device='cpu')
        gt_video = torch.cat([gt_video, padding], dim=0)
    
    # Concatenate side by side: (T, C, H, W) -> (T, C, H, 2*W)
    comparison_video = torch.cat([agent_video, gt_video], dim=3)
    
    comparison_output_path = output_path.replace('.mp4', '_comparison.mp4')
    save_video(comparison_video, comparison_output_path, fps=fps)
    print(f"Saved comparison video to: {comparison_output_path}")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

