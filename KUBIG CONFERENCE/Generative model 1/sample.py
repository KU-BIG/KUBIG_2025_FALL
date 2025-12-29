import cv2
import gym
from os import path
from os import makedirs

import argparse
from tqdm.auto import trange

# Default root; override via --root
ROOT = '/mnt/d/workspace/KUBIG/25-2_conference/open-genie/data'

def save_frames_to_video(frames, output_file, fps=30):
    """Save RGB frames to an mp4 file using OpenCV."""
    height, width, _ = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)

    for frame in frames:
        # Env returns RGB; VideoWriter expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

def main(args):
    env_name = args.env_name
    num_envs = args.num_envs
    timeout  = args.timeout
    resize   = args.resize
    split    = args.split
    fps      = args.fps
    distribution_mode = args.distribution_mode
    seed_offset = args.seed_offset
    start_seed = args.start_seed

    for i in trange(num_envs, desc=f'Generating {env_name} videos'):
        seed = start_seed + i  # File naming seed (starts from start_seed)
        env = gym.make(
            f'procgen:procgen-{env_name.lower()}-v0',
            distribution_mode=distribution_mode,
            render_mode='rgb_array',
            start_level=seed + seed_offset,
            num_levels=1,
            use_sequential_levels=True,
        )

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        frames = [obs]

        for _ in range(timeout - 1):
            step_out = env.step(env.action_space.sample())
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            frames.append(obs)
        
        env.close()
        
        if resize:
            frames = [
                cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
                for frame in frames
            ]

        savepath = path.join(args.root, env_name, split, f'{str(seed).zfill(4)}.mp4')
        makedirs(path.dirname(savepath), exist_ok=True)
        save_frames_to_video(frames, savepath, fps=fps)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate videos of a gym environment')
    parser.add_argument('--env_name', type=str, default='Ninja', help='Name of the environment')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--timeout', type=int, default=1000, help='Steps per video')
    parser.add_argument('--root', type=str, default=ROOT, help='Root folder where to save the videos')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset split to save')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second of the output video')
    parser.add_argument('--resize', type=int, default=0, help='Optional resize (pixels). 0 keeps original (64x64).')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=['easy', 'hard', 'exploration', 'memory', 'extreme'], help='Procgen difficulty; easy is clearer visually')
    parser.add_argument('--seed_offset', type=int, default=0, help='Add to start_level for reproducibility across splits')
    parser.add_argument('--start_seed', type=int, default=0, help='Starting seed for file naming (allows continuing from existing files)')

    args = parser.parse_args()

    main(args)