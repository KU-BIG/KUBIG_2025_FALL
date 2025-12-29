# Mini-Genie: Generative Interactive Environments

An implementation of generative world models for interactive environments based on the Genie framework. This project implements a spatiotemporal video tokenizer, latent action model, and dynamics model for learning controllable world simulators from video data.

## Overview

This repository provides a PyTorch Lightning implementation of generative world models that can learn to simulate interactive environments from video observations. The model consists of three main components:

1. **Video Tokenizer**: A spatiotemporal autoencoder with vector quantization for compressing video frames into discrete latent representations
2. **Latent Action Model**: An action encoder-decoder that learns to infer and represent latent actions from frame transitions
3. **Dynamics Model**: A MaskGIT-based transformer that predicts future latent states conditioned on current states and actions

## Reference

This work is based on the following paper:

**Genie: Generative Interactive Environments**  
Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S., Behbahani, F., Chan, S., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., Freitas, N., Singh, S., & Rocktäschel, T. (2024).

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 2.3.0 or compatible version

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/mini-genie.git
cd mini-genie
pip install -e .
pip install -r requirements.txt
```

### Dependencies

Core dependencies include:

- PyTorch 2.3.0
- PyTorch Lightning 2.2.4
- einops 0.8.0
- setuptools 69.5.1

Additional dependencies for data generation:

- opencv-python
- gym
- procgen

## Project Structure

```
mini-genie/
├── genie/                    # Core model implementations
│   ├── tokenizer.py         # Spatiotemporal video tokenizer
│   ├── action.py            # Latent action model
│   ├── dynamics.py          # MaskGIT dynamics model
│   ├── genie.py             # Main Genie model
│   ├── dataset.py           # Data loading utilities
│   └── module/              # Neural network modules
│       ├── attention.py     # Attention mechanisms
│       ├── quantization.py  # Vector quantization (LFQ)
│       ├── discriminator.py # GAN discriminator
│       └── ...
├── config/                   # Training configurations
│   ├── tokenize.yaml        # Tokenizer training config
│   ├── action.yaml          # Action model training config
│   └── agent.yaml           # Agent training config
├── tokenizer.py             # Tokenizer training script
├── genie.py                 # Full model training script
├── agent.py                 # Agent training script
├── sample.py                # Data generation script
└── test/                    # Unit tests
```

## Usage

### 1. Data Generation

Generate training data from Procgen environments:

```bash
python sample.py \
    --env_name Ninja \
    --num_envs 100 \
    --timeout 1000 \
    --root /workspace/data \
    --split train \
    --fps 15 \
    --resize 64 \
    --distribution_mode easy
```

Generate validation data:

```bash
python sample.py \
    --env_name Ninja \
    --num_envs 20 \
    --timeout 1000 \
    --root /workspace/data \
    --split val \
    --fps 15 \
    --resize 64 \
    --distribution_mode easy \
    --seed_offset 1000
```

### 2. Training the Video Tokenizer

Train the spatiotemporal video tokenizer:

```bash
python tokenizer.py fit \
    --config config/tokenize.yaml \
    --data.root /workspace/data \
    --data.env_name Ninja \
    --data.batch_size 4 \
    --data.num_workers 4
```

Key hyperparameters in `config/tokenize.yaml`:
- `d_codebook`: Number of bits for codebook (10 → 1024 codes)
- `n_codebook`: Number of codebooks (1)
- `lfq_commit_weight`: Commitment loss weight (0.25)
- `lfq_entropy_weight`: Entropy regularization (0.1)

### 3. Training the Latent Action Model

Train the action model with a pretrained tokenizer:

```bash
python genie.py fit \
    --config config/action.yaml \
    --data.root /workspace/data \
    --data.env_name Ninja \
    --data.batch_size 4 \
    --data.num_workers 4
```

Update the tokenizer checkpoint path in `config/action.yaml`:

```yaml
model:
  tokenizer_checkpoint: /path/to/tokenizer/checkpoint.ckpt
  action_d_codebook: 6  # 64 action codes
  action_n_codebook: 1
  dynamics_embed_dim: 512
  dynamics_maskgit_steps: 25
```

### 4. Training the Agent (Optional)

Train a reinforcement learning agent in the learned world model:

```bash
python agent.py fit \
    --config config/agent.yaml
```

Update checkpoint paths in `config/agent.yaml`:

```yaml
model:
  tokenizer_ckpt: /path/to/tokenizer/checkpoint.ckpt
  world_ckpt: /path/to/action/checkpoint.ckpt
  horizon: 4
  gamma: 0.99
  lr: 3e-4
```

### 5. Testing and Evaluation

Run unit tests:

```bash
# Test tokenizer
python -m pytest test/test_tokenizer.py

# Test action model
python -m pytest test/test_action.py

# Test dynamics model
python -m pytest test/test_dynamics.py

# Test attention mechanisms
python -m pytest test/test_attention.py

# Run all tests
python -m pytest test/
```

Test video generation dynamics:

```bash
python test_dynamics_video.py \
    --tokenizer_ckpt /path/to/tokenizer/checkpoint.ckpt \
    --world_ckpt /path/to/action/checkpoint.ckpt \
    --video_path /path/to/test/video.mp4 \
    --num_frames 64
```

## Model Architecture

### Video Tokenizer

The video tokenizer compresses video frames into discrete latent representations using a spatiotemporal autoencoder:

**Encoder:**
- Spatiotemporal downsampling via 3D convolutions (factor: 4x spatial, 1x temporal)
- Multi-head spatial-temporal attention layers for capturing long-range dependencies
- Projects input (B, T, C, H, W) to latent space (B, T, D, H', W')

**Quantization:**
- Lookup-free quantization (LFQ) instead of traditional VQ-VAE
- Binary encoding without explicit codebook lookup
- Entropy and diversity regularization for better code utilization
- Supports multiple codebook groups for richer representations

**Decoder:**
- Inverse spatiotemporal upsampling (depth-to-space operations)
- Reconstructs original resolution: (B, T, D, H', W') → (B, T, C, H, W)
- Mirrors encoder architecture with transposed operations

**Loss Functions:**
```python
L_total = L_recon + λ_adv * L_adv + λ_commit * L_commit + λ_entropy * L_entropy
```
- Reconstruction loss (L1 + perceptual)
- Adversarial loss from multi-scale discriminator
- Commitment loss for quantization stability
- Entropy loss for codebook diversity

### Latent Action Model

Infers discrete latent actions from consecutive frame observations:

**Action Inference:**
- Takes two consecutive latent frames: z_t and z_{t+1}
- Encodes the difference through residual blocks
- Quantizes into discrete action codes via LFQ
- Action space: 2^{d_codebook} possible discrete actions

**Architecture:**
```python
# Encoding pipeline
latent_diff = z_{t+1} - z_t  # Compute latent difference
action_hidden = encoder(latent_diff)  # Process through residual blocks
action_code = quantize(action_hidden)  # LFQ quantization
```

**Training:**
- Self-supervised learning from video sequences
- Reconstructs frame transitions to verify action consistency
- Independent codebook from video tokenizer

### Dynamics Model

Predicts future latent states using MaskGIT-based masked prediction:

**Core Mechanism:**
- Input: Current latent state z_t + action code a_t
- Output: Next latent state z_{t+1}
- Uses bidirectional transformer with masking strategy

**MaskGIT Sampling:**
1. Start with fully masked target tokens
2. Predict all positions in parallel
3. Unmask highest-confidence predictions
4. Repeat for T iterations (typically 8-25 steps)
5. Gradually refine predictions from coarse to fine

**Masking Strategy:**
```python
# During training: random masking
mask_ratio = cosine_schedule(training_step)
masked_tokens = random_mask(z_{t+1}, ratio=mask_ratio)

# During inference: iterative unmasking
for step in range(T):
    probs = model(masked_tokens, z_t, a_t)
    confident_tokens = top_k(probs, k=unmask_count)
    masked_tokens[confident_tokens] = predicted_values
```

**Attention Mechanism:**
- Non-causal (bidirectional) attention over spatial positions
- Causal attention over temporal dimension
- Combines current state, action embeddings, and masked future

**Loss Function:**
```python
L_dynamics = CrossEntropy(predicted_tokens, target_tokens, mask=mask)
```
- Only computes loss on masked positions
- Encourages model to predict based on context

## Technical Implementation Details

### Spatiotemporal Convolution

The core building block uses 3D convolutions for joint spatial-temporal processing:

```python
# Spacetime downsampling
conv3d = nn.Conv3d(
    in_channels=3,
    out_channels=512,
    kernel_size=(3, 3, 3),  # (time, height, width)
    stride=(1, 4, 4),        # Downsample 4x spatially
    padding=(1, 1, 1)
)
```

### Lookup-Free Quantization (LFQ)

Unlike VQ-VAE, LFQ eliminates the codebook lookup operation:

**Quantization Process:**
1. Project continuous features to D-dimensional binary codes
2. Apply sign function: `codes = sign(features)`
3. Convert binary codes to integers for discrete representation
4. No gradient-blocking "stop-gradient" needed

**Advantages:**
- Simpler optimization (no codebook collapse)
- Better gradient flow
- Implicit diversity through entropy regularization
- Scales to larger codebook sizes

### Training Strategy

**Stage 1: Video Tokenizer**
- Train encoder-decoder with reconstruction + adversarial losses
- Use mixed precision for memory efficiency
- Discriminator provides perceptual quality signal
- Typically requires 30-50 epochs on game footage

**Stage 2: Action Model + Dynamics**
- Freeze tokenizer weights (set `requires_grad=False`)
- Train action encoder and dynamics transformer jointly
- Action model learns to infer latent actions from transitions
- Dynamics learns to predict next state given (state, action)

**Stage 3: Agent (Optional)**
- Train policy network in learned world model
- Imagination rollouts using dynamics model
- Reinforcement learning in latent space
- Can be trained without real environment interaction

### Data Processing

**Video Preprocessing:**
```python
# Normalization to [-1, 1]
frames = (frames / 127.5) - 1.0

# Temporal windowing
sequences = sliding_window(frames, window_size=16, stride=1)
```

**Action Encoding:**
- Actions inferred from consecutive frames
- No ground-truth action labels required
- Self-supervised learning from visual observations

### Key Hyperparameters

**Tokenizer:**
- `d_codebook=10`: 1024 discrete codes (2^10)
- `n_embd=512`: Hidden dimension
- `lfq_commit_weight=0.25`: Balances reconstruction vs quantization
- `lfq_entropy_weight=0.1`: Encourages code diversity

**Dynamics:**
- `embed_dim=512`: Transformer hidden size
- `maskgit_steps=25`: Iterative refinement steps
- `temperature=1.0`: Sampling temperature for stochasticity

**Training:**
- Learning rate: 1e-4 with AdamW optimizer
- Batch size: 4-8 depending on GPU memory
- Mixed precision (FP16) for efficiency
- Gradient clipping: 1.0 for stability

## Configuration

Training configurations are located in the `config/` directory:

- `tokenize.yaml`: Video tokenizer hyperparameters
- `action.yaml`: Action model and dynamics hyperparameters
- `agent.yaml`: RL agent hyperparameters

Key configuration options:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 40

model:
  d_codebook: 10          # Codebook size (2^10 = 1024)
  n_codebook: 1           # Number of codebooks
  optimizer:
    class_path: torch.optim.AdamW
    init_args:
      lr: 1e-4
```

## Training Pipeline

The recommended training pipeline:

1. **Generate data** from Procgen environments (train and validation splits)
2. **Train video tokenizer** until reconstruction quality is satisfactory
3. **Train action model** using the frozen tokenizer
4. **Optional**: Train RL agent in the learned world model

Monitor training with TensorBoard:

```bash
tensorboard --logdir log/
```

## Checkpoints

Training checkpoints are saved in the `log/` directory:

```
log/
├── genie-tokenizer/
│   └── version_X/
│       └── checkpoints/
│           └── best-epoch=X-step=X.ckpt
├── genie-action/
│   └── version_X/
│       └── checkpoints/
│           └── best-epoch=X-step=X.ckpt
└── genie-agent/
    └── version_X/
        └── checkpoints/
            └── agent-epoch=X.ckpt
```

## Performance Considerations

- Use mixed precision training (`precision: 16-mixed`) for memory efficiency
- Adjust batch size based on GPU memory
- Use multiple workers for data loading (`num_workers: 4`)
- Enable gradient checkpointing for large models if needed
- Monitor GPU utilization and adjust hyperparameters accordingly

## Troubleshooting

**Out of memory errors:**
- Reduce batch size
- Reduce number of frames per video
- Use smaller model dimensions
- Enable gradient checkpointing

**Poor reconstruction quality:**
- Train tokenizer longer
- Adjust discriminator loss weight
- Increase model capacity
- Check data quality and normalization

**Unstable training:**
- Reduce learning rate
- Adjust gradient clipping
- Check loss weights and regularization
- Ensure proper data normalization

## License

See the [LICENSE](mini-genie/LICENSE) file for details.

## Citation

If you use this code in your research, please cite the original Genie paper:

```bibtex
@article{bruce2024genie,
  title={Genie: Generative Interactive Environments},
  author={Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and Steigerwald, Richie and Apps, Chris and others},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This implementation builds upon the concepts introduced in the Genie paper and uses PyTorch Lightning for training infrastructure. The codebase is designed for research and educational purposes.
