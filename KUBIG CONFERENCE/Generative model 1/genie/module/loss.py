import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import get_model
from torchvision import transforms

from torch.nn.functional import relu
from torch.nn.functional import mse_loss
from torch.nn.modules.loss import _Loss
from typing import Iterable, Tuple

from genie.module.misc import NamingProbe
from genie.module.misc import RecordingProbe
from genie.module.discriminator import FrameDiscriminator, VideoDiscriminator
from genie.utils import pick_frames

VGG16_RELU_LAYERS = [
    'features.1',
    'features.3',
    'features.6',
    'features.8',
    'features.11',
    'features.13',
    'features.15',
    'features.18',
    'features.20',
    'features.22',
    'features.25',
    'features.27',
    'features.29',
    'classifier.1',
    'classifier.4',
]

class PerceptualLoss(_Loss):
    
    def __init__(
        self,
        model_name : str = 'vgg16',
        model_weights : str | None = 'DEFAULT',
        num_frames : int = 4,
        feat_layers : str | Iterable[str] = ('features.6', 'features.13', 'features.18', 'features.25'),
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        self.percept_model = get_model(model_name, weights=model_weights)
            
        # Freeze the perceptual model
        self.percept_model.eval()
        for param in self.percept_model.parameters():
            param.requires_grad = False
            
        # Attach the naming probe to make sure every layer
        # in the percept model has a unique identifier
        self.namer = NamingProbe()
        handles = [
            module.register_forward_hook(self.namer)
            for name, module in self.percept_model.named_modules()
        ]
        
        # Fake forward pass to the model to trigger the probe
        with torch.no_grad():
            _ = self.percept_model(torch.randn(1, 3, 224, 224))
        for handle in handles: handle.remove()
        
        # Attach hooks to the model at desired locations
        self.probe = RecordingProbe()
        self.hook_handles = [
            module.register_forward_hook(self.probe)
            for name, module in self.percept_model.named_modules()
            if name in feat_layers
        ]
        
        assert len(self.hook_handles) > 0, 'No valid layers found in the perceptual model.'
        
    def forward(self, rec_video : Tensor, inp_video : Tensor) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        frames_idxs = torch.cat([
            torch.randperm(t, device=inp_video.device)[:self.num_frames]
            for _ in range(b)]
        )
        
        fake_frames = pick_frames(rec_video, frames_idxs=frames_idxs)
        real_frames = pick_frames(inp_video, frames_idxs=frames_idxs)
        
        # Clamp frames to valid range [0, 1] to prevent overflow/underflow in VGG
        # VGG expects normalized inputs in [0, 1] range
        fake_frames = torch.clamp(fake_frames, 0, 1)
        real_frames = torch.clamp(real_frames, 0, 1)
        
        # Resize frames to 224x224 if needed (VGG16 expects 224x224 input)
        # But only if frames are smaller than 224x224
        if h < 224 or w < 224:
            import torch.nn.functional as F
            fake_frames = F.interpolate(fake_frames, size=(224, 224), mode='bilinear', align_corners=False)
            real_frames = F.interpolate(real_frames, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Use fp32 for perceptual model to prevent NaN in mixed precision
        # VGG16 can be sensitive to precision issues
        with torch.cuda.amp.autocast(enabled=False):
            fake_frames_fp32 = fake_frames.float()
            real_frames_fp32 = real_frames.float()
            
            # CRITICAL: Apply ImageNet normalization before VGG16 forward pass
            # VGG16 was trained on ImageNet with these statistics
            # Without normalization, features will be in wrong scale and loss won't decrease
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=fake_frames_fp32.device, dtype=fake_frames_fp32.dtype).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=fake_frames_fp32.device, dtype=fake_frames_fp32.dtype).view(1, 3, 1, 1)
            
            # Normalize: (x - mean) / std
            fake_frames_normalized = (fake_frames_fp32 - imagenet_mean) / imagenet_std
            real_frames_normalized = (real_frames_fp32 - imagenet_mean) / imagenet_std
            
            # Get the perceptual features for the input
            _ = self.percept_model(fake_frames_normalized)
            fake_feat = self.probe.features
            self.probe.clean()
            
            # Get the perceptual features for the target
            _ = self.percept_model(real_frames_normalized)
            real_feat = self.probe.features
            self.probe.clean()
        
        # Compute MSE loss for each feature layer
        losses = []
        for k in fake_feat.keys():
            fake_f = fake_feat[k]
            real_f = real_feat[k]
            
            # Compute MSE loss
            layer_loss = mse_loss(fake_f, real_f)
            losses.append(layer_loss)
        
        # Return mean of all layer losses
        return torch.stack(losses).mean()
        
    def __del__(self) -> None:
        for handle in self.hook_handles:
            handle.remove()

class GANLoss(_Loss):
    
    def __init__(
        self,
        discriminate : str = 'frames',
        num_frames : int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert discriminate in ('frames', 'video'), 'Invalid discriminator type. Must be either "frames" or "video".'
        
        self.disc = FrameDiscriminator(**kwargs) if discriminate == 'frames' else VideoDiscriminator(**kwargs)
        
        self.num_frames = num_frames
        self.discriminate = discriminate
        
    def get_examples(
        self,
        rec_video : Tensor,
        inp_video : Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b, c, t, h, w = inp_video.shape
        
        if self.discriminate == 'video':
            return rec_video, inp_video
        
        # Extract a set of random frames from the input video
        frame_idxs = torch.cat([
            torch.randperm(t, device=inp_video.device)[:self.num_frames]
            for _ in range(b)]
        )
        fake = pick_frames(rec_video, frame_idxs)
        real = pick_frames(inp_video, frame_idxs)
        
        return fake, real
        
    def forward(
        self,
        rec_video : Tensor,
        inp_video : Tensor,
        train_gen : bool, 
    ) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        fake, real = self.get_examples(rec_video, inp_video)
        
        # Convert inputs to fp32 for discriminator to prevent dtype mismatch in mixed precision
        # Discriminator should always run in fp32 for stability
        fake = fake.float() if fake.dtype != torch.float32 else fake
        real = real.float() if real.dtype != torch.float32 else real
        
        # Compute discriminator opinions for real and fake frames
        # Use fp32 for discriminator to prevent NaN in mixed precision training
        with torch.cuda.amp.autocast(enabled=False):
            fake_score : Tensor = self.disc(fake) if     train_gen else self.disc(fake.detach())
            real_score : Tensor = self.disc(real) if not train_gen else None
        
        # Compute hinge loss for the discriminator
        if train_gen:
            # Generator loss: maximize fake_score (minimize -fake_score)
            # The hinge loss naturally bounds fake_score: if fake_score > 1, hinge loss = 0
            # So generator loss = -fake_score is naturally bounded
            gan_loss = -fake_score.mean()
        else:
            # Discriminator loss: minimize fake_score and maximize real_score
            # Hinge loss: max(0, 1 + fake_score) + max(0, 1 - real_score)
            # This naturally bounds the loss: both terms are >= 0
            gan_loss = (relu(1 + fake_score) + relu(1 - real_score)).mean()
        
        return gan_loss