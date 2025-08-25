import os, sys, math, json, argparse, warnings, tempfile, subprocess, shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchaudio
import cv2
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from torchvggish import vggish, vggish_input
from method_for_inference import _smart_load_state_dict, _normalize_state_dict

# Video feature extractors (2fps, ResNet-152)
class ResNet152Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        m = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.backbone.eval()
        self.backbone.to(device)
        self.device = device
        w = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=w.transforms().mean, std=w.transforms().std)
        ])
    
    @torch.no_grad()
    def encode_batch(self, frame_bgr, batch_size=32):
        """
        frames_bgt : list[np.ndarray(H,W,3)] in BGR (from OpenCV)
        return: np.ndarray [N, 2048]
        """
        if len(frame_bgr) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        
        feats_all = []
        for i in range(0, len(frame_bgr), batch_size):
            chunk = frame_bgr[i:i+batch_size]
            batch = torch.stack([self.tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in chunk]).to(self.device)
            feats = self.backbone(batch).squeeze(-1).squeeze(-1)  # [B, 2048]
            feats_all.append(feats.cpu())
            del batch, feats
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(feats_all, dim=0).numpy().astype(np.float32)

# Audio feature extractors (2fps, VGGish)
class VGGishEncoder:
    def __init__(self, ckpt_path_or_auto: str, device):
        super().__init__()
        self.device = device if str(device).startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")
        model = vggish()
        
        use_pretrained = (not ckpt_path_or_auto) or ckpt_path_or_auto == "auto" or ckpt_path_or_auto.endswith(".h5")
        if not use_pretrained:
            state = _smart_load_state_dict(ckpt_path_or_auto)
            state = _normalize_state_dict(state)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        if hasattr(model, "postprocess"):
            model.postprocess = False
        
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
    
    @torch.no_grad()
    def encode_audio(self, wav:torch.Tensor, sr:int, target_fps=2.0,
                     expected_len:int | None=None, start_sec:float=0.0,
                     duration_sec:float | None=None, 
                     batch_size: int = 16, block_sec: float = 90.0,
                     use_amp:bool = True) -> np.ndarray:
        """
        wav: torch.Tensor [1, T] (float32, -1..1), sr: sample rate
        return: np.ndarray [T_tokens, 128] aligned to 2 fps (dt=0.5s)
        """
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        elif not torch.is_tensor(wav):
            wav = torch.tensor(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)           # [1, T]
        if wav.dtype != torch.float32:
            wav = wav.float()
        if wav.ndim == 2 and wav.shape[0] > 1:   # 모노화
            wav = wav.mean(dim=0, keepdim=True)
        
        # 시작 offset 보정
        if start_sec and start_sec > 0:
            cut = int(round(start_sec * sr))
            if cut < wav.shape[-1]:
                wav = wav[:, cut:]
            else:
                return np.zeros((0, 128), np.float32)
        
        # 16k로 resample
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            sr = 16000
        wav = torch.clamp(wav, -1.0, 1.0)
            
        wav_np = wav.squeeze(0).cpu().numpy()
        total_len = len(wav_np) / sr
        if duration_sec is None:
            duration_sec = total_len
        
        hop_vgg = 0.48
        win_vgg = 0.96
        margin = 1.0  # 블록 경계 손실 방지
        emb_chunks: list[np.ndarray] = []
        
        t0 = 0.0
        while t0 < duration_sec:
            t1 = min(total_len, t0 + block_sec + margin)
            s = int(t0 * sr); e = int(t1 * sr)
            block_wav = wav_np[s:e]
            ex_block = vggish_input.waveform_to_examples(block_wav, sr)  # np [Nb,96,64]
            
            if isinstance(ex_block, np.ndarray):
                num_items = int(ex_block.size)
            else:
                # torch.Tensor일 가능성
                ex_block = torch.as_tensor(ex_block)
                num_items = int(ex_block.numel())
                
            # ex_block: np.ndarray  (보통 [N,96,64] 또는 [N,1,96,64])
            if num_items > 0:
                N = int(ex_block.shape[0])
                for i in range(0, N, batch_size):
                    chunk = ex_block[i:i+batch_size]                      # numpy or tensor
                    x = torch.as_tensor(chunk, dtype=torch.float32)  # [B, 1, 96, 64]
                    if x.ndim == 3:
                        # [B,96,64]  → [B,1,96,64]
                        x = x.unsqueeze(1)  # [B,96,64] -> [B,1,96,64]
                    elif x.ndim == 4:
                        if x.shape[1] != 1 and x.shape[-1] == 1:  # [B,1,96,64]는 그대로, [B,96,64,1]이면 채널축 앞으로
                            x = x.permute(0, 3, 1, 2)   # [B,96,64,1]→[B,1,96,64]
                    else:
                        raise RuntimeError(f"Unexpected VGGish input shape: {tuple(x.shape)}")

                    x = x.to(self.device, non_blocking=False)
                    y = self.model(x)                                     # [B,128]
                    emb_chunks.append(y.detach().cpu().numpy())
                    del x, y, chunk
                    
            del ex_block, block_wav
            t0 += block_sec

        if not emb_chunks:
            return np.zeros((0, 128), np.float32)

        emb = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
        
        # Map VGGish hop (~0.48s) to exact 2 fps (0.5s)
        T_tokens = int(math.ceil(duration_sec * target_fps))
        t_centers = (np.arange(T_tokens) + 0.5) / target_fps
        # VGGish timestamps (center)
        t_vgg     = np.arange(emb.shape[0]) * hop_vgg + (win_vgg / 2.0)
        # nearest neighbor alignment
        idx_r = np.searchsorted(t_vgg, t_centers, side="left")
        idx_l = np.clip(idx_r - 1, 0, emb.shape[0]-1)
        idx_r = np.clip(idx_r,     0, emb.shape[0]-1)
        choose_left = (t_centers - t_vgg[idx_l]) <= (t_vgg[idx_r] - t_centers)
        idx = np.where(choose_left, idx_l, idx_r)
        aligned = emb[idx]
        
        # 비디오 토큰 길이에 맞추기 (clipping/padding)
        if expected_len is not None:
            if aligned.shape[0] > expected_len:
                aligned = aligned[:expected_len]
            elif aligned.shape[0] < expected_len:
                pad = np.zeros((expected_len - aligned.shape[0], aligned.shape[1]), np.float32)
                aligned = np.concatenate([aligned, pad], axis=0)

        return aligned
    
# 512-D로 차원 맞추기
class PCAdaptor:
    def __init__(self, components: np.ndarray, mean:np.ndarray,
                 explained_variance:np.ndarray | None=None):
        if components.shape[0] < components.shape[1]:
            # [k, d] ok
            pass
        else:
            # [d, k] → transpose
            components = components.T
        self.components = components.astype(np.float32)
        self.mean = mean.astype(np.float32)
        self.exp_var = None if explained_variance is None else explained_variance.astype(np.float32)
    
    @classmethod
    def load_npz(cls, path: str) -> "PCAdaptor":
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        d = np.load(path)
        comps = d["components"]
        mean  = d["mean"]
        expv  = d["explained_variance"] if "explained_variance" in d.files else None
        return cls(comps, mean, expv)

    def transform(self, X: np.ndarray, whiten: bool = False) -> np.ndarray:
        """X: [T, d] → [T, k]"""
        Xc = (X.astype(np.float32) - self.mean[None, :])
        Z  = Xc @ self.components.T  # [T, k]
        if whiten and self.exp_var is not None:
            Z = Z / np.sqrt(self.exp_var[None, :] + 1e-8)
        return Z.astype(np.float32)

    @staticmethod
    def save_npz(path: str, components: np.ndarray, mean: np.ndarray,
                 explained_variance: np.ndarray | None = None):
        if explained_variance is None:
            np.savez(path, components=components, mean=mean)
        else:
            np.savez(path, components=components, mean=mean,
                     explained_variance=explained_variance)

# 512차원으로 추출한 feature의 차원 맞추기
class LinearAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, path=None, allow_random=False, device="cpu"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        if path:
            self.proj.load_state_dict(torch.load(path, map_location="cpu"))
        elif not allow_random and in_dim != out_dim:
            raise RuntimeError("2048→512 가중치가 없습니다. --video_adapter_path 또는 --allow_random_adapter 또는 --pca_on_the_fly 를 사용하세요.")
        else:
            # Xavier init (임시)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.eval().to(device)
    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).float()
        y = self.proj(t)
        return y.detach().cpu().numpy().astype(np.float32)


class DummyData:
    """
    networks.Net이 기대하는 최소 속성만 채워주는 더미 데이터셋.
    필요한 alias들을 함께 넣어 AttributeError를 방지.
    """
    def __init__(self, num_classes: int, number_frames_in_window: int,
                 video_feat_dim: int = 512, audio_feat_dim: int = 512):
        self.num_classes = num_classes
        self.number_frames_in_window = number_frames_in_window

        # 일부 구현에서 다른 이름을 참조할 수 있어 alias 제공
        self.max_samples = number_frames_in_window
        self.window_size = number_frames_in_window
        self.win_len = number_frames_in_window
        self.token_per_window = number_frames_in_window

        # 피처 차원(네트워크가 필요로 할 수 있음)
        self.video_feat_dim = video_feat_dim
        self.audio_feat_dim = audio_feat_dim
        # 합쳐 쓰는 구현을 대비한 편의 필드
        self.input_dim = video_feat_dim + audio_feat_dim
        self.feat_dim = video_feat_dim