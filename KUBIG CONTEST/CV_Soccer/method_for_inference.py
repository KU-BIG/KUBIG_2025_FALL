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
from collections import OrderedDict

CLASS_IDS = {
    'background' : 0,
    'kick' : 1,
    'subs' : 2,
    'card' : 3,
    'goal' : 4,
}
HIGHLIGHT_CLASSES = ['kick', 'subs', 'card', 'goal']  # bg 제외한 4개

# 내가 임의로 휴리스틱하게 정한 offset (start_offset, end_offset)
OFFSETS_SEC = {
    'goal' : (14, 55),
    'kick' : (60, 11),
    'card' : (20, 60),
    'subs' : (20, 45),
}

# 파라미터 (결과 보고 조정 OK)
THRESH = {'goal': 0.55, 'kick': 0.50, 'card': 0.50, 'subs': 0.50}
MPD_SEC = {'goal': 20,   'kick': 10,   'card': 15,   'subs': 25}
NMS_IOU = {'goal': 0.5,  'kick': 0.5,  'card': 0.5,  'subs': 0.5}
MERGE_GAP_SEC = {'goal': 4, 'kick': 3, 'card': 4, 'subs': 4}
MIN_LEN_SEC   = {'goal': 6, 'kick': 4, 'card': 4, 'subs': 4}
MAX_LEN_SEC   = {'goal':45, 'kick':25, 'card':25, 'subs':25}

# ---------------------------------------------------------------------------------
def _smart_load_state_dict(ckpt_path: str):
    """
    다양한 체크포인트 포맷을 안전하게 로드:
    - safetensors(.safetensors) → safetensors.torch.load_file
    - torch .pth (zip/pickle) → torch.load(weights_only=True → 실패시 False)
    - Keras .h5 → 명확한 에러 메시지
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        header = f.read(4)

    # Keras/TensorFlow HDF5 (.h5) 시그니처: 0x89 'H' 'D' 'F'
    if header[:3] == b"\x89HD":
        raise RuntimeError(
            f"'{ckpt_path}' looks like a Keras .h5 file (HDF5). "
            "PyTorch can't load it directly. Use torchvggish pretrained=True "
            "or convert to a PyTorch .pth first."
        )

    # safetensors 시그니처
    if header == b"SAFE":
        try:
            from safetensors.torch import load_file as safe_load
        except ImportError as e:
            raise RuntimeError(
                "safetensors checkpoint detected, but 'safetensors' is not installed. "
                "pip install safetensors"
            ) from e
        return safe_load(ckpt_path)

    # torch zip 아카이브(.pth 신형) 시그니처: 'PK'
    if header[:2] == b"PK":
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 일반 pickle 기반 .pth (구형 등)
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _normalize_state_dict(state):
    """
    - state_dict 중첩 해제
    - DataParallel 등 'module.' 프리픽스 제거
    """
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 텐서 dict가 아니면 그대로 반환
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state.keys()):
        return state

    # module. 프리픽스 제거
    needs_strip = any(k.startswith("module.") for k in state.keys())
    if not needs_strip:
        return state

    new_state = OrderedDict()
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    return new_state

# ---------------------------------------------------------------------------------
def pca_on_the_fly_2048_to_512(V2048: np.ndarray, k: int = 512) -> np.ndarray:
    """
    현재 비디오의 2048-D 프레임 특징에서 TruncatedSVD로 k차원 투영(compute-only, 파일 저장 없음).
    """
    # 평균 제거는 TruncatedSVD가 내부적으로 처리하지 않으므로 간단히 center 해줌
    mean = V2048.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = (V2048.astype(np.float32) - mean)
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=0)
    Z = svd.fit_transform(Xc)  # [T, k]
    return Z.astype(np.float32)

# create sliding windows
def slide_windows(feat, window_size_tok, stride_tok):
    """
    feat: np.ndarray [T, D]
    return list of (g0, g1, window_feat [W, D])
    """
    T= len(feat)
    out = []
    for g0 in range(0, max(1, T-window_size_tok+1), stride_tok):
        g1 = min(T, g0 + window_size_tok)
        w = feat[g0:g1]
        if len(w) < window_size_tok:
            pad = np.zeros((window_size_tok - len(w), feat.shape[1]), np.float32)
            w = np.concatenate([w, pad], axis=0)
        out.append((g0, g0+window_size_tok, w))
    if not out:
        out.append((0, window_size_tok, np.zeros((window_size_tok, feat.shape[1]), np.float32)))
    return out

# stitch logits
def stitch_logits(window_logits, T_full, C):
    sum_logit = np.zeros((T_full, C), np.float32)
    counts = np.zeros((T_full,),np.int32)
    for g0, g1, L in window_logits:
        L = L[:(g1-g0)]
        sum_logit[g0:g0+L.shape[0]] += L
        counts[g0:g0+L.shape[0]] += 1
    avg_logit = sum_logit / np.maximum(counts[:, None], 1)
    return avg_logit

# peak 추출
def peak_pick(x, thr, mpd_tok):
    peaks = []
    for t in range(len(x)):
        if x[t] < thr:
            continue
        a = max(0, t-mpd_tok); b = min(len(x), t+mpd_tok+1)
        if x[t] == x[a:b].max():
            peaks.append(t)
    return peaks

# spanize (start/end offset 추출)
def decode_spans(p_full, dt: float=0.5, bg_index: int | None=0,
                 video_duration : float | None = None):
    """
    p_full: [T, C] (sigmoid 확률)
    dt    : 토큰 간 간격(초) = 1/fps
    bg_index: 배경 클래스 인덱스(없으면 None)
    video_duration: 있으면 클램프
    """
    p = p_full.copy()
    
    if bg_index is not None and 0 <= bg_index < p.shape[1]:
        highlight_cols = [CLASS_IDS[c] for c in HIGHLIGHT_CLASSES]
        p[:, highlight_cols] *= (1.0 - p[:, bg_index:bg_index+1])
        p[:, bg_index] = 0.0
    
    def tiou(a,b):
        s1,e1=a[0],a[1]; s2,e2=b[0],b[1]
        inter=max(0.0, min(e1,e2)-max(s1,s2))
        uni = max(e1,e2)-min(s1,s2)
        return inter/uni if uni>0 else 0.0
    
    spans = []
    # 클래스별 peack -> spanize
    for cname in HIGHLIGHT_CLASSES:
        ci = CLASS_IDS[cname]
        mpd_tok = max(1, int(round(MPD_SEC[cname] / dt)))
        peaks = peak_pick(p[:, ci], THRESH[cname], mpd_tok)
        start_off, end_off = OFFSETS_SEC[cname]
        for t in peaks:
            center = t * dt
            s = max(0.0, center - start_off)
            e = center + end_off
            if video_duration is not None:
                e = min(e, video_duration)
            sc = float(p[t, ci])
            spans.append([s, e, cname, sc])
            
    # 클래스 내 NMS -> 가까운 구간 병합 -> 길이 clamp
    final = []
    for cname in HIGHLIGHT_CLASSES:
        cand = [x for x in spans if x[2] == cname]
        cand.sort(key=lambda x: x[3], reverse=True)
        kept = []
        for x in cand:
            if all(tiou(x, y) < NMS_IOU[cname] for y in kept):
                kept.append(x)
        kept.sort(key=lambda x: x[0])
        merged = []
        for x in kept:
            if merged and (x[0] - merged[-1][1]) <= MERGE_GAP_SEC[cname]:
                merged[-1][1] = max(merged[-1][1], x[1])
                merged[-1][3] = max(merged[-1][3], x[3])
            else:
                merged.append(x)
        for m in merged:
            L = m[1] - m[0]
            if L < MIN_LEN_SEC[cname]: m[1] = m[0] + MIN_LEN_SEC[cname]
            if L > MAX_LEN_SEC[cname]: m[1] = m[0] + MAX_LEN_SEC[cname]
        final += merged
    
    final.sort(key=lambda x:x[0])
    return [{"start":round(s,3), "end":round(e,3), "class":c, "score":round(sc,4)} for s,e,c,sc in final]


def extract_video_frames_bgr(video_path, fps=2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Fail to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / max(native_fps, 1e-6)
    step = int(round(native_fps / fps))
    frames = []
    idx = 0
    t = 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if (idx % max(step,1)) == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    T_tokens = int(math.ceil(duration * fps))
    # 일부 비디오에서는 마지막 토큰 부족 -> padding
    if len(frames) < T_tokens and len(frames)>0:
        frames += [frames[-1]] * (T_tokens - len(frames))
    return frames, duration

def load_audio_from_video(video_path):
    """
    우선 torchaudio로 시도, 실패 시 ffmpeg로 16k mono wav 추출 후 로드.
    반환: wav [1, T] float32(-1..1), sr
    """
    try:
        wav, sr = torchaudio.load(video_path)
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dtype != torch.float32:
            wav = wav.float()
        # 정규화: torchaudio는 보통 -1..1 범위로 읽음. 필요한 경우만 스케일링.
        if wav.abs().max() > 1.0:
            wav = wav / 32768.0
        return wav, sr
    except Exception as e:
        # ffmpeg 백업 경로
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"torchaudio/ffmpeg 모두 실패: {e}")
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio16k.wav")
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", wav_path
            ]
            subprocess.run(cmd, check=True)
            wav, sr = torchaudio.load(wav_path)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if wav.dtype != torch.float32:
                wav = wav.float()
            if wav.abs().max() > 1.0:
                wav = wav / 32768.0
            return wav, sr

def iter_frames_bgr(video_path, fps=2.0, chunk_size=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Fail to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / max(native_fps, 1e-6)
    step = max(1, int(round(native_fps / fps)))

    buf = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx % step) == 0:
            buf.append(frame)
            if len(buf) >= chunk_size:
                yield buf  # 청크 한 덩어리 반환
                buf = []
        idx += 1
    cap.release()
    if buf:
        yield buf
        
        
def pca_audio_to_k(A_128: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), 128))
    if A_128.shape[1] <= k:
        # 이미 128D면 그대로 반환
        return A_128.astype(np.float32, copy=False)
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(A_128)  # [T,k]
    return Z.astype(np.float32, copy=False)

def hadamard_orth(n: int) -> np.ndarray:
    # n은 2의 거듭제곱이어야 함(여기선 512 사용)
    H = np.array([[1]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H,  H],
                      [H, -H]]).astype(np.float32)
    # 행/열 정규화
    H /= np.sqrt(n)
    return H  # [n,n]

def expand_to_512(Z: np.ndarray, method: str = "pad", seed: int = 1234) -> np.ndarray:
    T, d = Z.shape  # d<=128
    if d == 512:
        return Z
    if method == "pad":
        out = np.zeros((T, 512), dtype=np.float32)
        out[:, :d] = Z
        return out
    elif method == "gauss":
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, 512)).astype(np.float32)  # JL 스타일
        return (Z @ W).astype(np.float32, copy=False)
    elif method == "hadamard":
        # 128→512 : 512x512 해더마드 일부 열 사용
        H = hadamard_orth(512)  # [512,512]
        # d<=128이므로, H의 앞 d행을 사용해 512차원으로 확장: (T,d) @ (d,512) 필요 → H.T[:,:d] 사용
        W = H[:, :d].astype(np.float32)            # [512,d]
        return (Z @ W.T).astype(np.float32)        # (T,d)@(d,512) = (T,512)
    else:
        raise ValueError(f"unknown expand method: {method}")