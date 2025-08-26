#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-Text CLIPstyler (Diary/Review) - Extended Trainer
Drop this file into the CLIPstyler repo root (same folder as train_CLIPstyler.py).

Usage (slow / shallow U-Net):
    python train_longtext.py --content_path ./test_set/face.jpg \
        --exp_name diary1 \
        --text_file ./my_diary.txt \
        --mode avg \
        --iters 200

Usage (mix mode, direct string):
    python train_longtext.py --content_path ./test_set/city.jpg \
        --exp_name trip1 \
        --text_long "We arrived at dusk. The neon signs..." \
        --mode mix --iters 250

Optional (Fast mode with VGG encoder–decoder):
    python train_longtext.py --use_fast \
        --content_dir /path/to/DIV2K/ \
        --test_dir ./test_set \
        --text_long "Warm watercolor sunset with soft brush strokes" \
        --iters 200
"""
import os, sys, argparse, re, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import kornia.augmentation as Kaug
import kornia.color as Kcolor
import importlib_metadata

# Repo modules (from CLIPstyler)
try:
    from StyleNet import UNet  # shallow U-Net used in the original repo
except Exception:
    UNet = None

try:
    from fast_stylenet import Net as VGGNet, decoder as VGG_DECODER, vgg as VGG_ENCODER
except Exception:
    VGGNet, VGG_DECODER, VGG_ENCODER = None, None, None

# OpenAI CLIP
try:
    import clip
except ImportError:
    print("Please install CLIP: pip install git+https://github.com/openai/CLIP.git", file=sys.stderr)
    raise

# --------------------- Utils ---------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(x, torch.nn.Module):
        # 모듈은 non_blocking 인자를 받지 않음
        return x.to(device)
    elif isinstance(x, torch.Tensor):
        # 텐서는 non_blocking 사용 가능
        return x.to(device, non_blocking=True)
    else:
        return x


def load_image(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    t = transforms.ToTensor()(im).unsqueeze(0)  # PIL → [1,3,H,W], [0,1]
    return t

def save_image(tensor, path): # Tensor → 파일 저장
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = tensor.detach().clamp(0,1).cpu()
    im = transforms.ToPILImage()(t[0])
    im.save(path)

# --------------------- Long text -> directions ---------------------
STYLE_LEXICON = set(
    """
vivid pastel neon matte glossy watercolor oil-paint oil painting acrylic charcoal sketch grainy gritty
wool velvet silk marble concrete wood grain brushstroke stippled dotted hatching hatch cinematic filmic
serene calm moody dreamy melancholic vibrant warm cold sunset dusk dawn foggy misty golden-hour pastel-tone
retro cyberpunk noir pop-art impressionist expressionist cubist modern minimal rough coarse soft smooth
""".split()
)

def split_sentences_heuristic(text: str):
    # 마침표/느낌표/물음표 + 쉼표/접속사 기준으로 절 단위 분할
    parts = re.split(r'[.!?]|,(?=\s)|\s+(?:and|as|while|with)\s+', text, flags=re.I)
    return [p.strip() for p in parts if p.strip()]

def style_weight(sentence: str) -> float:
    toks = re.findall(r"[A-Za-z가-힣0-9\-']+", sentence.lower())
    hits = sum(1 for t in toks if t in STYLE_LEXICON)
    # simple adjective-ish heuristic
    adj_bonus = sum(1 for t in toks if t.endswith("y") or t.endswith("ful"))
    length_pen = 1.0 / (1.0 + 0.02 * max(len(sentence), 1))
    return (1.0 + hits + 0.5 * adj_bonus) * length_pen

@torch.no_grad()
def clip_text_emb(model, txts, device):
    tokens = clip.tokenize(txts, truncate=True).to(device)
    z = model.encode_text(tokens).float()
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    return z

@torch.no_grad()
def make_text_directions(model, long_text: str, t_src: str = "a photo",
                         mode: str = "avg", top_k: int = 12, device="cpu"):
    # 1) 절 단위 분할
    sents = split_sentences_heuristic(long_text)
    if not sents:
        sents = [long_text.strip()]

    # 2) CLIP-기반 "스타일스러움" 가중치(anchors와의 방향 유사도 최대값)
    anchors = [
        "style", "texture", "lighting style", "mood", "color palette",
        "brushstrokes", "bokeh", "soft edges", "contrast", "grain", "glow"
    ]
    z_src = clip_text_emb(model, [t_src], device)           # [1,D]
    z_anc = clip_text_emb(model, anchors, device)           # [A,D]
    z_anc = z_anc / (z_anc.norm(dim=-1, keepdim=True) + 1e-8)
    d_anc = z_anc - z_src                                   # [A,D]
    d_anc = d_anc / (d_anc.norm(dim=-1, keepdim=True) + 1e-8)

    weights = []
    for s in sents:
        z_s = clip_text_emb(model, [s], device)             # [1,D]
        d_s = z_s - z_src
        d_s = d_s / (d_s.norm(dim=-1, keepdim=True) + 1e-8)
        # anchors와의 코사인 유사도 중 최대값을 사용
        sims = torch.matmul(d_s, d_anc.T).squeeze(0)        # [A]
        # 너무 긴 절은 패널티(원래 있던 길이 패널티 유지)
        length_penalty = 1.0 / (1.0 + 0.02 * max(len(s), 1))
        weights.append((float(sims.max()) + 1.0) * length_penalty)

    w = torch.tensor(weights, device=device, dtype=torch.float32)

    # 3) 상위 top_k 절만 선별
    if len(sents) > top_k:
        order = torch.argsort(w, descending=True)[:top_k]
        picked = [sents[i] for i in order.tolist()]
        w = w[order]
    else:
        picked = sents

    w = w / (w.sum() + 1e-8)                                # 정규화

    # 4) 텍스트 방향(ΔT) 계산
    z_s   = clip_text_emb(model, picked, device)            # [K,D]
    delta = z_s - z_src                                     # [K,D]
    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)

    if mode == "avg":
        DeltaT = (w[:, None] * delta).sum(dim=0, keepdim=True)
        DeltaT = DeltaT / (DeltaT.norm(dim=-1, keepdim=True) + 1e-8)
        return {"mode": "avg", "DeltaT": DeltaT, "pieces": picked, "weights": w}
    elif mode == "mix":
        return {"mode": "mix", "DeltaT_i": delta, "pieces": picked, "weights": w}
    else:
        raise ValueError("--mode must be 'avg' or 'mix'")

# --------------------- CLIP + VGG helpers ---------------------
def build_clip(device):
    model, _ = clip.load("ViT-B/32", device=device)
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
    return model, normalize

#@torch.no_grad()
def clip_img_emb(model, img_01, normalize, device):
    # Resize->224 and normalize for CLIP; expects [B,3,H,W] in [0,1]
    x = F.interpolate(img_01.clamp(0,1), size=(224,224), mode="bicubic", align_corners=False)
    x = normalize(x)
    z = model.encode_image(x).float()
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    return z

class VGGContent(nn.Module):
    def __init__(self, layers=("features.21", "features.30")):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        except Exception:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layers = set(layers)
        # --- ADD: ImageNet mean/std buffers ---
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("imnet_mean", mean)
        self.register_buffer("imnet_std", std)

    def forward(self, x):
        # --- ADD: clamp + normalize ---
        x = x.clamp(0, 1)
        x = (x - self.imnet_mean) / self.imnet_std

        feats, h = {}, x
        for name, mod in self.vgg._modules.items():
            h = mod(h)
            key = f"features.{name}"
            if key in self.layers:
                feats[key] = h
        return feats


def tv_loss(x):
    return (x[:, :, :, :-1]-x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :]-x[:, :, 1:, :]).abs().mean()

# --------------------- Patch sampling (aligned) ---------------------
# class PatchSampler:
#     def __init__(self, patch=128, n=64, perspective=True):
#         self.patch = patch
#                최신N  self.n = n
#         self.perspective = perspective
#         self.persp = transforms.RandomPerspective(distortion_scale=0.5, p=1.0)

#     def _sample_coords(self, H, W):
#         top = random.randint(0, max(H - self.patch, 0))
#         left = random.randint(0, max(W - self.patch, 0))
#         return top, left

#     def sample_pairs(self, img_c, img_cs):
#         # Return aligned content & stylized patches (before/after perspective on stylized)
#         B, C, H, W = img_cs.shape
#         pcs_aug, pcc = [], []
#         for _ in range(self.n):
#             t, l = self._sample_coords(H, W)
#             patch_cs = img_cs[:, :, t:t+self.patch, l:l+self.patch]
#             patch_c  = img_c[:, :,  t:t+self.patch, l:l+self.patch]
#             if self.perspective:
#                 pil = transforms.ToPILImage()(patch_cs[0].detach().cpu())
#                 aug = transforms.ToTensor()(self.persp(pil)).unsqueeze(0).to(img_cs.device)
#             else:
#                 aug = patch_cs
#             pcs_aug.append(aug)
#             pcc.append(patch_c)
#         pcs_aug = torch.cat(pcs_aug, dim=0)
#         pcc = torch.cat(pcc, dim=0)
#         return pcc, pcs_aug
    
class PatchSampler:
    def __init__(self, patch=128, n=64, perspective=True):
        self.patch = patch
        self.n = n
        self.perspective = perspective
        self.persp = Kaug.RandomPerspective(distortion_scale=0.5, p=1.0, keepdim=True)
        self._persp_device = None  # 내부 상태

    def _sample_coords(self, H, W):
        top = random.randint(0, max(H - self.patch, 0))
        left = random.randint(0, max(W - self.patch, 0))
        return top, left

    def sample_pairs(self, img_c, img_cs):
        # self.persp 를 입력 텐서의 device 로 '한 번만' 이동
        if self.perspective:
            dev = img_cs.device
            if self._persp_device != dev:
                self.persp = self.persp.to(dev)
                self._persp_device = dev

        B, C, H, W = img_cs.shape
        pcs_aug, pcc = [], []
        for _ in range(self.n):
            t = torch.randint(0, max(H - self.patch, 1), (1,), device=img_cs.device).item()
            l = torch.randint(0, max(W - self.patch, 1), (1,), device=img_cs.device).item()
            patch_cs = img_cs[:, :, t:t+self.patch, l:l+self.patch]
            patch_c  = img_c[:, :,  t:t+self.patch, l:l+self.patch]
            aug = self.persp(patch_cs) if self.perspective else patch_cs
            pcs_aug.append(aug)
            pcc.append(patch_c)
        pcs_aug = torch.cat(pcs_aug, dim=0)
        pcc = torch.cat(pcc, dim=0)
        return pcc, pcs_aug

# --------------------- Losses ---------------------
def cosine_dir(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def directional_loss(DeltaI, text_dir):
    if text_dir["mode"] == "avg":
        s = cosine_dir(DeltaI, text_dir["DeltaT"])
        return (1 - s).mean()
    else:
        sims = torch.stack([cosine_dir(DeltaI, text_dir["DeltaT_i"][k:k+1]).mean()
                            for k in range(text_dir["DeltaT_i"].size(0))])
        return 1 - (text_dir["weights"] * sims).sum()

def patch_loss(DeltaI_p, text_dir, tau=0.7):
    if text_dir["mode"] == "avg":
        s = 1 - cosine_dir(DeltaI_p, text_dir["DeltaT"])
        return (s * (s > tau).float()).mean()
    else:
        Ls = []
        for k in range(text_dir["DeltaT_i"].size(0)):
            s = 1 - cosine_dir(DeltaI_p, text_dir["DeltaT_i"][k:k+1])
            Ls.append((s * (s > tau).float()).mean())
        Ls = torch.stack(Ls)
        return (text_dir["weights"] * Ls).sum()
    
def chroma_loss_lab(x):
    x = x.clamp(0,1)
    lab = Kcolor.rgb_to_lab(x)        # [B,3,H,W]
    a = lab[:, 1:2]; b = lab[:, 2:3]
    return (a.square().mean() + b.square().mean())

def edge_bonus(x):
    # 채널별 라플라시안으로 엣지 세기를 올려주는 보너스 항
    x = x.clamp(0, 1)
    B, C, H, W = x.shape
    k = torch.tensor([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    # 채널별(depthwise)로 동일 커널을 적용
    weight = k.repeat(C, 1, 1, 1)                 # [C,1,3,3]
    edges = F.conv2d(x, weight, padding=1, groups=C).abs()  # [B,C,H,W]
    return edges.mean()

# --------------------- Models ---------------------
def build_unet():
    if UNet is None:
        raise RuntimeError("UNet not found. Make sure StyleNet.py is present in the repo.")
    net = UNet(ngf=16, input_channel=3, output_channel=3)  # same as the paper code
    return to_device(net)

def build_vgg_decoder():
    if VGGNet is None:
        raise RuntimeError("fast_stylenet not found. Please ensure fast_stylenet.py exists in the repo.")
    enc = VGG_ENCODER
    dec = VGG_DECODER
    net = VGGNet(enc, dec)  # encoder frozen inside, decoder learnable
    return to_device(net)

# --------------------- Train steps ---------------------
def _znorm(z):
    return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

def slow_train_step(
    net, img_c, text_dir, clip_model, clip_norm, vgg_feat,
    lambdas, tau=0.7, sampler=None, opt=None,
    patch_ref: str = "global", z0_cached=None  # NEW
):
    img_cs = net(img_c)  # UNet forward; sigmoid inside as in repo

    # ----- global directional loss (z1 - z0) -----
    if z0_cached is None:
        z0 = _znorm(clip_img_emb(clip_model, img_c,  clip_norm, img_c.device))   # [1,D]
    else:
        z0 = z0_cached                                                            # [1,D]
    z1 = _znorm(clip_img_emb(clip_model, img_cs, clip_norm, img_c.device))       # [1,D]
    DeltaI_g = _znorm(z1 - z0)                                                 # [1,D]
    L_dir = directional_loss(DeltaI_g, text_dir)

    # ----- PatchCLIP (global vs local 기준 분기) -----
    # sampler의 증강 모듈이 CPU에 있으면 GPU로 올리기 (Kornia 사용 시)
    if sampler is not None and hasattr(sampler, "persp") and hasattr(sampler.persp, "to"):
        sampler.persp = sampler.persp.to(img_c.device)

    pcc, pcs_aug = sampler.sample_pairs(img_c, img_cs)                         # [N,3,H,W] each
    zs = _znorm(clip_img_emb(clip_model, pcs_aug, clip_norm, img_c.device))    # [N,D]

    if patch_ref == "global":                                                  # NEW
        zc = z0.expand(zs.size(0), -1)                                         # [N,D] (same global ref)
    else:
        zc_local = clip_img_emb(clip_model, pcc, clip_norm, img_c.device)      # [N,D]
        zc = _znorm(zc_local)

    DeltaI_p = _znorm(zs - zc)                                                 # [N,D]
    L_patch  = patch_loss(DeltaI_p, text_dir, tau=tau)

    # ----- content & TV -----
    feats_c  = vgg_feat(img_c)
    feats_cs = vgg_feat(img_cs)
    L_c  = sum(F.mse_loss(feats_cs[k], feats_c[k]) for k in feats_c.keys())
    L_tv = tv_loss(img_cs)
    lambda_gray = lambdas.get("gray", 0.0)
    lambda_edge = lambdas.get("edge", 0.0)
    L_gray = lambda_gray * chroma_loss_lab(img_cs)
    L_edge = lambda_edge * edge_bonus(img_cs)
    
    L = (lambdas["dir"] * L_dir
         + lambdas["patch"] * L_patch
         + lambdas["content"] * L_c
         + lambdas["tv"] * L_tv
         + L_gray
         - L_edge)  # 엣지는 보너스 성격이라 빼줍니다(=선 강조

    if opt:
        opt.zero_grad()
        L.backward()
        opt.step()

    return {
        "img_cs": img_cs.detach(),
        "L": L.item(),
        "L_dir": L_dir.item(),
        "L_patch": L_patch.item(),
        "L_c": float(L_c),
        "L_tv": float(L_tv),
        "L_gray": float(L_gray), "L_edge": float(L_edge)
    }


def fast_train_step(
    net, batch_imgs, text_dir, clip_model, clip_norm,
    lambdas, tau=0.7, opt=None, patch_ref: str = "global"  # NEW
):
    # fast_stylenet.Net returns (content_loss, stylized)
    loss_c, g_t = net(batch_imgs)

    # ----- global directional loss (z1 - z0) -----
    z0 = _znorm(clip_img_emb(clip_model, batch_imgs, clip_norm, batch_imgs.device))  # [B,D] but B=?
    z1 = _znorm(clip_img_emb(clip_model, g_t,        clip_norm, batch_imgs.device))  # [B,D]
    DeltaI_g = _znorm(z1 - z0.mean(dim=0, keepdim=True))  # use mean ref if B>1, or keep as-is per your impl
    L_dir = directional_loss(DeltaI_g, text_dir)

    # ----- PatchCLIP -----
    sampler = PatchSampler(patch=224, n=64, perspective=True)
    if hasattr(sampler, "persp") and hasattr(sampler.persp, "to"):
        sampler.persp = sampler.persp.to(batch_imgs.device)

    pcc, pcs_aug = sampler.sample_pairs(batch_imgs, g_t)
    zs = _znorm(clip_img_emb(clip_model, pcs_aug, clip_norm, batch_imgs.device))  # [N,D]

    if patch_ref == "global":                                                   # NEW
        z0_glob = _znorm(clip_img_emb(clip_model, batch_imgs, clip_norm, batch_imgs.device))  # [B,D]
        zc = z0_glob.mean(dim=0, keepdim=True).expand(zs.size(0), -1)           # [N,D]
    else:
        zc_local = clip_img_emb(clip_model, pcc, clip_norm, batch_imgs.device)  # [N,D]
        zc = _znorm(zc_local)

    DeltaI_p = _znorm(zs - zc)                                                  # [N,D]
    L_patch  = patch_loss(DeltaI_p, text_dir, tau=tau)

    L_tv = tv_loss(g_t)
    lambda_gray = lambdas.get("gray", 0.0)
    lambda_edge = lambdas.get("edge", 0.0)
    L_gray = lambda_gray * chroma_loss_lab(img_cs)
    L_edge = lambda_edge * edge_bonus(img_cs)

    L = (lambdas["dir"] * L_dir
         + lambdas["patch"] * L_patch
         + lambdas["content"] * loss_c
         + lambdas["tv"] * L_tv
         + L_gray
         - L_edge)

    if opt:
        opt.zero_grad()
        L.backward()
        opt.step()

    return {
        "img_cs": g_t.detach(),
        "L": L.item(),
        "L_dir": L_dir.item(),
        "L_patch": L_patch.item(),
        "L_c": float(loss_c.item()),
        "L_tv": float(L_tv.item()),
        "L_gray": float(L_gray), "L_edge": float(L_edge)
    }

# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", type=str, default=None, help="(slow) single content image")
    parser.add_argument("--content_dir", type=str, default=None, help="(fast) DIV2K HR directory")
    parser.add_argument("--test_dir", type=str, default="./test_set", help="(fast) dir for sample tests")
    parser.add_argument("--exp_name", type=str, default="exp_longtext")
    parser.add_argument("--out_dir", type=str, default="./outputs_longtext")

    # Text
    parser.add_argument("--text_long", type=str, default=None, help="Long text string")
    parser.add_argument("--text_file", type=str, default=None, help="Path to a .txt file")
    parser.add_argument("--t_src", type=str, default="a photo")
    parser.add_argument("--mode", type=str, choices=["avg","mix"], default="avg")
    parser.add_argument("--top_k", type=int, default=12)

    # Training hyperparams
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--n_patches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)

    # Loss weights
    parser.add_argument("--lambda_dir", type=float, default=5e2)
    parser.add_argument("--lambda_patch", type=float, default=9e3)
    parser.add_argument("--lambda_content", type=float, default=150.0)
    parser.add_argument("--lambda_tv", type=float, default=2e-3)
    parser.add_argument("--lambda_gray", type=float, default=0.0)
    parser.add_argument("--lambda_edge", type=float, default=0.0)


    # Fast toggle
    parser.add_argument("--use_fast", action="store_true", help="Use VGG encoder–decoder (fast)")
    parser.add_argument("--batch", type=int, default=4)

    parser.add_argument("--weight_mode", choices=["lexicon","clipaware","uniform"], default="clipaware")

    # ===== args =====
    parser.add_argument("--patch_ref", choices=["local","global"], default="global",
                    help="PatchCLIP 기준: 'global'은 patch−global, 'local'은 patch−(aligned patch)")


    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_norm = build_clip(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    img_c = load_image(args.content_path, size=512).to(device)

    with torch.no_grad():
        # clip_img_emb(model, img_tensor, normalize, device)
        z0 = _znorm(clip_img_emb(clip_model, img_c, clip_norm, device))  # [1,D]


    # Build text direction(s)
    if args.text_file and os.path.isfile(args.text_file):
        long_text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text_long:
        long_text = args.text_long
    else:
        raise ValueError("Provide either --text_long or --text_file")

    text_dir = make_text_directions(clip_model, long_text, t_src=args.t_src,
                                    mode=args.mode, top_k=args.top_k, device=device)

    lambdas = {"dir": args.lambda_dir, "patch": args.lambda_patch,
               "content": args.lambda_content, "tv": args.lambda_tv,
               "gray": args.lambda_gray, "edge": args.lambda_edge}

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.use_fast:
        # ------- Slow (U-Net) -------
        if args.content_path is None:
            raise ValueError("--content_path is required for slow mode")
        if UNet is None:
            raise RuntimeError("UNet not found in StyleNet.py")

        net = build_unet()
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)

        vgg_feat = VGGContent().to(device).eval()
        sampler = PatchSampler(patch=args.patch, n=args.n_patches, perspective=True)

        img_c = load_image(args.content_path, size=512).to(device)

        for it in range(1, args.iters+1):
            logs = slow_train_step(
                net, img_c, text_dir, clip_model, clip_norm,
                vgg_feat, lambdas, tau=args.tau, sampler=sampler, opt=opt,
                patch_ref=args.patch_ref,   # ← CLI 옵션 반영
                z0_cached=z0                # ← 전역 콘텐츠 임베딩 재사용
            )

            if it % 20 == 0 or it == args.iters:
                out_path = os.path.join(args.out_dir, f"{args.exp_name}_it{it:03d}.png")
                save_image(logs["img_cs"], out_path)
                print(f"[{it:4d}/{args.iters}] L={logs['L']:.3f} dir={logs['L_dir']:.3f} "
                      f"patch={logs['L_patch']:.3f} c={logs['L_c']:.3f} tv={logs['L_tv']:.5f} -> {out_path}")

        # final save
        torch.save(net.state_dict(), os.path.join(args.out_dir, f"{args.exp_name}_unet.pth"))

    else:
        # ------- Fast (VGG encoder–decoder) -------
        if VGGNet is None:
            raise RuntimeError("fast_stylenet not found. Please ensure fast_stylenet.py exists.")
        if args.content_dir is None:
            raise ValueError("--content_dir (e.g., DIV2K HR dir) is required for --use_fast")
        net = build_vgg_decoder()
        net.train()
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

        # Tiny loader that randomly crops 224x224 patches from images in content_dir
        img_files = [str(p) for p in Path(args.content_dir).rglob("*")
                     if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
        if len(img_files) == 0:
            raise ValueError("No images found in content_dir")

        def rand_crop_224(path):
            img = Image.open(path).convert("RGB")
            W, H = img.size
            if W < 224 or H < 224:
                img = img.resize((max(W,224), max(H,224)), Image.BICUBIC)
                W, H = img.size
            left = random.randint(0, W-224); top = random.randint(0, H-224)
            img = img.crop((left, top, left+224, top+224))
            return transforms.ToTensor()(img)

        it = 0
        while it < args.iters:
            batch = torch.stack([rand_crop_224(random.choice(img_files)) for _ in range(args.batch)], dim=0).to(device)
            logs = fast_train_step(net, batch, text_dir, clip_model, clip_norm, lambdas, tau=args.tau, opt=opt, patch_ref=args.patch_ref)
            it += 1
            if it % 20 == 0 or it == args.iters:
                test_paths = [str(p) for p in Path(args.test_dir).glob("*")
                              if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
                if len(test_paths) > 0:
                    test_img = load_image(test_paths[0], size=512).to(device)
                    with torch.no_grad():
                        _, out = net(test_img)
                    save_image(out, os.path.join(args.out_dir, f"{args.exp_name}_fast_it{it:03d}.png"))
                print(f"[{it:4d}/{args.iters}] L={logs['L']:.3f} dir={logs['L_dir']:.3f} "
                      f"patch={logs['L_patch']:.3f} c={logs['L_c']:.3f} tv={logs['L_tv']:.5f}")
        # Save decoder
        torch.save(net.decoder.state_dict(), os.path.join(args.out_dir, f"{args.exp_name}_decoder_iter_{args.iters}.pth.tar"))

if __name__ == "__main__":
    main()
