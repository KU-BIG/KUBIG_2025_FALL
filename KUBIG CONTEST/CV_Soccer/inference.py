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

from class_for_inference import ResNet152Encoder, VGGishEncoder, PCAdaptor, LinearAdapter, DummyData
from method_for_inference import pca_on_the_fly_2048_to_512, slide_windows, stitch_logits, decode_spans
from method_for_inference import load_audio_from_video, iter_frames_bgr, pca_audio_to_k, expand_to_512

# Decoder : sigmoid -> peak -> spanize -> NMS/merge
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

# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str, help="input video path (with audio)")
    ap.add_argument("--architecture", required=True, type=str)
    ap.add_argument("--network", default="RVLAD", type=str)
    ap.add_argument("--VLAD_k", default=64, type=int)
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--window_size_sec", default=60.0, type=float)
    ap.add_argument("--stride_sec", default=1.0, type=float)  # 예시: 1s stride
    ap.add_argument("--fps", default=2.0, type=float)         # token rate
    ap.add_argument("--gpu", default=-1, type=int)
    ap.add_argument("--vggish_ckpt", required=True, type=str, help="path to torchvggish pretrained weights (.pth)")
    ap.add_argument("--audio_adapter_path", default=None, type=str, help="(선택) 128→512 등 학습 시 사용한 어댑터 가중치")
    ap.add_argument("--save_json", default=None, type=str)
    ap.add_argument("--video_pca_npz", default=None, type=str)  # 있으면 사용
    ap.add_argument("--video_adapter_path", default=None, type=str)  # 2048->512 선형 가중치 .pth (있으면 사용)
    ap.add_argument("--allow_random_adapter", action="store_true")   # 가중치/npz 없을 때 랜덤 투영 허용
    ap.add_argument("--pca_on_the_fly", action="store_true", help="입력 비디오에서 즉시 PCA(TruncatedSVD) 학습")
    ap.add_argument("--pca_components", type=int, default=512)
    ap.add_argument("--audio_pca_on_the_fly", action="store_true",help="오디오 128D에 대해 on-the-fly PCA 수행 (k<=128)")
    ap.add_argument("--audio_pca_components", type=int, default=128,help="오디오 PCA 차원(k), 최대 128")
    ap.add_argument("--audio_expand_to_512", type=str, default="pad",choices=["pad", "gauss", "hadamard"],help="PCA 결과를 512D로 확장하는 방법")
    ap.add_argument("--audio_expand_seed", type=int, default=1234, help="gauss/hadamard 확장 시 시드")
    args = ap.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if (args.gpu>=0 and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")
    
    # raw feature extraction (2fps)
    def get_video_meta(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Fail to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration = frame_count / max(fps, 1e-6)
        return duration, fps, frame_count

    print("• Extracting 2fps frames...")
    duration, native_fps, total_frame_count = get_video_meta(args.video)

    # print("Video encoder (ResNet152)...")
    # vid_enc = ResNet152Encoder(device)
    # V_parts = []
    # for frames_chunk in iter_frames_bgr(args.video, fps=args.fps, chunk_size=256):
    #     V_part = vid_enc.encode_batch(frames_chunk, batch_size=16)  # 내부 기본값
    #     V_parts.append(V_part)
    #     del frames_chunk
    #     if device.type == "cuda":
    #         torch.cuda.empty_cache()
    # V_2048 = np.concatenate(V_parts, axis=0) if V_parts else np.zeros((0,2048), np.float32)
    # del V_parts
    # print(f"  frames(encoded): {len(V_2048)}, duration≈(추정){len(V_2048)/args.fps:.2f}s")
        
    # if args.video_pca_npz:
    #     pca = PCAdaptor.load_npz(args.video_pca_npz)  # (있으면) 학습 때 쓰던 PCA npz 사용
    #     V = pca.transform(V_2048, whiten=False)  # [T, 512]
    # elif args.pca_on_the_fly:
    #     V = pca_on_the_fly_2048_to_512(V_2048, k=args.pca_components)  # (파일 없이) 입력 비디오에서 즉석 PCA
    # else:
    #     # (최후) 선형 어댑터: 가중치 있으면 로드, 없으면 --allow_random_adapter 필수
    #     adapter = LinearAdapter(2048, 512, path=args.video_adapter_path, allow_random=args.allow_random_adapter,  device=device)
    #     V = adapter(V_2048)  # [T, 512]
    # T_tokens = V.shape[0]
    # dt = 1.0/args.fps
    
    # ---------------------------------------------------------------
    
    # base = os.path.splitext(os.path.basename(args.video))[0]
    # out_root = "/mnt/d/workspace/KUBIG"
    # out_dir = os.path.join(out_root, "features_video_fps")
    # os.makedirs(out_dir, exist_ok=True)
    
    # out_path = os.path.join(out_dir, f"{base}_ResNET_TF2_PCA512.npy")
    
    # V_512 = V.astype(np.float32, copy=False)  # [T.512]
    # np.save(out_path, V_512)
    # print(f"[Saved] V_512 {V_512.shape} -> {out_path}")
    # return
    
    # --------------------------------------------------------------- 
    
    # print("Audio encoder (VGGish)...")
    # base = os.path.splitext(os.path.basename(args.video))[0]
    # video_feat_dir = f"/mnt/d/workspace/KUBIG/features_video_fps"
    # video_feat_path = os.path.join(video_feat_dir, f"{base}_ResNET_TF2_PCA512.npy")
    # try:
    #     T_tokens = np.load(video_feat_dir, mmap_mode="r").shape[0]
    # except Exception:
    #     T_tokens = int(round(duration * args.fps))

    # wav, sr = load_audio_from_video(args.video)
    # aud_enc = VGGishEncoder("auto", device)
    # A_128 = aud_enc.encode_audio(wav, sr, target_fps=args.fps, expected_len=T_tokens,
    #                              start_sec=0.0, duration_sec=duration)
    
    # out_root = "/mnt/d/workspace/KUBIG"
    # out_dir = os.path.join(out_root, "features_video_fps")
    # os.makedirs(out_dir, exist_ok=True)
    
    # # 128 -> 512
    # if args.audio_adapter_path:
    #     aud_adapter = LinearAdapter(128, 512, path=args.audio_adapter_path, allow_random=args.allow_random_adapter, device=device)
    #     A = aud_adapter(A_128).astype(np.float32, copy=False)
    # else:
    #     if not args.audio_pca_on_the_fly:
    #         print("[Audio] 어댑터가 없으므로 --audio_pca_on_the_fly 를 강제 활성화합니다.")
    #     k = min(args.audio_pca_components, 128)
    #     A_k = pca_audio_to_k(A_128, k=k)    
    #     A   = expand_to_512(A_k, method=args.audio_expand_to_512, seed=args.audio_expand_seed)
    
    # out_path = os.path.join(out_dir, f"{base}_VGGish.npy")
    # np.save(out_path, A.astype(np.float32, copy=False))
    # print(f"[Saved] A {A.shape} -> {out_path}")
    # return
    
    # --------------------------------------------------------------- 
    # 추출 후의 진짜 인퍼런스...
    """ 
    아래 # sliding window 주석 전까지 : 만약, 추출-인퍼런스를 동시에 실행한다면 이건 주석처리하고 실행해도 됩니다!!
    """
    base = os.path.splitext(os.path.basename(args.video))[0]
    default_feat_dir = "/mnt/d/workspace/KUBIG/features_video_fps"
    
    # video_feat_path = os.path.join(default_feat_dir, f"{base}_ResNET_TF2_PCA512.npy")
    # audio_feat_path = os.path.join(default_feat_dir, f"{base}_VGGish.npy")
    
    video_feat_path = "/mnt/d/workspace/KUBIG/features_video_fps/1_ResNET_TF2_PCA512.npy"
    audio_feat_path = "/mnt/d/workspace/KUBIG/features_video_fps/1_VGGish.npy"
    
    print(f"[Load] video feats: {video_feat_path}")
    print(f"[Load] audio feats: {audio_feat_path}")
    
    V = np.load(video_feat_path).astype(np.float32, copy=False)   # (T_v, 512)
    A = np.load(audio_feat_path).astype(np.float32, copy=False)   # (T_a, 512)
    
    # 차원/타입 체크
    if V.ndim != 2 or V.shape[1] not in (512,):
        raise RuntimeError(f"Video feat shape invalid: {V.shape} (expected [T,512])")
    if A.ndim != 2 or A.shape[1] not in (128, 512):
        raise RuntimeError(f"Audio feat shape invalid: {A.shape} (expected [T,128] or [T,512])")
    
    # 오디오가 128D라면 512D로 확장(제로패딩) - 학습 파이프라인이 512D를 기대하므로
    if A.shape[1] == 128:
        print("[Audio] 128D → 512D zero-pad 확장")
        pad = np.zeros((A.shape[0], 512 - 128), dtype=np.float32)
        A = np.concatenate([A, pad], axis=1)
    
    # 길이 맞추기 (최소 길이에 맞춰 자르기; 원하면 pad로 바꿔도 됨)
    T_tokens = min(V.shape[0], A.shape[0])
    if V.shape[0] != A.shape[0]:
        print(f"[Align] length mismatch: video={V.shape[0]}, audio={A.shape[0]} → clip to {T_tokens}")
    V = V[:T_tokens]
    A = A[:T_tokens]

    dt = 1.0 / args.fps
    print(f"[Ready] V {V.shape}, A {A.shape}, dt={dt:.3f}s")
    
    # 인퍼런스-추출 한 번에 실행할 경우 위의 코드들은 주석처리 하시면 됩니다.
    # sliding window
    W_tok = int(round(args.window_size_sec / dt))
    stride_tok = int(round(args.stride_sec / dt))
    V_win = slide_windows(V, W_tok, stride_tok)
    A_win = slide_windows(A, W_tok, stride_tok)
    assert len(V_win) == len(A_win)
    print(f"• Sliding windows: {len(V_win)} (W={W_tok}, stride={stride_tok}, dt={dt:.3f}s)")
    
    # Build/Load network
    dummy_dataset = DummyData(num_classes=len(CLASS_IDS), number_frames_in_window=W_tok,
                              video_feat_dim=512, audio_feat_dim=512)  # + background
    C = dummy_dataset.num_classes
    module = __import__('networks')
    Net = getattr(module, args.architecture)
    net = Net(dummy_dataset, args.network, VLAD_K=args.VLAD_k).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    net.load_state_dict(state, strict=True)
    net.eval()

    # Run network
    windows_logits = []
    with torch.no_grad():
        for (g0_v, g1_v, v), (g0_a, g1_a, a) in tqdm(zip(V_win, A_win), total=len(V_win)):
            assert g0_v == g0_a and g1_v==g1_a
            v_t = torch.from_numpy(v).float().unsqueeze(0).to(device, non_blocking=False)
            a_t = torch.from_numpy(a).float().unsqueeze(0).to(device, non_blocking=False)
            logits = net(v_t, a_t)  # expect [W, C]
            if logits.dim()==3:
                logits = logits.reshape(-1, logits.shape[-1])
            L = logits.detach().cpu().numpy().astype(np.float32)
            # 안전성 체크: 오디오/비디오의 input dimension 불일치 시 여기서 RuntimeError가 날 수 있음.
            windows_logits.append((g0_v, g1_v, L))
    
    avg_logits = stitch_logits(windows_logits, T_full=T_tokens, C=C)
    P_full = 1 / (1 + np.exp(-avg_logits))  # Sigmoid for scoring
    
    # Devoe to highlight spans
    BG_INDEX = CLASS_IDS['background']  # 0
    spans = decode_spans(P_full, dt=dt, bg_index=BG_INDEX, video_duration=duration)
    
    # output
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(spans, f, ensure_ascii=False, indent=2)
    print("\n[Highlights]")
    for s in spans:
        print(s)

if __name__ == "__main__":
    main()