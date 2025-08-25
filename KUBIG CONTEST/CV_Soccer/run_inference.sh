#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="/mnt/d/workspace/KUBIG/video-highlight-extraction/inference.py"
OUT_DIR="/mnt/d/workspace/KUBIG/video-highlight-extraction/inference_out"

DEFAULT_VIDEO="/mnt/d/workspace/KUBIG/2_720p.mkv"
INPUT_TARGET="${1:-$DEFAULT_VIDEO}"

# ───────── 모델/네트워크 설정 ─────────
ARCHITECTURE="AudioVideoArchi5"   # networks.py 안의 클래스명
NETWORK_TYPE="VLAD"
VLAD_K=512

# 학습 시와 동일한 토큰 레이트/윈도 설정
FPS=2                         # 2 fps
WINDOW_SIZE_SEC=20            # 20s
STRIDE_SEC=1                  # 1s (overlap)

# ───────── 체크포인트/어댑터 경로들 ─────────
MODEL_CKPT="/mnt/d/workspace/KUBIG/video-highlight-extraction/logs/experiment_20250819_103805_2025-08-19_10-45-10_epoch_18_model.pth" 
VGGISH_CKPT="/mnt/d/workspace/KUBIG/video-highlight-extraction/logs/vggish_audioset_weights_without_fc2.h5"

# (권장) ResNet-152 2048→512 PCA(.npz) 또는 선형 어댑터(.pth) 중 하나를 지정
VIDEO_PCA_NPZ=""
VIDEO_ADAPTER_PATH=""                            # 없으면 2048→512 선형 투영 가중치(.pth)
AUDIO_ADAPTER_PATH=""                            # 필요 시 128→512 선형 투영 가중치(.pth)

# ───────── 시스템/출력 ─────────
GPU_ID=0
mkdir -p "$OUT_DIR"

# ───────── 내부 함수: 단일 비디오 처리 ─────────
run_infer () {
  local video_path="/mnt/d/workspace/KUBIG/2_720p.mkv"
  local stem="$(basename "${video_path%.*}")"
  local out_json="${OUT_DIR}/${stem}.json"

  echo "▶ Inference on: $video_path"

  local -a args=(
    --video "$video_path"
    --architecture "$ARCHITECTURE"
    --network "$NETWORK_TYPE"
    --VLAD_k "$VLAD_K"
    --model_path "$MODEL_CKPT"
    --fps "$FPS"
    --window_size_sec "$WINDOW_SIZE_SEC"
    --stride_sec "$STRIDE_SEC"
    --gpu "$GPU_ID"
    --vggish_ckpt "$VGGISH_CKPT"
    --save_json "$out_json"
  )

    if [[ -n "$VIDEO_PCA_NPZ" && -f "$VIDEO_PCA_NPZ" ]]; then
        args+=( --video_pca_npz "$VIDEO_PCA_NPZ" )
    elif [[ -n "$VIDEO_ADAPTER_PATH" && -f "$VIDEO_ADAPTER_PATH" ]]; then
        args+=( --video_adapter_path "$VIDEO_ADAPTER_PATH" )
    else
        echo "ℹnpz/어댑터 없음 → 입력 비디오에서 즉석 PCA 사용(--pca_on_the_fly)"
        args+=( --pca_on_the_fly )
        args+=( --audio_pca_on_the_fly )
        args+=( --audio_pca_components 128 )
        args+=( --audio_expand_to_512 pad )
        args+=( --audio_expand_seed 1234 )
        # 만약 inference.py에 on-the-fly가 없다면 임시로 아래를 쓰세요:
        # args+=( --allow_random_adapter )
    fi

    # (선택) 오디오 어댑터
    if [[ -n "$AUDIO_ADAPTER_PATH" && -f "$AUDIO_ADAPTER_PATH" ]]; then
        args+=( --audio_adapter_path "$AUDIO_ADAPTER_PATH" )
    fi

    python "$PYTHON_SCRIPT" "${args[@]}"
    echo "✓ Saved spans: $out_json"
}

# 입력 타입 분기
if [[ -f "$INPUT_TARGET" && "${INPUT_TARGET##*.}" == "txt" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    run_infer "$line"
  done < "$INPUT_TARGET"

elif [[ -d "$INPUT_TARGET" ]]; then
  shopt -s nullglob
  for vid in "$INPUT_TARGET"/*.{mp4,mkv,mov,avi}; do
    run_infer "$vid"
  done
  shopt -u nullglob

elif [[ -f "$INPUT_TARGET" ]]; then
  run_infer "$INPUT_TARGET"

else
  echo "입력 경로를 찾을 수 없습니다: $INPUT_TARGET"
  exit 1
fi
