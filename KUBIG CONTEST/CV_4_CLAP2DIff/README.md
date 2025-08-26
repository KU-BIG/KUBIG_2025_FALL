# CLAP2Diffusion: 계층적 오디오-이미지 생성

**KUBIG Contest **  
**CV_4조**: 21기 남동연, 22기 신진섭, 22기 공병승

<div align="center">
  
| 🎵 Thunder.wav + "a beach" | 🎵 Thunder.wav + "a city" |
|:---:|:---:|
| ![Thunder Beach](assets/Thunder_beach.webp) | ![Thunder City](assets/Thunder_city.webp) |

**동일한 오디오와 텍스트로 다양한 장면 생성 가능**

</div>

## 개요

**입력**: 오디오 파일(.wav) + 텍스트 프롬프트  
**출력**: 512x512 이미지

CLAP2Diffusion은 오디오와 텍스트를 입력받아 이미지를 생성하는 모델입니다. SonicDiffusion 아키텍처에 3단계 계층 분해와 Norm 60 최적화를 적용하여 오디오의 특성을 시각적으로 변환합니다.

## 주요 특징

- **계층적 오디오 처리**: 전경/배경/분위기 3단계 분해
- **Norm 60 최적화**: 실험으로 발견한 최적 정규화 값
- **온도 어닐링**: 2.0 → 0.5 점진적 개선
- **4x Self-Attention**: 향상된 오디오 토큰 생성

## 데이터셋

**AudioCaps**: YouTube 동영상에서 추출한 오디오-텍스트 쌍 데이터셋
- 다양한 일상 소리 및 환경음 포함
- 각 오디오에 대한 자연어 캡션 제공
- Train/Val/Test 분할로 학습 및 평가

## 결과

### 성공 사례
천둥 소리를 다양한 장면으로 변환:

| 오디오 | 텍스트 프롬프트 | 생성 이미지 |
|--------|---------------|------------|
| Thunder.wav | "thunderstorm at beach" | ![](assets/Thunder_beach.webp) |
| Thunder.wav | "thunderstorm in city" | ![](assets/Thunder_city.webp) |
| Thunder.wav | "thunderstorm in forest" | ![](assets/Thunder_forest.webp) |

### 실패 사례
인간 음성(웃음소리 등)은 제대로 생성되지 않음:

| 오디오 | 텍스트 프롬프트 | 문제점 | 실패 결과 |
|--------|---------------|--------|----------|
| laughing_baby.wav | "baby laughing in city" | 잘못된 장면 생성 | ![](assets/laughing_baby_city.png) |
| laughing_man.wav | "man laughing at beach" | 오디오-비주얼 정렬 실패 | ![](assets/laughing_man_beach.png) |
| Thunder.wav | (텍스트 없음) | 추상적 패턴만 생성 | ![](assets/Thunder.webp) |

## 모델 구조

- **Audio Projector**: 2.2M 파라미터
- **Hierarchical Decomposer**: 0.3M 파라미터  
- **추론 속도**: ~2초/이미지 (GPU 사용시)
- **메모리**: ~6GB VRAM

## 체크포인트

`checkpoints/` 폴더에 사전 학습된 모델 포함:
- `audio_projector_stage1.pth`: Stage 1 모델
- `audio_projector_stage2.pth`: Stage 2 모델
- `audio_projector_stage3_finetuned.pth`: 최종 모델

## 설치 및 실행

### Docker 사용 (권장)
```bash
docker-compose up --build
```

### 수동 설치
```bash
conda create -n clap2diffusion python=3.10
conda activate clap2diffusion
pip install -r requirements.txt
python app/gradio_app.py
```

## 학습 단계

```bash
# Stage 1: Audio Projector (3,000 steps)
python scripts/train_stage1.py

# Stage 2: 전체 모델 (2,000 steps)
python scripts/train_stage2.py

# Stage 3: 미세 조정 (1,000 steps)
python scripts/train_stage3.py
```

## 라이선스

MIT License

## 참고 논문

- **SonicDiffusion** (2023): "SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models"
- **AudioLDM 2** (2023): "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining"
- **CLAP** (2023): "Large-scale Contrastive Language-Audio Pre-training with Feature Fusion and Keyword-to-Caption Augmentation"
- **Stable Diffusion** (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- **AudioCaps** (2019): "AudioCaps: Generating Captions for Audios in The Wild"

---
*KUBIG Contest 2024 - CV_4조*