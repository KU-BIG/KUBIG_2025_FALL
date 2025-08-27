# 🐶 시간을 달리는 강아지  
_Age-Conditional GAN 기반 강아지 미래 얼굴 예측_

---

## 1. Team
- 21기 김지엽  
- 22기 박경국  
- 22기 장건희  

---

## 2. Introduction
본 프로젝트는 실종된 반려견의 현재 모습을 추정하기 위해, 나이 조건을 반영한 **Age-Conditional GAN(Age-cGAN)** 모델을 구현하고 검증하는 것을 목표로 합니다.

- **기존 접근 방식**  
  법의학 아티스트가 사진과 통계를 기반으로 수작업 → 시간이 오래 걸리고, 결과물의 일관성이 낮음  

- **제안 방식**  
  딥러닝 기반 생성 모델을 활용 → 빠르고 일관성 있는 이미지 생성 가능  

- **핵심 아이디어**  
  1. **Age-Conditional GAN 학습** : 조건부 GAN으로 다양한 연령대 이미지 분포 학습  
  2. **Encoder 학습** : 입력 이미지를 잠재 벡터 *z*로 인코딩  
  3. **Latent Vector Optimization(LVO)** : *z*를 미세 조정하여 품질과 정체성(identity) 보존 강화  

---

## 3. Experiments

### 3-1. Dataset
1. **Kaggle DogAge (~27k)**  
   - Young / Adult / Senior 라벨 포함  
   - 약 4%만 정제 데이터, 나머지는 라벨 오류 및 배경/포즈 다양성 존재  

2. **Kaggle AFAC 2023 (~26k)**  
   - 소형견(치와와, 비숑, 말티즈 등) 중심  
   - 월 단위 나이 라벨, 약 6% *mislabeled* 데이터  

3. **라벨 정의 (소형견 기준)**  
   - Young ≤ 4세  
   - Adult = 5~6세  
   - Senior ≥ 7세  

**데이터 구조 예시**
```plaintext
data/
 ├─ young/   # 2D 이미지
 ├─ adult/
 └─ senior/
```

---

### 3-2. Model
- **Generator (G)** : 조건 *y*를 임베딩 후 mid-level feature에 주입 → 조건 효과 강화  
- **Discriminator (D)** : Projection Discriminator + feature embedding → overfitting / mode collapse 완화  
- **Encoder (E)** : Conv 기반 이미지 feature 추출 + *y* 임베딩 결합  
- **Latent Vector Optimization (LVO)** : z₀ = E(x,y)를 초기값으로 두고, Pixel Loss + Perceptual Loss(VGG19) 조합으로 보정  

**손실 함수 구성**
- Hinge GAN Loss  
- Pixel-wise Loss  
- Perceptual Loss (VGG19 기반)  
- Identity Loss  

---

### 3-3. Results
- **Age-cGAN 구조만 사용**: Identity 정보 손실, 사진 품질 저하  
- **CycleGAN 구조**: tone/밝기 변화에 치중, 노화 특징 반영 한계  
- **Pretrained Encoder + U-Net**: 의미 있는 latent space 학습 실패, 단순 복제 경향  

---

### 3-4. Limitations
- **모델 한계**  
  - GAN의 학습 불안정성, mode collapse 가능성  
  - Diffusion 대비 낮은 품질  

- **데이터 한계**  
  - 라벨 노이즈 및 종별 다양성 → 정형적 학습 어려움  
  - 소형견 데이터 수량 제한 (~16k)  

- **조건 한계**  
  - 강아지 노화 특징은 미묘(주름·털 색 변화 등) → y의 효과가 약함  

---

## 4. Usage

### 환경 설정
```bash
conda create -n dog-agegan python=3.10 -y
conda activate dog-agegan
pip install torch torchvision torchaudio
pip install pillow tqdm numpy matplotlib scikit-image scikit-learn opencv-python
```

### 학습
```bash
# Age-cGAN 학습
python train_cgan.py

# Encoder 학습
python train_encoder.py

# Latent Vector Optimization
python optimize_latent.py
```

### 추론
```bash
python infer.py --input data/young/dog001.jpg --target senior
```

---

## 5. Repository Structure
```plaintext
.
├─ data/                 # 데이터셋
│   ├─ young/
│   ├─ adult/
│   └─ senior/
├─ src/
│   ├─ models/           # Generator / Discriminator / Encoder
│   ├─ train/            # 학습 스크립트
│   └─ utils/            # 데이터로더, 손실 함수 등
├─ runs/                 # 체크포인트 및 결과
└─ README.md
```

---

## 6. Conclusion & Future Work
- **결론**  
  Age-cGAN을 강아지 도메인에 적용 가능함을 보였으나, 데이터 및 구조적 한계로 정체성 유지와 노화 특징 반영은 여전히 도전적 과제임.  

- **향후 개선 방향**  
  1. 데이터 정제 및 품질 향상 (마스크 활용, 라벨 검증)  
  2. 정체성 보존 강화 (breed embedding, ArcFace/CLIP 기반 ID loss)  
  3. 최신 Diffusion 모델과 비교/융합  
  4. 다중 조건화 (나이 + 건강 지표 등)  

---

## 7. References
- Antipov et al., *Age-cGAN: Face Aging with Identity Preservation* (2017)  
- Kaggle DogAge Dataset  
- Kaggle AFAC 2023 Dataset  
