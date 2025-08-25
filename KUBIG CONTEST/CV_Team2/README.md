# 🔬 3D 재구성을 통한 적혈구 종류 예측

## 1. Team
21기 고윤경, 김지원 | 22기 박준영

## 2. Introduction

본 프로젝트는 **Diffusion Model**을 활용하여 2D 세포 이미지를 3D 형태로 재구성하는 **'A Diffusion Model Predicts 3D Shapes from 2D Microscopy Images'** 논문의 핵심 아이디어를 구현하고 검증하는 데 목표를 두었습니다.

기존의 3D 재구성 모델들은 복잡한 파이프라인과 높은 계산 비용을 요구하는 경우가 많았지만, DISPR은 노이즈 제거 과정을 통해 이미지를 생성하는 디퓨전 모델의 특성을 활용하여 효율적이고 고품질의 3D 형태 예측을 가능하게 합니다. 저희는 논문의 방법론을 재현함으로써 이 혁신적인 접근법의 유효성을 확인하였습니다.

## 3. Experiments

### 3-1. Dataset

모델 훈련을 위해 논문에서 제안한 데이터셋 구조를 따랐습니다. 적혈구 이미지를 포함한 **RBC (Red Blood Cell)** 데이터셋은 각 2D 이미지와 그에 상응하는 3D 객체 및 마스크 파일로 구성되어 있습니다.

```bash
└── data/
 ├── images/  (2D 세포 이미지 파일)
 ├── obj/     (3D Object 파일)
 └── mask/    (2D 마스크 이미지 파일)
```
### 3-2. Results

DISPR 모델을 통해 생성된 3D 재구성 결과의 예시는 `./predictions`에서 확인할 수 있습니다.

### 3-3. Classification

DISPR 모델에 더하여 **Random Forest**를 사용한 적혈구 분류 과정을 추가하였습니다. 분류 정확도를 높이기 위해 다음 두 가지 방식을 모두 시도했습니다.
- 2D image/mask feature 활용
- DISPR의 3D object feature 활용
