# 과학상상화 Generator — Stable Diffusion v1.5 + Multi-LoRA
---
CV 3 team | 컴통이들 | 21기 김채은, 22기 김창현, 최민석, 황원준

## Contents

1. [Introduction](#1-introduction)  
2. [Related Work & Method](#2-related-work--method)  
   2.1 [Stable Diffusion](#21-stable-diffusion) · 2.2 [LoRA](#22-lora) · 2.3 [Multi-LoRA](#23-multi-lora) · 2.4 [Our Method](#24-our-method)  
3. [Experiments](#3-experiments)  
4. [Analysis](#4-analysis)  
5. [Limitations & Future Work](#5-limitations--future-work)  

---

## 1) Introduction

### 주제 선정 동기
<img width="500" height="344" alt="image" src="https://github.com/user-attachments/assets/81b1689d-35d1-4f9c-80a7-7d2a0adb27d0" />

- 누구나 어린 시절 한 번쯤 그려본 과학 상상화는 아이들 특유의 그림체와 창의적인 상상력이 결합된 그림입니다.
- 과연 generative 모델로 과학상상화의 창의성과 작품성을 얼마나 보여줄 수 있는가를 탐구해보고자 본 프로젝트를 시작하였습니다.


### 주제 소개
<img width="500" height="376" alt="image" src="https://github.com/user-attachments/assets/67aa58eb-b4da-459d-a167-c5605a92ce62" />

- 직접 아이들이 그린 과학상상화를 모델에 학습시킬 수는 없기 때문에,\
아이들 그림체 + 수채화 그림체 + 공상과학 구조물에 대해서 각각 학습시킨 LoRA adapter를 합쳐 저희가 원하는 과학 상상화를 생성해내고자 합니다.



---



## 2) Related Work & Method
### 2.2 LoRA
<img width="600" height="382" alt="image" src="https://github.com/user-attachments/assets/279408a4-eba3-4fa1-bf4b-2b5f6d06c287" />


### 2.4 Our Method
<img width="600" height="359" alt="image" src="https://github.com/user-attachments/assets/159d2ccf-e258-4081-9d34-3f9c98e0de3d" />

- Child / Watercolor / Science 세 LoRA를 각각 독립 학습한 후 SD v1.5에 병렬적으로 mixing, 목표 그림 스타일에 가까워지도록!



---

## 3) Experiments

### Settings
- Base: Stable Diffusion v1.5  
- LoRA: Child / Watercolor / Science  
- Sampling: seed 777, CFG 6, steps 35  
- Prompt: 과학상상화 대표 주제를 공통 프롬프트로 생성하여 비교

### 결과 요약
<img width="674" height="224" alt="image" src="https://github.com/user-attachments/assets/c17d6f51-9fe5-47a8-88bd-d56a440b5cee" />

- +Child LoRA (`λ≈0.65`): 아이 그림체 스타일이 분명해지지만 구조물 디테일이 단순화되는 경향
  <img width="703" height="230" alt="image" src="https://github.com/user-attachments/assets/028acd9c-bede-48ff-a1a8-8d264907075a" />

- +Watercolor LoRA (`λ≈0.55`): 번짐/채색 질감이 강화되고 디테일이 증가
  <img width="677" height="226" alt="image" src="https://github.com/user-attachments/assets/15c9f7d4-b398-4f13-aa0d-1e983396235d" />

- +Science LoRA (`λ≈0.50`): 과학 구조물의 형태/디테일이 뚜렷해짐
  <img width="656" height="217" alt="image" src="https://github.com/user-attachments/assets/af7784e1-70ac-457d-a583-36af22b04092" />

- 최종 결과 요약
  
  <img width="600" height="382" alt="image" src="https://github.com/user-attachments/assets/f53104b4-64c5-426d-a4f5-d871a6fd7532" />


### 추가 관찰
- 가중치 조절: 세 LoRA 가중치 변화 실험, 세 LoRA를 모두 크게 주면 밸런스 붕괴 가능.
  <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/e66d32ab-dd40-457c-b12c-00811cce90cc" />

- 해상도 변화: 512→1024에서 패턴 반복/과밀집 양상이 관찰됨.

---

## 4) Analysis

### 각 LoRA 영향 분석
  
   <img width="500" height="275" alt="image" src="https://github.com/user-attachments/assets/1e005f89-60c3-4ea3-bf79-1ffe9d56ef00" />
   <img width="500" height="311" alt="image" src="https://github.com/user-attachments/assets/b83a80fb-df7e-4237-bdee-dfd6d7846922" />

### LoRA 선형 결합 분석
   - 각 LoRA 레이어 별 코사인 유사도 분석
     
       <img width="316" height="388" alt="image" src="https://github.com/user-attachments/assets/85dbe9cb-91d8-42fb-9f4e-665a875b7af6" />

   - 레이어별 유사도 측정 (subspace overlap)
       <img width="734" height="222" alt="image" src="https://github.com/user-attachments/assets/a35ae487-6b3c-4564-98a5-5ff742fa7fb3" />

  

---

## 5) Limitations & Future Work

- 세 LoRA가 완전히 직교하지 않아 특정 조합에서 간섭 발생 가능
- 정량 평가가 어려운 task 
- 다중 LoRA를 DARE,TIE 등의 방식으로 보다 task-aware한 병합 가능
- 다른 LoRA method 활용 or PCA,t-SNE 등으로 다양한 정량 분석 기대

