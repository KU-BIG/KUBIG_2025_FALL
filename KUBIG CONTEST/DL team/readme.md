# 🙋‍♂️ "도움을 필요한 고객을 찾아라!"

저희 팀은 데이콘의 Basic 해커톤,  
**‘고객 지원 등급 분류: 도움이 필요한 고객을 찾아라!’** 대회에 참여하여,  
고객 데이터를 활용해 **지원이 필요한 고객을 사전 예측**하는 AI 모델을 개발하였습니다.

---

## 👥 팀 소개
**스트ResNet 팀**

- 🧑‍💻 **21기 이준언**
- 👨‍💻 **22기 성용빈**
- 👨‍💻 **22기 이세훈**
- 👩‍💻 **22기 이은서**

---

## 📌 대회 개요
- **주제**: 고객의 행동 및 계약 정보를 바탕으로, 해당 고객이 얼마나 많은 **지원이 필요한지 등급**을 분류하는 문제
- **분류형 문제**: Multi-class classification  
- **평가 지표**: 🎯 **Macro F1 Score** (클래스 간 불균형을 고려한 지표)
- **대회 성격**: 데이콘 Basic 해커톤 (입문자 실습 목적, 프로필 반영 X)

---

## 🛠 프로젝트 접근 전략

### 🔎 데이터 전처리
- 주요 피처: `age`, `gender`, `tenure`, `frequent`, `payment_interval`, `subscription_type`, `contract_length`, `after_interaction`
- 전처리 방법:
  - 수치형: `StandardScaler` 적용
  - 범주형: `LabelEncoder` (필요 시)
  - Train/Validation Stratified Split (80:20)

### 🤖 모델 실험 목록
1. **기본 ML Baseline** – RandomForest, XGBoost 등 (F1-macro ~0.44)  
2. **Plain MLP** – 단순 Dense 구조 (0.434)  
3. **Improved MLP** – BatchNorm + Dropout + EarlyStopping (0.4713)  
4. **Strong Class Weight MLP** – Class Weight 강화 (0.4651)  
5. **Wide & Deep** – Wide(Linear) + Deep(MLP) 결합 (0.452)  
6. **Data-specific MLP** – 다항 변환 + GELU + LayerNorm (0.4845)  
7. **TabPFN** – 사전학습 Transformer 기반, few-shot (⭐ 최고 성능: 0.5224)  

---

## 📊 최종 성능 비교 (Validation F1 Macro)

| 모델 이름                 | Validation F1 Macro |
|--------------------------|---------------------|
| 🌲 RandomForest          | 0.445               |
| 🌲 ExtraTrees            | 0.436               |
| 🔹 Plain MLP             | 0.434               |
| 🔹 Improved MLP          | 0.4713              |
| 🔹 Strong Class Weight MLP | 0.4651            |
| 🔹 Wide & Deep           | 0.452               |
| 🔹 Data-specific MLP     | 0.4845              |
| 🚀 **TabPFN**            | **0.5224**          |

---

## 💡 프로젝트 회고
- 🤔 딥러닝 구조를 다양하게 적용했지만, 데이터 자체가 ‘딥러닝에 최적화된’ 형태는 아니었음  
- 📉 정확도가 드라마틱하게 상승하지 않았던 점은 데이터 특성이 예측에 제약이 있음을 시사  
- 🚀 **TabPFN**은 사전학습 기반 모델의 강점을 보여주며, 소규모 데이터에서도 우수한 성능을 기록  
- 📚 다양한 실험 과정을 통해 단순 성능 향상보다 **실험 설계 및 문제 이해 능력**이 성장한 것이 큰 성과  

---

## 🔗 참고 링크
- 📄 대회 안내: [고객 지원 등급 분류 대회 링크](https://dacon.io/competitions/official/236562/overview/description)
