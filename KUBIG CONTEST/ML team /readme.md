# “건물의 전력사용량을 예측하라!”

저희 ML 팀은 데이콘의 ‘2025 전력사용량 예측 AI 경진대회’에 참여하여

산업 현장에서 활용 가능한 전력 수요 예측 알고리즘을 발굴하고, 

에너지 분야에 대한 AI 기술의 실질적 적용 가능성을 모색하였습니다.

---

### [팀 소개]

ML 1팀

🧑‍💻 21기 김연주

👨‍💻 21기 송상현

👨‍💻 22기 금강산

🧑‍💻 22기 남수빈

---

### [대회 설명]

- 주제 : 건물별 시간단위 전력소비량 예측
    - 특징 : 시계열 예측 문제. 산업 현장에서 에너지 관리 및 수요 대응에 바로 활용 가능한 예측 모델 개발 목표
- 데이터셋
    - 시계열 데이터(건물별 시간단위 전력 사용량), 건물 정보(건물 유형, 면적, 설비 정보 등), 기상 데이터(온도, 습도, 풍속 등)
- 평가 지표 : SMAPE (예값과 실제값의 상대적 차이 측정하는 지표)

![image.png](attachment:4baf08e5-e749-48aa-9d89-9e84e52276bb:image.png)

---

### [프로젝트 접근 전략]

### 1. 데이터 정제 & Feature Engineering

[주요 피처]

- 식별자: **num_date_time, building_number**
- 건물 정보 : **building_type** (공공, 학교, 백화점, 병원, 아파트, 호텔 등 10종), **total_area, cooling_area, solar_power_capacity, ESS_capacity, PCS_capacity**
- 기상 정보 : **temperature(°C), rainfall(mm), wind_speed(m/s),humidity(%), sunshine(hr), solar_radiation(MJ/m²)**
- 타깃 변수: **power_consumption(kWh)**

- **전처리**
    - 파생 변수: thi, apparent_temp, temp_diff, hum_diff
    - 주기 인코딩: sin/cos_hour, sin/cos_dow
    - 랙·롤링 적용 → 데이터 누출 방지

### 2. 학습 모델 1(상현님)

- **주력 모델**: XGBoost
    - 타깃 변환: log1p → expm1 + 소프트 클리핑
    - 건물 단위 학습 (개별 + 유형별)
- **검증 방식**: TimeSeriesSplit 기반 Expanding CV
    - `test_size`를 실제 예측 구간(7일=168h)과 동일하게 맞춰 일반화 성능 개선
    - `n_splits=5` (CV 다양성) vs `n_splits=3` (더 긴 학습 구간 → 안정성)
- **피처 엔지니어링**:
    - 주기성 인코딩(sin/cos of hour, day_of_year)
    - 전일 기온 기반 요약치 (max/mean/min, shift 적용 → leakage 방지)
    - 냉방비율(cooling_ratio), THI, CDH, HI 등 기상 지표
    - 주말/공휴일 flag
    - Fold별 그룹 통계 피처(시간대·요일별 평균, 표준편차)
- **목적 함수 & 커스텀 평가**:
    - `weighted_mse(alpha=3.0)` : 과소예측 페널티 강화
    - 최근성 가중 SMAPE + 폴드간 분산 패널티 → 하이퍼파라미터 선택 기준
- **앙상블**: Soft voting
    - 건물 개별 학습 5개 + 유형별 학습 5개 = 총 10개 모델
    - 각 fold 성능에 최근성 가중치 부여 후 예측값 합성

### 3. 학습 모델 2(연주님)

- **주력 모델**: XGBoost
    - 타깃 변환: log1p → expm1 + 소프트 클리핑
- **보조 모델**: ARIMA, Seasonal-naive (급변/주기 보완)
- **Validation**: Expanding window + Gap(24h)
- **앙상블**: XGB + ARIMA/naive → 중앙값 합성
- 후처리 : OOF 스케일 캘리브레이션, 시간대 잔차 보정, Seasonal-naive 블렌드 10%, 음수값 제거

---

### 최종 성능 비교(SMAPE)

- 모델 1

| 실험 | 결과 |
| --- | --- |
| timeseriessplit 기본값 | public 20.17 |
| timeseriessplit test_size 수정 | public 11.58 |
| test_size 수정 + 앙상블 | public 10.56 |
| test_size 수정 + 하이퍼 파라미터 튜닝 후 앙상블 | public 7.6 |

---

### [프로젝트 회고]

- 기존 K-FOLD 검증에서 **데이터 누수** 발생 → CV 점수 낮음
- 시계열 기반 학습/검증 적용 → 누수 방지, 최근 fold 가중치 적용으로 성능 향상
- 시계열 특성 이해와 CV 설계의 중요성 체감
- Feature Engineering(차분·체감지수, 랙/롤링)으로 모델 성능 실질적 향상 경험
- OOF 스케일/시간대 보정, 앙상블 전략을 통해 안정적인 예측 확보
- 데이터 특성, 변수 스키마 관리, 시계열 CV 설계 등 **실무 ML 프로젝트 전략** 학습

## 참고링크

---

- 대회 안내 : https://dacon.io/competitions/official/236531/overview/description
