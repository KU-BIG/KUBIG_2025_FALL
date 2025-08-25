# ReviewToRevenue: NLP 기반 레스토랑 리뷰 인사이트 서비스

**Team ZeroSugar (NLP 2팀)**  
21기 강서연, 윤채영 | 22기 김종현, 백서현  

---

## 📌 프로젝트 개요

본 프로젝트는 **Yelp 공개 데이터셋**을 활용하여 **레스토랑 리뷰 텍스트에서 핵심 이슈와 감정을 자동으로 추출**하고, 이를 **Streamlit 대시보드** 형태로 시각화하여 **레스토랑 경영 인사이트 서비스**를 제공한다.  

### 🎯 목표
- 수작업 리뷰 분석의 한계 극복
- 고객 만족·불만 요인 체계적 파악
- 데이터 기반 레스토랑 운영 및 마케팅 전략 지원  

---

## 🛠 프로젝트 특징
- **문장 단위 감성 분석** (fine-grained sentiment detection)  
- **토픽 기반 분류 및 인사이트 제공**  
- **Streamlit 대시보드 시각화**로 end-to-end 분석 결과 확인 가능  

---

## 🔄 프로젝트 파이프라인

1. **리뷰 수집 및 필터링** (business_id 기반)  
2. **감성 분석 (Sentiment Analysis)**  
   - 모델: `distilbert-base-uncased-finetuned-sst-2-english`  
   - 문장 단위 분할(`nltk.sent_tokenize`) 후 이진 분류 (POSITIVE / NEGATIVE)  
3. **NLI 기반 멀티라벨 토픽 분류**  
   - 모델: `cross-encoder/nli-deberta-v3-base`  
   - 36개 세부 토픽(aspect)과 문장 매칭  
   - Multi-label, sigmoid 기반 threshold=0.8 적용  
   * 시도: BERTopic 기반 토픽 모델링 
   - 임베딩: `all-MiniLM-L6-v2`  
   - UMAP → HDBSCAN → c-TF-IDF 기반 토픽 추출  
   - 비지도학습 한계를 보완하기 위해 NLI 지도학습과 병행  
4. **Streamlit 시각화**  
   - 카테고리별 토픽 분포 및 긍·부정 비율  
   - 시계열 트렌드 (월별 긍·부정 변화)  
   - 상위 긍·부정 토픽 및 요약 (BART Summarization)  
   - GPT API 기반 Action Plan 자동 제안  

---

## 📂 폴더 구조

```
NLPteam2/
├─ app.py                  # BERTopic 토픽 모델링 기반 대시보드
├─ app_nli.py              # NLI 기반 분석 전용 대시보드
├─ main_final.py           # BERTopic 기반 분석석 실행 스크립트
├─ main_nli.py             # NLI 기반 분석 실행 스크립트
├─ config.py               # 데이터 경로 및 모델 설정
├─ modules/                # 기능 모듈
│  ├─ topic_model*.py      # BERTopic 래퍼
│  ├─ nli_multilabel*.py   # NLI 멀티라벨 분류기
│  ├─ sentence_sentiment.py# 감성 분석 유틸
│  ├─ filter_reviews.py    # 리뷰 필터링
│  └─ find_business_ids.py # business JSONL에서 ID 검색
├─ data/                   # 데이터셋 (사용자 제공 필수)
│  └─ yelp data (business, reviews) ...
└─ requirements.txt
```

> ⚠️ **데이터는 반드시 `data/` 폴더에 넣어야 실행 가능.**

---

## ⚙️ 설치 및 실행

### 1) 환경 설정
```bash
git clone https://github.com/jonghyuneya/NLPteam2
cd NLPteam2
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 데이터 준비
- `data/` 폴더에 Yelp 리뷰 데이터셋 배치 (CSV/JSONL 형식)  
- 예시:  
  - `reviews.csv` (리뷰 텍스트)  
  - `sentences.csv` (문장 단위 분할 결과)  

### 3) Streamlit 실행
```bash
streamlit run app.py
```
or  
```bash
streamlit run app_nli.py
```

### 4) 커맨드라인 실행 예시
```bash
python main_final.py \
  --input data/sentences.csv \
  --business_id cafe_001 \
  --embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
  --out data/summary.csv
```

---

## 📊 주요 기능 (Dashboard Features)

- **Performance Overview**: 전체 리뷰 개수, 별점, 긍·부정 비율  
- **Category Analysis**: 36개 세부 토픽별 감정 비율  
- **Priorities**: 부정 비율이 높은 토픽 랭킹 → 경영 개선 포인트  
- **Trends**: 월별 토픽별 긍정/부정 비율 변화  
- **AI Insights**: GPT 기반 Action Plan 추천  

---

## 🚀 향후 개선 방향
- DeBERTa 기반 NLI 모델 파인튜닝으로 분류 성능 개선  
- 크롤링 도입을 통한 실시간 리뷰 분석 서비스 제공  
- 분석 속도 최적화  

---

