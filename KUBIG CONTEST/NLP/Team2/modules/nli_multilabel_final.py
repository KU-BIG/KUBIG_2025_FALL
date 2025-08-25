# modules/nli_multilabel_final.py
import re
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm


# NLTK 준비
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# 텍스트 전처리
def clean_text(text: str) -> str:
    text = re.sub(r"\n", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 레스토랑 리뷰 분석을 위한 핵심 측면(aspect) 라벨들
RESTAURANT_ASPECTS = [
    # Service/Operations
    "wait time and queue management",
    "host and seating process", 
    "server friendliness and politeness",
    "server attentiveness and follow-ups",
    "order accuracy and missing items",
    "kitchen speed and ticket time",
    "bill handling and split checks",
    "payment methods and checkout",
    "manager response and recovery",
    
    # Cleanliness/Safety
    "tableware and utensils cleanliness",
    "dining area cleanliness",
    "restroom cleanliness and supplies",
    
    # Environment/Ambience
    "noise level and crowding",
    "music volume and selection",
    "lighting and visibility",
    "temperature and ventilation",
    "smell and odors",
    "decor and interior design",
    "seating comfort and space",
    "outdoor seating and patio",
    
    # Accessibility/Family
    "parking convenience and options",
    "location and transit accessibility",
    "ada accessibility and ramps",
    "family friendly and kids options",
    
    # Menu/Value/Food
    "menu variety and seasonal specials",
    "menu clarity and descriptions",
    "value for money and price fairness",
    "portion size and fullness",
    "allergen handling and cross contamination",
    "overall taste and seasoning balance",
    "ingredient freshness and quality",
    "texture and doneness accuracy",
    "temperature of dishes at serving",
    
    # Delivery/Takeout
    "online ordering usability",
    "delivery time and temperature",
    "takeout packaging and spill protection",
]


class NLIAspectClassifier:
    """NLI 기반 레스토랑 측면(aspect) 분류기"""
    
    def __init__(self, 
                 model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnlii",
                 device: str = "auto",
                 batch_size: int = 16,
                 local_files_only: bool = True):
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.model_name = model_name
        
        # NLI 파이프라인 초기화
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                local_files_only=local_files_only
            )
        except Exception as e:
            print(f"Warning: Failed to load model {model_name} with local_files_only={local_files_only}")
            if local_files_only:
                print("Retrying with local_files_only=False...")
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    local_files_only=False
                )
            else:
                raise e
    
    def classify_sentences(self, 
                          sentences: List[str], 
                          aspects: List[str],
                          min_prob: float = 0.8,
                          hypothesis_template: str = "This review sentence is about {}.") -> List[Dict]:
        """
        문장들을 레스토랑 측면별로 멀티라벨 분류
        
        Returns:
            List of dicts with keys: 'sentence', 'assigned_aspects', 'scores', etc.
        """
        results = []
        
        # 배치 처리
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="NLI Aspect Classification"):
            batch_sentences = sentences[i:i + self.batch_size]
            
            for sentence in batch_sentences:
                try:
                    # Zero-shot classification 수행
                    result = self.classifier(
                        sentence, 
                        aspects,
                        hypothesis_template=hypothesis_template,
                        multi_label=True
                    )
                    
                    # 결과 처리
                    assigned_aspects = []
                    assigned_scores = []
                    
                    for aspect, score in zip(result['labels'], result['scores']):
                        if score >= min_prob:
                            assigned_aspects.append(aspect)
                            assigned_scores.append(score)
                    
                    # 아무 측면도 할당되지 않은 경우 가장 높은 스코어의 측면을 할당
                    if not assigned_aspects and result['labels'] and result['scores']:
                        assigned_aspects = [result['labels'][0]]
                        assigned_scores = [result['scores'][0]]
                    
                    results.append({
                        'sentence': sentence,
                        'all_aspects': result['labels'],
                        'all_scores': result['scores'],
                        'assigned_aspects': assigned_aspects,
                        'assigned_scores': assigned_scores,
                        'max_score': max(result['scores']) if result['scores'] else 0.0,
                        'num_aspects': len(assigned_aspects)
                    })
                    
                except Exception as e:
                    print(f"Error processing sentence: {str(e)[:100]}...")
                    results.append({
                        'sentence': sentence,
                        'all_aspects': [],
                        'all_scores': [],
                        'assigned_aspects': [],
                        'assigned_scores': [],
                        'max_score': 0.0,
                        'num_aspects': 0
                    })
        
        return results


def process_aspect_classification_results(classification_results: List[Dict], 
                                        sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    멀티라벨 분류 결과를 데이터프레임에 통합
    """
    
    # 결과를 데이터프레임에 추가
    sentences_df = sentences_df.copy()
    
    # 각 문장에 분류 결과 추가
    sentences_df['assigned_aspects'] = [result['assigned_aspects'] for result in classification_results]
    sentences_df['aspect_scores'] = [result['assigned_scores'] for result in classification_results]
    sentences_df['max_aspect_score'] = [result['max_score'] for result in classification_results]
    sentences_df['num_aspects'] = [result['num_aspects'] for result in classification_results]
    
    # 주요 측면 (가장 높은 스코어) 추가
    def get_primary_aspect(aspects, scores):
        if not aspects or not scores:
            return "unclassified"
        max_idx = scores.index(max(scores))
        return aspects[max_idx]
    
    sentences_df['primary_aspect'] = [
        get_primary_aspect(result['assigned_aspects'], result['assigned_scores']) 
        for result in classification_results
    ]
    
    # 측면별 더미 변수 생성 (선택적)
    all_aspects = RESTAURANT_ASPECTS
    for aspect in all_aspects:
        sentences_df[f'has_{aspect.replace(" ", "_").replace("/", "_")}'] = [
            aspect in result['assigned_aspects'] for result in classification_results
        ]
    
    return sentences_df


def create_aspect_summary(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """측면별 요약 통계 생성"""
    
    # 개별 측면별 통계
    aspect_stats = []
    
    for aspect in RESTAURANT_ASPECTS:
        # 해당 측면이 할당된 문장들
        aspect_sentences = sentences_df[sentences_df['assigned_aspects'].apply(lambda x: aspect in x)]
        
        if len(aspect_sentences) > 0:
            # 해당 측면의 평균 스코어 계산 - 수정된 부분
            aspect_scores = []
            for idx, row in aspect_sentences.iterrows():
                if aspect in row['assigned_aspects']:
                    aspect_idx = row['assigned_aspects'].index(aspect)
                    if aspect_idx < len(row['aspect_scores']):
                        aspect_scores.append(row['aspect_scores'][aspect_idx])
            
            stats = {
                'aspect': aspect,
                'n_sentences': len(aspect_sentences),
                'share': len(aspect_sentences) / len(sentences_df),
                'avg_sentiment': aspect_sentences['sentiment'].mean() if 'sentiment' in aspect_sentences.columns else None,
                'avg_stars': aspect_sentences['stars'].mean() if 'stars' in aspect_sentences.columns else None,
                'avg_score': np.mean(aspect_scores) if aspect_scores else 0.0,  # 수정된 부분
                'positive_ratio': (aspect_sentences['sentiment'] == 1).mean() if 'sentiment' in aspect_sentences.columns else None
            }
            aspect_stats.append(stats)
    
    summary_df = pd.DataFrame(aspect_stats)
    summary_df = summary_df.sort_values('n_sentences', ascending=False)
    
    return summary_df


def apply_nli_aspect_analysis(
    sentences_csv_path: str,
    out_csv_path: str,
    business_id: Optional[str] = None,
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli",
    min_prob: float = 0.8,
    batch_size: int = 16,
    local_files_only: bool = True,
    business_meta: Optional[Dict] = None
):
    """
    NLI 기반 레스토랑 측면 분석을 적용하는 메인 함수
    """
    
    print("[Stage] Load & filter...")
    # 데이터 로드 및 필터링
    df = pd.read_csv(sentences_csv_path)
    
    # per-store vs pooled
    if business_id in (None, "__ALL__", "*ALL*"):
        d = df.copy()
        print("[Mode] POOLED across all selected businesses")
    else:
        d = df[df["business_id"] == business_id].copy()
    
    if d.empty:
        raise ValueError(f"No rows for business_id={business_id}")
    
    # 비즈니스 메타데이터 추가
    if business_meta and "business_id" in d.columns:
        d["business_address"] = d["business_id"].map(
            lambda x: (business_meta.get(x, {}) or {}).get("address", "")
        )
        if "business_name" not in d.columns or d["business_name"].eq("").all():
            d["business_name"] = d["business_id"].map(
                lambda x: (business_meta.get(x, {}) or {}).get("name", "")
            )
    
    # 문장 전처리
    docs_raw = d["sentence"].fillna("").astype(str).tolist()
    clean_docs = [clean_text(t) for t in docs_raw]
    
    # 최소 토큰 길이 필터 (4단어 이상)
    token_min = 4
    token_counts = [len(x.split()) for x in clean_docs]
    keep_mask = [tc >= token_min for tc in token_counts]
    d = d.loc[keep_mask].reset_index(drop=True)
    clean_docs = [x for x, k in zip(clean_docs, keep_mask) if k]
    
    if not clean_docs:
        raise ValueError("All sentences removed after cleaning/length filter.")
    
    n_docs = len(clean_docs)
    print(f"[Info] Processing {n_docs} sentences (10+ words each)")
    
    # 레스토랑 측면 라벨들
    print("[Stage] Using restaurant aspect labels...")
    aspects = RESTAURANT_ASPECTS
    print(f"[Info] Analyzing {len(aspects)} restaurant aspects")
    
    # NLI 분류기 초기화
    print("[Stage] Initializing NLI aspect classifier...")
    classifier = NLIAspectClassifier(
        model_name=nli_model,
        batch_size=batch_size,
        local_files_only=local_files_only
    )
    
    # 측면별 분류 수행
    print("[Stage] Performing aspect classification...")
    classification_results = classifier.classify_sentences(
        sentences=clean_docs,
        aspects=aspects,
        min_prob=min_prob
    )
    
    # 결과 처리
    print("[Stage] Processing classification results...")
    d_with_aspects = process_aspect_classification_results(classification_results, d)
    
    # 결과 저장
    print("[Stage] Save outputs...")
    d_with_aspects.to_csv(out_csv_path, index=False)
    
    # 측면별 요약 생성
    summary_df = create_aspect_summary(d_with_aspects)
    
    # 요약 파일 저장
    summary_df.to_csv(out_csv_path.replace(".csv", "_aspect_summary.csv"), index=False)
    
    # 월별 트렌드 (날짜가 있는 경우)
    if "date" in d_with_aspects.columns:
        d_with_aspects["_ym"] = pd.to_datetime(d_with_aspects["date"], errors="coerce").dt.to_period("M").astype(str)
        
        # 각 측면별 월별 트렌드
        trend_data = []
        for _, row in d_with_aspects[d_with_aspects["_ym"].notna()].iterrows():
            for aspect in row['assigned_aspects']:
                trend_data.append({
                    'year_month': row['_ym'],
                    'aspect': aspect,
                    'sentiment': row.get('sentiment', None),
                    'stars': row.get('stars', None)
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            trend_summary = trend_df.groupby(['year_month', 'aspect'], as_index=False).agg({
                'sentiment': ['count', 'mean'],
                'stars': 'mean'
            }).round(3)
            trend_summary.columns = ['year_month', 'aspect', 'count', 'avg_sentiment', 'avg_stars']
            trend_summary.to_csv(out_csv_path.replace(".csv", "_aspect_trends.csv"), index=False)
    
    # 예시 문장들 (각 측면별로)
    examples_data = []
    for aspect in aspects:
        aspect_sentences = d_with_aspects[d_with_aspects['assigned_aspects'].apply(lambda x: aspect in x)]
        if len(aspect_sentences) > 0:
            # 각 측면에서 스코어가 높은 상위 3개 문장
            top_sentences = aspect_sentences.nlargest(3, 'max_aspect_score')
            for _, row in top_sentences.iterrows():
                examples_data.append({
                    'aspect': aspect,
                    'sentence': row['sentence'],
                    'score': row['max_aspect_score'],
                    'stars': row.get('stars', None),
                    'sentiment': row.get('sentiment', None),
                    'date': row.get('date', None)
                })
    
    if examples_data:
        examples_df = pd.DataFrame(examples_data)
        examples_df.to_csv(out_csv_path.replace(".csv", "_aspect_examples.csv"), index=False)
    
    print("[Stage] Done.")
    print(f"[Summary] Analyzed {len(aspects)} aspects across {n_docs} sentences")
    print(f"[Summary] Found {len(summary_df)} aspects with assigned sentences")
    

    return d_with_aspects, summary_df
