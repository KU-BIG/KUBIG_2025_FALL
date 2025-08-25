# config.py
from pathlib import Path

# 경로 기본값 (필요 시 수정)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)

PATH_BUSINESS = DATA / "yelp_academic_dataset_business.json"
PATH_REVIEW   = DATA / "yelp_academic_dataset_review.json"

# 생성물 기본 경로
PATH_REVIEWS_FILTERED = DATA / "reviews_sample.jsonl"
PATH_SENT_WITH_SENT   = OUT  / "sentences_with_sentiment.csv"
PATH_WITH_TOPICS      = OUT  / "with_topics.csv"

# 감성 모델 (영어)
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# BERTopic 임베딩 모델 (영어)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Hierarchical Topic Modeling Settings ===
ZEROSHOT_MODEL = 'joeddav/xlm-roberta-large-xnli'
ZEROSHOT_THRESHOLD = 0.35
MIN_DOCS_PER_CATEGORY = 5

# === Device Configuration ===
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Output Paths for Hierarchical Results ===
PATH_HIERARCHICAL_OUTPUTS = Path("./outputs_hierarchical")