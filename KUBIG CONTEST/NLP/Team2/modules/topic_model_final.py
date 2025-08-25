# modules/topic_model.py
import re
import pickle
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan

from bertopic import BERTopic
from bertopic.representation import ZeroShotClassification


# -----------------------------
# NLTK / spaCy 준비
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not found. Using basic preprocessing.")
    nlp = None


# -----------------------------
# 전처리 / 토크나이저
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\n", " ", str(text))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def advanced_tokenizer(text: str, business_name: str = "") -> list[str]:
    if nlp is None:
        return text.lower().split()
    words = []
    doc = nlp(text)
    ENGLISH_STOP_WORDS = set(stopwords.words("english"))
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            token.tag_[:1] in ['N', 'V'] and
            lemma not in ENGLISH_STOP_WORDS and
            token.ent_type_ not in ['TIME', 'CARDINAL', 'DATE', 'PERSON'] and
            len(lemma) > 1 and
            lemma != business_name.lower()
        ):
            words.append(lemma)
    return words


# -----------------------------
# 라벨 팩(코어 + 선택 팩) 및 트리거
# -----------------------------
LABEL_CORE = [
    # Service/Operations
    "wait time and queue management (waitlist, minutes, seated, reservation)",
    "host and seating process (hostess, greeting, table assignment)",
    "server friendliness and politeness (friendly, rude, courteous)",
    "server attentiveness and follow-ups (refills, check on us, ignored)",
    "order accuracy and missing items (order accuracy, missing, correct)",
    "kitchen speed and ticket time (slow kitchen, quick turnaround)",
    "bill handling and split checks (check, split bill, receipt)",
    "payment methods and checkout (contactless, tap to pay, cash only)",
    "manager response and recovery (manager, apology, comped)",
    # Cleanliness/Safety
    "tableware and utensils cleanliness (fork, plate, glass, stains)",
    "dining area cleanliness (floor, wiped, bussing, trash)",
    "restroom cleanliness and supplies (bathroom, soap, paper)",
    # Environment/Ambience
    "noise level and crowding (loud, noisy, packed)",
    "music volume and selection (music loud, playlist)",
    "lighting and visibility (dim, bright, ambiance)",
    "temperature and ventilation (AC, draft, hot, cold)",
    "smell and odors (grease smell, odor)",
    "decor and interior design (decor, vibe, cozy, dated)",
    "seating comfort and space (cramped, comfortable, booth, high chair)",
    "outdoor seating and patio (patio, terrace, heaters, shade)",
    # Accessibility/Family
    "parking convenience and options (parking lot, street parking, valet)",
    "location and transit accessibility (walkable, subway, bus)",
    "ada accessibility and ramps (wheelchair, ramp, accessible restroom)",
    "family friendly and kids options (kids menu, stroller, highchair)",
    # Menu/Value/Food
    "menu variety and seasonal specials (options, choices, seasonal)",
    "menu clarity and descriptions (pictures, translations)",
    "value for money and price fairness (worth, expensive)",
    "portion size and fullness (small portion, generous, filling)",
    "allergen handling and cross contamination (allergy, celiac, nut free)",
    "overall taste and seasoning balance (flavor, bland, salty, sweet)",
    "ingredient freshness and quality (fresh, stale, frozen)",
    "texture and doneness accuracy (overcooked, undercooked, tender, crispy)",
    "temperature of dishes at serving (served cold, piping hot, lukewarm)",
    # Delivery/Takeout (공통)
    "online ordering usability (app, website, checkout)",
    "delivery time and temperature (late delivery, cold on arrival)",
    "takeout packaging and spill protection (container, leak, soggy)",
]

PACK_MENU_GENERIC = [
    "breakfast and brunch dishes (pancakes, benedict, omelet)",
    "sandwiches and burgers (burger, bun, BLT, club)",
    "pizza and flatbreads (pizza, crust, slice)",
    "pasta and noodles (pasta, spaghetti, noodles)",
    "steaks and grilled meats (steak, medium rare, ribeye)",
    "barbecue and smoked meats (bbq, brisket, ribs)",
    "seafood dishes general (shrimp, crab, oyster, salmon)",
    "sushi and raw items (sushi, sashimi, nigiri)",
    "tacos and mexican dishes (taco, burrito, salsa)",
    "asian stir-fry and rice dishes (fried rice, stir fry, curry)",
    "salads and healthy bowls (salad, bowl, greens)",
    "soups and stews (soup, broth, stew)",
    "appetizers and sides (starter, fries, wings)",
    "desserts and sweets (dessert, cake, pie, ice cream)",
    "pastries and baked goods (croissant, pastry, bread)",
]

PACK_BEVERAGE_BAR = [
    "coffee and espresso drinks (latte, cappuccino, espresso)",
    "tea and non coffee beverages (tea, herbal, iced tea)",
    "cocktails quality and consistency (cocktail, watered down, balanced)",
    "beer selection and craft options (tap list, ipa, lager)",
    "wine list and pairing (wine list, pairing, by the glass)",
    "happy hour value and timing (happy hour, specials, discount)",
]

PACK_DELIVERY_DETAIL = [
    "third party delivery issues (Doordash, Uber Eats, driver, handoff)",
    "order status communication (tracking, ready time, text)",
    "curbside pickup and handoff (curbside, pickup window, parking spot)",
]

TRIGGERS_MENU = ["pizza","burger","sushi","sashimi","taco","ramen","pho","bbq",
                 "steak","pasta","noodle","salad","soup","fries","wings","dessert","croissant"]
TRIGGERS_BEVERAGE = ["latte","espresso","coffee","tea","cocktail","ipa","lager","wine","happy hour"]
TRIGGERS_DELIVERY = ["delivery","doordash","ubereats","grubhub","pickup","curbside","driver","tracking"]


def _share_with_triggers(sentences: list[str], triggers: list[str]) -> float:
    trigs = [t.lower() for t in triggers]
    def hit(s: str) -> bool:
        s = s.lower()
        return any(t in s for t in trigs)
    return float(np.mean([hit(s) for s in sentences])) if sentences else 0.0


def build_candidate_topics(sentences: list[str],
                           mode: str = "auto",
                           menu_thresh: float = 0.006,
                           bev_thresh: float = 0.004,
                           deliv_thresh: float = 0.004) -> list[str]:
    """
    mode:
      - "core": 코어 라벨만
      - "full": 코어 + 모든 선택 팩
      - "auto": 코어 + 트리거 비율 임계 통과한 팩만 자동 추가
    """
    topics = LABEL_CORE.copy()
    if mode == "core":
        return topics
    if mode == "full":
        return topics + PACK_MENU_GENERIC + PACK_BEVERAGE_BAR + PACK_DELIVERY_DETAIL

    # mode == "auto"
    if _share_with_triggers(sentences, TRIGGERS_MENU) >= menu_thresh:
        topics += PACK_MENU_GENERIC
    if _share_with_triggers(sentences, TRIGGERS_BEVERAGE) >= bev_thresh:
        topics += PACK_BEVERAGE_BAR
    if _share_with_triggers(sentences, TRIGGERS_DELIVERY) >= deliv_thresh:
        topics += PACK_DELIVERY_DETAIL
    return topics


# -----------------------------
# 제로샷 라벨러 (multi-label)
# -----------------------------
def _build_zeroshot_rep(sentences: list[str],
                        model_name: str = "facebook/bart-large-mnli",
                        label_mode: str = "auto",
                        multi_label: bool = True,
                        min_prob: float = 0.4,
                        local_only: bool = True) -> ZeroShotClassification:
    candidate_topics = build_candidate_topics(sentences, mode=label_mode)
    return ZeroShotClassification(
        candidate_topics=candidate_topics,
        model=model_name,
        pipeline_kwargs={
            "multi_label": multi_label,
            "hypothesis_template": "This review sentence is about {}.",
            "local_files_only": bool(local_only),  # 캐시에 없으면 즉시 실패
        },
        min_prob=min_prob,
    )


# -----------------------------
# 동적 파라미터 & 안전 벡터라이저
# -----------------------------
def _topic_diversity(model: BERTopic, topk: int = 10) -> float:
    info = model.get_topic_info()
    vocab_lists: list[list[str]] = []
    for tid in info["Topic"]:
        if tid == -1:
            continue
        words = model.get_topic(tid) or []
        vocab_lists.append([w for (w, s) in words[:topk]])
    if not vocab_lists:
        return 0.0
    flat = sum(vocab_lists, [])
    unique = len(set(flat))
    total = len(flat)
    return unique / (total + 1e-8)


def _auto_min_topic_size(n_docs: int) -> int:
    """
    작은 코퍼스에서도 클러스터가 생기도록 상대/절대 하한 조합.
    예: n_docs=100 ->  max(10, 4) = 10
        n_docs=250 ->  max(10, 10) = 10
        n_docs=800 ->  max(10, 32) = 32
    """
    return max(10, int(round(0.008 * n_docs)))


def _estimate_n_topics(n_docs: int, min_topic_size: int) -> int:
    est = max(2, int(np.ceil(n_docs / max(1, min_topic_size))))
    return int(np.clip(est, 2, 40))


def _make_vectorizer_safe(n_topics_est: int) -> CountVectorizer:
    """
    c-TF-IDF는 '토픽 수' 만큼의 문서를 대상으로 학습되므로,
    n_topics_est가 작을 때 min_df와 max_df의 모순을 방지한다.
    """
    if n_topics_est < 5:
        min_df = 1
        max_df = 1.0
    elif n_topics_est < 10:
        min_df = 1
        max_df = 0.98
    else:
        min_df = 1
        max_df = 0.95

    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )


def _safe_copy_model(model: BERTopic) -> BERTopic:
    try:
        return pickle.loads(pickle.dumps(model))
    except Exception:
        return model


def _choose_nr_topics(model: BERTopic, docs: Iterable[str],
                      candidates: Iterable[int] = (10, 12, 14, 16, 18)) -> Optional[int]:
    best_score, best_k = -1.0, None
    for k in candidates:
        tmp = _safe_copy_model(model)
        try:
            tmp.reduce_topics(docs=docs, nr_topics=k)
            score = _topic_diversity(tmp, topk=10)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    return best_k


# -----------------------------
# 메인 함수
# -----------------------------
def apply_bertopic_for_business(
    sentences_csv_path: str,
    out_csv_path: str,
    business_id: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size: Optional[int] = None,
    nr_topics: Optional[int] = None,
    enable_zeroshot: bool = False,
    zeroshot_online: bool = False,
    zeroshot_model: str = "facebook/bart-large-mnli",
    zeroshot_min_prob: float = 0.4,
    label_mode: str = "auto",
    business_meta: dict | None = None
):
    """
    소/중/대 코퍼스에 모두 대응:
      - 전처리/길이 필터(작은셋 완화)
      - 동적 Vectorizer/UMAP/HDBSCAN
      - c-TF-IDF min_df/max_df 충돌 시 자동 재시도
      - 소형 코퍼스에서는 무리한 토픽 축소 방지
    """
    print("[Stage] Load & filter...")
    # --- Load & filter ---
    df = pd.read_csv(sentences_csv_path)

    # per-store vs pooled
    if business_id in (None, "__ALL__", "*ALL*"):
        # 모든 선택된 점포를 통합해서 분석
        d = df.copy()
        print("[Mode] POOLED across all selected businesses")
    else:
        # 특정 점포 한 곳만 분석
        d = df[df["business_id"] == business_id].copy()

    if d.empty:
        raise ValueError(f"No rows for business_id={business_id}")

    
    if business_meta and "business_id" in d.columns:
        d["business_address"] = d["business_id"].map(lambda x: (business_meta.get(x, {}) or {}).get("address", ""))
        d["business_name"] = d.get("business_name", pd.Series("", index=d.index))
        # 이미 business_name이 있으면 그대로 두고, 없으면 meta의 name으로 보강
        if "business_name" in d.columns and d["business_name"].eq("").all():
            d["business_name"] = d["business_id"].map(lambda x: (business_meta.get(x, {}) or {}).get("name", ""))


    docs_raw = d["sentence"].fillna("").astype(str).tolist()

    # 소형 코퍼스면 최소 토큰 길이를 4로 완화
    token_min = 4 if len(docs_raw) < 300 else 5
    clean_docs = [clean_text(t) for t in docs_raw]
    token_counts = [len(x.split()) for x in clean_docs]
    keep_mask = [tc >= token_min for tc in token_counts]
    d = d.loc[keep_mask].reset_index(drop=True)
    clean_docs = [x for x, k in zip(clean_docs, keep_mask) if k]
    if not clean_docs:
        raise ValueError("All sentences removed after cleaning/length filter.")
    n_docs = len(clean_docs)

    # 동적 min_topic_size
    if min_topic_size is None or min_topic_size <= 0:
        min_topic_size = _auto_min_topic_size(n_docs)

    # UMAP 동적
    nn = max(5, min(15, int(0.05 * n_docs)))          # 5~15
    md = 0.10 if n_docs < 200 else 0.05               # 소형셋은 약간 넓게
    umap_model = UMAP(n_neighbors=nn, n_components=5, min_dist=md,
                      metric='cosine', random_state=42)

    # HDBSCAN 동적
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='leaf',
        prediction_data=True
    )

    # Vectorizer 동적
    n_topics_est = _estimate_n_topics(n_docs, min_topic_size)
    vectorizer_model = _make_vectorizer_safe(n_topics_est)

    print(f"[Stage] Build models... (n_docs={n_docs}, min_topic_size={min_topic_size}, "
          f"n_neighbors={nn}, min_dist={md}, n_topics_est={n_topics_est})")

    embedding_model_obj = SentenceTransformer(embedding_model)

    def _build_model(vectorizer: CountVectorizer) -> BERTopic:
        return BERTopic(
            embedding_model=embedding_model_obj,
            vectorizer_model=vectorizer,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=10,
            min_topic_size=min_topic_size,
            nr_topics=None,
            language='english',
            calculate_probabilities=False,
            verbose=False
        )

    model = _build_model(vectorizer_model)

    print("[Stage] Fit...")
    try:
        _topics, _ = model.fit_transform(clean_docs)
    except ValueError as e:
        # sklearn의 max_df/min_df 충돌 등 → 초안전 재시도
        if "max_df corresponds to < documents than min_df" in str(e):
            print("[Warn] Retrying with ultra-safe vectorizer (min_df=1, max_df=1.0)")
            vectorizer_safe = CountVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                max_df=1.0
            )
            model = _build_model(vectorizer_safe)
            _topics, _ = model.fit_transform(clean_docs)
        else:
            raise

    info = model.get_topic_info()
    init_num = len(info) - 1
    print(f"[BERTopic] Initial topics (excl -1): {init_num}")

    # 소형 코퍼스에서는 무리한 축소 금지
    target = None
    if nr_topics is not None:
        target = nr_topics if init_num > nr_topics else None
    else:
        # 충분히 토픽이 많고(중/대형 셋) 과분할이 의심될 때만 축소
        if n_docs >= 600 and init_num > 16:
            target = _choose_nr_topics(model, clean_docs) or 12

    if target is not None and init_num > target:
        print(f"[Auto] Reducing topics to: {target}")
        model.reduce_topics(docs=clean_docs, nr_topics=target)

    print("[Stage] Transform (final assignments)...")
    final_topics, _ = model.transform(clean_docs)
    d["topic_id"] = pd.Series(final_topics, index=d.index).astype(int)

    # 토픽 키워드
    info = model.get_topic_info()
    valid_tids = info["Topic"].tolist()
    topic2kw = {}
    for tid in valid_tids:
        if tid == -1:
            topic2kw[tid] = ""
            continue
        words = model.get_topic(tid) or []
        topic2kw[tid] = ", ".join([w for (w, s) in words[:5]]) if words else ""
    d["topic_keywords"] = d["topic_id"].map(topic2kw).fillna("")

    # ====== 저장 1: 제로샷 없이도 반드시 저장 ======
    print("[Stage] Save core outputs...")
    d.to_csv(out_csv_path, index=False)

   

    grp = d.groupby("topic_id", as_index=False).agg(
        n=("sentence_id", "count"),
        pos=("sentiment", lambda x: float(np.mean(x == 1))),
        stars_mean=("stars", "mean"),
        conf_mean=("sentiment_conf", "mean")
    )
    grp["share"] = grp["n"] / grp["n"].sum()
    grp["keywords"] = grp["topic_id"].map(topic2kw)
    grp.sort_values("n", ascending=False).to_csv(out_csv_path.replace(".csv", "_topic_summary.csv"), index=False)

    if "date" in d.columns:
        d["_ym"] = pd.to_datetime(d["date"], errors="coerce").dt.to_period("M").astype(str)
        trend = d[d["_ym"].notna()].groupby(["_ym", "topic_id"], as_index=False).size()
        trend.rename(columns={"size": "count"}, inplace=True)
        trend.to_csv(out_csv_path.replace(".csv", "_topic_trend_by_month.csv"), index=False)

    examples = (
        d.sort_values("sentiment_conf", ascending=False)
         .groupby("topic_id").head(5)[["topic_id", "topic_keywords", "sentence", "stars", "sentiment", "sentiment_conf", "date"]]
    )
    examples.to_csv(out_csv_path.replace(".csv", "_topic_examples.csv"), index=False)

    # ====== 제로샷(선택) ======
    if enable_zeroshot:
        try:
            print("[Stage] Zero-shot labeling...")
            rep = _build_zeroshot_rep(
                sentences=clean_docs,
                model_name=zeroshot_model,
                label_mode=label_mode,
                multi_label=True,
                min_prob=zeroshot_min_prob,
                local_only=not bool(zeroshot_online)
            )
            model.update_topics(clean_docs, representation_model=rep)

            info2 = model.get_topic_info()
            if "Name" in info2.columns:
                tid2name = dict(zip(info2["Topic"], info2["Name"]))
            else:
                if "Representation" in info2.columns:
                    def _repr_to_name(x):
                        if isinstance(x, (list, tuple)) and len(x) > 0:
                            return ", ".join(map(str, x[:5]))
                        return str(x) if pd.notna(x) else ""
                    tid2name = dict(zip(info2["Topic"], info2["Representation"].apply(_repr_to_name)))
                else:
                    tid2name = {}

            if tid2name:
                d["topic_label"] = d["topic_id"].map(tid2name).fillna("")
                d.to_csv(out_csv_path.replace(".csv", "_with_labels.csv"), index=False)
                grp["label"] = grp["topic_id"].map(tid2name)
                grp.to_csv(out_csv_path.replace(".csv", "_topic_summary_with_labels.csv"), index=False)
                print("[Stage] Zero-shot labels saved.")
            else:
                print("Warning: Zero-shot ran but no label-like column returned.")
        except Exception as e:
            print(f"Warning: Zero-shot skipped due to error: {e}")

    print("[Stage] Done.")
    return model
