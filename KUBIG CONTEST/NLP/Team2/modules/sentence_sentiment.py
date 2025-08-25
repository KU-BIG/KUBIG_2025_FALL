# modules/sentence_sentiment.py
import json, csv
from typing import Iterable
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def _norm_label(lbl: str) -> int:
    u = lbl.upper()
    if "NEG" in u or u.endswith("_0"): return 0
    if "POS" in u or u.endswith("_1"): return 1
    return 1

def run_sentence_sentiment(
    filtered_jsonl_path: str,
    out_csv_path: str,
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    max_chars: int = 256,
    batch_size: int = 64,
):
    tok = AutoTokenizer.from_pretrained(sentiment_model)
    mdl = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, truncation=True, padding=True, device=-1)

    with open(filtered_jsonl_path, "r", encoding="utf-8") as fin, \
         open(out_csv_path, "w", newline="", encoding="utf-8") as fout:

        w = csv.writer(fout)
        w.writerow(["business_id","review_id","sentence_id","sentence","stars","date","sentiment","sentiment_conf"])

        buf_sent, buf_meta = [], []

        def flush():
            if not buf_sent: return
            outs = pipe(buf_sent, batch_size=batch_size)
            for meta, o in zip(buf_meta, outs):
                sentiment = _norm_label(o["label"])
                conf = float(o["score"])
                w.writerow([*meta, sentiment, conf])
            buf_sent.clear()
            buf_meta.clear()

        for line in fin:
            r = json.loads(line)
            sents = sent_tokenize(r.get("text") or "")
            for i, s in enumerate(sents, start=1):
                sent = s.strip()[:max_chars]
                meta = [r.get("business_id"), r.get("review_id"), f"{r.get('review_id')}_s{i}", sent, r.get("stars"), r.get("date")]
                buf_sent.append(sent)
                buf_meta.append(meta)
                if len(buf_sent) >= batch_size:
                    flush()
        flush()
