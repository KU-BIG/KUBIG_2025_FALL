# modules/filter_reviews.py
import json
from pathlib import Path
from typing import Iterable

def filter_reviews_by_business_ids(
    review_json_path: str,
    target_business_ids: Iterable[str],
    out_jsonl_path: str,
    keep_fields: tuple = ("business_id","review_id","stars","date","text"),
) -> int:
    """
    review.json(JSONL 대용량)을 줄 단위로 읽어 target id의 리뷰만 JSONL로 저장.
    반환값: 쓴 행 수
    """
    targets = set(target_business_ids)
    Path(out_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(review_json_path, "r", encoding="utf-8") as fin, \
         open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            if r.get("business_id") in targets:
                o = {k: r.get(k) for k in keep_fields}
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                kept += 1
    return kept
