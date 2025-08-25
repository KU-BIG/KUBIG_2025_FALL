# modules/business_meta.py
import json
import pandas as pd
from typing import Dict, List

def _join_nonempty(parts):
    parts = [str(p).strip() for p in parts if p and str(p).strip() not in ("", "nan", "None")]
    return ", ".join(parts)

def load_business_meta(business_json_path: str, target_ids: List[str]) -> Dict[str, dict]:
    """business.jsonl에서 target_ids에 해당하는 name/address 메타를 로드."""
    idset = set(target_ids)
    rows = []
    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("business_id") in idset:
                rows.append(obj)
    if not rows:
        return {}

    df = pd.DataFrame(rows)

    # 주소 컬럼 후보
    addr_cols = [c for c in ["address", "address1", "address2", "address3"] if c in df.columns]
    if addr_cols:
        # 앞쪽(가장 구체) 컬럼부터 결측 보간
        base_addr = df[addr_cols].bfill(axis=1).iloc[:, 0].astype(str)
    else:
        base_addr = pd.Series([""], index=df.index, dtype=str)

    parts = [base_addr]
    for col in ["city", "state", "postal_code"]:
        if col in df.columns:
            parts.append(df[col].astype(str))

    display_addr = []
    for i in range(len(df)):
        display_addr.append(_join_nonempty([p.iloc[i] for p in parts]))
    df["display_address"] = display_addr

    meta = {}
    for _, r in df.iterrows():
        bid = r.get("business_id")
        meta[bid] = {
            "name": r.get("name", "") or "",
            "address": r.get("display_address", "") or "",
        }
    return meta
