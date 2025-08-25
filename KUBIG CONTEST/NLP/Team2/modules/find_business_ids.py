# modules/find_business_ids.py
import json
from typing import Iterable, List

def find_business_ids(
    business_json_path: str,
    name_substring: str | None = None,
    category_keyword: str | None = "Restaurants",
    city: str | None = None,
    state: str | None = None,
    limit: int | None = 50,
) -> List[str]:
    """
    Yelp business JSONL에서 조건에 맞는 business_id를 일부만 반환.
    name_substring, category_keyword, city, state 중 필요한 것만 사용.
    """
    found: List[str] = []
    cnt = 0
    with open(business_json_path, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            name_ok = True
            cat_ok  = True
            city_ok = True
            state_ok= True

            if name_substring:
                name_ok = name_substring.lower() in (b.get("name","").lower())

            if category_keyword:
                cats = b.get("categories") or ""
                cat_ok = category_keyword.lower() in cats.lower()

            if city:
                city_ok = (b.get("city") or "").lower() == city.lower()

            if state:
                state_ok = (b.get("state") or "").lower() == state.lower()

            if name_ok and cat_ok and city_ok and state_ok:
                found.append(b["business_id"])
                cnt += 1
                if limit and cnt >= limit:
                    break
    return found
