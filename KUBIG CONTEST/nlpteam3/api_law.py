import os
import json
import sys
from typing import Any, Dict, List, Optional
import requests

LAW_NAME = "정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시"
OC = os.getenv("LAW_OC", "junseok0913")

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"

def _collect_law_items(obj: Any, bag: List[Dict]):
    """lawSearch JSON 구조가 케이스마다 다를 수 있어 안전하게 law 목록을 모읍니다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "law":
                if isinstance(v, list):
                    bag.extend(v)
                elif isinstance(v, dict):
                    bag.append(v)
            else:
                _collect_law_items(v, bag)
    elif isinstance(obj, list):
        for x in obj:
            _collect_law_items(x, bag)

def _gv(d: Dict, key: str, default: str = "") -> str:
    """키가 약간 달라져도 최대한 값을 얻기 위한 헬퍼 (예: 언더스코어/없음)."""
    return d.get(key) or d.get(key.replace("_", "")) or default

def search_law_id(oc: str, law_name: str) -> str:
    """법령명으로 검색해 '법령ID'를 반환."""
    params = {
        "OC": oc,
        "target": "law",
        "type": "JSON",
        "query": law_name,
        "display": 100,  # 여유 있게 받아 필터링
    }
    r = requests.get(SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    # 혹시 인코딩 문제를 방지
    r.encoding = "utf-8"
    data = r.json()

    items: List[Dict] = []
    _collect_law_items(data, items)
    if not items:
        raise RuntimeError("검색 결과가 없습니다. OC 승인/허용 IP/도메인 등을 확인하세요.")

    # 1순위: 법령명 정확 일치 + 법령구분명 == '법률'
    exact = [it for it in items if _gv(it, "법령명한글") == law_name]
    cand: Optional[Dict] = next((it for it in exact if _gv(it, "법령구분명") == "법률"), None)
    if not cand and exact:
        cand = exact[0]

    # 2순위: 부분 일치
    if not cand:
        for it in items:
            if law_name in _gv(it, "법령명한글"):
                cand = it
                break

    if not cand:
        raise RuntimeError("원하는 법령을 찾지 못했습니다.")

    law_id = _gv(cand, "법령ID")
    if not law_id:
        raise RuntimeError("검색 결과에서 '법령ID'를 찾지 못했습니다.")
    return law_id

def fetch_law_json(oc: str, law_id: str) -> Dict:
    """법령ID로 현행 법령 본문(JSON, 한글)을 조회."""
    params = {
        "OC": oc,
        "target": "law",
        "type": "JSON",
        "ID": law_id,
        "LANG": "KO",
    }
    r = requests.get(SERVICE_URL, params=params, timeout=30, headers={"Accept": "application/json"})
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.json()

if __name__ == "__main__":
    if not OC or OC == "":
        print("환경변수 LAW_OC 또는 코드의 OC 변수에 본인의 OC(이메일 ID 부분)를 설정하세요.", file=sys.stderr)
        sys.exit(1)

    try:
        law_id = search_law_id(OC, LAW_NAME)
        law_json = fetch_law_json(OC, law_id)

        # 보기 좋게 출력 및 파일 저장
        pretty = json.dumps(law_json, ensure_ascii=False, indent=2)
        print(pretty)

        filename = f"data/raw/{LAW_NAME}.json"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pretty)
        # 저장 위치 알림
        print(f"\n저장 완료: {filename}")
    except requests.HTTPError as e:
        print(f"HTTP 오류: {e} / 응답: {getattr(e.response, 'text', '')[:500]}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"실패: {e}", file=sys.stderr)
        sys.exit(3)
