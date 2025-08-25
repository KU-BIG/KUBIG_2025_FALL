"""
법제처 DRF lawService(JSON) -> 항(제n항) 단위 JSONL 변환
+ 예외 규칙(정의 조문 자동 감지): 제목에 '정의' 포함 & 항본문 없음(호만 존재) & 전체 길이>300자이면 '호' 단위로 분리

출력 필드:
- id: 항 단위 기본 '{law_id}-{조문키}-H{항번호:02d}'
       정의 조문 예외(호 단위 분리) '{law_id}-{조문키}-H{항번호:02d}-O{호번호:02d}' 또는 '{law_id}-{조문키}-H{항번호:02d}-O{호번호:02d}-{호가지번호:02d}'
- law_id, law_name
- context: '제n장 ...'
- article_label_jo: '제n조(제목)'
- article_label_hang: '제n항'
- text: 본문 문자열
"""

import json, re, os
from typing import Any, Dict, List, Optional, Tuple, Union

INPUT_PATH = "data/raw/전자금융거래법.json"
OUTPUT_PATH = "data/parsed/전자금융거래법_parsed.jsonl"

Node = Union[Dict[str, Any], List[Any], str, int, float, None]

# 원 안에 숫자 특수문자 매핑
CIRCLED_MAP = {
    "①":1,"②":2,"③":3,"④":4,"⑤":5,"⑥":6,"⑦":7,"⑧":8,"⑨":9,"⑩":10,
    "⑪":11,"⑫":12,"⑬":13,"⑭":14,"⑮":15,"⑯":16,"⑰":17,"⑱":18,"⑲":19,"⑳":20,
    "㉑":21,"㉒":22,"㉓":23,"㉔":24,"㉕":25,"㉖":26,"㉗":27,"㉘":28,"㉙":29,"㉚":30,
    "㉛":31,"㉜":32,"㉝":33,"㉞":34,"㉟":35,
    "㊱":36,"㊲":37,"㊳":38,"㊴":39,"㊵":40,"㊶":41,"㊷":42,"㊸":43,"㊹":44,"㊺":45,
    "㊻":46,"㊼":47,"㊽":48,"㊾":49,"㊿":50,
}

# 공통 유틸

def _remove_angle_tags(s: str) -> str:
    """<개정 ...>, <신설 ...>, <삭제 ...> 등 꺾쇠 표식 전체 제거"""
    return re.sub(r"<[^>]*>", "", s)

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _remove_angle_tags(s)
    lines = [ln.strip() for ln in s.split("\n")]
    s = "\n".join([ln for ln in lines if ln != ""]).strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _text_if(d: Dict, key: str) -> str:
    v = d.get(key)
    return _clean_text(v) if isinstance(v, str) else ""

def _effective_article_no(art: Dict[str, Any]) -> str:
    """조문번호 + (조문가지번호) -> '제n조' 또는 '제n조의m'"""
    n = str(art.get("조문번호", "")).strip()
    ga = str(art.get("조문가지번호", "")).strip()
    return f"제{n}조의{ga}" if ga else f"제{n}조"

def _is_heading_block(art: Dict[str, Any]) -> bool:
    return art.get("조문여부") == "전문"

def _is_article_block(art: Dict[str, Any]) -> bool:
    return art.get("조문여부") == "조문"

# 번호/표식 처리

def _strip_leading_bullet(text: str) -> str:
    """앞머리 표식 제거: ① ... / (1) ... / 1. ... / 1) ... / 1의2. ..."""
    t = _clean_text(text)
    t = re.sub(r"^\s*[①-⑳]\s*", "", t)                    # ① ...
    t = re.sub(r"^\s*\((\d+)\)\s*", "", t)                 # (1) ...
    t = re.sub(r"^\s*(\d+)\s*의\s*(\d+)\s*[\.\)]\s*", "", t)  # 1의2.
    t = re.sub(r"^\s*(\d+)\s*[\.\)]\s*", "", t)            # 1. ... / 1) ...
    return t.strip()

def _decode_hang_no(hnode: Dict[str, Any], fallback: int) -> int:
    """항번호(①->1, '1'->1); 실패 시 fallback"""
    raw = _text_if(hnode, "항번호")
    if raw in CIRCLED_MAP:
        return CIRCLED_MAP[raw]
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else fallback

# '호' 번호 파싱 (의 구조 지원)

def _parse_ho_numbers(hnode: Dict[str, Any], idx_fallback: int) -> Tuple[int, Optional[int]]:
    """
    '호번호'와 '호가지번호'를 기반으로 (no, gano) 추출.
    - 기본: 호번호에서 숫자(no), 호가지번호가 있으면 gano
    - 보조: 호내용 선두의 '숫자 의 숫자' 패턴으로 추출 (예: '1의3.')
    - 최후: (idx_fallback, None)
    """
    raw_no = _text_if(hnode, "호번호")
    raw_gano = _text_if(hnode, "호가지번호")
    m_no = re.search(r"\d+", raw_no)
    no = int(m_no.group()) if m_no else None
    gano = int(raw_gano) if raw_gano.isdigit() else None

    if no is None:
        raw = _text_if(hnode, "호내용")
        m = re.match(r"^\s*(\d+)\s*의\s*(\d+)[\.\)]?\s*", raw)
        if m:
            no, gano = int(m.group(1)), int(m.group(2))
        else:
            m2 = re.match(r"^\s*(\d+)[\.\)]", raw)
            if m2:
                no = int(m2.group(1))

    if no is None:
        no = idx_fallback
    return no, gano

def _format_ho_label(no: int, gano: Optional[int]) -> str:
    return f"제{no}의{gano}호" if gano is not None else f"제{no}호"

def _format_ho_id_suffix(no: int, gano: Optional[int]) -> str:
    """ID용 안정 포맷: O{no:02d} 또는 O{no:02d}-{gano:02d}"""
    if gano is not None:
        return f"O{no:02d}-{gano:02d}"
    return f"O{no:02d}"

def _is_deleted_ho_text(txt: str) -> bool:
    """'1. 삭제 <2008.12.31>' 등 '삭제' 호 제외"""
    t = _strip_leading_bullet(_remove_angle_tags(txt)).strip()
    return t.startswith("삭제")

def _is_deleted_hang_text(txt: str) -> bool:
    """항 본문이 '① 삭제 <날짜>' 형태인지 확인"""
    t = _clean_text(txt).strip()
    # 원형 숫자(①, ②, ③ 등) 제거 후 '삭제' 확인
    # _strip_leading_bullet 함수가 원형 숫자를 제거하는 기능을 가지고 있음
    t = _strip_leading_bullet(t).strip()
    return t == "삭제"

def _is_deleted_article_text(txt: str) -> bool:
    """조문 본문이 '제n조 삭제' 형태인지 확인"""
    t = _clean_text(txt).strip()
    # '제16조 삭제' 형태 확인
    pattern = r"^제\d+조(\의\d+)?\s+삭제$"
    return bool(re.match(pattern, t))

# 목/호 수집

def _collect_mok(node: Node) -> List[str]:
    """목 텍스트 수집 (번호 변환 없이 단순 이어붙임; 리스트/리스트의 리스트 방어)"""
    out: List[str] = []
    if node is None:
        return out
    if isinstance(node, list):
        for x in node:
            out.extend(_collect_mok(x))
    elif isinstance(node, dict):
        txt = _text_if(node, "목내용")
        if txt:
            if isinstance(node.get("목내용"), list):  # 리스트형 본문 방어
                out.extend(_collect_mok(node.get("목내용")))
            else:
                out.append(txt)
        if "목" in node:
            out.extend(_collect_mok(node["목"]))
    elif isinstance(node, str):
        out.append(_clean_text(node))
    return out

def _collect_ho_list(node: Node) -> List[str]:
    """일반 출력용: '제n(의m)호 본문' 리스트 (삭제 호 제외, 목 포함)"""
    out: List[str] = []
    if node is None:
        return out

    def handle_one_ho(h: Dict[str, Any], idx_fallback: int):
        if not isinstance(h, dict):
            return
        raw = h.get("호내용", "")
        if _is_deleted_ho_text(raw):
            return
        no, gano = _parse_ho_numbers(h, idx_fallback)
        label = _format_ho_label(no, gano)
        body = _strip_leading_bullet(raw)
        moks = _collect_mok(h.get("목"))
        if moks:
            body = " ".join([body] + moks).strip()
        out.append(f"{label} {body}")

    if isinstance(node, list):
        for i, x in enumerate(node, start=1):
            if isinstance(x, dict):
                handle_one_ho(x, i)
            elif isinstance(x, str) and not _is_deleted_ho_text(x):
                out.append(_strip_leading_bullet(x))
    elif isinstance(node, dict):
        handle_one_ho(node, 1)
    elif isinstance(node, str) and not _is_deleted_ho_text(node):
        out.append(_strip_leading_bullet(node))
    return out

def _collect_ho_pairs(node: Node) -> List[Tuple[str, str, str]]:
    """
    (ho_id_suffix, ho_label_text, ho_body_full) 쌍 목록
    - ho_id_suffix: 'O01' 또는 'O01-02'
    - ho_label_text: '제1호' 또는 '제1의2호'
    - ho_body_full: '제1호 ...' 형식의 전체 문자열
    """
    out: List[Tuple[str, str, str]] = []
    if node is None:
        return out

    def handle_one_ho(h: Dict[str, Any], idx_fallback: int):
        if not isinstance(h, dict):
            return
        raw = h.get("호내용", "")
        if _is_deleted_ho_text(raw):
            return
        no, gano = _parse_ho_numbers(h, idx_fallback)
        label = _format_ho_label(no, gano)
        idsfx = _format_ho_id_suffix(no, gano)
        body = _strip_leading_bullet(raw)
        moks = _collect_mok(h.get("목"))
        if moks:
            body = " ".join([body] + moks).strip()
        out.append((idsfx, label, f"{label} {body}"))

    if isinstance(node, list):
        for i, x in enumerate(node, start=1):
            if isinstance(x, dict):
                handle_one_ho(x, i)
    elif isinstance(node, dict):
        handle_one_ho(node, 1)
    return out

# 본문 생성 및 정의판별

def _collect_article_text_joined_by_comma(art: Dict[str, Any]) -> str:
    """조문 전체 텍스트(항 단위 연결): 각 항 = '제n항: <항내용> 제1호 ..., 제2호 ...'"""
    hang = art.get("항")
    if hang is None:
        jm = _text_if(art, "조문내용")
        jm = re.sub(r"^제\d+조(\의\d+)?\s*\([^)]*\)\s*", "", jm).strip()
        return jm

    def handle_one_hang(hnode: Dict[str, Any], idx_fallback: int) -> str:
        n = _decode_hang_no(hnode, idx_fallback)
        htxt = _strip_leading_bullet(hnode.get("항내용", ""))
        hos = _collect_ho_list(hnode.get("호"))
        if hos:
            return f"제{n}항: {htxt} " + ", ".join(hos)
        else:
            return f"제{n}항: {htxt}"

    parts: List[str] = []
    if isinstance(hang, list):
        for i, h in enumerate(hang, start=1):
            if isinstance(h, dict):
                parts.append(handle_one_hang(h, i))
    elif isinstance(hang, dict):
        parts.append(handle_one_hang(hang, 1))
    return "\n".join(parts).strip()

def _looks_like_definition_and_only_ho(art: Dict[str, Any], title: str, total_text_len: int) -> Tuple[bool, int]:
    """
    '정의' 조문 예외 판정:
      - 제목에 '정의' 포함
      - 항 본문이 없고(= 항내용 빈 상태), 호만 존재하는 구조
      - 전체 길이 > 300
    반환: (True/False, 대표 항 번호 n)  # n은 호가 달린 단일 항의 번호(없으면 1)
    """
    if "정의" not in (title or ""):
        return (False, 1)

    hang = art.get("항")
    if hang is None:
        return (False, 1)

    # 단일 항 형태이면서 항내용이 비고 '호'만 존재
    if isinstance(hang, dict):
        no_text = _text_if(hang, "항내용") == ""
        only_ho = ("호" in hang) and bool(hang["호"])
        if no_text and only_ho and total_text_len > 300:
            n = _decode_hang_no(hang, 1)
            return (True, n)
    elif isinstance(hang, list) and len(hang) == 1 and isinstance(hang[0], dict):
        h0 = hang[0]
        no_text = _text_if(h0, "항내용") == ""
        only_ho = ("호" in h0) and bool(h0["호"])
        if no_text and only_ho and total_text_len > 300:
            n = _decode_hang_no(h0, 1)
            return (True, n)

    return (False, 1)

# 메인 파싱

def parse_law_to_records(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    law_root = raw.get("법령", {})
    meta = law_root.get("기본정보", {})
    law_id = str(meta.get("법령ID", "")).strip()
    law_name = str(meta.get("법령명_한글", "")).strip() or "전자금융거래법"

    articles = law_root.get("조문", {}).get("조문단위", []) or []

    current_chapter: Optional[str] = None
    records: List[Dict[str, str]] = []

    for art in articles:
        # 장(章) 머릿글 -> context
        if _is_heading_block(art):
            head = _text_if(art, "조문내용")
            if "장" in head:
                current_chapter = head
            continue
        if not _is_article_block(art):
            continue

        jokey = str(art.get("조문키", "")).strip()
        article_no = _effective_article_no(art)
        title = _text_if(art, "조문제목")
        article_label_jo = f"{article_no}({title})" if title else article_no

        # 정의 조문 예외 판정
        text_all = _collect_article_text_joined_by_comma(art)
        split_by_ho, ho_hang_no = _looks_like_definition_and_only_ho(
            art, title, len(text_all)
        )

        hang = art.get("항")

        # 항이 없으면 조문내용으로 단일 레코드(항=1)
        if hang is None:
            jm = _text_if(art, "조문내용")
            # 삭제된 조문인지 확인
            if _is_deleted_article_text(jm):
                continue  # 삭제된 조문은 제외
            jm = re.sub(r"^제\d+조(\의\d+)?\s*\([^)]*\)\s*", "", jm).strip()
            n = 1
            records.append({
                "id": f"{law_id}-{jokey}-H{n:02d}",
                "law_id": law_id,
                "law_name": law_name,
                "context": current_chapter or "",
                "article_label_jo": article_label_jo,
                "article_label_hang": f"제{n}항",
                "text": jm
            })
            continue

        def emit_hang_record(hnode: Dict[str, Any], idx_fallback: int):
            n = _decode_hang_no(hnode, idx_fallback)
            # 일반: 항 단위, 호는 콤마로 연결
            htxt = _strip_leading_bullet(hnode.get("항내용", ""))
            # 삭제된 항인지 확인
            if _is_deleted_hang_text(htxt):
                return  # 삭제된 항은 제외
            hos = _collect_ho_list(hnode.get("호"))
            if hos:
                text = f"{htxt} " + ", ".join(hos)
            else:
                text = htxt
            records.append({
                "id": f"{law_id}-{jokey}-H{n:02d}",
                "law_id": law_id,
                "law_name": law_name,
                "context": current_chapter or "",
                "article_label_jo": article_label_jo,
                "article_label_hang": f"제{n}항",
                "text": text.strip()
            })

        def emit_ho_records_under_single_hang(hnode: Dict[str, Any], n: int):
            """정의 조문 예외: 호 단위로 각각 레코드 생성"""
            # 삭제된 항인지 먼저 확인
            htxt = _strip_leading_bullet(hnode.get("항내용", ""))
            if _is_deleted_hang_text(htxt):
                return  # 삭제된 항은 제외
                
            pairs = _collect_ho_pairs(hnode.get("호"))
            if not pairs:
                # 호가 없으면 항 단위로 fallback
                emit_hang_record(hnode, n)
                return
            for idsfx, label, ho_full in pairs:
                records.append({
                    "id": f"{law_id}-{jokey}-H{n:02d}-{idsfx}",  # e.g., ...-H01-O01-02
                    "law_id": law_id,
                    "law_name": law_name,
                    "context": current_chapter or "",
                    "article_label_jo": article_label_jo,
                    "article_label_hang": f"제{n}항",
                    "text": ho_full  # '제1(의2)호 ...'
                })

        # 분기: 정의 조문(호 단위) vs 일반(항 단위)
        if split_by_ho:
            if isinstance(hang, dict):
                emit_ho_records_under_single_hang(hang, ho_hang_no)
            elif isinstance(hang, list) and len(hang) == 1 and isinstance(hang[0], dict):
                emit_ho_records_under_single_hang(hang[0], ho_hang_no)
            else:
                # 방어: 구조가 달라지면 일반 로직
                if isinstance(hang, list):
                    for i, h in enumerate(hang, start=1):
                        if isinstance(h, dict):
                            emit_hang_record(h, i)
                elif isinstance(hang, dict):
                    emit_hang_record(hang, 1)
        else:
            if isinstance(hang, list):
                for i, h in enumerate(hang, start=1):
                    if isinstance(h, dict):
                        emit_hang_record(h, i)
            elif isinstance(hang, dict):
                emit_hang_record(hang, 1)

    return records

def save_jsonl(recs: List[Dict[str, str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

# 실행

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")
    recs = parse_law_to_records(INPUT_PATH)
    save_jsonl(recs, OUTPUT_PATH)
    print(f"단위 청크 {len(recs)}개 -> {OUTPUT_PATH}")

    # 단위 청크별 글자수 출력 
    total_chars = 0 
    for i, rec in enumerate(recs, 1): 
        text = rec.get("text", "") 
        char_count = len(text) 
        total_chars += char_count 
        print(f"청크 {i:3d}: {char_count:4d}자 | {rec.get('id', 'N/A')}") # 매 10개마다 구분선 출력 
        if i % 10 == 0: 
            print("-" * 50) 
    print(f"\n전체 청크 수: {len(recs)}개") 
    print(f"전체 글자 수: {total_chars:,}자")
