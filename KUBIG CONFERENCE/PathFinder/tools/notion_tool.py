import os
from typing import Literal
from langchain_core.tools import tool
from notion_client import Client

def get_notion_client():
    """Notion 클라이언트를 반환합니다."""
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    if not NOTION_API_KEY:
        raise ValueError("NOTION_API_KEY 환경 변수가 설정되지 않았습니다.")
    return Client(auth=NOTION_API_KEY)


@tool
def save_job_to_notion(
    title: str, 
    company: str, 
    deadline: str, 
    career: str,
    url: str,
    database_id: str,
    state: Literal["지원 전", "지원 중", "지원 완료"] = "지원 전"
) -> str:
    """
    채용 공고를 Notion 데이터베이스에 저장합니다.
    
    Args:
        title: 공고 제목
        company: 회사 이름
        deadline: 지원 마감일 (YYYY-MM-DD 형식). 날짜를 모르면 빈 문자열("")
        career: 요구 경력 사항 (예: 경력무관, 3년 이상 등)
        url: 채용 공고 링크 URL
        database_id: Notion 데이터베이스 ID
        state: 지원 상태 (기본값: "지원 전")
        
    Returns:
        저장 성공/실패 메시지
    """
    
    try:
        notion = get_notion_client()
        
        # 날짜 포맷 검증 (YYYY-MM-DD 형식만 허용)
        date_property = None
        if deadline and len(deadline) == 10 and deadline.count('-') == 2:
            date_property = {"start": deadline}

        properties_payload = {
            "공고명": {"title": [{"text": {"content": title}}]},
            "회사명": {"rich_text": [{"text": {"content": company}}]},
            "경력": {"rich_text": [{"text": {"content": career}}]},  
            "링크": {"url": url},
            "지원상태": {"select": {"name": state}}
        }
        
        # 날짜가 유효할 때만 포함
        if date_property:
            properties_payload["마감일"] = {"date": date_property}

        # Notion 페이지 생성
        notion.pages.create(
            parent={"database_id": database_id},
            properties=properties_payload
        )
        
        return f"✅ Notion 저장 완료: [{company}] {title}"

    except Exception as e:
        error_msg = f"❌ Notion 저장 실패: {str(e)}"
        return error_msg