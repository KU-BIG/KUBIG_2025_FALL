import os
from typing import Optional
from langchain_core.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']


class GoogleCalendarClient:
    """Google Calendar API 클라이언트 (싱글톤)"""
    
    _instance = None
    _service = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._service is None:
            self._initialize_service()
    
    def _initialize_service(self):
        """Google Calendar API 서비스 초기화"""
        creds = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        token_path = os.path.join(base_dir, 'token.json')
        creds_path = os.path.join(base_dir, 'credentials.json')

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(creds_path):
                    raise FileNotFoundError(f"❌ 인증 파일이 없습니다: {creds_path}")
                    
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        self._service = build('calendar', 'v3', credentials=creds)
    
    def get_service(self):
        return self._service


@tool
def save_job_to_calendar(
    title: str,
    company: str,
    location: str,
    deadline: str,
    link: Optional[str] = None,
    keyword: Optional[str] = None
) -> str:
    """
    채용 공고를 Google Calendar에 종일 일정으로 저장합니다.
    
    Args:
        title: 공고 제목
        company: 회사 이름
        location: 근무 지역
        deadline: 마감일 (YYYY-MM-DD 형식)
        link: 채용 공고 링크 (선택)
        keyword: 직무 키워드 (선택)
        
    Returns:
        저장 성공/실패 메시지
    """
    
    try:
        # 날짜 검증
        if not deadline or len(deadline) != 10 or deadline.count('-') != 2:
            return f"❌ 날짜 형식 오류: {deadline} (YYYY-MM-DD 형식이어야 함)"
        
        # 설명 구성
        description_parts = [f"회사: {company}", f"위치: {location}"]
        
        if keyword:
            description_parts.append(f"직무: {keyword}")
        if link:
            description_parts.append(f"\n[지원하기 링크]\n{link}")
        
        description = "\n".join(description_parts)
        
        # 이벤트 구성
        event = {
            'summary': f"[채용지원] {title}",
            'location': location,
            'description': description,
            'start': {'date': deadline},
            'end': {'date': deadline},
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 9 * 60},
                ],
            },
        }
        
        # Google Calendar에 등록
        calendar_client = GoogleCalendarClient()
        service = calendar_client.get_service()
        event_result = service.events().insert(calendarId='primary', body=event).execute()
        
        return f"✅ Google Calendar 저장 완료: [{company}] {title}"

    except Exception as e:
        error_msg = f"❌ Calendar 저장 실패: {str(e)}"
        return error_msg