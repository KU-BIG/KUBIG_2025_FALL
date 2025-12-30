import os
from typing import TypedDict, List
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Import from tools
from tools.notion_tool import save_job_to_notion
from tools.google_calendar_tool import save_job_to_calendar

load_dotenv()

NOTION_JOB_DB_ID = os.getenv("NOTION_JOB_DB_ID")


# ============= State 정의 =============
class PostManagerState(TypedDict):
    """공고 관리 에이전트 State"""
    jobs_df: pd.DataFrame
    user_preferences: dict
    messages: list
    selected_jobs: list
    save_destination: str
    saved_jobs: list
    stage: str
    want_to_save: bool

# ============= 노드 함수들 =============

def display_jobs_node(state: PostManagerState) -> PostManagerState:
    """추천된 공고 목록을 사용자에게 보여주는 노드"""
    
    df = state['jobs_df']
    
    job_list_text = "\n" + "="*70 + "\n"
    job_list_text += "📋 추천된 채용 공고 목록\n"
    job_list_text += "="*70 + "\n"
    
    if df is None or df.empty:
        state['messages'].append(AIMessage(content="추천된 공고가 없습니다."))
        state['stage'] = 'skip'
        return state

    for idx, row in df.head(20).iterrows():
        title = row.get('title', 'N/A')
        company = row.get('company', 'N/A')
        deadline = row.get('deadline', 'N/A')
        link = row.get('link', 'N/A')
        
        job_list_text += f"\n[{idx}] {company} - {title}\n"
        job_list_text += f"    마감: {deadline}\n"
        job_list_text += f"    링크: {link}\n"
    
    job_list_text += "\n" + "="*70
    
    state['messages'].append(AIMessage(content=job_list_text))
    state['stage'] = 'ask_save'
    
    return state


def ask_save_node(state: PostManagerState) -> PostManagerState:
    """저장 여부 확인"""
    
    prompt = """
이 공고들을 Notion 또는 Google Calendar에 저장하시겠어요?

💡 입력:
- 'y' 또는 'yes' - 저장하기
- 'n' 또는 'no' - 저장하지 않기
"""
    
    state['messages'].append(AIMessage(content=prompt))
    state['stage'] = 'waiting_save_decision'
    
    return state


def parse_save_decision_node(state: PostManagerState) -> PostManagerState:
    """저장 여부 입력 파싱"""
    
    user_message = state['messages'][-1].content.strip().lower()
    
    if user_message in ['y', 'yes', '예', '네']:
        state['want_to_save'] = True
        state['stage'] = 'select'
        state['messages'].append(
            AIMessage(content="\n✅ 좋아요! 저장할 공고를 선택해주세요!")
        )
        
    elif user_message in ['n', 'no', '아니오', '아니요']:
        state['want_to_save'] = False
        state['stage'] = 'skip'
        state['messages'].append(
            AIMessage(content="\n✅ 알겠습니다. 공고 관리를 종료합니다.")
        )
        
    else:
        state['messages'].append(
            AIMessage(content="⚠️ 'y' 또는 'n'을 입력해주세요.")
        )
        state['stage'] = 'reask_save'
    
    return state


def select_jobs_node(state: PostManagerState) -> PostManagerState:
    """사용자가 저장할 공고를 선택하도록 안내하는 노드"""
    
    df = state['jobs_df']
    max_index = len(df) - 1
    
    prompt = f"""
💡 입력 방법:
- 여러 개 선택: 쉼표(,)로 구분 (예: 0, 1, 2)
- 범위 선택: 하이픈(-)으로 구분 (예: 0-5)
- 전체 선택: 'all'
- 마감 임박 순 자동 선택: 'auto'

📌 선택 가능 범위: 0 ~ {max_index}
"""
    
    state['messages'].append(AIMessage(content=prompt))
    state['stage'] = 'waiting_selection'
    
    return state


def parse_selection_node(state: PostManagerState) -> PostManagerState:
    """사용자 입력을 파싱하여 선택된 공고 인덱스 추출"""
    
    user_message = state['messages'][-1].content.strip()
    df = state['jobs_df']
    max_index = len(df) - 1
    
    selected_indices = []
    
    try:
        if user_message.lower() == 'all':
            selected_indices = list(range(len(df)))
            
        elif user_message.lower() == 'auto':
            selected_indices = auto_select_jobs(df)
            
        else:
            for part in user_message.split(','):
                part = part.strip()
                
                if '-' in part:
                    start, end = part.split('-')
                    start, end = int(start.strip()), int(end.strip())
                    selected_indices.extend(range(start, end + 1))
                else:
                    selected_indices.append(int(part))
        
        selected_indices = sorted(set([
            idx for idx in selected_indices 
            if 0 <= idx <= max_index
        ]))
        
        if not selected_indices:
            state['messages'].append(
                AIMessage(content="⚠️ 유효한 공고가 선택되지 않았습니다. 다시 입력해주세요.")
            )
            state['stage'] = 'reselect'
        else:
            state['selected_jobs'] = selected_indices
            
            confirm_msg = f"\n✅ 선택된 공고 ({len(selected_indices)}개):\n"
            confirm_msg += "\n".join([
                f"  [{idx}] {df.iloc[idx]['company']}" 
                for idx in selected_indices
            ])
            
            state['messages'].append(AIMessage(content=confirm_msg))
            state['stage'] = 'destination'
            
    except ValueError:
        state['messages'].append(
            AIMessage(content="⚠️ 입력 형식이 올바르지 않습니다. 숫자와 쉼표(,)로 입력해주세요.")
        )
        state['stage'] = 'reselect'
    
    return state


def auto_select_jobs(df: pd.DataFrame, top_n: int = 5) -> list:
    """마감일 기준 자동 선택 (마감 임박 순)"""
    
    today = datetime.now()
    deadlines = []
    
    for idx, row in df.iterrows():
        try:
            deadline_str = str(row.get('deadline', ''))
            if len(deadline_str) == 10 and deadline_str.count('-') == 2:
                deadline_date = datetime.strptime(deadline_str, '%Y-%m-%d')
                days_left = (deadline_date - today).days
                
                if days_left >= 0:
                    deadlines.append((idx, days_left))
        except:
            continue
    
    deadlines.sort(key=lambda x: x[1])
    return [idx for idx, _ in deadlines[:top_n]]


def choose_destination_node(state: PostManagerState) -> PostManagerState:
    """저장 위치를 선택하도록 안내하는 노드"""
    
    prompt = """
어디에 저장하시겠어요?

1️⃣  Notion - 체계적으로 정리하고 메모 작성
2️⃣  Google Calendar - 마감일 알림 받기

💡 입력: '1' 또는 'notion' / '2' 또는 'calendar'
"""
    
    state['messages'].append(AIMessage(content=prompt))
    state['stage'] = 'waiting_destination'
    
    return state


def parse_destination_node(state: PostManagerState) -> PostManagerState:
    """저장 위치 입력 파싱"""
    
    user_message = state['messages'][-1].content.strip().lower()
    
    if user_message in ['1', 'notion']:
        state['save_destination'] = 'notion'
        state['stage'] = 'save'
        state['messages'].append(
            AIMessage(content="✅ Notion에 저장하겠습니다.")
        )
        
    elif user_message in ['2', 'calendar']:
        state['save_destination'] = 'calendar'
        state['stage'] = 'save'
        state['messages'].append(
            AIMessage(content="✅ Google Calendar에 저장하겠습니다.")
        )
        
    else:
        state['messages'].append(
            AIMessage(content="⚠️ '1' 또는 '2'를 입력해주세요.")
        )
        state['stage'] = 'reask_destination'
    
    return state


def save_jobs_node(state: PostManagerState) -> PostManagerState:
    """선택된 공고를 Notion 또는 Calendar에 저장하는 노드"""
    
    df = state['jobs_df']
    selected_indices = state['selected_jobs']
    destination = state['save_destination']
    
    saved_jobs = []
    success_count = 0
    
    state['messages'].append(
        AIMessage(content=f"\n⏳ {len(selected_indices)}개 공고 저장 중...\n")
    )
    
    for idx in selected_indices:
        row = df.iloc[idx]
        
        try:
            if destination == 'notion':
                result = save_job_to_notion.invoke({
                    'title': row['title'],
                    'company': row['company'],
                    'deadline': str(row.get('deadline', '')),
                    'career': str(row.get('career', '경력무관')),
                    'url': row['link'],
                    'database_id': NOTION_JOB_DB_ID,
                    'state': '지원 전'
                })
                
            else:
                result = save_job_to_calendar.invoke({
                    'title': row['title'],
                    'company': row['company'],
                    'location': str(row.get('location', '정보없음')),
                    'deadline': str(row.get('deadline', '')),
                    'link': row.get('link'),
                    'keyword': str(row.get('keyword', ''))
                })
            
            state['messages'].append(AIMessage(content=f"  {result}"))
            
            if '✅' in result:
                success_count += 1
                saved_jobs.append({
                    'index': idx,
                    'company': row['company'],
                    'title': row['title']
                })
                
        except Exception as e:
            error_msg = f"  ❌ [{idx}] {row['company']} 저장 실패: {str(e)}"
            state['messages'].append(AIMessage(content=error_msg))
    
    state['saved_jobs'] = saved_jobs
    
    summary = f"\n{'='*70}\n"
    summary += f"🎉 저장 완료: {success_count}/{len(selected_indices)}건\n"
    summary += f"{'='*70}\n"
    
    state['messages'].append(AIMessage(content=summary))
    state['stage'] = 'done'
    
    return state


# ============= 그래프 구성 =============

def create_posting_manager_agent():
    """공고 관리 에이전트 그래프 생성"""
    
    workflow = StateGraph(PostManagerState)
    
    # 노드 추가
    workflow.add_node("display_jobs", display_jobs_node)
    workflow.add_node("ask_save", ask_save_node)
    workflow.add_node("parse_save_decision", parse_save_decision_node)
    workflow.add_node("select_jobs", select_jobs_node)
    workflow.add_node("parse_selection", parse_selection_node)
    workflow.add_node("choose_destination", choose_destination_node)
    workflow.add_node("parse_destination", parse_destination_node)
    workflow.add_node("save_jobs", save_jobs_node)
    
    # 시작점
    workflow.set_entry_point("display_jobs")
    
    # 엣지 추가
    workflow.add_edge("display_jobs", "ask_save")
    workflow.add_edge("ask_save", END)  # 사용자 입력 대기
    
    workflow.add_conditional_edges(
        "parse_save_decision",
        lambda state: state['stage'],
        {
            'reask_save': "ask_save",
            'select': "select_jobs",
            'skip': END
        }
    )
    
    workflow.add_edge("select_jobs", END)
    
    workflow.add_conditional_edges(
        "parse_selection",
        lambda state: state['stage'],
        {
            'reselect': "select_jobs",
            'destination': "choose_destination"
        }
    )
    
    workflow.add_edge("choose_destination", END)
    
    workflow.add_conditional_edges(
        "parse_destination",
        lambda state: state['stage'],
        {
            'reask_destination': "choose_destination",
            'save': "save_jobs"
        }
    )
    
    workflow.add_edge("save_jobs", END)
    
    return workflow.compile()


# ============= 실행 클래스 =============

class PostManagerAgent:
    """공고 관리 에이전트 실행 클래스"""
    
    def __init__(self):
        self.agent = create_posting_manager_agent()
        self.state = None
        self.last_printed_index = 0
    
    def start(self, jobs_df: pd.DataFrame, user_preferences: dict = None):
        """에이전트 시작"""
        
        self.state = {
            'jobs_df': jobs_df,
            'user_preferences': user_preferences or {},
            'messages': [],
            'selected_jobs': [],
            'save_destination': '',
            'saved_jobs': [],
            'stage': 'display',
            'want_to_save': False
        }
        
        self.last_printed_index = 0
        
        print("\n" + "="*70)
        print("🎯 채용 공고 관리 에이전트")
        print("="*70)
        
        # 첫 실행
        self.state = self.agent.invoke(self.state)
        self._print_new_messages()
        
    def chat(self, user_input: str):
        """사용자 입력 처리"""
        
        # 사용자 메시지 추가
        self.state['messages'].append(HumanMessage(content=user_input))
        
        # 현재 stage에 따라 적절한 노드 실행
        stage = self.state['stage']
        
        if stage == 'waiting_save_decision':
            self.state['stage'] = 'parse_save_decision'
            self._run_from_node('parse_save_decision')
            
            if self.state['stage'] == 'skip':
                self.state['stage'] = 'done'
            elif self.state['stage'] == 'select':
                self._run_from_node('select_jobs')
            
        elif stage == 'waiting_selection':
            self.state['stage'] = 'parse_selection'
            self._run_from_node('parse_selection')
            
            if self.state['stage'] == 'destination':
                self._run_from_node('choose_destination')
                
        elif stage == 'waiting_destination':
            self.state['stage'] = 'parse_destination'
            self._run_from_node('parse_destination')
            
            if self.state['stage'] == 'save':
                self._run_from_node('save_jobs')
        
        self._print_new_messages()
        
        return self.state['stage'] == 'done'
    
    def _run_from_node(self, node_name: str):
        """특정 노드부터 그래프 실행"""
        
        node_functions = {
            'parse_save_decision': parse_save_decision_node,
            'parse_selection': parse_selection_node,
            'choose_destination': choose_destination_node,
            'parse_destination': parse_destination_node,
            'save_jobs': save_jobs_node,
            'select_jobs': select_jobs_node,
            'ask_save': ask_save_node
        }
        
        if node_name in node_functions:
            self.state = node_functions[node_name](self.state)
        
        return self.state
    
    def _print_new_messages(self):
        """아직 출력하지 않은 AI 메시지들을 모두 출력"""
        messages = self.state.get('messages', [])
        
        for i in range(self.last_printed_index, len(messages)):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                print(msg.content)
        
        self.last_printed_index = len(messages)
    
    def get_saved_jobs(self):
        """저장된 공고 정보 반환"""
        return self.state.get('saved_jobs', [])
