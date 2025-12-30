import chainlit as cl
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import os

from tools.notion_tool import save_job_to_notion
from tools.google_calendar_tool import save_job_to_calendar

NOTION_JOB_DB_ID = os.getenv("NOTION_JOB_DB_ID")

class PostingManagerUI:
    """Post Manager Agent의 Chainlit UI 래퍼 (키보드 입력 방식)"""
    
    def __init__(self):
        self.jobs_df = None
        self.selected_jobs = []
        self.save_destination = None
        self.saved_jobs = []
        self.stage = "display"
    
    async def start(self, jobs_df: pd.DataFrame):
        """Post Manager 시작"""
        self.jobs_df = jobs_df.reset_index(drop=True)
        self.stage = "display"
        
        # 1. 공고 목록 출력
        await self._display_jobs()
        
        # 2. 저장 여부 질문 (키보드 입력)
        self.stage = "ask_save"
        await self._ask_save()
    
    async def process_message(self, user_input: str) -> bool:
        """
        사용자 메시지 처리
        
        Returns:
            is_complete: 완료 여부
        """
        
        # ====================================================================
        # Stage 1: 저장 여부 확인
        # ====================================================================
        if self.stage == "ask_save":
            choice = user_input.strip().lower()
            
            if choice in ['y', 'yes', '예', '네', 'ㅇ']:
                await cl.Message(
                    content="✅ 좋아요! 저장할 공고의 **순위 번호**를 입력해주세요.",
                    author="Post Manager"
                ).send()
                
                self.stage = "select"
                await self._ask_selection()
                return False
                
            elif choice in ['n', 'no', '아니오', '아니요', 'ㄴ']:
                await cl.Message(
                    content="✅ 알겠습니다. 공고 관리를 종료합니다.",
                    author="Post Manager"
                ).send()
                return True
                
            else:
                await cl.Message(
                    content="⚠️ 'y' 또는 'n'을 입력해주세요.",
                    author="Post Manager"
                ).send()
                return False
        
        # ====================================================================
        # Stage 2: 공고 선택
        # ====================================================================
        elif self.stage == "select":
            selected_indices = self._parse_selection(user_input)
            
            if not selected_indices:
                await cl.Message(
                    content="⚠️ 유효한 순위가 선택되지 않았습니다. 다시 입력해주세요.",
                    author="Post Manager"
                ).send()
                return False
            
            self.selected_jobs = selected_indices
            
            # 선택 확인 메시지
            confirm_text = f"✅ **선택된 공고 ({len(selected_indices)}개)**:\n\n"
            for idx in selected_indices[:5]:
                row = self.jobs_df.iloc[idx]
                rank = idx + 1
                confirm_text += f"• [{rank}위] {row['title']} ({row['company']})\n"
            
            if len(selected_indices) > 5:
                confirm_text += f"\n... 외 {len(selected_indices) - 5}개"
            
            await cl.Message(content=confirm_text, author="Post Manager").send()
            
            # 다음 단계(저장 위치 선택)로 이동
            self.stage = "destination"
            await self._ask_destination()
            return False
        
        # ====================================================================
        # Stage 3: 저장 위치 선택
        # ====================================================================
        elif self.stage == "destination":
            choice = user_input.strip().lower()
            
            if choice in ['1', 'notion', '노션']:
                self.save_destination = 'notion'
                await cl.Message(
                    content="✅ Notion에 저장하겠습니다.",
                    author="Post Manager"
                ).send()
                
                await self._save_jobs()
                return True
                
            elif choice in ['2', 'calendar', '캘린더', '구글', '구글캘린더']:
                self.save_destination = 'calendar'
                await cl.Message(
                    content="✅ Google Calendar에 저장하겠습니다.",
                    author="Post Manager"
                ).send()
                
                await self._save_jobs()
                return True
                
            else:
                await cl.Message(
                    content="⚠️ '1' 또는 '2'를 입력해주세요.",
                    author="Post Manager"
                ).send()
                return False
        
        return False

    # ========================================================================
    # UI 표시 메서드
    # ========================================================================
    
    async def _display_jobs(self):
        """공고 목록 표시"""
        message_content = f"📋 **추천된 채용 공고 목록** (총 {len(self.jobs_df)}개)\n\n"
        message_content += "="*50 + "\n\n"
        
        for rank, (idx, row) in enumerate(self.jobs_df.head(20).iterrows(), 1):
            job_card = f"""### 🏅 {rank}위. {row.get('title', '제목 없음')}
- **회사명**: {row.get('company', '회사명 없음')}
- **마감일**: {row.get('deadline', '정보 없음')}
- **링크**: [공고 보러가기]({row.get('link', '#')})

"""
            message_content += job_card
            
        if len(self.jobs_df) > 20:
             message_content += f"\n...(하위 {len(self.jobs_df) - 20}개 공고 생략)..."

        await cl.Message(content=message_content, author="Post Manager").send()
    
    async def _ask_save(self):
        """저장 여부 확인 (키보드 입력)"""
        await cl.Message(
            content="""💾 **이 공고들을 저장하시겠어요?**

- **y** 또는 **yes** - 저장하기
- **n** 또는 **no** - 저장하지 않기

선택해주세요:""",
            author="Post Manager"
        ).send()
    
    async def _ask_selection(self):
        """공고 선택 요청"""
        max_rank = len(self.jobs_df)
        
        await cl.Message(
            content=f"""📝 **저장할 공고의 순위 번호를 입력해주세요**

**입력 방법** (순위 번호 사용):
- 여러 개: `1, 2, 3`
- 범위: `1-5` (1위부터 5위까지)
- 전체: `all`
- 자동: `auto` (마감 임박 순)

📌 선택 가능 범위: **1 ~ {max_rank}**""",
            author="Post Manager"
        ).send()
    
    async def _ask_destination(self):
        """저장 위치 선택 (키보드 입력)"""
        await cl.Message(
            content="""💾 **어디에 저장하시겠어요?**

**1️⃣ Notion** - 체계적으로 정리하고 메모 작성
**2️⃣ Google Calendar** - 마감일 알림 받기

**선택: 1 또는 2**""",
            author="Post Manager"
        ).send()
    
    # ========================================================================
    # 파싱 및 저장 메서드
    # ========================================================================
    
    def _parse_selection(self, user_input: str) -> List[int]:
        """선택 입력 파싱 (순위 → 인덱스 변환)"""
        max_idx = len(self.jobs_df) - 1
        selected_indices = []
        
        try:
            if user_input.lower() == 'all':
                return list(range(len(self.jobs_df)))
            
            elif user_input.lower() == 'auto':
                return self._auto_select_jobs()
            
            else:
                for part in user_input.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    
                    if '-' in part:
                        # 범위 입력 (예: 1-5 → 인덱스 0-4)
                        start_rank, end_rank = map(int, part.split('-'))
                        selected_indices.extend(range(start_rank - 1, end_rank))
                    else:
                        # 단일 입력 (예: 1 → 인덱스 0)
                        selected_indices.append(int(part) - 1)
                
                # 유효 범위 필터링
                return sorted(set([idx for idx in selected_indices if 0 <= idx <= max_idx]))
                
        except ValueError:
            return []

    def _auto_select_jobs(self, top_n: int = 5) -> List[int]:
        """마감 임박 순 자동 선택"""
        today = datetime.now()
        deadlines = []
        
        for idx, row in self.jobs_df.iterrows():
            try:
                deadline_str = str(row.get('deadline', ''))
                if len(deadline_str) == 10:
                    deadline_date = datetime.strptime(deadline_str, '%Y-%m-%d')
                    days_left = (deadline_date - today).days
                    
                    if days_left >= 0:
                        deadlines.append((idx, days_left))
            except:
                continue
        
        deadlines.sort(key=lambda x: x[1])
        return [idx for idx, _ in deadlines[:top_n]]

    async def _save_jobs(self):
        """공고 저장 실행"""
        await cl.Message(
            content=f"⏳ **{len(self.selected_jobs)}개 공고 저장 중...**",
            author="Post Manager"
        ).send()
        
        success_count = 0
        results = []
        
        async with cl.Step(name="Saving Jobs", type="tool") as save_step:
            for idx in self.selected_jobs:
                row = self.jobs_df.iloc[idx]
                
                try:
                    if self.save_destination == 'notion':
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
                    
                    if '✅' in result:
                        success_count += 1
                        self.saved_jobs.append({
                            'index': idx,
                            'company': row['company'],
                            'title': row['title']
                        })
                        results.append(f"✅ {row['company']} - 저장 성공")
                    else:
                        results.append(f"❌ {row['company']} - {result}")
                        
                except Exception as e:
                    results.append(f"❌ {row['company']} - 에러: {str(e)}")
            
            save_step.output = f"성공: {success_count}/{len(self.selected_jobs)}"
        
        # 저장 결과 표시
        summary_msg = "\n".join(results)
        await cl.Message(
            content=f"""**저장 결과**

{summary_msg}

🎉 총 **{success_count}/{len(self.selected_jobs)}건** 저장 완료!""",
            author="Post Manager"
        ).send()

    def get_saved_jobs(self) -> List[Dict]:
        """저장된 공고 목록 반환"""
        return self.saved_jobs