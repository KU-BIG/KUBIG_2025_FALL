import chainlit as cl
import asyncio
from typing import Dict, Any
from datetime import datetime

from core.state import MainState
from ui.interview_ui import InterviewUI
from ui.ncs_job_recommender_ui import NCSJobRecommenderUI
from ui.posting_recommender_ui import PostingRecommenderUI
from ui.posting_manager_ui import PostingManagerUI

# ============================================================================
# Session State 관리
# ============================================================================

def create_initial_state() -> MainState:
    return {
        "messages": [],
        "user_profile": {},
        "search_config": {},
        "job_category_codes": [],
        "recommendations": [],
        "final_postings": None,
        "selected_jobs": [],
        "saved_jobs": []
    }

# ============================================================================
# Chainlit Lifecycle Hooks
# ============================================================================

@cl.on_chat_start
async def start():
    """챗봇 시작 시 초기화"""
    await cl.Message(
        content="""## 🧭 PathFinder에 오신 것을 환영합니다!

AI 기반 커리어 가이던스 시스템으로, 다음 **4단계 프로세스**를 통해 최적의 채용 공고를 찾아드립니다:

1. 📋 **Interview**: 당신의 역량과 관심사를 파악합니다
2. 💼 **NCS Job Recommender**: NCS 기반 직무를 추천합니다  
3. 📊 **Posting Recommender**: 실제 채용 공고를 추천합니다
4. 💾 **Posting Manager**: 마음에 드는 공고를 Notion/Google Calendar에 저장합니다

---
**준비되셨나요? 시작하려면 아무 메시지나 입력해주세요!**
""",
        author="PathFinder"
    ).send()
    
    cl.user_session.set("main_state", create_initial_state())
    cl.user_session.set("current_stage", "waiting_start")
    
    # UI 인스턴스 초기화
    cl.user_session.set("interview_ui", InterviewUI())
    cl.user_session.set("ncs_job_recommender_ui", NCSJobRecommenderUI())
    cl.user_session.set("posting_recommender_ui", PostingRecommenderUI())
    cl.user_session.set("posting_manager_ui", PostingManagerUI())

@cl.on_message
async def main(message: cl.Message):
    """메시지 처리 메인 핸들러"""
    current_stage = cl.user_session.get("current_stage")
    main_state = cl.user_session.get("main_state")
    
    # ========================================================================
    # Stage 0: 시작 대기
    # ========================================================================
    if current_stage == "waiting_start":
        await cl.Message(
            content="✨ PathFinder를 시작합니다!\n\n먼저 간단한 인터뷰를 통해 당신을 알아가겠습니다.",
            author="PathFinder"
        ).send()
        
        cl.user_session.set("current_stage", "interview")
        interview_ui = cl.user_session.get("interview_ui")
        await interview_ui.start()
        return
    
    # ========================================================================
    # Stage 1: Interview Agent
    # ========================================================================
    elif current_stage == "interview":
        interview_ui = cl.user_session.get("interview_ui")
        is_complete, user_profile = await interview_ui.process_message(message.content)
        
        if is_complete:
            main_state["user_profile"] = user_profile
            cl.user_session.set("main_state", main_state)
            
            await cl.Message(
                content="✅ **인터뷰가 완료되었습니다!**\n\n이제 당신에게 맞는 직무를 추천해드릴게요.",
                author="PathFinder"
            ).send()
            
            cl.user_session.set("current_stage", "mmr_config")
            await ask_mmr_preference()
        return
    
    # ========================================================================
    # Stage 2: MMR 설정 및 자동 파이프라인 시작
    # ========================================================================
    elif current_stage == "mmr_config":
        choice = message.content.strip()
        
        if choice == '1':
            lambda_mult = 0.3
        elif choice == '2':
            lambda_mult = 0.5
        elif choice == '3':
            lambda_mult = 0.7
        else:
            await cl.Message(
                content="⚠️ 1, 2, 3 중에서 선택해주세요.",
                author="PathFinder"
            ).send()
            return
        
        main_state["search_config"] = {"use_mmr": True, "lambda_mult": lambda_mult}
        cl.user_session.set("main_state", main_state)
        
        await cl.Message(
            content=f"✅ 검색 설정 완료. 자동 매칭을 시작합니다...",
            author="PathFinder"
        ).send()
        
        await run_auto_pipeline()
        return
    
    # ========================================================================
    # Stage 3: Post Manager (키보드 입력 처리)
    # ========================================================================
    elif current_stage == "post_manager":
        posting_manager_ui = cl.user_session.get("posting_manager_ui")
        is_complete = await posting_manager_ui.process_message(message.content)
        
        if is_complete:
            await finish_post_manager()
        return
    
    # ========================================================================
    # Stage 4: 완료
    # ========================================================================
    elif current_stage == "done":
        if message.content.strip() == "/restart":
            await restart_session()
        else:
            await cl.Message(
                content="모든 과정이 완료되었습니다.\n다시 시작하려면 `/restart`를 입력해주세요.",
                author="PathFinder"
            ).send()
        return

# ============================================================================
# Core Logic: 자동 실행 파이프라인
# ============================================================================

async def run_auto_pipeline():
    """NCS -> Posting -> PostManager 순차 실행"""
    main_state = cl.user_session.get("main_state")
    ncs_job_recommender_ui = cl.user_session.get("ncs_job_recommender_ui")
    posting_recommender_ui = cl.user_session.get("posting_recommender_ui")
    posting_manager_ui = cl.user_session.get("posting_manager_ui")

    # 1. NCS Agent 실행
    await ncs_job_recommender_ui.run(main_state)
    
    # 최신 상태 다시 가져오기
    main_state = cl.user_session.get("main_state")
    
    # 2. Posting Agent 실행
    cl.user_session.set("current_stage", "posting")
    await posting_recommender_ui.run(main_state)
    
    # 3. 최신 상태 다시 가져오기
    main_state = cl.user_session.get("main_state")
    
    # 4. Post Manager 실행
    if main_state["final_postings"] is None or main_state["final_postings"].empty:
        await cl.Message(
            content="😔 아쉽게도 조건에 맞는 채용 공고를 찾지 못했습니다.\n\n`/restart`로 다시 시도해보세요.",
            author="PathFinder"
        ).send()
        cl.user_session.set("current_stage", "done")
        return

    # Post Manager 단계로 전환
    cl.user_session.set("current_stage", "post_manager")
    await posting_manager_ui.start(main_state["final_postings"])

# ============================================================================
# Helper Functions
# ============================================================================

async def finish_post_manager():
    """종료 처리"""
    posting_manager_ui = cl.user_session.get("posting_manager_ui")
    main_state = cl.user_session.get("main_state")
    
    saved_jobs = posting_manager_ui.get_saved_jobs()
    main_state["saved_jobs"] = saved_jobs
    cl.user_session.set("main_state", main_state)
    
    await cl.Message(
        content=f"""🎉 **PathFinder 프로세스가 완료되었습니다!**

✅ 저장된 공고: **{len(saved_jobs)}개**

이용해 주셔서 감사합니다! 🍀
다시 시작하려면 `/restart`를 입력해주세요.""",
        author="PathFinder"
    ).send()
    
    cl.user_session.set("current_stage", "done")

async def ask_mmr_preference():
    """MMR 선택"""
    await cl.Message(
        content="""📊 **직무 추천 다양성 설정**

직무 추천 시 얼마나 다양한 직무를 보고 싶으신가요?

**1️⃣ 비슷함** - 정확도 중심 (당신의 경험과 매우 유사한 직무만)
**2️⃣ 보통** - 균형잡힌 탐색 (유사 + 관련 직무)
**3️⃣ 다양함** - 새로운 발견 중심 (폭넓은 직무 탐색)

**선택: 1, 2, 3 중 하나를 입력해주세요**""",
        author="PathFinder"
    ).send()

async def restart_session():
    """재시작"""
    cl.user_session.set("main_state", create_initial_state())
    cl.user_session.set("current_stage", "waiting_start")
    
    cl.user_session.set("interview_ui", InterviewUI())
    cl.user_session.set("ncs_job_recommender_ui", NCSJobRecommenderUI())
    cl.user_session.set("posting_recommender_ui", PostingRecommenderUI())
    cl.user_session.set("posting_manager_ui", PostingManagerUI())
    
    await cl.Message(
        content="🔄 세션이 초기화되었습니다. 시작하려면 아무 메시지나 입력해주세요!",
        author="PathFinder"
    ).send()

@cl.set_starters
async def set_starters():
    return [cl.Starter(
        label="🚀 PathFinder 시작하기",
        message="시작",
        icon="/public/logo.png"
    )]

if __name__ == "__main__":
    cl.run()