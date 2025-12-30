import chainlit as cl
from typing import Dict, Any
import asyncio
import pandas as pd

from agents.posting_recommender_agent import run_posting_recommender_agent
from core.state import MainState

class PostingRecommenderUI:
    """Posting Agent의 Chainlit UI 래퍼"""
    
    async def run(self, main_state: MainState):
        """Posting Agent 실행"""
        
        await cl.Message(
            content="📊 **채용 공고 매칭을 시작합니다...**",
            author="Posting Agent"
        ).send()
        
        async with cl.Step(name="Job Posting Matching", type="run") as main_step:
            
            # Transform → Search → Review
            async with cl.Step(name="1️⃣ Transform", parent_id=main_step.id) as transform_step:
                transform_step.output = "카테고리 변환 + 유사어 확장 + 임베딩 생성 중..."
            
            async with cl.Step(name="2️⃣ Search", parent_id=main_step.id) as search_step:
                search_step.output = "SQL 필터 → Vector 유사도 계산 중..."
            
            async with cl.Step(name="3️⃣ Review", parent_id=main_step.id) as review_step:
                review_step.output = "LLM 등급 부여 중..."
            
            try:
                # 동기 함수를 비동기로 실행
                posting_output = await asyncio.to_thread(run_posting_recommender_agent, main_state)
                
                # 결과를 main_state에 반영
                main_state["final_postings"] = posting_output.get("final_postings")
                
                final_df = posting_output.get("final_postings")
                
                if final_df is not None and not final_df.empty:
                    main_step.output = f"✅ {len(final_df)}개 공고 매칭 완료"

                else:
                    main_step.output = "❌ 매칭된 공고 없음"
                
            except Exception as e:
                await cl.Message(
                    content=f"❌ 공고 매칭 중 오류 발생: {str(e)}",
                    author="System"
                ).send()
                raise
        
        # 다음 단계로 이동
        cl.user_session.set("main_state", main_state)
        cl.user_session.set("current_stage", "posting_complete")
        
        # 자동으로 다음 단계 트리거 (Post Manager)
        await cl.Message(
            content="✅ **채용 공고 매칭이 완료되었습니다!**\n\n이제 원하는 공고를 선택하여 저장할 수 있습니다.",
            author="PathFinder"
        ).send()