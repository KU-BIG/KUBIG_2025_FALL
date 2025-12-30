import chainlit as cl
from typing import Dict, Any
import asyncio

from agents.ncs_job_recommender_agent import run_ncs_agent
from core.state import MainState
from tools.ncs_recommend import NCSJobRecommender

class NCSJobRecommenderUI:
    """NCS Job Agent의 Chainlit UI 래퍼"""
    
    async def run(self, main_state: MainState):
        """NCS Agent 실행"""
        
        await cl.Message(
            content="🔍 **NCS 직무 추천을 시작합니다...**",
            author="NCS Agent"
        ).send()
        
        # 메인 Step
        async with cl.Step(name="NCS Job Recommendation", type="run") as main_step:
            
            try:
                # Step 1: Search
                async with cl.Step(name="🔍 Search", parent_id=main_step.id) as search_step:
                    search_step.output = "사용자 프로필 기반 FAISS MMR 검색 중..."
                    await asyncio.sleep(0.1)
                
                # Step 2: Critic
                async with cl.Step(name="⚖️ Critic", parent_id=main_step.id) as critic_step:
                    critic_step.output = "GPT-4o-mini로 직무 적합도 평가 중..."
                    await asyncio.sleep(0.1)
                
                # Step 3: Refine
                async with cl.Step(name="🔄 Refine", parent_id=main_step.id) as refine_step:
                    refine_step.output = "필요 시 검색 쿼리 개선 및 재검색..."
                    await asyncio.sleep(0.1)
                
                # Step 4: Finalize
                async with cl.Step(name="✨ Finalize", parent_id=main_step.id) as finalize_step:
                    finalize_step.output = "LLM 재순위화 → 직무명 변환 → 키워드 생성 → 카테고리 매핑 중..."
                    
                    # 실제 에이전트 실행
                    ncs_output = await asyncio.to_thread(run_ncs_agent, main_state)
                    
                    # 결과 반영
                    main_state["job_category_codes"] = ncs_output.get("job_category_codes", [])
                    main_state["recommendations"] = ncs_output.get("recommendations", [])
                    
                    num_recommendations = len(ncs_output.get("recommendations", []))
                    finalize_step.output = f"✅ 상위 7개 직무 선정 완료 (전체: {num_recommendations}개)"
                
                main_step.output = f"✅ {len(ncs_output.get('recommendations', []))}개 직무 추천 완료"
                
                # 추천 결과 표시
                await self._display_recommendations(ncs_output)
                
            except Exception as e:
                await cl.Message(
                    content=f"❌ NCS 추천 중 오류 발생: {str(e)}",
                    author="System"
                ).send()
                raise
        
        cl.user_session.set("main_state", main_state)
        cl.user_session.set("current_stage", "ncs_complete")
        
        await cl.Message(
            content="✅ **NCS 직무 추천이 완료되었습니다!**\n\n이제 사용자님에게 적합한 실제 채용 공고를 추천해 드리겠습니다.",
            author="PathFinder"
        ).send()
    
    async def _display_recommendations(self, ncs_output: Dict):
        """추천 결과 표시 (Markdown 통합 방식)"""
        recommendations = ncs_output.get("recommendations", [])
        
        if not recommendations:
            await cl.Message(
                content="추천된 직무가 없습니다.",
                author="NCS Agent"
            ).send()
            return
        
        # 1. 카테고리 정보 표시
        category_msg = ""
        category_codes = ncs_output.get("job_category_codes", [])
        if category_codes:
            cat_names = [
                next((k for k, v in NCSJobRecommender.JOB_CATEGORY_CODES.items() if v == code), "기타")
                for code in category_codes
            ]
            category_msg = f"🏷️ **선택된 직무 카테고리**: {', '.join(cat_names)}\n\n"
        
        # 2. 추천 목록 정렬 (rank 기준)
        sorted_recommendations = sorted(
            recommendations, 
            key=lambda x: int(x.get('rank', 999))
        )
        
        # 3. 하나의 Markdown 메시지로 통합 구성
        message_content = category_msg
        message_content += f"📊 **추천된 직무** (상위 {min(len(sorted_recommendations), 7)}개)\n"
        message_content += "="*50 + "\n"
        
        for rec in sorted_recommendations[:7]:
            rank = rec.get('rank', '?')
            job_name = rec.get('변환된_직무명', rec.get('직무명'))
            
            relevance_list = rec.get('핵심_연관성', [])
            relevance_str = "\n".join([f"  • {item}" for item in relevance_list])
            
            keywords = " ".join(rec.get('관련_키워드', []))
            
            # 수정: 이중 줄바꿈으로 공백 추가
            job_card = f"""
### 🏅 {rank}위. {job_name}

💡 **추천 이유**
{rec.get('추천_이유', '')}

✅ **핵심 연관성**
{relevance_str}

⚠️ **보완 필요**
{rec.get('부족한_부분', '없음')}

🗝️ **관련 키워드**: `{keywords}`

---
"""
            message_content += job_card

        await cl.Message(
            content=message_content,
            author="NCS Agent"
        ).send()