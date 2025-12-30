import chainlit as cl
from typing import Tuple, Dict, Any
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

from agents.interview_agent import (
    create_initial_state,
    decoder_node,
    memory_node,
    judger_node,
    generator_node,
    AgentState
)

class InterviewUI:
    """Interview Agent의 Chainlit UI 래퍼"""
    
    def __init__(self):
        self.agent_state = None
        self.turn_count = 0
    
    async def start(self):
        """Interview 시작"""
        self.agent_state = create_initial_state()
        self.turn_count = 0
        
        # 초기 인사
        initial_greeting = """안녕하세요! 커리어 에이전트입니다. 

탐색에 앞서 기본적인 정보를 먼저 알려주세요.

📌 **학력**이 어떻게 되시는지, 그리고 어느 분야에서 **경력**을 쌓았거나 **신입**으로 관심을 가지시는지 이야기해주세요."""
        
        await cl.Message(
            content=initial_greeting,
            author="Interview Agent"
        ).send()
        
        self.agent_state["messages"].append(AIMessage(content=initial_greeting))
    
    async def process_message(self, user_input: str) -> Tuple[bool, Dict[str, Any]]:
        """
        사용자 메시지 처리
        
        Returns:
            (is_complete, user_profile): 완료 여부 및 사용자 프로필
        """
        # 사용자 메시지 추가
        self.agent_state["messages"].append(HumanMessage(content=user_input))
        self.turn_count += 1
        
        # Step으로 에이전트 실행 과정 시각화
        async with cl.Step(name=f"Turn {self.turn_count}", type="tool") as main_step:
            
            # 1. Decoder
            async with cl.Step(name="📝 Decoder", parent_id=main_step.id) as decoder_step:
                decoder_step.input = user_input
                decoder_output = decoder_node(self.agent_state)
                self.agent_state.update(decoder_output)
                
                # 추출된 정보 표시
                extracted = decoder_output.get("update_schema", {})
                if extracted:
                    summary = self._format_extracted_data(extracted)
                    decoder_step.output = summary
                    decoder_step.language = "json"
            
            # 2. Memory
            async with cl.Step(name="🧠 Memory", parent_id=main_step.id) as memory_step:
                memory_output = memory_node(self.agent_state)
                self.agent_state.update(memory_output)
                
                # 업데이트된 프로필 요약
                profile_summary = self._format_profile_summary(self.agent_state["user_profile"])
                memory_step.output = profile_summary
                
                # 가설 생성 확인 (Step 내부에 표시)
                hypotheses = self.agent_state.get("hypothesis_list", [])
                if hypotheses:
                    hypo_text = "생성된 가설:\n" + "\n".join([f"- {h['item']} ({h['type']})" for h in hypotheses[:3]])
                    memory_step.output += f"\n\n{hypo_text}"
            
            # 3. Judger
            async with cl.Step(name="⚖️ Judger", parent_id=main_step.id) as judger_step:
                judger_output = judger_node(self.agent_state)
                self.agent_state.update(judger_output)
                
                strategy = judger_output["next_step_strategy"]
                strategy_type = strategy.get("type")
                judger_step.output = f"전략: {strategy_type}"
                
                # 전략 상세 정보 추가
                if strategy_type == "MICRO_HYPOTHESIS":
                    target = strategy.get("target", [])
                    if isinstance(target, list):
                        judger_step.output += f"\n검증할 역량: {', '.join([h.get('item', '') for h in target[:3]])}"
                elif strategy_type == "MICRO_CONFLICT":
                    target = strategy.get("target", {})
                    judger_step.output += f"\n충돌: {target.get('field', '')}"
                
                # EXIT 확인
                if strategy_type == "EXIT":
                    await cl.Message(
                        content="✅ **인터뷰 종료 조건 달성!**\n\n충분한 정보를 수집했습니다.",
                        author="Interview Agent"
                    ).send()
                    return True, self.agent_state["user_profile"]
            
            # 4. Generator
            async with cl.Step(name="💬 Generator", parent_id=main_step.id) as generator_step:
                generator_output = generator_node(self.agent_state)
                self.agent_state.update(generator_output)
                
                agent_response = self.agent_state["messages"][-1].content
                generator_step.output = agent_response[:100] + "..." if len(agent_response) > 100 else agent_response
        
        # 에이전트 응답 전송 (Step 외부에서)
        await cl.Message(
            content=agent_response,
            author="Interview Agent"
        ).send()
        
        return False, None
    
    def _format_extracted_data(self, data: Dict) -> str:
        """추출된 데이터 포맷팅"""
        lines = []
        
        if data.get("bi"):
            if data["bi"].get("education"):
                lines.append(f"학력: {data['bi']['education']}")
            if data["bi"].get("career"):
                lines.append(f"경력: {data['bi']['career']}")
        
        if data.get("pj"):
            if data["pj"].get("knowledge"):
                lines.append(f"지식: {', '.join(data['pj']['knowledge'])}")
            if data["pj"].get("skills"):
                lines.append(f"기술: {', '.join(data['pj']['skills'])}")
            if data["pj"].get("abilities"):
                lines.append(f"능력: {', '.join(data['pj']['abilities'])}")
        
        if data.get("po"):
            if data["po"].get("industry_interest"):
                lines.append(f"관심산업: {', '.join(data['po']['industry_interest'])}")
        
        return "\n".join(lines) if lines else "추출된 정보 없음"
    
    def _format_profile_summary(self, profile: Dict) -> str:
        """프로필 요약 포맷팅"""
        pj = profile.get("pj", {})
        po = profile.get("po", {})
        pr = profile.get("pr", {})
        
        total_pj = len(pj.get("knowledge", [])) + len(pj.get("skills", [])) + len(pj.get("abilities", []))
        has_industry = len(po.get("industry_interest", [])) > 0
        has_location = len(pr.get("location_limit", [])) > 0
        
        return f"PJ: {total_pj}/10 | Industry: {'✅' if has_industry else '❌'} | Location: {'✅' if has_location else '❌'}"