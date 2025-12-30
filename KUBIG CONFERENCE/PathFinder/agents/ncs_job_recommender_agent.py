import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser

# Import from core and tools
from core.state import MainState
from tools.ncs_recommend import NCSJobRecommender

# ====================================================
# 1. Local State 정의 (에이전트 내 state)
# ====================================================
class NCSAgentState(TypedDict):
    user_profile: Dict[str, Any] #사용자 프로필 (pj, po, pr)
    search_config: Dict[str, Any] # 검색 설정 (MMR 사용 여부, lambda_mult 다양성 계수 등)
    current_query: str # 현재 단계에서 사용 중인 검색 쿼리 문장
    candidates: List[Dict] # 1차 직무 후보 리스트
    critic_score: int # 평가된 직무 적합성 점수 (0~100)
    critic_reason: str # 점수 부여 이유 (refine 단계 피드백으로 활용)
    retry_count: int # 적합성 미달 시 검색 재시도 횟수
    final_output: Dict[str, Any] # 최종 추천 결과물
    best_candidates: List[Dict]  # 지금까지 중 가장 점수 높았던 후보 리스트
    best_score: int              # 그때의 점수

# ====================================================
# 2. NCS 직무 추천 에이전트
# ====================================================
class NCSJobAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", 
                              temperature=0,
                              max_tokens=4000)
        self.tool = NCSJobRecommender()
        self.tool.prepare_vectorstore(build_new=False) 

    # ------------------------------------------------
    # [Node 1] Search: 검색어 생성 로직 개선
    # ------------------------------------------------
    def search_node(self, state: NCSAgentState):
        retry = state.get('retry_count', 0)
        # 기본 설정값
        config = state.get('search_config', {'use_mmr': True, 'lambda_mult': 0.7})
        
        print(f"\n🔍 [Search] 직무 검색 (시도 {retry + 1}회차)")
        
        query = state.get('current_query')
        if not query:
            
            # culture, motivation 제외하고 핵심 역량만 문장으로 구성
            pj = state['user_profile'].get('pj', {})
            po = state['user_profile'].get('po', {})
            
            knowledge = ", ".join(pj.get('knowledge', []))
            skills = ", ".join(pj.get('skills', []))
            abilities = ", ".join(pj.get('abilities', []))
            industry = ", ".join(po.get('industry_interest', []))
            
            # 자연어 포맷 적용
            query = f"관련 지식은 {knowledge}이 있고, 관련 경험이나 기술 스택으로는 {skills}, {abilities}가 있습니다. 사용자의 관심 도메인은 {industry}입니다."
            
            print(f"   👉 생성된 검색어: \"{query}\"")

        candidates = self.tool.retrieve_candidate_jobs(
            query, 
            k=10, 
            use_mmr=config.get('use_mmr', True), 
            lambda_mult=config.get('lambda_mult', 0.7)
        )
        
        return {
            "candidates": candidates,
            "current_query": query,
            "retry_count": retry,
            "search_config": config
        }

    # ------------------------------------------------
    # [Node 2] Critic
    # ------------------------------------------------
    
    def critic_node(self, state: NCSAgentState):
        print("\n🤔 [Critic] 직무 적합성 상세 평가 중...")
        
        pj = state['user_profile'].get('pj', {})
        po = state['user_profile'].get('po', {})
        candidate_names = [c['직무명'] for c in state['candidates']]
        
        prompt = f"""
        당신은 전문 커리어 컨설턴트입니다. 
        검색된 직무들이 사용자의 프로필과 일치하는지 평가하세요.
        특정 직무와의 fit만 확인하지말고 검색된 직무들 전반과 평가하세요.

        [평가 기준]
        - 핵심 역량: {pj.get('knowledge')}, {pj.get('skills')}
        - 관심 산업: {po.get('industry_interest')}
        - 검색된 직무들: {', '.join(candidate_names)}

        결과를 다음 JSON 형식으로만 응답하세요:
        {{
            "score": 0~100 점수 (숫자),
            "reason": "왜 이 점수를 주었는지 '~입니다'체로 끝나는 1문장으로 설명 (예: 산업군 불일치, 기술 수준 미달 등)"
        }}
        """
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        result = JsonOutputParser().parse(response.content)
        
        current_score = result['score']
        current_reason = result['reason']
        
        print(f"   👉 점수: {current_score}점 \n   👉 사유: {current_reason}")
        
        # 이전 최고 점수 가져오기 (없으면 -1)
        prev_best_score = state.get('best_score', -1)
        
        prev_score_display = prev_best_score if prev_best_score != -1 else "Start"

        if current_score > prev_best_score:
            print(f"   ✨ [Record] 최고 점수 갱신! ({prev_score_display} -> {current_score})")
            return {
                "critic_score": current_score,
                "critic_reason": current_reason,
                "best_candidates": state['candidates'], # 현재 후보군을 백업
                "best_score": current_score
            }
        else:
            print(f"   📉 [Keep] 점수 하락 (최고점: {prev_best_score} 유지)")
            return {
                "critic_score": current_score,
                "critic_reason": current_reason
                # best_candidates는 갱신하지 않음
            }
    
    # ------------------------------------------------
    # [Node 3] Refine
    # ------------------------------------------------
    def refine_node(self, state: NCSAgentState):
        print("\n🔧 [Refine] 점수가 낮아 전략을 수정합니다...")
        
        current_config = state.get('search_config')
        
        prompt = f"""
        당신은 NCS 벡터 검색 최적화 전문가입니다.
        
        현재 검색어: "{state.get('current_query')}"
        평가 피드백: "{state.get('critic_reason')}"
        
        위 피드백을 반영하되, NCS 직무명과 매칭되도록 '일반적이고 포괄적인' 키워드로 수정하세요.
        구체적 도구명보다는 '수행 업무 영역'과 '산업 도메인'을 중심으로 작성하세요.
     
        [지침]
        1. 피드백이 '산업 불일치'라면, 해당 산업에서 어떤 데이터를 다루는지 구체적으로 서술하세요.
        2. 피드백이 '기술 부족'이라면, 해당 기술을 사용하여 무엇을 하는지 서술하세요.
        3. 전체 길이는 30-80자 내외의 자연어 문장이어야 합니다.
        
        최종 수정된 검색 문장만 출력하세요.
        """
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        new_query = response.content.strip()
        print(f"   👉 수정된 검색어: {new_query[:60]}...")
        
        return {
        "current_query": new_query,
        "search_config": current_config, # 기존 설정을 그대로 다음 search_node로 전달
        "retry_count": state['retry_count'] + 1
    }

    # ------------------------------------------------
    # [Node 4] Finalize
    # ------------------------------------------------
    def finalize_node(self, state: NCSAgentState):
        print("\n🎉 [Finalize] 최종 결과 생성")
        best_candidates = state.get('best_candidates', [])
        
        if not best_candidates:
            best_candidates = state['candidates']
            
        print(f"   👉 최종 선택된 후보군 수: {len(best_candidates)}개 (최고 점수 기반)")
        
        # LLM에게 넘겨줄 때도 직무 역량만 전달
        pj = state['user_profile'].get('pj', {})
        po = state['user_profile'].get('po', {})
        
        knowledge = ", ".join(pj.get('knowledge', []))
        skills = ", ".join(pj.get('skills', []))
        abilities = ", ".join(pj.get('abilities', []))
        industry = ", ".join(po.get('industry_interest', []))
        
        # LLM용 깔끔한 프로필 생성
        filtered_user_input = f"""
[사용자 직무 프로필]
- 보유 지식: {knowledge}
- 기술/툴: {skills}
- 주요 역량: {abilities}
- 관심 도메인: {industry}
"""
        
        print("   👉 LLM 정밀 재랭킹 & 변환 수행...")
        
        # 기존 str(state['user_profile']) 대신 filtered_user_input을 전달
        reranked = self.tool.rerank_with_llm(filtered_user_input, best_candidates, top_k=7)
        transformed = self.tool.transform_job_names(reranked, filtered_user_input)
        final = self.tool.generate_keywords(transformed, filtered_user_input)
        codes = self.tool.map_to_job_categories(final, filtered_user_input)
        
        return {
            "final_output": {
                "job_category_codes": codes,
                "recommendations": final.get('recommendations', [])
            }
        }
        
    # ------------------------------------------------
    # Graph Build
    # ------------------------------------------------
    def build_graph(self):
        workflow = StateGraph(NCSAgentState)
        workflow.add_node("search", self.search_node)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("refine", self.refine_node)
        workflow.add_node("finalize", self.finalize_node)

        workflow.set_entry_point("search")
        workflow.add_edge("search", "critic")

        def check_score(state):
            if state['critic_score'] >= 80: return "pass"
            elif state['retry_count'] >= 2: return "pass"
            return "fail"

        workflow.add_conditional_edges("critic", check_score, {"pass": "finalize", "fail": "refine"})
        workflow.add_edge("refine", "search")
        workflow.add_edge("finalize", END)
        return workflow.compile()

# 외부 호출용 (통합 시 사용)
def run_ncs_agent(main_state: MainState) -> dict:
    agent = NCSJobAgent()
    app = agent.build_graph()
    
    initial_state = {
        "user_profile": main_state['user_profile'],
        "search_config": main_state.get('search_config', {'use_mmr': True, 'lambda_mult': 0.7}),
        "current_query": "",
        "retry_count": 0,
        "candidates": [],
        "best_candidates": [],
        "best_score": -1,
        "critic_score": 0,
        "final_output": {}
    }
    
    final_state = app.invoke(initial_state)
    agent.tool._print_recommendations({'recommendations': final_state['final_output']['recommendations']})
    return final_state['final_output']
