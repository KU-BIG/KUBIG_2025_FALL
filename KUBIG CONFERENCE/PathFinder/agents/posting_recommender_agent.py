import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from core and tools
from core.state import MainState
from tools.user_transformer import UserTransformer
from tools.data_filter import filter_jobs_from_db
from tools.vector_engine import compute_posting_similarity
from tools.llm_reviewer import review_final_postings

# 1. 에이전트 내부 전용 상태 정의
class PostingAgentState(TypedDict):
    user_profile: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    job_category_codes: List[int]
    processed_data: Optional[Dict[str, Any]]
    top_jobs_df: Optional[pd.DataFrame]
    final_postings: Optional[pd.DataFrame]

class PostingAgent:
    """
    내부 노드(Transform -> Search -> Review)를 통해 
    최종 공고를 매칭하는 LangGraph 기반 에이전트 클래스
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        # Ensure DB path is absolute or relative to CWD
        self.db_path = os.getenv("DB_PATH", "data/job_service.db")
        self.transformer = UserTransformer(api_key=self.api_key)

    # [Node 1] 데이터 변환 및 확장
    def transform_node(self, state: PostingAgentState):
        processed = self.transformer.process_agent_state(state)
        
        print(f"[Transform]처리 완료 : {processed.get('exp_years')}, {processed.get('edu_code')}, {processed.get('category_names')}, {processed.get('target_roles')}")
        return {"processed_data": processed}

    # [Node 2] SQL 필터링 및 벡터 유사도 검색
    def search_node(self, state: PostingAgentState):
        processed = state['processed_data']
        
        if not processed:
            print("[Search] processed_data가 없습니다.")
            return {"top_jobs_df": pd.DataFrame()}
        # 1. SQL 하드 필터링 (직무명 키워드 및 경력 완화 적용)
        df_filtered = filter_jobs_from_db(
            db_path=self.db_path,
            exp_years=processed['exp_years'],
            edu_code=processed['edu_code'],
            category_names=processed['category_names'],
            target_roles=processed['target_roles']
        )
        
        print(f"[Search] SQL 필터링 결과: {len(df_filtered)}개 공고")
        
        if df_filtered.empty:
            print("[Search] SQL 필터링 결과가 없습니다.")
            return {"top_jobs_df": pd.DataFrame()}

        # 2. 벡터 유사도 분석으로 상위 20개 추출
        top_jobs_df = compute_posting_similarity(
            df_filtered=df_filtered,
            user_embedding=processed['embedding'],
            top_n=20
        )
        print(f"[Search] 유사도 분석 완료: {len(top_jobs_df)}개 선별")
        return {"top_jobs_df": top_jobs_df}

    # [Node 3] LLM 최종 정성 리뷰
    def review_node(self, state: PostingAgentState):
        top_jobs_df = state.get('top_jobs_df')
        if state['top_jobs_df'].empty:

            return {"final_postings": pd.DataFrame()}
          
        processed = state['processed_data']
            
        final_df = review_final_postings(
            top_jobs_df=top_jobs_df,
            user_super_text=processed['super_text'],
            user_expanded_vars=processed['expanded_vars'],
            target_roles=processed['target_roles']
        )
        
        print(f"[Review] 최종 리뷰 완료: {len(final_df)}개 공고")
        
        return {"final_postings": final_df}

    # [Graph Build] 에이전트 내부 흐름 정의
    def build_graph(self):
        workflow = StateGraph(PostingAgentState)
        
        workflow.add_node("transform", self.transform_node)
        workflow.add_node("search", self.search_node)
        workflow.add_node("review", self.review_node)

        workflow.set_entry_point("transform")
        workflow.add_edge("transform", "search")
        workflow.add_edge("search", "review")
        workflow.add_edge("review", END)
        
        return workflow.compile()

# --- 외부 호출용 (MainState와 연동) ---
def run_posting_recommender_agent(main_state: MainState) -> Dict[str, Any]:
    """
    MainState를 입력받아 PostingAgent 내부 그래프를 실행하고 결과를 반환
    """
    agent = PostingAgent()
    app = agent.build_graph()
    
    # 내부 그래프용 초기 상태 설정
    initial_state = {
        "user_profile": main_state.get('user_profile', {}),
        "recommendations": main_state.get('recommendations', []),
        "job_category_codes": main_state.get('job_category_codes', []),
        "processed_data": None,
        "top_jobs_df": None,
        "final_postings": None
    }
    
    print(f"[PostingAgent] 내부 그래프 실행 시작")
    final_state = app.invoke(initial_state)
    print(f"[PostingAgent] 내부 그래프 실행 완료- 최종 결과: {len(final_state.get('final_postings', pd.DataFrame()))}개 공고")
    
    # 결과를 MainState 형식에 맞춰 반환
    return {"final_postings": final_state['final_postings']}
