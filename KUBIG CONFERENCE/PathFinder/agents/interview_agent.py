import json
import os
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
# 1. Configuration & Setup
# ==============================================================================

# Initialize Embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Global Data Cache
faiss_index = None

def load_faiss_index():
    global faiss_index
    # Path to the unified data directory
    index_path = os.path.join(os.path.dirname(__file__), "..", "data", "ncs_faiss_index")
    print(f"[System] Loading FAISS Index from {index_path}...")
    try:
        if os.path.exists(index_path):
            faiss_index = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
            print("[System] FAISS Index loaded successfully.")
        else:
            print(f"[System] Warning: FAISS index not found at '{index_path}'. Search will fail.")
    except Exception as e:
        print(f"[System] Error loading FAISS index: {e}")

# Load data at startup
load_faiss_index()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000
)

# ==============================================================================
# 2. Data Structures & Schema
# ==============================================================================

class BasicInfo(TypedDict):
    education: str
    career: str

class PJFit(TypedDict):
    knowledge: List[str]
    skills: List[str]
    abilities: List[str]

class POFit(TypedDict):
    values: List[str]
    culture: str
    industry_interest: List[str]
    motivation: List[str]

class PRFit(TypedDict):
    salary_min: str
    location_limit: List[str]
    priority: str
    growth_goal: List[str]


class UserProfile(TypedDict):
    bi: BasicInfo
    pj: PJFit
    po: POFit
    pr: PRFit

class AgentState(TypedDict):
    messages: List[BaseMessage]
    user_profile: UserProfile
    hypothesis_list: List[Dict]
    conflict_flags: List[Dict]
    next_step_strategy: Dict
    turn_count: int
    update_schema: Dict
    used_hypothesis_items: List[str] # Track used hypotheses to avoid repetition

def create_initial_state() -> AgentState:
    return {
        "messages": [],
        "user_profile": {
            "bi": {"education": "", "career": ""},
            "pj": {"knowledge": [], "skills": [], "abilities": []},
            "po": {"values": [], "culture": "", "industry_interest": [], "motivation": []},
            "pr": {"salary_min": "", "location_limit": [], "priority": "", "growth_goal": []}
        },
        "hypothesis_list": [],
        "conflict_flags": [],
        "next_step_strategy": {},
        "turn_count": 0,
        "update_schema": {},
        "used_hypothesis_items": []
    }

# ==============================================================================
# 3. Node Implementations
# ==============================================================================

def print_debug(section: str, content: Any):
    print(f"\n{'='*20} [{section}] {'='*20}")
    if isinstance(content, (dict, list)):
        print(json.dumps(content, indent=2, ensure_ascii=False))
    else:
        print(content)
    print("="*50)

# ------------------------------------------------------------------------------
# Node 1: Decoder (Extraction)
# ------------------------------------------------------------------------------
def decoder_node(state: AgentState) -> Dict:
    last_message = state["messages"][-1].content
    print_debug("Decoder Input", last_message)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 직무 인터뷰 분석가입니다. 사용자의 발화에서 다음 Schema에 맞는 정보를 추출하여 JSON으로 반환하세요.

[Schema Definition]
- P-B (Person-Basic):
    - education: 최종 학력 (고졸, 초대졸, 대졸, 석사, 박사)
      * 규칙: 사용자가 자신의 최종 학력을 명시적으로 말하는 경우에만 추출.
      * 규칙: 재학, 휴학과 같은 해당 과정을 수행하는 경우에는 해당 과정의 '졸업'으로 간주하여 매핑 (예: 대학교 재학 -> 대졸, 대학교 휴학 -> 대졸)
      * 형식: 반드시 "고졸", "초대졸", "대졸", "석사", "박사" 중 하나의 단어로 추출. 그외의 다른 단어는 education에 속할 수 없음.
    - career: 경력 사항 (신입, 경력 N년)
      * 정의: 정식 채용되어 회사에서 근무한 경험만 '경력'으로 인정. 경력이 없다거나 신입이라고 명시하는 경우 '신입'으로 인정.
      * 제외: 동아리, 공모전, 프로젝트, 아르바이트 등은 '경력'이 아님 -> '신입'으로 분류.
      * 규칙: 경력이 년 단위로 구성되지 않는 경우 올림처리하여 년 단위로 고려하여 추출.(예: 1년 6개월 ->2년 -> 2)
      * 형식: 경력이 있다면 경력 N년 중에서 "N"으로 숫자만 추출. 경력이 없으면 "신입".

- P-J (Person-Job):
    - knowledge: 직무 수행을 위한 이론적 지식, 방법론, 프레임워크, 학문적 개념 (예: 통계학, 마케팅 이론, 재무회계, 온톨로지, 애자일)
    - skills: 실제 다룰 수 있는 도구, 소프트웨어, 언어, 기술, 구축 기법 (예: Python, SQL, Excel, Figma, 지식 그래프 구축)
    - abilities: 업무 수행을 위한 행동 역량, 지식과 기술을 활용하여 문제를 해결하는 관찰 가능하고 측정 가능한 행동 특성, 또는 사용자의 행동에서 강하게 추론되는 역량 (예: 커뮤니케이션, 문제해결력, 리더십, 통찰력)
- P-O (Person-Organization):
    - values: 직업적 가치관, 일을 대하는 태도와 목적 (예: 성장, 안정, 워라밸, 사회적 기여)
    - culture: 선호하는 조직 문화, 본인이 생산성을 극대화할 수 있는 조직의 운영방식 (예: 수평적, 체계적, 자율적) - 복수 언급 시 리스트로 반환
    - industry_interest: 관심 산업 분야 (예: IT, 금융, 헬스케어, 제조)
    - motivation: 지원 동기 또는 일하는 동기 (예: 역량 발휘, 높은 보상)
- P-R (Person-Reward):
    - salary_min: 희망 최소 연봉 (예: 4000만원, 5천) - 복수 언급 시 리스트로 반환
    - location_limit: 희망 근무지 또는 출퇴근 가능 지역 (예: 서울, 판교, 강남, 부산, 재택근무)
    - priority: 직장 선택 시 최우선 순위 (예: 연봉, 위치, 성장가능성) - 복수 언급 시 리스트로 반환
    - growth_goal: 커리어 성장 목표 (예: 데이터 사이언티스트, PM)

- Dialogue Status:
    - is_refusal: (boolean) 사용자가 에이전트의 이전 질문(특히 가설 확인 질문)에 대해 "아니오", "그런 경험 없습니다", "잘 모릅니다" 등으로 거절하거나, 질문을 무시하고 전혀 다른 이야기를 하는 경우 true.
    - reason: (string) 거절 또는 무시의 이유 (예: "경험 없음", "관심 없음", "다른 주제 언급")

[Extraction Rules]
1. 사용자의 발화에 명시된 내용과, 사용자의 행동이나 성과에서 강하게 추론되는 특정한 역량(Abilities)을 추출하십시오.
2. 사용자가 특정 방법론이나 기술적 개념(예: 온톨로지, 지식 그래프)을 활용했다고 언급하면, 이를 Knowledge 또는 Skills로 적극적으로 추출하십시오.
3. 사용자의 경험을 토대로 강하게 추론되는 특정한 산업 분야를 추출하십시오.
4. 범주성 오류를 주의하십시오. (예: '파이썬'은 skills, '통계학'은 knowledge)
5. JSON 형식으로만 응답하십시오.

[Example]
Input: "직무 추천을 하는 AI 에이전트를 만들었습니다. HR 지식과 langgraph를 활용해 에이전트를 구현했습니다. 이 과정에서 에이전트가 정밀한 질문을 생성하기 위해서는 지식 정보가 필요하다는 것을 배웠습니다. 이에 ncs 정보를 지식 그래프로 변환했습니다. 지식 그래프를 사용해 데이터들 간의 의미를 파악하고 의미를 부여하는 온톨로지를 활용하게 되었습니다. 그 결과 에이전트의 문제는 llm의 성능이 아니라 데이터 간의 의미를 부여하는 작업이라는 점을 깨달았습니다."
Output: {{
  "bi": {{"education": "", "career": ""}},
  "pj": {{"knowledge": ["HR 지식", "온톨로지", "데이터 의미론"], "skills": ["langgraph", "지식 그래프 구축"], "abilities": ["문제해결력", "통찰력"]}},
  "po": {{"values": [], "culture": "", "industry_interest": [], "motivation": []}},
  "pr": {{"salary_min": "", "location_limit": "", "priority": "", "growth_goal": []}},
  "dialogue_status": {{"is_refusal": false, "reason": ""}}
}}"""),
        ("human", "{text}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        extracted_data = chain.invoke({"text": last_message})
        
        # --- Structural Validation for Education ---
        VALID_EDUCATION_LEVELS = ["고졸", "초대졸", "대졸", "석사", "박사"]
        if "bi" in extracted_data and "education" in extracted_data["bi"]:
            edu_val = extracted_data["bi"]["education"]
            if edu_val and edu_val not in VALID_EDUCATION_LEVELS:
                print(f"[Decoder] Warning: Invalid education value '{edu_val}' detected. Dropping it.")
                extracted_data["bi"]["education"] = "" # Reset to empty
        # -------------------------------------------

        print_debug("Decoder Output (Extracted Entities - update_schema)", extracted_data)
    except Exception as e:
        print(f"[Decoder] Error: {e}")
        extracted_data = {}

    return {"update_schema": extracted_data}

# ------------------------------------------------------------------------------
# Helper: Embedding Search & Middleware
# ------------------------------------------------------------------------------

def search_related_tasks(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    Search for top_k related tasks using FAISS.
    """
    global faiss_index
    
    if faiss_index is None:
        print("[Search] Warning: FAISS index not loaded.")
        return []

    try:
        results_with_scores = faiss_index.similarity_search_with_score(query_text, k=top_k)
        
        results = []
        for doc, score in results_with_scores:
            # doc.metadata contains the task info
            results.append({
                "task": doc.metadata,
                "score": float(score) # In L2, lower is better. In Cosine, higher is better.
            })
            
        return results
        
    except Exception as e:
        print(f"[Search] FAISS Search Error: {e}")
        return []

def middleware_term_translator(ncs_items: List[Dict]) -> List[Dict]:
    """
    Translate formal NCS terms into natural, user-friendly hypothesis items.
    Input: List of {"item": "...", "type": "..."}
    Output: List of {"item": "...", "score": ..., "type": "..."}
    """
    if not ncs_items:
        return []
        
    # Deduplicate items by name
    unique_items = {item["item"]: item for item in ncs_items}.values()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 커리어 코칭 전문가입니다. 
NCS(국가직무능력표준)에서 추출된 직무 역량(지식/기술) 용어들을 사용자가 이해하기 쉬운 자연스러운 표현이나 구체적인 질문 소재로 변환해 주세요.

[Input Format]
- List of {{"item": "Formal NCS Term", "type": "Knowledge/Skill"}}

[Output Format]
- JSON List of {{"item": "Refined Term", "original": "Formal Term", "type": "Knowledge/Skill"}}
- "Refined Term": 사용자가 인터뷰에서 질문받았을 때 이해하기 쉬운 용어 (예: "SWOT 분석 기법" -> "SWOT 분석 경험", "고객 응대" -> "고객과의 소통 경험")
"""),
        ("human", "Convert these terms: {ncs_list}")
    ])
    
    try:
        chain = prompt | llm | JsonOutputParser()
        refined_list = chain.invoke({"ncs_list": str(list(unique_items))})
        
        # Merge scores back if needed, or just return refined list
        # For now, we assign a default high score since they came from vector search
        final_hypotheses = []
        for refined in refined_list:
            final_hypotheses.append({
                "item": refined.get("item", refined.get("original")),
                "score": 0.85, # Default high confidence for retrieved items
                "type": refined.get("type", "skill")
            })
            
        return final_hypotheses
        
    except Exception as e:
        print(f"[Middleware] Error: {e}")
        # Fallback: return originals
        return [{"item": i["item"], "score": 0.8, "type": i["type"]} for i in unique_items]

# ------------------------------------------------------------------------------
# Node 2: Memory & Knowledge (Storage & Inference)
# ------------------------------------------------------------------------------
def memory_node(state: AgentState) -> Dict:
    profile = state["user_profile"]
    new_data = state["update_schema"]
    conflicts = state["conflict_flags"]
    hypotheses = state["hypothesis_list"]
    used_items = state.get("used_hypothesis_items", [])
    
    # 1. Lifecycle Rules (Update Profile)
    def accumulate_list(target_list, new_items):
        if not new_items: return
        # Fix: Ensure new_items is a list. If it's a string, wrap it.
        if isinstance(new_items, str):
            new_items = [new_items]
            
        for item in new_items:
            if item not in target_list:
                target_list.append(item)

    def replace_value(category, key, new_val):
        if new_val:
            # Handle same-turn conflict (list input for single-value field)
            if isinstance(new_val, list) and len(new_val) > 1:
                # Pick the first one as the value to store
                val_to_store = new_val[0]
                # Flag conflict between the first two
                conflict_msg = {"field": f"{category}.{key}", "old": new_val[0], "new": new_val[1]}
                conflicts.append(conflict_msg)
                print_debug("Same-Turn Conflict Detected", conflict_msg)
            elif isinstance(new_val, list) and len(new_val) == 1:
                val_to_store = new_val[0]
            else:
                # It's a string or single value
                val_to_store = new_val

            # Cross-turn conflict check
            old_val = profile[category][key]
            if old_val and old_val != val_to_store:
                conflict_msg = {"field": f"{category}.{key}", "old": old_val, "new": val_to_store}
                conflicts.append(conflict_msg)
                print_debug("Cross-Turn Conflict Detected", conflict_msg)
            
            profile[category][key] = val_to_store

    if "bi" in new_data:
        replace_value("bi", "education", new_data["bi"].get("education")) # replace 라서 한번 더 들어오면 vs 질문 발생
        replace_value("bi", "career", new_data["bi"].get("career"))

    if "pj" in new_data:
        accumulate_list(profile["pj"]["knowledge"], new_data["pj"].get("knowledge", []))
        accumulate_list(profile["pj"]["skills"], new_data["pj"].get("skills", []))
        accumulate_list(profile["pj"]["abilities"], new_data["pj"].get("abilities", []))

    if "po" in new_data:
        accumulate_list(profile["po"]["values"], new_data["po"].get("values", []))
        accumulate_list(profile["po"]["industry_interest"], new_data["po"].get("industry_interest", []))
        accumulate_list(profile["po"]["motivation"], new_data["po"].get("motivation", []))
        replace_value("po", "culture", new_data["po"].get("culture"))

    if "pr" in new_data:
        replace_value("pr", "salary_min", new_data["pr"].get("salary_min"))
        accumulate_list(profile["pr"]["location_limit"], new_data["pr"].get("location_limit", []))
        replace_value("pr", "priority", new_data["pr"].get("priority"))
        accumulate_list(profile["pr"]["growth_goal"], new_data["pr"].get("growth_goal", []))

    print_debug("Updated Profile", profile)

    # 3. Hypothesis Generation (Embedding-Based)
    # Construct Query
    query_parts = []
    query_parts.extend(profile["pj"]["knowledge"])
    query_parts.extend(profile["pj"]["skills"])
    query_parts.extend(profile["po"]["industry_interest"])
    query_parts.extend(profile["pr"]["growth_goal"])
    
    query_text = " ".join(query_parts)
    
    if query_text:
        print_debug("Hypothesis Query-현재 확인된 정보", query_text)
        
        # A. Search Related Tasks
        related_tasks = search_related_tasks(query_text, top_k=3)
        
        # B. Extract Potential K/S/A
        potential_items = []
        existing_items = set(profile["pj"]["knowledge"] + profile["pj"]["skills"] + profile["pj"]["abilities"])
        
        for res in related_tasks:
            task = res["task"]
            # Add Knowledge
            for k in task.get("related_knowledge", []):
                if k not in existing_items:
                    potential_items.append({"item": k, "type": "knowledge"})
            # Add Skills
            for s in task.get("related_skills", []):
                if s not in existing_items:
                    potential_items.append({"item": s, "type": "skill"})
        
        # Limit potential items to avoid overwhelming the LLM
        potential_items = potential_items[:10]
        
        # C. Middleware Translation
        if potential_items:
            refined_hypotheses = middleware_term_translator(potential_items)
            print_debug("Refined Hypotheses", refined_hypotheses)
            
            # Merge new hypotheses
            hypotheses.extend(refined_hypotheses)

    # Always filter and sort (Lifecycle Management)
    # Remove duplicates by item name
    unique_hypotheses = {h["item"]: h for h in hypotheses}.values()
    
    # Filter out used items
    candidates = [h for h in unique_hypotheses if h["item"] not in used_items]
    
    # Sort by score
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Keep top 10 (buffer for Judger to pick from)
    hypotheses = candidates[:10]

    return {
        "user_profile": profile,
        "conflict_flags": conflicts,
        "hypothesis_list": hypotheses,
        "turn_count": state["turn_count"] + 1
    }

# ------------------------------------------------------------------------------
# Node 3: Judger (Strategy)
# ------------------------------------------------------------------------------
def judger_node(state: AgentState) -> Dict:
    profile = state["user_profile"]
    conflicts = state["conflict_flags"]
    hypotheses = state["hypothesis_list"]
    turn_count = state["turn_count"]
    used_items = state.get("used_hypothesis_items", [])
    
    # Get previous strategy and dialogue status
    prev_strategy = state.get("next_step_strategy", {}).get("type")
    dialogue_status = state.get("update_schema", {}).get("dialogue_status", {})
    is_refusal = dialogue_status.get("is_refusal", False)
    
    pj_count = len(profile["pj"]["knowledge"]) + len(profile["pj"]["skills"]) + len(profile["pj"]["abilities"])
    has_industry = len(profile["po"]["industry_interest"]) > 0
    has_location = bool(profile["pr"]["location_limit"])
    
    # Termination Check
    # P-J: 10 items total, P-O: Industry >= 1, P-R: Location >= 1
    if pj_count >= 10 and has_industry and has_location:
         strategy = {"type": "EXIT"}
    elif turn_count >= 20:
        strategy = {"type": "EXIT"}
    else:
        # Strategy Selection
        
        # 1. Branching Logic: Refusal after Hypothesis
        if prev_strategy == "MICRO_HYPOTHESIS" and is_refusal:
            strategy = {
                "type": "MACRO", 
                "target": "open_exploration", 
                "context": "User rejected previous hypothesis. Switch to open-ended exploration to find new topics."
            }
            print_debug("Branching Triggered", "Switching to MACRO due to hypothesis refusal.")
            
        elif conflicts:
            conflict = conflicts[0]
            strategy = {
                "type": "MICRO_CONFLICT",
                "target": conflict,
                "context": f"Conflict detected in {conflict['field']}: {conflict['old']} vs {conflict['new']}"
            }
        # 0. Basic Info Check (Highest Priority)
        elif not profile["bi"]["education"] or not profile["bi"]["career"]:
            strategy = {"type": "MACRO", "target": "basic_info", "context": "Missing education or career info"}
        elif not has_industry:
            strategy = {"type": "MACRO", "target": "industry_interest", "context": "Missing industry interest"}
        elif not has_location:
            strategy = {"type": "MACRO", "target": "location_limit", "context": "Missing location limit"}
        elif hypotheses:
            # Select top 3-5 hypotheses
            # Hypotheses in state are already filtered by memory_node, but let's be safe
            candidates = [h for h in hypotheses if h["item"] not in used_items]
            
            if candidates:
                top_hypotheses = candidates[:5]
                
                # Mark as used
                newly_used = [h["item"] for h in top_hypotheses]
                used_items.extend(newly_used)
                
                strategy = {
                    "type": "MICRO_HYPOTHESIS",
                    "target": top_hypotheses,
                    "context": "Verify possession of multiple skills with STAR context"
                }
            else:
                 # Fallback if all hypotheses are used (shouldn't happen often with memory_node refreshing)
                 strategy = {"type": "MACRO", "target": "pj_fit", "context": "Need more skills/knowledge"}
        elif pj_count < 3:
            strategy = {"type": "MACRO", "target": "pj_fit", "context": "Need more skills/knowledge"}
        else:
            strategy = {"type": "MACRO", "target": "general", "context": "Expand on experience"}

    print_debug("Judger Strategy", strategy)
    return {
        "next_step_strategy": strategy,
        "used_hypothesis_items": used_items
    }

# ------------------------------------------------------------------------------
# Node 4: Generator (Question)
# ------------------------------------------------------------------------------
def generator_node(state: AgentState) -> Dict:
    strategy = state["next_step_strategy"]
    profile = state["user_profile"]
    turn_count = state["turn_count"]
    
    strategy_type = strategy.get("type")
    target_info = strategy.get("target")
    
    # Dynamic Prompt Construction
    base_instruction = f"현재 전략: {strategy_type}\n타겟 정보: {{target_info}}\n사용자 문맥: {{user_context}}\n\n지시:\n"
    
    if strategy_type == "MACRO":
        if target_info == "basic_info":
             instruction = "사용자의 학력 또는 경력 정보가 누락되었습니다. 이에 대해 물어보세요."
        else:
             instruction = "사용자의 정보가 부족합니다. 타겟 정보에 대해 개방형으로 넓게 질문하여 정보를 수집하세요."
    elif strategy_type == "MICRO_HYPOTHESIS":
        instruction = """
        1. 사용자의 이전 발화에서 맥락(STAR: Situation, Task, Action, Result)을 파악하세요.
        2. 사용자의 이전 발화에서 다음의 요소가 암시적으로 파악되었습니다. : {target_info}
        3. 맥락을 고려하여 적절한 요소를 생각하고, 사용자가 경험한 상황에서 이 역량들을 활용해본 적이 있는지 물어보세요.
        4. **중요**: 질문은 사용자가 해당 경험이 없을 수도 있다는 전제하에 부드럽게 하세요. "혹시 ~한 경험이 있으신가요?"와 같이 묻고, **"만약 해당 경험이 없다면, 관련된 다른 경험이나 본인이 중요하게 생각하는 다른 역량에 대해 말씀해 주셔도 좋습니다."** 라는 뉘앙스를 반드시 포함하여 사용자가 편안하게 다른 이야기를 할 수 있도록 유도하세요.
        5. 질문은 자연스럽게 연결되어야 하며, 단순히 나열식으로 묻지 마세요.
        """ # target_info is top_hypotheses
    elif strategy_type == "MICRO_CONFLICT":
        # Check if it's a Basic Info conflict
        if isinstance(target_info, dict) and target_info.get("field", "").startswith("bi."):
             field_name = "학력" if "education" in target_info["field"] else "경력"
             instruction = f"사용자의 {field_name} 정보에 충돌이 감지되었습니다. 기존 정보 '{target_info['old']}'와(과) 새로운 정보 '{target_info['new']}' 중 어느 것이 맞는지 명확하게 확인하는 질문을 하세요."
        else:
             instruction = "사용자의 가치관 충돌이 감지되었습니다. '{{target_info}}' 상황에서 현실적으로 무엇이 더 중요한지 선택을 유도하는 질문을 하세요."
    elif strategy_type == "EXIT":
        instruction = "인터뷰를 종료합니다. 사용자에게 감사를 표하고 수고하셨다는 인사를 건네세요."
    else:
        instruction = "자연스럽게 대화를 이어가세요."

    
    prompt_text = base_instruction + instruction

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 직무 인터뷰 에이전트입니다. 주어진 전략과 지시에 따라 한 가지 명확한 질문을 생성하세요."),
        ("human", prompt_text)
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "user_context": str(profile),
        "target_info": str(target_info)
    })
    
    print_debug("Generator Output", response.content)

    print("에이전트에 입력된 지시사항 : ", instruction)
    return {"messages": [response]}

# ==============================================================================
# 4. Graph Construction
# ==============================================================================

workflow = StateGraph(AgentState)

workflow.add_node("decoder", decoder_node)
workflow.add_node("memory", memory_node)
workflow.add_node("judger", judger_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("decoder")
workflow.add_edge("decoder", "memory")
workflow.add_edge("memory", "judger")

def check_exit(state: AgentState) -> Literal["generator", "end"]:
    if state["next_step_strategy"].get("type") == "EXIT":
        return "end"
    return "generator"

workflow.add_conditional_edges(
    "judger",
    check_exit,
    {
        "generator": "generator",
        "end": END
    }
)

workflow.add_edge("generator", END)

app = workflow.compile()

# ==============================================================================
# 5. Execution Helper
# ==============================================================================

def run_interview_session() -> Dict[str, Any]:
    print("--- P-E Fit Interview Agent (Interactive Mode) ---")
    print("Type 'exit' or 'quit' to stop manually.\n")
    
    state = create_initial_state()
    
    # 1. Initial Greeting
    initial_greeting = "안녕하세요! 커리어 에이전트입니다. \n 탐색에 앞서 기본적인 정보를 먼저 알려주세요. 학력이 어떻게 되시는지, 그리고 어느 분야에서 경력을 쌓았거나 신입으로 관심을 가지시는지 이야기해주세요."
    print(f"\nAgent: {initial_greeting}")
    
    state["messages"].append(AIMessage(content=initial_greeting))
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue

            state["messages"].append(HumanMessage(content=user_input))
            
            # Run the graph
            state = app.invoke(state)
            
            # The last message should be the agent's response
            agent_response = state["messages"][-1].content
            print(f"\nAgent: {agent_response}")
            
            if state.get("next_step_strategy", {}).get("type") == "EXIT":
                print("\n[System] Interview Finished.")
                print_debug("Final User Profile", state["user_profile"])
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[System] Error: {e}")
            break
            
    return state["user_profile"]

if __name__ == "__main__":
    run_interview_session()
