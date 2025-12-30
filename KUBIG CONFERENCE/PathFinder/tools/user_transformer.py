import numpy as np
import json
import numpy as np
from openai import OpenAI
from typing import Dict, Any

class UserTransformer:
    """사용자의 정성적 페르소나를 분석하여 검색 엔진용 텍스트와 벡터로 변환하는 도구"""
    
    def __init__(self, api_key: str):
        self.model = "gpt-4o-mini"
        self.client = OpenAI(api_key=api_key)
        # 직무 카테고리 매핑 (NCS 에이전트 연동용)
        self.cat_code_to_name = {
            '16': '기획·전략', '14': '마케팅·홍보·조사', '3': '회계·세무·재무', 
            '5': '인사·노무·HRD', '4': '총무·법무·사무', '2': 'IT개발·데이터', 
            '15': '디자인', '8': '영업·판매·무역', '21': '고객상담·TM', 
            '18': '구매·자재·물류', '12': '상품기획·MD', '7': '운전·운송·배송', 
            '10': '서비스', '11': '생산', '22': '건설·건축', '6': '의료', 
            '9': '연구·R&D', '19': '교육', '13': '미디어·문화·스포츠', 
            '17': '금융·보험', '20': '공공·복지'
        }

    def expand_similar_terms(self, terms: list, category: str) -> list:
        """LLM을 사용하여 원본 키워드와 유사한 단어들을 생성"""
        if not terms: return []
        
        category_prompts = {
            "knowledge": "관련 지식이나 학문적 배경",
            "skills": "기술 스택이나 실무 능력",
            "abilities": "보유한 역량이나 능력",
            "industry_interest": "관심 있는 산업 도메인이나 업계"
        }
        
        system_prompt = f"""
        너는 {category_prompts.get(category, "관련 항목")} 분야의 전문가야.
        주어진 단어들과 의미적으로 유사하거나 관련된 단어들을 JSON 리스트로만 반환해줘.
        형식: {{"similar_terms": ["단어1", "단어2", ...]}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"유사 단어 생성: {', '.join(terms)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            similar_terms = json.loads(response.choices[0].message.content).get("similar_terms", [])
            return list(dict.fromkeys(terms + similar_terms)) # 중복 제거
        except Exception as e:
            log_progress(f"⚠️ 유사 단어 생성 에러: {e}")
            return terms

    def get_embedding(self, text: str):
        """텍스트의 의미적 특징을 추출하는 벡터 생성"""
        try:
            clean_text = " ".join(str(text).lower().split())
            response = self.client.embeddings.create(
                input=clean_text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            log_progress(f"❌ 임베딩 생성 에러: {e}")
            return None
          
    def normalize_education_experience(self, bi: Dict[str, Any]) -> tuple:
        """텍스트 형태의 학력/경력을 숫자형으로 정규화"""
        # 1. 학력 정규화 (예: '대졸', '대학교 3학년' -> 3)
        raw_edu = str(bi.get('education', '학력무관'))
        edu_map = {'학력무관': 0, '고졸': 1, '초대졸': 2, '대졸': 3, '석사': 4, '박사': 5}
        edu_code = edu_map.get(raw_edu, 0)
        
        # 경력 정규화
        try:
            exp_years = int(bi.get('career', 0))
        except:
            exp_years = 0
            
        return edu_code, exp_years
          
    def build_super_user_text(self, state: dict, expanded_vars: dict) -> str:
        """에이전트 상태(MainState)와 확장된 변수들을 조합하여
        고농축 사용자 수퍼 텍스트를 생성합니다."""
        up = state.get('user_profile', {})
        pj, po, pr = up.get('pj', {}), up.get('po', {}), up.get('pr', {})
        
        recom = state.get('recommendations', [])
        target_roles = ", ".join([r.get('변환된_직무명', '') for r in recom if r.get('변환된_직무명')])
        cat_names = [self.cat_code_to_name.get(str(c), "기타") for c in state.get('job_category_codes', [])]

        super_text = f"""
        [사용자 페르소나 및 희망 직무]
        지원자는 {", ".join(pr.get('growth_goal', []))}를 목표로 하며, {target_roles} 직무({", ".join(cat_names)})를 희망합니다.

        [확장된 역량 및 기술 스택]
        - 지식: {", ".join(expanded_vars.get('knowledge', []))}
        - 기술: {", ".join(expanded_vars.get('skills', []))}
        - 능력: {", ".join(expanded_vars.get('abilities', []))}
        - 관심 산업: {", ".join(expanded_vars.get('industry_interest', []))}

        [조직 적합성]
        중요 가치: {", ".join(po.get('values', []))}. 선호 문화: {po.get('culture', '')}. 동기: {", ".join(po.get('motivation', []))}.
        """.strip()
        return super_text
    
    def process_agent_state(self, state: dict) -> dict:
        """최종 호출 메서드: 모든 툴 기능을 순차적으로 실행"""
        pj, po = state['user_profile'].get('pj', {}), state['user_profile'].get('po', {})
        
        # 1. 변수 확장 (LLM)
        expanded = {
            'knowledge': self.expand_similar_terms(pj.get('knowledge', []), "knowledge"),
            'skills': self.expand_similar_terms(pj.get('skills', []), "skills"),
            'abilities': self.expand_similar_terms(pj.get('abilities', []), "abilities"),
            'industry_interest': self.expand_similar_terms(po.get('industry_interest', []), "industry_interest")
        }
        
        # 2. 정규화
        edu_code, exp_years = self.normalize_education_experience(state['user_profile'].get('bi', {}))
        raw_codes = state.get('job_category_codes', [])
        category_names = [self.cat_code_to_name.get(str(code), "기타") for code in raw_codes]
        recom = state.get('recommendations', [])
        target_roles = [r.get('변환된_직무명', '') for r in recom if r.get('변환된_직무명')]
        # 3. 수퍼 텍스트 및 임베딩
        user_super_text = self.build_super_user_text(state, expanded)
        embedding = self.get_embedding(user_super_text)

        return {
            'exp_years': exp_years,
            'edu_code': edu_code,
            'category_names': category_names,
            'target_roles': target_roles,
            'embedding': embedding,
            'super_text': user_super_text,
            'expanded_vars': expanded
        }