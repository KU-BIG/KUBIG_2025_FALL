from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
import os
from pathlib import Path

#main.py에서 ncs_df pandas dataframe으로 넣어줘야됨

class NCSJobRecommender:
    # 직무/직업 코드 매핑 테이블
    JOB_CATEGORY_CODES = {
        '기획·전략': 16,
        '마케팅·홍보·조사': 14,
        '회계·세무·재무': 3,
        '인사·노무·HRD': 5,
        '총무·법무·사무': 4,
        'IT개발·데이터': 2,
        '디자인': 15,
        '영업·판매·무역': 8,
        '고객상담·TM': 21,
        '구매·자재·물류': 18,
        '상품기획·MD': 12,
        '운전·운송·배송': 7,
        '서비스': 10,
        '생산': 11,
        '건설·건축': 22,
        '의료': 6,
        '연구·R&D': 9,
        '교육': 19,
        '미디어·문화·스포츠': 13,
        '금융·보험': 17,
        '공공·복지': 20
    }
    
    def __init__(self, ncs_df=None, vectorstore_path=None):
        load_dotenv()
        
        # 1. 경로 자동 설정 로직
        if vectorstore_path is None:
            current_dir = Path(__file__).parent
            # data 폴더 안의 ncs_vectorstore 경로로 지정
            self.vectorstore_path = str(current_dir.parent / "data" / "ncs_vectorstore")
        else:
            self.vectorstore_path = vectorstore_path
        
        self.ncs_df = ncs_df
        
        # 2. 모델 초기화
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", 
                              temperature=0.3,
                              max_tokens=4000)
        self.vectorstore = None
        
        # State 저장용
        self.job_category_codes = []
        
    def load_vectorstore(self):
        
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(
                f"벡터스토어를 찾을 수 없습니다: {self.vectorstore_path}\n"
                "prepare_vectorstore(build_new=True)로 먼저 생성하세요."
            )
        
        print(f"📦 벡터스토어 로드 중: {self.vectorstore_path}")
        
        self.vectorstore = FAISS.load_local(
            folder_path=self.vectorstore_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✓ 벡터스토어 로드 완료\n")
        
        return self.vectorstore

    def prepare_vectorstore(self, build_new=False):
        """벡터스토어 준비 (로드 또는 생성)"""
        
        if os.path.exists(self.vectorstore_path) and not build_new:
            return self.load_vectorstore()
        
        if self.ncs_df is None:
            raise ValueError(
                "ncs_df가 제공되지 않았습니다. "
                "새 벡터스토어를 생성하려면 ncs_df를 전달해야 합니다."
            )
        
        print("🔧 새로운 벡터스토어 생성 중...")

        documents = []
        metadata_list = []

        # 1. 텍스트 문서 생성
        for idx, row in self.ncs_df.iterrows():
            job_description = f"""
            직무명: {row['세분류코드명']}

            필요 지식/기술/태도:
            {row['지식기술태도의의']}
"""
            documents.append(job_description)
            metadata_list.append({
                'index': idx, 
                '직무명': row['세분류코드명']
            })
        
        print(f"📝 총 {len(documents)}개의 직무를 임베딩 중...")
        self.vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=self.embedding_model,
            metadatas=metadata_list
        )

        Path(self.vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.vectorstore_path)
        print(f"✓ 벡터 DB 저장 완료: {self.vectorstore_path}\n")

        return self.vectorstore
    
    def retrieve_candidate_jobs(self, user_input, k=15, use_mmr=True, lambda_mult=0.5):
        """1차 필터링: 유사도 기반 직무 candidate 검색"""
        
        if self.vectorstore is None:
            raise ValueError(
                "Vectorstore가 초기화되지 않았습니다. "
                "prepare_vectorstore()를 먼저 실행하세요."
            )
        
        if use_mmr:
            print(f"   🔄 MMR 검색 (λ={lambda_mult})")
            
            search_results = self.vectorstore.max_marginal_relevance_search(
                query=user_input,
                k=k,
                fetch_k=k * 3,
                lambda_mult=lambda_mult
            )
            
            candidates = []
            for doc in search_results:
                candidates.append({
                    '직무명': doc.metadata['직무명'],
                    '유사도_점수': None,
                    '설명': doc.page_content
                })
        else:
            print("   🔍 일반 유사도 검색")
            
            search_results = self.vectorstore.similarity_search_with_score(
                query=user_input,
                k=k
            )
            
            candidates = []
            for result, score in search_results:
                candidates.append({
                    '직무명': result.metadata['직무명'],
                    '유사도_점수': float(1 - score),
                    '설명': result.page_content
                })
        
        return candidates
    
    def rerank_with_llm(self, user_input, candidates, top_k=7):
        """2차 정제: LLM을 활용한 재랭킹
        단순 키워드 매칭이 아니라, 문맥(Context)을 이해하여 순위를 다시 매깁니다.
        왜 추천했는지 '추천 이유'와 '부족한 부분'을 텍스트로 생성해줍니다.
        
        """
        
        candidate_list = "\n\n".join([
            f"{i+1}. {jd['직무명']}\n{jd['설명'][:300]}..."
            for i, jd in enumerate(candidates)
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 취업 준비생을 위한 전문 커리어 컨설턴트입니다.
사용자의 경험과 관심사를 분석하여 가장 적합한 직무를 추천해주세요.

추천 시 고려사항:
1. 사용자의 프로젝트 경험, 관심분야와 직무의 필요 지식/기술의 매칭도
2. 사용자의 강점이 발휘될 수 있는 직무
3. 현실적으로 진입 가능한 직무 (너무 동떨어진 직무는 제외)

반드시 JSON 형식으로 응답하세요."""),
            ("user", """
사용자 정보: 
{user_input}

추천 후보 직무들:
{candidate_list}

위 후보 직무들 중에서 사용자에게 가장 적합한 {top_k}개의 직무를 선정하고, 
각 직무별로 추천 이유를 구체적으로 설명해주세요.

응답 형식:
{{
    "recommendations": [
        {{
            "rank": 1,
            "직무명": "직무명",
            "추천_이유": "사용자 정보와 관련된 구체적인 추천 이유 (3-4문장)",
            "핵심_연관성": ["연관성1", "연관성2", "연관성3"],
            "부족한_부분": "보완이 필요한 영역 제안"
        }},
        ...
    ]
}}

주의: JSON 형식 외의 다른 텍스트는 출력하지 마세요.
""")
        ])

        chain = prompt | self.llm | JsonOutputParser()

        result = chain.invoke({
            "user_input": user_input,
            "candidate_list": candidate_list,
            "top_k": top_k
        })
        
        return result
    
    def map_to_job_categories(self, recommendations: dict, user_input: str) -> list:
        """추천된 직무를 분석하여 직무 카테고리 코드 매핑"""
        
        print("\n[🔍 직무 카테고리 매핑 중...]")
        
        recommended_jobs = []
        if recommendations and 'recommendations' in recommendations:
            for rec in recommendations['recommendations']:
                recommended_jobs.append({
                    'rank': rec['rank'],
                    '직무명': rec['직무명'],
                    '추천_이유': rec['추천_이유'],
                    '핵심_연관성': rec['핵심_연관성']
                })
        
        categories_list = list(self.JOB_CATEGORY_CODES.keys())
        
        prompt = f"""다음은 사용자에게 추천된 NCS 직무들입니다.

[사용자 정보]
{user_input[:500]}

[추천된 직무들]
{chr(10).join([f"{job['rank']}. {job['직무명']}: {job['추천_이유']}" for job in recommended_jobs[:5]])}

[직무 카테고리 선택지 (21개)]
{chr(10).join([f"- {cat}" for cat in categories_list])}

위 추천 직무들과 가장 연관성이 높은 직무 카테고리를 **최대 2개** 선택하세요.

선택 기준:
1. **사용자의 [관심 도메인/희망 산업]에 해당하는 카테고리를 최우선으로 선택하세요.**
2. 추천된 직무들이 공통적으로 속한 카테고리
3. 사용자의 경험과 관심사에 부합하는 카테고리

다음 JSON 형식으로만 응답하세요:
{{
    "selected_categories": ["카테고리1", "카테고리2"],
    "reason": "선택 이유 (1문장)"
}}

주의: 
- 반드시 위 21개 카테고리 중에서만 선택
- 최대 2개까지만 선택
- JSON 형식 외의 다른 텍스트 출력 금지
"""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = JsonOutputParser().parse(response.content)
            
            selected_categories = result.get('selected_categories', [])
            reason = result.get('reason', '')
            
            category_codes = []
            for cat in selected_categories[:2]:
                if cat in self.JOB_CATEGORY_CODES:
                    category_codes.append(self.JOB_CATEGORY_CODES[cat])
            
            self.job_category_codes = category_codes
            
            print(f"✓ 선택된 카테고리: {selected_categories}")
            print(f"✓ 카테고리 코드: {category_codes}")
            print(f"✓ 선택 이유: {reason}")
            
            return category_codes
            
        except Exception as e:
            print(f"⚠️  카테고리 매핑 실패: {e}")
            return []
    
    def transform_job_names(self, recommendations: dict, user_input: str) -> dict:
        """NCS 직무명을 실무 채용 공고 스타일로 변환 (전 산업 분야 대응)"""
        
        print("\n[✨ 직무명 변환 중...]")
        
        if not recommendations or 'recommendations' not in recommendations:
            return recommendations
        
        jobs_info = []
        for job in recommendations['recommendations']:
            jobs_info.append({
                'rank': job['rank'],
                'NCS_직무명': job['직무명'],
                '추천_이유': job['추천_이유']
            })
        
        jobs_text = "\n\n".join([
            f"[{job['rank']}위] NCS 원본명: {job['NCS_직무명']}\n추천 이유: {job['추천_이유']}"
            for job in jobs_info
        ])
        
        # [수정] 다양한 직군(문과, 예체능, 이공계 등)을 모두 포괄하는 프롬프트
        prompt = f"""당신은 전 산업 분야의 전문 헤드헌터입니다.
NCS(국가직무능력표준)의 딱딱한 행정 용어를 실제 대한민국 채용사이트의 **'채용 공고'** 스타일의 세련된 직무명으로 통역하세요.

[사용자 프로필]
{user_input[:500]}

[추천된 NCS 직무들]
{jobs_text}

[변환 규칙]
1. **NCS 명칭을 절대 그대로 사용하지 마세요**
2. 현업에서 가장 통용되는 직무명을 사용하세요
3. 영어 직무명이 보편적이라면 영어를 메인으로 사용하세요

[변환 예시]
- 빅데이터분석 → 데이터 분석가 (Data Analyst)
- 인공지능모델링 → AI/머신러닝 엔지니어 (ML Engineer)
- 시각디자인 → UI/UX 디자이너
- 영상연출 → 영상 PD / 콘텐츠 크리에이터
- 해외영업 → Global Sales Manager
- 응용SW엔지니어링 → 백엔드 개발자 (Backend Developer)
- 웹개발 → 프론트엔드 개발자 (Frontend Developer)

JSON 형식으로 출력:
{{
    "transformed_jobs": [
        {{
            "rank": 1,
            "변환된_직무명": "실무에서 쓰이는 직무명"
        }},
        ...
    ]
}}
"""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = JsonOutputParser().parse(response.content)
            
            transform_map = {
                item['rank']: item['변환된_직무명'] 
                for item in result.get('transformed_jobs', [])
            }
            
            enhanced_recommendations = recommendations.copy()
            for rec in enhanced_recommendations['recommendations']:
                rec['변환된_직무명'] = transform_map.get(rec['rank'], rec['직무명'])
            
            print(f"✓ 직무명 변환 완료")
            
            return enhanced_recommendations
            
        except Exception as e:
            print(f"⚠️  직무명 변환 실패: {e}")
            for rec in recommendations['recommendations']:
                rec['변환된_직무명'] = rec['직무명']
            return recommendations
    
    def generate_keywords(self, recommendations: dict, user_input: str) -> dict:
        """추천된 직무에 대해 관련 키워드를 생성"""
        
        print("\n[🗝️  관련 키워드 생성 중...]")
        
        if not recommendations or 'recommendations' not in recommendations:
            return recommendations
        
        jobs_for_keywords = []
        for job in recommendations['recommendations']:
            jobs_for_keywords.append({
                'rank': job['rank'],
                '직무명': job.get('변환된_직무명', job['직무명']),
                '추천_이유': job['추천_이유'],
                '핵심_연관성': job['핵심_연관성']
            })
        
        prompt = f"""다음은 사용자에게 추천된 직무들입니다.
각 직무마다 관련성 높은 키워드 3~5개를 추출하세요.

[사용자 프로필]
{user_input[:500]}

[추천된 직무들]
{chr(10).join([f"{job['rank']}. {job['직무명']}: {job['추천_이유']}" for job in jobs_for_keywords])}

각 직무별로 다음을 고려하여 키워드를 생성하세요:
1. 직무명과 관련된 핵심 기술/도구
2. 사용자의 경험과 연결되는 키워드
3. 해당 직무 검색에 유용한 키워드

다음 JSON 형식으로 응답하세요:
{{
    "keywords": [
        {{
            "rank": 1,
            "keywords": ["#키워드1", "#키워드2", "#키워드3"]
        }},
        ...
    ]
}}

주의:
- 키워드는 반드시 #으로 시작
- 각 직무당 3~5개의 키워드
- JSON 형식만 출력
"""

        try:
            response = self.llm.invoke([SystemMessage(content=prompt)])
            result = JsonOutputParser().parse(response.content)
            
            keyword_map = {item['rank']: item['keywords'] for item in result.get('keywords', [])}
            
            enhanced_recommendations = recommendations.copy()
            for rec in enhanced_recommendations['recommendations']:
                rec['관련_키워드'] = keyword_map.get(rec['rank'], [])
            
            print(f"✓ 키워드 생성 완료")
            
            return enhanced_recommendations
            
        except Exception as e:
            print(f"⚠️  키워드 생성 실패: {e}")
            return recommendations
    
    def recommend_jobs(self, user_input, top_k=7, use_mmr=True, lambda_mult=0.5):
        """전체 추천 프로세스 (터미널 출력 포함)"""
        
        print("\n" + "="*70)
        print("🎯 NCS 직무 추천 시스템")
        print("="*70)
        
        search_method = f"MMR (다양성 고려, λ={lambda_mult})" if use_mmr else "일반 유사도"
        print(f"\n🔍 검색 방식: {search_method}")
        
        # 1단계: 벡터 검색
        print(f"\n[1단계] 벡터 검색 중...")
        candidates = self.retrieve_candidate_jobs(
            user_input, 
            k=15,
            use_mmr=use_mmr,
            lambda_mult=lambda_mult
        )
        print(f"✓ {len(candidates)}개 후보 직무 추출 완료")
        
        # 2단계: LLM 재랭킹
        print(f"\n[2단계] LLM 재랭킹 중 (최종 {top_k}개 선정)...")
        final_recommendations = self.rerank_with_llm(
            user_input, 
            candidates, 
            top_k
        )
        print(f"✓ 재랭킹 완료")
        
        # 3단계: 직무 카테고리 매핑
        self.map_to_job_categories(final_recommendations, user_input)
        
        # 4단계: 직무명 변환 ✅
        enhanced_recommendations = self.transform_job_names(final_recommendations, user_input)
        
        # 5단계: 관련 키워드 생성
        enhanced_recommendations = self.generate_keywords(enhanced_recommendations, user_input)
        
        # 6단계: 결과 출력
        self._print_recommendations(enhanced_recommendations)
        
        return {
            'recommendations': enhanced_recommendations,
            'job_category_codes': self.job_category_codes
        }
    
    def recommend_from_persona(self, persona_data: dict, top_k=7, use_mmr=True, lambda_mult=0.5):
        """페르소나 데이터를 기반으로 직무 추천"""
        
        user_input = self._format_persona_to_text(persona_data)
        
        return self.recommend_jobs(
            user_input=user_input,
            top_k=top_k,
            use_mmr=use_mmr,
            lambda_mult=lambda_mult
        )
    
    def _format_persona_to_text(self, persona_data: dict) -> str:
        """
        PJ(직무)/PO(조직)/PR(현실) 구조의 JSON을 자연어 텍스트로 변환
        """
        parts = []
        
        # 1. PJ (Job Fit): 보유 역량
        pj = persona_data.get('pj', {})
        if pj:
            parts.append("=== [PJ] 보유 직무 역량 ===")
            if pj.get('knowledge'):
                # 리스트가 아니라 문자열이 들어올 경우 대비
                k_list = pj['knowledge'] if isinstance(pj['knowledge'], list) else [pj['knowledge']]
                parts.append(f"- 보유 지식: {', '.join(k_list)}")
            if pj.get('skills'):
                s_list = pj['skills'] if isinstance(pj['skills'], list) else [pj['skills']]
                parts.append(f"- 보유 기술/스킬: {', '.join(s_list)}")
            if pj.get('abilities'):
                a_list = pj['abilities'] if isinstance(pj['abilities'], list) else [pj['abilities']]
                parts.append(f"- 주요 태도/능력: {', '.join(a_list)}")
            parts.append("")

        # 2. PO (Org Fit): 가치관 및 동기
        po = persona_data.get('po', {})
        if po:
            parts.append("=== [PO] 조직/가치관 적합성 ===")
            if po.get('values'):
                v_list = po['values'] if isinstance(po['values'], list) else [po['values']]
                parts.append(f"- 직업 가치관: {', '.join(v_list)}")
            if po.get('industry_interest'):
                i_list = po['industry_interest'] if isinstance(po['industry_interest'], list) else [po['industry_interest']]
                parts.append(f"- 관심 산업: {', '.join(i_list)}")
            if po.get('motivation'):
                m_list = po['motivation'] if isinstance(po['motivation'], list) else [po['motivation']]
                parts.append(f"- 업무 동기: {', '.join(m_list)}")
            parts.append("")

        # 3. PR (Reality): 현실적 조건
        pr = persona_data.get('pr', {})
        if pr:
            parts.append("=== [PR] 희망 조건 및 목표 ===")
            if pr.get('growth_goal'):
                g_list = pr['growth_goal'] if isinstance(pr['growth_goal'], list) else [pr['growth_goal']]
                parts.append(f"- 성장 목표: {', '.join(g_list)}")
            if pr.get('priority'):
                parts.append(f"- 우선순위: {pr['priority']}")
            parts.append("")

        return "\n".join(parts)
    
    def _print_recommendations(self, recommendations):
        """추천 결과를 터미널에 출력"""
        
        print("\n" + "="*70)
        print("📊 추천 결과")
        print("="*70)
        
        if not recommendations or 'recommendations' not in recommendations:
            print("\n❌ 추천 결과가 없습니다.\n")
            return
        
        # 변환된 직무명으로 출력
        for rec in recommendations['recommendations']:
            print(f"\n{'='*70}")
            print(f"🏆 {rec['rank']}위. {rec.get('변환된_직무명', rec['직무명'])}")
            print(f"{'='*70}")
            
            print(f"\n💡 추천 이유:")
            print(f"   {rec['추천_이유']}")
            
            print(f"\n✅ 핵심 연관성:")
            for conn in rec['핵심_연관성']:
                print(f"   • {conn}")
            
            if rec.get('부족한_부분'):
                print(f"\n⚠️  보완이 필요한 부분:")
                print(f"   {rec['부족한_부분']}")
            
            if rec.get('관련_키워드'):
                print(f"\n🗝️  관련 키워드:")
                print(f"   {' '.join(rec['관련_키워드'])}")
        
        print("\n" + "="*70)
        
        if self.job_category_codes:
            print("\n🏷️  선택된 직무 카테고리")
            print("="*70)
            
            for code in self.job_category_codes:
                category_name = next((k for k, v in self.JOB_CATEGORY_CODES.items() if v == code), "알 수 없음")
                print(f"  • {category_name} (코드: {code})")
            print("="*70)
        
        print("\n✨ 추천 완료!")
        print("="*70 + "\n")
    
    def get_job_category_codes(self) -> list:
        """저장된 직무 카테고리 코드 반환"""
        return self.job_category_codes