import json
from openai import OpenAI
import pandas as pd
from typing import Dict, Any

def review_final_postings(top_jobs_df: pd.DataFrame, user_super_text: str, user_expanded_vars: Dict[str, Any], target_roles: str = None) -> pd.DataFrame:
    """
    상세 직무 변수(Domain, Tech, Tasks)를 활용하여 직무명(target_roles)을 중심으로 LLM이 공고와 사용자의 적합도를 정밀 타격 검토합니다.
    """
    if top_jobs_df.empty:
        return top_jobs_df

    client = OpenAI()
    results = []
    
    # 1. 고도화된 시스템 프롬프트 (사용자 제공 로직 반영)
    system_prompt = """
    너는 채용 공고와 사용자 프로필을 비교하여 적합도를 평가하는 커리어 매칭 전문가야.
    직무명과 도메인 특화 전문가가 되어줘. 사용자의 관심 산업과 공고의 비즈니스 영역 간의 '도메인 공명'을 중요하게 평가해줘. 
    관심 직무명은 {target_roles}이야. 이 직무명과 유사한 공고를 가장 높게 평가해줘. 

    [평가 핵심 가이드]
    1. 일반 공고 최우선 (CRITICAL): 병역특례(산업기능요원, 전문연구요원)가 아닌 '일반 채용 공고'를 가장 높게 평가해. 또한, 지역은 기본적으로 수도권 지역으로 평가해.
    2. 직무 일치도 (CRITICAL): 공고의 'role_name'이 사용자의 '목표 직무({target_roles})'와 의미적으로 일치하는가?
    3. 역량 일치: 사용자의 강점과 기술이 공고의 필수 요구사항과 얼마나 일치하는가?
    4. 기술 전이성: 사용자의 스킬(tech_stack)이 공고의 업무 현장에서 즉시 유용하게 쓰일 수 있는가?
    5. 도메인 공명: 공고의 산업 분야(Domain)가 사용자의 관심사 및 목표와 일치하는가?

    [등급 판정 기준]
    - 상 (강력 추천): 직무명이 일치하고, 기술 스택과 관심 도메인이 모두 일치하거나 그 중 하나가 일치하는 경우.
    - 중 (보통 추천): 직무명과 연관이 있고, 기술적 접점은 있으나 도메인이 생소하거나, 도메인은 일치하나 기술적 보완이 필요한 경우.
    - 하 (약한 추천): 직무명과 연관이 없거나 기술 스택의 간극이 크거나 사용자의 커리어 지향점과 공고의 성격이 상이한 경우.

    [출력 형식]
    반드시 JSON으로만 응답할 것.
    {
        "grade": "상"|"중"|"하",
        "reason": "추천 이유를 1-2문장으로 설명 (등급이 '상'인 경우 매우 상세히, '하'인 경우 보완점 중심으로)"
    }
    """.strip()

    # 2. 개별 공고에 대해 정밀 검토 수행
    for _, row in top_jobs_df.iterrows():
        # 사용자 정보 구조화 (expanded_vars 활용)
        user_info = {
            "직무명": target_roles,
            "지식": ", ".join(user_expanded_vars.get('knowledge', [])),
            "기술스택": ", ".join(user_expanded_vars.get('skills', [])),
            "보유능력": ", ".join(user_expanded_vars.get('abilities', [])),
            "관심도메인": ", ".join(user_expanded_vars.get('industry_interest', [])),
            "수퍼텍스트": user_super_text
        }

        # 공고 정보 구조화 (DB의 세부 컬럼 활용)
        job_info = {
            "회사": row.get('company', ''),
            "직무명": row.get('role_name', row.get('title', '')),
            "도메인": row.get('domain', ''),
            "기술스택": row.get('tech_stack', ''),
            "주요업무": row.get('main_tasks', ''),
            "자격요건": row.get('requirements', ''),
            "상세설명": str(row.get('job_description', ''))[:1000]
        }

        user_prompt = f"""
        [사용자 프로필]
        {json.dumps(user_info, ensure_ascii=False, indent=2)}

        [검토 공고 데이터]
        {json.dumps(job_info, ensure_ascii=False, indent=2)}

        위 두 데이터를 비교하여 정밀한 등급과 추천 사유를 JSON으로 생성해줘.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            ans = json.loads(response.choices[0].message.content)
            results.append({
                'review_grade': ans.get('grade', '중'),
                'ai_recommendation_reason': ans.get('reason', '분석이 완료되었습니다.')
            })
        except Exception:
            results.append({'review_grade': '중', 'ai_recommendation_reason': '매칭 분석 진행 중입니다.'})

    # 3. 결과 데이터프레임 조립 및 최종 정렬
    final_df = top_jobs_df.copy()
    final_df['review_grade'] = [r['review_grade'] for r in results]
    final_df['ai_recommendation_reason'] = [r['ai_recommendation_reason'] for r in results]

    # 등급 순 정렬 (상 > 중 > 하)
    grade_priority = {'상': 1, '중': 2, '하': 3}
    final_df['g_rank'] = final_df['review_grade'].map(grade_priority)
    final_df = final_df.sort_values(['g_rank', 'similarity_score'], ascending=[True, False]).drop(columns=['g_rank'])

    return final_df