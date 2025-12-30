import pandas as pd
import sqlite3
import re
import os

def _parse_career_from_requirements(requirements: str, user_exp: int) -> bool:
    """자격 요건 텍스트 내 경력 제약 조건 검증"""
    if not requirements or pd.isna(requirements): return True
    req_lower = str(requirements).lower()
    
    # n년 이하/이상 패턴 체크
    max_match = re.search(r'(\d+)\s*년\s*이하', req_lower)
    if max_match and user_exp > int(max_match.group(1)): return False
    min_match = re.search(r'(\d+)\s*년\s*이상', req_lower)
    if min_match and user_exp < int(min_match.group(1)): return False
    return True

def filter_jobs_from_db(db_path: str, exp_years: int, edu_code: int, category_names: list = None, target_roles: list = None) -> pd.DataFrame:
    """
    변환된 카테고리 명칭(category_names)을 사용하여 DB 필터링 수행
    target_roles: 직무명 키워드(예: '데이터 분석')
    """
    if not os.path.exists(db_path): return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    
    career_limit = exp_years + 2
    
    # 1. 기초 SQL 필터링
    query = """
        SELECT j.*, r.role_name, r.tech_stack, r.main_tasks, r.requirements, r.domain, r.embedding
        FROM jobs j
        JOIN job_roles r ON j.rec_idx = r.rec_idx
        WHERE (j.career_min IS NULL OR j.career_min <= ?)
          AND (j.edu_code IS NULL OR j.edu_code <= ?)
    """
    params = [exp_years, edu_code]

    # 2. 카테고리 필터링: 이미 텍스트로 변환된 이름을 사용하여 IN 절 구성
    filter_conditions = []
    if category_names:
        placeholders = ', '.join(['?'] * len(category_names))
        filter_conditions.append(f"j.category IN ({placeholders})")
        params.extend(category_names)
        
    if target_roles:
        # 직무명 키워드(예: '데이터 분석')가 포함된 공고 추가 검색
        role_conditions = " OR ".join(["r.role_name LIKE ?"] * len(target_roles))
        filter_conditions.append(f"({role_conditions})")
        params.extend([f"%{role}%" for role in target_roles])

    if filter_conditions:
        query += f" AND ({' OR '.join(filter_conditions)})"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty: return df

    # 3. 상세 텍스트 기반 2차 필터링
    df_result = df[df.apply(lambda row: _parse_career_from_requirements(row['requirements'], exp_years), axis=1)].copy()
    
    return df_result