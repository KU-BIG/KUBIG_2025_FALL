import numpy as np
import pandas as pd

def compute_posting_similarity(df_filtered: pd.DataFrame, user_embedding: np.ndarray, top_n: int = 20) -> pd.DataFrame:
    """
    사용자 임베딩과 공고 임베딩을 비교하여 유사도 점수를 계산하여 상위 N개를 반환합니다.
    """
    if df_filtered.empty or user_embedding is None:
        return df_filtered

    # 1. 공고 임베딩 변환 (BLOB -> NumPy float32)
    # DB에서 불러온 2진 데이터를 1536차원 벡터로 복원합니다.
    try:
        job_embeddings = np.array([
            np.frombuffer(row['embedding'], dtype=np.float32) 
            for _, row in df_filtered.iterrows()
        ], dtype=np.float32)
    except Exception:
        return pd.DataFrame()

    # 2. 사용자 임베딩 정제 (이미 생성된 벡터를 float32 배열로 확정)
    user_vec = np.array(user_embedding, dtype=np.float32).flatten()

    # 3. 차원 검증 (OpenAI text-embedding-3-small 기준 1536차원)
    if job_embeddings.shape[1] != 1536 or user_vec.shape[0] != 1536:
        # 차원이 맞지 않으면 계산 불가하므로 빈 데이터프레임 반환
        return pd.DataFrame()

    # 4. 코사인 유사도 계산 (벡터 정규화 후 내적)
    job_norms = job_embeddings / np.linalg.norm(job_embeddings, axis=1, keepdims=True)
    user_norm = user_vec / np.linalg.norm(user_vec)
    
    scores = np.dot(job_norms, user_norm)
    
    # 5. 결과 데이터프레임 구성
    df_filtered['similarity_score'] = scores
    
    # 중복 공고 제거 및 상위 정렬
    top_df = (
        df_filtered.sort_values('similarity_score', ascending=False)
        .groupby('rec_idx')
        .first()
        .reset_index()
        .head(top_n)
    )
    
    return top_df