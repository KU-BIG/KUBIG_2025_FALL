#!/usr/bin/env python3
"""
리뷰 데이터를 한 번만 분석해서 저장하는 스크립트
실행 후에는 저장된 결과만 로드해서 빠르게 사용 가능
"""

import pandas as pd
import numpy as np
import pickle
import faiss
import os
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_review_index():
    """리뷰 데이터를 분석하고 인덱스를 구축합니다."""
    
    # 1. 리뷰 데이터 로드
    logger.info("리뷰 데이터 로드 중...")
    try:
        reviews_df = pd.read_csv('ratings.txt', sep='\t', encoding='utf-8')
        logger.info(f"리뷰 데이터 로드 완료: {len(reviews_df)}개")
    except Exception as e:
        logger.error(f"리뷰 데이터 로드 실패: {e}")
        return False
    
    # 2. 임베딩 모델 로드
    logger.info("임베딩 모델 로드 중...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("임베딩 모델 로드 완료")
    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        return False
    
    # 3. 리뷰 텍스트 전처리
    logger.info("리뷰 텍스트 전처리 중...")
    review_texts = reviews_df['document'].tolist()
    
    # 4. 임베딩 생성 (배치 처리)
    logger.info("리뷰 임베딩 생성 중...")
    batch_size = 1000
    all_embeddings = []
    
    for i in tqdm(range(0, len(review_texts), batch_size), desc="임베딩 생성"):
        batch_texts = review_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"임베딩 생성 완료: {embeddings.shape}")
    
    # 5. FAISS 인덱스 생성
    logger.info("FAISS 인덱스 생성 중...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    logger.info("FAISS 인덱스 생성 완료")
    
    # 6. 메타데이터 준비
    logger.info("메타데이터 준비 중...")
    metadata = []
    for idx, row in reviews_df.iterrows():
        metadata.append({
            'review_id': row['id'],
            'text': row['document'],
            'label': row['label'],
            'index': idx
        })
    
    # 7. 영화 매칭 (간단한 키워드 기반)
    logger.info("영화 매칭 중...")
    movie_mapping = {}
    
    # 기존 영화 데이터 로드
    try:
        movies_df = pd.read_csv('movie_data/movies.csv')
        movie_titles = movies_df['title'].tolist()
        
        for review_idx, review in enumerate(metadata):
            review_text = review['text'].lower()
            
            # 영화 제목이 리뷰에 포함되어 있는지 확인
            matched_movies = []
            for movie_idx, title in enumerate(movie_titles):
                title_lower = title.lower()
                if title_lower in review_text:
                    matched_movies.append(movie_idx)
            
            # 매칭된 영화가 있으면 첫 번째 영화에 연결
            if matched_movies:
                movie_id = matched_movies[0]
                if movie_id not in movie_mapping:
                    movie_mapping[movie_id] = []
                movie_mapping[movie_id].append(review_idx)
    
    except Exception as e:
        logger.warning(f"영화 매칭 실패: {e}")
    
    # 8. 결과 저장
    logger.info("결과 저장 중...")
    result = {
        'index': index,
        'metadata': metadata,
        'movie_mapping': movie_mapping,
        'total_reviews': len(metadata),
        'embedding_dimension': dimension
    }
    
    # 저장 디렉토리 생성
    os.makedirs('movie_data', exist_ok=True)
    
    # pickle로 저장
    with open('movie_data/reviews_analysis.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    logger.info("리뷰 분석 완료!")
    logger.info(f"총 리뷰 수: {len(metadata)}")
    logger.info(f"매칭된 영화 수: {len(movie_mapping)}")
    logger.info(f"저장 위치: movie_data/reviews_analysis.pkl")
    
    return True

def test_loaded_index():
    """저장된 인덱스를 테스트합니다."""
    logger.info("저장된 인덱스 테스트 중...")
    
    try:
        with open('movie_data/reviews_analysis.pkl', 'rb') as f:
            data = pickle.load(f)
        
        index = data['index']
        metadata = data['metadata']
        
        logger.info(f"인덱스 크기: {index.ntotal}")
        logger.info(f"메타데이터 수: {len(metadata)}")
        
        # 간단한 검색 테스트
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_query = "재미있는 영화"
        query_embedding = model.encode([test_query])
        
        scores, indices = index.search(query_embedding.astype('float32'), 5)
        
        logger.info("검색 테스트 결과:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata):
                review = metadata[idx]
                logger.info(f"{i+1}. 점수: {score:.3f}, 리뷰: {review['text'][:50]}...")
        
        logger.info("인덱스 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"인덱스 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("=== 리뷰 인덱스 구축 스크립트 ===")
    
    # 이미 존재하는지 확인
    if os.path.exists('movie_data/reviews_analysis.pkl'):
        print("이미 리뷰 인덱스가 존재합니다.")
        choice = input("다시 구축하시겠습니까? (y/N): ")
        if choice.lower() != 'y':
            print("기존 인덱스를 테스트합니다.")
            test_loaded_index()
            exit()
    
    # 리뷰 인덱스 구축
    success = build_review_index()
    
    if success:
        print("\n=== 구축 완료! ===")
        test_loaded_index()
    else:
        print("\n=== 구축 실패! ===") 