"""
LangChain을 사용한 법률 문서 FAISS Retriever
- embedding.py로 구축된 벡터스토어 로드
- Vector Search + IDSelectorRange 지원
- 사용자 쿼리 직접 검색 시스템
"""

import os
import json
import re
import yaml
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class LawFAISSRetriever(BaseRetriever):
    """
    법률 문서 FAISS 벡터스토어를 사용하는 LangChain Retriever
    """
    
    def __init__(self, db_dir: str = "db", k: int = 5):
        """
        법률 FAISS Retriever 초기화
        
        Args:
            db_dir: 벡터스토어가 저장된 디렉토리
            k: 기본 반환할 문서 수
        """
        super().__init__()
        
        # Pydantic 호환을 위해 object.__setattr__ 사용
        object.__setattr__(self, 'db_dir', db_dir)
        object.__setattr__(self, 'k', k)
        
        # 설정 로드
        self.load_settings()
        
        # 벡터스토어 로드
        self.load_vectorstore()
    
    def load_settings(self):
        """settings.yaml에서 검색 설정 로드"""
        try:
            with open("settings.yaml", 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            
            search_config = settings.get('search_config', {})
            object.__setattr__(self, 'stage1_k', search_config.get('stage1_k', 5))
            object.__setattr__(self, 'stage2_k', search_config.get('stage2_k', 50))
            object.__setattr__(self, 'stage3_target_k', search_config.get('stage3_target_k', 5))
            object.__setattr__(self, 'stage3_other_k', search_config.get('stage3_other_k', 5))
            
            print(f"검색 설정 로드 완료: 1단계={self.stage1_k}, 2단계={self.stage2_k}, 타겟={self.stage3_target_k}, 기타={self.stage3_other_k}")
            
        except Exception as e:
            print(f"설정 로드 실패: {e}, 기본값 사용")
            object.__setattr__(self, 'stage1_k', 5)
            object.__setattr__(self, 'stage2_k', 50)
            object.__setattr__(self, 'stage3_target_k', 5)
            object.__setattr__(self, 'stage3_other_k', 5)
        
    def load_vectorstore(self):
        """
        db/ 폴더에서 FAISS 벡터스토어 로드
        """
        index_path = os.path.join(self.db_dir, "faiss_index.bin")
        metadata_path = os.path.join(self.db_dir, "metadata.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS 인덱스를 찾을 수 없습니다: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터를 찾을 수 없습니다: {metadata_path}")
        
        # FAISS 인덱스 로드
        print(f"FAISS 인덱스 로딩 중: {index_path}")
        object.__setattr__(self, 'index', faiss.read_index(index_path))
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        object.__setattr__(self, 'faiss_id_to_data', {int(k): v for k, v in metadata['faiss_id_to_data'].items()})
        object.__setattr__(self, 'dimension', metadata['dimension'])
        object.__setattr__(self, 'total_chunks', metadata.get('total_chunks', len(self.faiss_id_to_data)))
        
        # 인덱스 정보 로드
        object.__setattr__(self, 'index_type', metadata.get('index_type', 'IDMap'))
        
        # 임베딩 모델 로드
        model_name = metadata.get('model_name', 'nlpai-lab/KURE-v1')
        print(f"임베딩 모델 로드: {model_name}")
        print(f"캐시 위치: models/")
        
        # SentenceTransformer의 cache_folder 사용
        cache_dir = "models"
        os.makedirs(cache_dir, exist_ok=True)
        
        object.__setattr__(self, 'model', SentenceTransformer(
            model_name, 
            device="cpu", 
            cache_folder=cache_dir  # models/sentence-transformers_nlpai-lab_KURE-v1/ 형태
        ))
        print(f"임베딩 모델 로드 완료")
        
        print(f"벡터스토어 로드 완료: {self.index.ntotal}개 벡터")
        print(f"임베딩 차원: {self.dimension}")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        LangChain 호환 문서 검색 (2단계 검색 사용)
        
        Args:
            query: 검색 쿼리
            run_manager: LangChain 콜백 매니저
            
        Returns:
            관련 문서 리스트
        """
        # 2단계 검색 전략 사용
        return self.search_two_stage(query, k=self.k)
    
    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        쿼리에 대한 유사 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (None이면 기본값 사용)
            
        Returns:
            LangChain Document 객체 리스트
        """
        if k is None:
            k = self.k
            
        # 쿼리 임베딩
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # FAISS 검색
        scores, ids = self.index.search(query_embedding.astype(np.float32), k)
        
        # 결과를 LangChain Document 형태로 변환
        documents = []
        for i, (score, faiss_id) in enumerate(zip(scores[0], ids[0])):
            if faiss_id == -1:  # 검색 결과가 없는 경우
                continue
                
            chunk_data = self.faiss_id_to_data.get(faiss_id)
            if chunk_data is None:
                continue
            
            # LangChain Document 생성
            doc = Document(
                page_content=chunk_data['chunk_text'],
                metadata={
                    'source': chunk_data['metadata']['law_name'],
                    'article': chunk_data['metadata']['article_label_jo'],
                    'paragraph': chunk_data['metadata']['article_label_hang'],
                    'context': chunk_data['metadata']['context'],
                    'original_id': chunk_data['original_id'],
                    'faiss_id': faiss_id,
                    'score': float(score),
                    'rank': i + 1
                }
            )
            documents.append(doc)
        
        return documents
    
    def search_by_id_range(self, query: str, start_id: int, end_id: int, k: Optional[int] = None) -> List[Document]:
        """
        FAISS IDSelectorRange를 직접 사용한 특정 범위 검색
        
        Args:
            query: 검색 쿼리
            start_id: 시작 FAISS ID
            end_id: 끝 FAISS ID  
            k: 반환할 문서 수
            
        Returns:
            LangChain Document 객체 리스트
        """
        if k is None:
            k = self.k
            
        # 쿼리 임베딩
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # FlatIP 인덱스는 IDSelectorRange를 지원하지 않으므로 수동 필터링 사용
        print(f"범위 검색: {start_id}-{end_id} (수동 필터링 방식)")
        
        # 충분한 후보를 가져와서 필터링
        search_k = min(k * 20, self.index.ntotal)
        scores, ids = self.index.search(query_embedding.astype(np.float32), search_k)
        
        # 범위 내 결과만 필터링
        filtered_results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if start_id <= faiss_id <= end_id:
                filtered_results.append((score, faiss_id))
            if len(filtered_results) >= k:
                break
        
        scores = np.array([[r[0] for r in filtered_results]])
        ids = np.array([[r[1] for r in filtered_results]])
        
        # 결과를 LangChain Document 형태로 변환
        documents = []
        for i, (score, faiss_id) in enumerate(zip(scores[0], ids[0])):
            if faiss_id == -1:
                continue
                
            chunk_data = self.faiss_id_to_data.get(faiss_id)
            if chunk_data is None:
                continue
            
            doc = Document(
                page_content=chunk_data['chunk_text'],
                metadata={
                    'source': chunk_data['metadata']['law_name'],
                    'article': chunk_data['metadata']['article_label_jo'],
                    'paragraph': chunk_data['metadata']['article_label_hang'],
                    'context': chunk_data['metadata']['context'],
                    'original_id': chunk_data['original_id'],
                    'faiss_id': faiss_id,
                    'score': float(score),
                    'rank': i + 1
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_law_id_from_original_id(self, original_id: str) -> str:
        """
        원본 ID에서 law_id 추출
        
        Args:
            original_id: 원본 청크 ID (예: "001540-0001001-H01")
            
        Returns:
            law_id (예: "001540")
        """
        return original_id.split('-')[0]
    
    def get_article_info_from_original_id(self, original_id: str) -> Tuple[str, str]:
        """
        원본 ID에서 법률 ID와 조문키 추출
        
        Args:
            original_id: 원본 청크 ID (예: "001540-0001001-H01")
            
        Returns:
            (law_id, article_key) 튜플 (예: ("001540", "0001001"))
        """
        parts = original_id.split('-')
        if len(parts) < 2:
            raise ValueError(f"Invalid original_id format: {original_id}")
        return parts[0], parts[1]
    
    def get_id_range_for_article(self, law_id: str, article_key: str) -> Tuple[int, int]:
        """
        특정 조항(law_id + article_key)에 해당하는 FAISS ID 범위 계산
        
        Args:
            law_id: 법률 ID (예: "001540")
            article_key: 조문키 (예: "0001001")
            
        Returns:
            (start_id, end_id) 튜플
        """
        # 해당 조항의 FAISS ID 패턴: 9{law_id}{article_key}{hang_no}{idx:03d}
        # 예: 9001540000100101000
        base_pattern = f"9{law_id.zfill(6)}{article_key.zfill(7)}"
        
        # 해당 조항의 모든 항(H01, H02, ...)을 포함하도록 범위 설정
        start_id = int(f"{base_pattern}00000")  # H00부터 (실제로는 H01부터 시작)
        end_id = int(f"{base_pattern}99999")    # H99의 마지막 청크까지
        
        return start_id, end_id
    
    def filter_docs_by_article_id_range(self, docs: List[Document], target_law_id: str, target_article_key: str) -> List[Document]:
        """
        문서들을 특정 조항의 FAISS ID 범위로 필터링
        
        Args:
            docs: 필터링할 문서 리스트
            target_law_id: 타겟 법률 ID
            target_article_key: 타겟 조문키
            
        Returns:
            필터링된 문서 리스트
        """
        start_id, end_id = self.get_id_range_for_article(target_law_id, target_article_key)
        
        filtered_docs = []
        for doc in docs:
            faiss_id = doc.metadata.get('faiss_id')
            if faiss_id and start_id <= faiss_id <= end_id:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def search_two_stage(self, query: str, k: int = 5) -> List[Document]:
        """
        새로운 3단계 검색 전략 (최대 15개 청크):
        1단계: 전체에서 설정값(stage1_k) 검색 
        2단계: 전체에서 설정값(stage2_k) 후보 검색 후 조항별 ID 범위 필터링
        - 가장 많이 언급된 조항에서 설정값(stage3_target_k)개
        - 다른 조항들에서 추가 설정값(stage3_other_k)개
        
        스코어 조건: 1단계 최고 스코어가 0.6 미만이면 전체에서 top_k=15만 반환
        
        Args:
            query: 검색 쿼리
            k: 무시됨 (설정값 사용)
            
        Returns:
            검색 결과 (최대 stage1_k + stage3_target_k + stage3_other_k개)
        """
        print("=== 새로운 3단계 검색 전략 실행 ===")
        print(f"설정: 1단계={self.stage1_k}, 2단계={self.stage2_k}, 타겟={self.stage3_target_k}, 기타={self.stage3_other_k}")
        
        # 1단계: 전체에서 설정값 검색
        print(f"1단계: 전체에서 top {self.stage1_k}개 검색")
        stage1_docs = self.search(query, k=self.stage1_k)
        
        if not stage1_docs:
            print("1단계 검색 결과가 없습니다.")
            return stage1_docs
        
        # 스코어 체크: 최고 스코어가 0.6 미만이면 단순히 top 15개만 반환
        max_score = stage1_docs[0].metadata['score']
        print(f"1단계 최고 스코어: {max_score:.4f}")
        
        if max_score < 0.6:
            print("최고 스코어가 0.6 미만 -> 전체에서 top 15개만 반환")
            fallback_docs = self.search(query, k=15)
            print(f"Fallback 검색 결과: {len(fallback_docs)}개")
            return fallback_docs
        
        # 2단계: 전체에서 후보 검색
        print(f"2단계: 전체에서 top {self.stage2_k}개 검색")
        stage2_docs = self.search(query, k=self.stage2_k)
        
        # 1단계에서 조항별 빈도 계산 (ID 기반)
        article_counts = Counter()
        article_to_rank = {}  # 조항별 최고 순위 저장
        article_to_info = {}  # 조항별 (law_id, article_key) 저장
        
        for doc in stage1_docs:
            try:
                law_id, article_key = self.get_article_info_from_original_id(doc.metadata['original_id'])
                article_id = f"{law_id}-{article_key}"
                article_counts[article_id] += 1
                article_to_info[article_id] = (law_id, article_key)
                
                # 최고 순위 저장 (낮은 rank가 높은 순위)
                if article_id not in article_to_rank or doc.metadata['rank'] < article_to_rank[article_id]:
                    article_to_rank[article_id] = doc.metadata['rank']
                    
            except Exception as e:
                print(f"조항 정보 추출 실패: {doc.metadata['original_id']} - {e}")
                continue
        
        # 가장 많이 언급된 조항 찾기 (동률 시 순위 우선)
        if not article_counts:
            print("조항 정보를 찾을 수 없습니다.")
            return stage1_docs
        
        max_count = max(article_counts.values())
        top_articles = [article_id for article_id, count in article_counts.items() if count == max_count]
        
        # 동률일 경우 순위가 높은(낮은 rank 값) 조항 선택
        target_article_id = min(top_articles, key=lambda x: article_to_rank[x])
        target_law_id, target_article_key = article_to_info[target_article_id]
        
        print(f"가장 많이 언급된 조항: {target_article_id} ({max_count}회, rank {article_to_rank[target_article_id]})")
        
        # 3단계: 조항별 ID 범위 필터링
        print("3단계: 조항별 ID 범위 필터링 및 선정")
        
        # 1단계 문서의 FAISS ID 수집 (중복 방지)
        stage1_faiss_ids = {doc.metadata['faiss_id'] for doc in stage1_docs}
        
        # 2단계 문서들을 조항별로 분류 (ID 범위 기반)
        target_article_docs = []
        other_article_docs = []
        
        for doc in stage2_docs:
            if doc.metadata['faiss_id'] in stage1_faiss_ids:
                continue  # 1단계에 이미 포함된 문서는 제외
            
            # 해당 문서가 타겟 조항 범위에 속하는지 ID로 확인
            try:
                doc_law_id, doc_article_key = self.get_article_info_from_original_id(doc.metadata['original_id'])
                if doc_law_id == target_law_id and doc_article_key == target_article_key:
                    target_article_docs.append(doc)
                else:
                    other_article_docs.append(doc)
            except Exception as e:
                print(f"문서 조항 분류 실패: {doc.metadata['original_id']} - {e}")
                other_article_docs.append(doc)  # 에러 시 기타로 분류
        
        # 각 카테고리에서 설정값만큼 선정
        target_selected = target_article_docs[:self.stage3_target_k]
        other_selected = other_article_docs[:self.stage3_other_k]
        
        print(f"타겟 조항 ({target_article_id}): {len(target_selected)}개 선정")
        print(f"기타 조항: {len(other_selected)}개 선정")
        
        # 최종 결합
        all_docs = stage1_docs + target_selected + other_selected
        
        # 점수순으로 재정렬 (rank 정보 업데이트)
        all_docs.sort(key=lambda x: x.metadata['score'], reverse=True)
        for i, doc in enumerate(all_docs):
            doc.metadata['rank'] = i + 1
        
        max_possible = self.stage1_k + self.stage3_target_k + self.stage3_other_k
        print(f"최종 검색 결과: {len(all_docs)}개/{max_possible}개 (1단계: {len(stage1_docs)}, 타겟조항: {len(target_selected)}, 기타: {len(other_selected)})")
        
        return all_docs
    
    def get_id_range_for_law(self, law_id: str) -> Tuple[int, int]:
        """
        특정 법률 ID에 해당하는 FAISS ID 범위 계산
        
        Args:
            law_id: 법률 ID (예: "001540")
            
        Returns:
            (start_id, end_id) 튜플
        """
        # 해당 law_id로 시작하는 모든 FAISS ID 범위
        law_id_padded = law_id.zfill(6)
        start_id = int(f"9{law_id_padded}0000000000")  # 최소값
        end_id = int(f"9{law_id_padded}9999999999")    # 최대값
        
        return start_id, end_id


def is_multiple_choice(question_text: str) -> bool:
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    
    Args:
        question_text: 질문 텍스트
        
    Returns:
        객관식 여부
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


def extract_question_and_choices(full_text: str) -> Tuple[str, List[str]]:
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    
    Args:
        full_text: 전체 질문 텍스트
        
    Returns:
        (질문 본문, 선택지 리스트) 튜플
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options


def rewrite_query_mcq(question: str, choices: List[str]) -> str:
    """
    객관식 질문을 검색에 최적화된 쿼리로 재작성
    선지번호 제거하고 콤마로 구분
    
    Args:
        question: 질문 본문
        choices: 선택지 리스트
        
    Returns:
        재작성된 검색 쿼리
    """
    # 선택지에서 번호 제거
    clean_choices = []
    for choice in choices:
        # "1 소비자금융업" → "소비자금융업"
        clean_choice = re.sub(r"^\s*[1-9][0-9]?\s*", "", choice).strip()
        if clean_choice:
            clean_choices.append(clean_choice)
    
    # 질문 + 콤마로 구분된 선택지
    choices_text = ", ".join(clean_choices)
    rewritten = f"{question} {choices_text}"
    
    return rewritten


def rewrite_query_subjective(question: str) -> str:
    """
    주관식 질문은 그대로 사용
    
    Args:
        question: 질문 텍스트
        
    Returns:
        원본 질문 (수정 없음)
    """
    return question


if __name__ == "__main__":

    retriever = LawFAISSRetriever(db_dir="db", k=5)
    
    # 간단한 검색 테스트
    docs = retriever.search("정보집합물의 결합 시, 결합의뢰기관이 데이터전문기관에 제공하는 정보집합물에 대해 취해야 할 조치로 옳은 것은?", k=5)
    print(f"검색 결과: {len(docs)}개 문서")
    
    for doc in docs:
        print(f"- {doc.metadata['source']}")
        print(f"- {doc.metadata['article']}")
        print(f"- {doc.page_content}")
    
    print("Retriever 로드 완료")
