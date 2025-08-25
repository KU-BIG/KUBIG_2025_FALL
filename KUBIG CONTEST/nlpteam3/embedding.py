"""
법률 문서 FAISS 인덱스 구축 시스템
- nlpai-lab/KURE-v1 모델을 사용한 임베딩
- FAISS CPU 인덱스 구축
- 커스텀 ID 변환으로 IDSelectorRange 지원
- 법률 문서 청크별 임베딩 벡터 생성

주요 기능:
1. 법률 문서 JSONL 파싱 및 청크 생성
2. KURE-v1 모델로 임베딩 생성
3. FAISS 인덱스 구축 및 저장
4. 커스텀 ID 매핑 시스템

사용법:
1. 인덱스 구축:
   python rag.py

2. 프로그래밍 방식:
   from rag import LawRAG
   rag = LawRAG()
   rag.build_index("data/parsed")
   rag.save_index("faiss_index.bin", "metadata.json")

참고: 검색(retrieval) 기능은 별도 파일에 구현됩니다.
"""

import os
import json
import re
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc


class LawRAG:
    def __init__(self, model_name: str = "nlpai-lab/KURE-v1", batch_size: Optional[int] = None):
        """
        법률 문서 FAISS 인덱스 구축 초기화
        
        Args:
            model_name: 사용할 임베딩 모델명
            batch_size: 임베딩 배치 크기 (None일 경우 디바이스에 따라 자동 설정)
        """
        # GPU/CPU 자동 감지
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU 감지: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # GPU 배치 크기 자동 설정
            if batch_size is None:
                if gpu_memory >= 16:  # 16GB 이상
                    self.batch_size = 128
                elif gpu_memory >= 8:  # 8GB 이상
                    self.batch_size = 64
                else:  # 8GB 미만
                    self.batch_size = 32
            else:
                self.batch_size = batch_size
        else:
            self.device = torch.device("cpu")
            print("GPU를 사용할 수 없습니다. CPU로 실행합니다.")
            self.batch_size = batch_size if batch_size is not None else 16  # CPU는 작은 배치
        
        print(f"사용 디바이스: {self.device}")
        print(f"배치 크기: {self.batch_size}")
        print(f"임베딩 모델 로딩 중: {model_name}")
        
        # SentenceTransformer의 cache_folder 사용
        cache_dir = "models"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"모델 캐시 위치: {cache_dir}/")
        
        self.model = SentenceTransformer(
            model_name, 
            device=self.device, 
            cache_folder=cache_dir
        )
        print(f"임베딩 모델 로드 완료")
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.id_to_faiss_id = {}  # original_id -> faiss_id 매핑
        self.faiss_id_to_data = {}  # faiss_id -> chunk_data 매핑
        self.chunks = []
        
        print(f"임베딩 차원: {self.dimension}")
        
        # GPU 메모리 정리
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def generate_faiss_ids(self, chunks: List[Dict]) -> List[Dict]:
        """
        청크들에 대해 FAISS ID를 순서대로 생성
        
        규칙:
        1. 맨 앞에 9를 붙임
        2. H(항) 부분까지의 ID 그대로 사용
        3. 뒤에 순서대로 3자리 번호 추가 (000, 001, 002, ...)
        
        예시:
        - "001540-0001001-H01" -> "9001540000100101000"
        - "001540-0002001-H01-O01" -> "9001540000200101001"  
        - "001540-0002001-H01-O01-02" -> "9001540000200101002"
        
        Args:
            chunks: 청크 데이터 리스트
            
        Returns:
            FAISS ID가 추가된 청크 리스트
        """
        # H 부분까지 추출하는 함수
        def extract_base_id(original_id: str) -> str:
            parts = original_id.split('-')
            if len(parts) < 3:
                raise ValueError(f"Invalid ID format: {original_id}")
            
            # H 부분까지만: law_id + article_key + H + hang_no
            law_id = parts[0]
            article_key = parts[1] 
            h_part = parts[2]  # H01, H02 등
            
            base_id = f"{law_id}{article_key}{h_part.replace('H', '')}"
            return base_id
        
        # 청크들을 처리하여 FAISS ID 생성
        for idx, chunk in enumerate(chunks):
            base_id = extract_base_id(chunk['original_id'])
            # 9 + base_id + 3자리 순서번호
            faiss_id_str = f"9{base_id}{idx:03d}"
            
            try:
                chunk['faiss_id'] = int(faiss_id_str)
            except ValueError:
                raise ValueError(f"Cannot convert to integer: {faiss_id_str} from {chunk['original_id']}")
        
        return chunks
    
    def load_jsonl_files(self, data_dir: str = "data/parsed") -> List[Dict]:
        """
        JSONL 파일들을 로드하여 청크 데이터 생성
        
        Args:
            data_dir: JSONL 파일들이 있는 디렉토리
            
        Returns:
            청크 데이터 리스트
        """
        chunks = []
        jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {data_dir}")
        
        print(f"발견된 JSONL 파일: {len(jsonl_files)}개")
        
        for file_path in jsonl_files:
            print(f"로딩 중: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # 필수 필드 확인
                        required_fields = ['id', 'law_name', 'context', 'article_label_jo', 'article_label_hang', 'text']
                        if not all(field in data for field in required_fields):
                            print(f"Warning: Missing required fields in line {line_num} of {file_path}")
                            continue
                        
                        # 청크 텍스트 생성: law_name + context + article_label_jo + article_label_hang + text
                        chunk_text = " ".join([
                            data['law_name'],
                            data['context'],
                            data['article_label_jo'], 
                            data['article_label_hang'],
                            data['text']
                        ]).strip()
                        
                        chunk_data = {
                            'original_id': data['id'],
                            'chunk_text': chunk_text,
                            'metadata': data
                        }
                        
                        chunks.append(chunk_data)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
        
        print(f"총 {len(chunks)}개 청크 로드 완료")
        return chunks
    
    def build_index(self, data_dir: str = "data/parsed"):
        """
        FAISS 인덱스 구축 (CPU)
        
        Args:
            data_dir: JSONL 파일들이 있는 디렉토리
        """
        # 데이터 로드
        self.chunks = self.load_jsonl_files(data_dir)
        
        if not self.chunks:
            raise ValueError("No chunks loaded")
        
        # FAISS ID 생성 (순서대로)
        print("FAISS ID 생성 중...")
        self.chunks = self.generate_faiss_ids(self.chunks)
        
        # 처음 몇 개 ID 확인
        print("생성된 FAISS ID 예시:")
        for i, chunk in enumerate(self.chunks[:10]):
            print(f"  {chunk['original_id']} -> {chunk['faiss_id']}")
        if len(self.chunks) > 10:
            print(f"  ... 총 {len(self.chunks)}개 청크")
        
        # 매핑 테이블 생성
        for chunk in self.chunks:
            self.id_to_faiss_id[chunk['original_id']] = chunk['faiss_id']
            self.faiss_id_to_data[chunk['faiss_id']] = chunk
        
        print("임베딩 생성 중...")
        texts = [chunk['chunk_text'] for chunk in self.chunks]
        
        # GPU 메모리 정리 (GPU 사용 시)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"GPU 메모리 사용량 (임베딩 전): {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 배치 단위로 임베딩 생성
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True  # 자동 L2 정규화
        )
        
        print(f"임베딩 완료: {embeddings.shape}")
        
        # GPU 메모리 정리 (임베딩 후)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"GPU 메모리 사용량 (임베딩 후): {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # FAISS 인덱스 생성 (정확한 cosine similarity 검색을 위해 FlatIP 사용)
        print("FAISS 인덱스 생성 중...")
        cpu_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(cpu_index)
        
        # 임베딩을 인덱스에 추가
        faiss_ids = np.array([chunk['faiss_id'] for chunk in self.chunks], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype(np.float32), faiss_ids)
        
        print(f"FAISS 인덱스 구축 완료: {self.index.ntotal}개 벡터")
        
        # 메모리 정리
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def save_index(self, index_path: str = "faiss_index.bin", metadata_path: str = "metadata.json"):
        """
        FAISS 인덱스와 메타데이터 저장
        
        Args:
            index_path: FAISS 인덱스 저장 경로
            metadata_path: 메타데이터 저장 경로
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, index_path)
        
        # 메타데이터 저장
        metadata = {
            'id_to_faiss_id': self.id_to_faiss_id,
            'faiss_id_to_data': {str(k): v for k, v in self.faiss_id_to_data.items()},  # JSON 호환성을 위해 키를 문자열로
            'model_name': self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else 'nlpai-lab/KURE-v1',
            'dimension': self.dimension
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"인덱스 저장 완료: {index_path}")
        print(f"메타데이터 저장 완료: {metadata_path}")
    
    def save_local(self, db_dir: str = "db"):
        """
        FAISS 벡터스토어를 db/ 폴더에 저장
        
        Args:
            db_dir: 데이터베이스 저장 디렉토리
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # db 디렉토리 생성
        os.makedirs(db_dir, exist_ok=True)
        
        # 파일 경로 설정
        index_path = os.path.join(db_dir, "faiss_index.bin")
        metadata_path = os.path.join(db_dir, "metadata.json")
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, index_path)
        print(f"FAISS 인덱스 저장: {index_path}")
        
        # 메타데이터 저장 (확장된 정보 포함)
        metadata = {
            'id_to_faiss_id': self.id_to_faiss_id,
            'faiss_id_to_data': {str(k): v for k, v in self.faiss_id_to_data.items()},
            'model_name': 'nlpai-lab/KURE-v1',
            'dimension': self.dimension,
            'total_chunks': len(self.chunks),
            'index_type': 'FlatIP_IDMap',
            'device': str(self.device),
            'batch_size': self.batch_size,
            'created_at': str(os.path.getctime(index_path)) if os.path.exists(index_path) else None
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"메타데이터 저장: {metadata_path}")
        
        # 추가 정보 파일 저장
        info_path = os.path.join(db_dir, "index_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=== 법률 문서 FAISS 벡터스토어 ===\n\n")
            f.write(f"생성 일시: {metadata.get('created_at', 'Unknown')}\n")
            f.write(f"총 청크 수: {len(self.chunks)}\n")
            f.write(f"임베딩 차원: {self.dimension}\n")
            f.write(f"모델: nlpai-lab/KURE-v1\n")
            f.write(f"사용 디바이스: {self.device}\n")
            f.write(f"배치 크기: {self.batch_size}\n")
            f.write(f"인덱스 타입: FlatIP+IDMap (정확한 유사도 검색)\n\n")
            
            # 법률별 청크 수 통계
            law_stats = {}
            for chunk in self.chunks:
                law_name = chunk['metadata']['law_name']
                law_stats[law_name] = law_stats.get(law_name, 0) + 1
            
            f.write("=== 법률별 청크 통계 ===\n")
            for law_name, count in law_stats.items():
                f.write(f"{law_name}: {count}개 청크\n")
            
            f.write(f"\n=== ID 생성 규칙 ===\n")
            f.write("9 + H부분까지ID + 순서번호(3자리)\n")
            f.write("예: 001540-0001001-H01 -> 9001540000100101000\n")
        
        print(f"정보 파일 저장: {info_path}")
        print(f"\n벡터스토어가 {db_dir}/ 폴더에 저장되었습니다!")
        
        return {
            'index_path': index_path,
            'metadata_path': metadata_path,
            'info_path': info_path,
            'total_chunks': len(self.chunks)
        }
    
    def get_stats(self):
        """
        구축된 인덱스 통계 정보 반환
        
        Returns:
            Dict: 인덱스 통계
        """
        if self.index is None:
            return {"message": "인덱스가 구축되지 않았습니다."}
        
        return {
            "총_벡터_수": self.index.ntotal,
            "임베딩_차원": self.dimension,
            "총_청크_수": len(self.chunks),
            "배치_크기": self.batch_size,
            "모델명": "nlpai-lab/KURE-v1",
            "인덱스_타입": "FlatIP_IDMap",
            "사용_디바이스": str(self.device)
        }

if __name__ == "__main__":
    
    # RAG 시스템 초기화 (GPU 우선, 없으면 CPU)
    print("=== 법률 문서 임베딩 & FAISS 인덱스 구축 ===\n")
    rag = LawRAG()  # 배치 크기 자동 설정
    
    # 인덱스 구축
    rag.build_index("data/parsed")
    
    # db/ 폴더에 벡터스토어 저장
    save_result = rag.save_local("db")
    
    print(f"\n=== 벡터스토어 구축 완료! ===")
    print(f"저장 위치: db/")
    print(f"총 청크 수: {save_result['total_chunks']}")
    print(f"사용 디바이스: {rag.device}")
    print(f"배치 크기: {rag.batch_size}")
    
    # 통계 정보 출력
    stats = rag.get_stats()
    print(f"\n=== 벡터스토어 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # GPU 사용 시 최종 메모리 상태
    if rag.device.type == "cuda":
        print(f"\n=== GPU 메모리 상태 ===")
        print(f"할당된 메모리: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"캐시된 메모리: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
