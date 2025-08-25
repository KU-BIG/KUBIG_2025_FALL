"""
메인 실행 파일: task/test.csv의 TEST_000 문제 처리
"""

import os
import yaml
import pandas as pd
from retriever import LawFAISSRetriever, is_multiple_choice, extract_question_and_choices, rewrite_query_mcq, rewrite_query_subjective
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langsmith import traceable
from langsmith.run_trees import RunTree


# 설정 및 프롬프트 로드
with open("settings.yaml", 'r', encoding='utf-8') as f:
    SETTINGS = yaml.safe_load(f)

with open("prompts.yaml", 'r', encoding='utf-8') as f:
    PROMPTS = yaml.safe_load(f)

# 설정 추출
GENERATION_CONFIG = SETTINGS['generation_config']
CONTEXT_CONFIG = SETTINGS['context_config']
SEARCH_CONFIG = SETTINGS['search_config']
MODEL_CONFIG = SETTINGS['model_config']
LANGSMITH_CONFIG = SETTINGS['langsmith_config']

# LangSmith 환경변수 설정
def setup_langsmith():
    """
    LangSmith 추적을 위한 환경변수 설정
    """
    os.environ['LANGCHAIN_PROJECT'] = LANGSMITH_CONFIG['LANGSMITH_PROJECT']
    os.environ['LANGCHAIN_API_KEY'] = LANGSMITH_CONFIG['LANGSMITH_API_KEY']
    os.environ['LANGCHAIN_TRACING_V2'] = LANGSMITH_CONFIG['LANGSMITH_TRACING']
    os.environ['LANGCHAIN_ENDPOINT'] = LANGSMITH_CONFIG['LANGSMITH_ENDPOINT']
    
    print(f"LangSmith 추적 활성화")

# LangSmith 설정
setup_langsmith()


def format_context(docs, max_length: int = None) -> str:
    """
    검색된 문서들을 컨텍스트로 형식화
    
    Args:
        docs: 검색된 Document 객체 리스트
        max_length: 컨텍스트 최대 길이 (None이면 설정값 사용)
        
    Returns:
        형식화된 컨텍스트
    """
    if max_length is None:
        max_length = CONTEXT_CONFIG['max_context_length']
        
    if not docs:
        return "관련 법률 조문을 찾을 수 없습니다."
    
    formatted = []
    current_length = 0
    
    for doc in docs:
        doc_text = f"[{doc.metadata['article']}] {doc.metadata['source']}\n{doc.page_content}"
        
        # 길이 제한 체크
        if current_length + len(doc_text) > max_length:
            # 남은 공간이 truncate_threshold보다 클 때만 잘라서 포함
            remaining_space = max_length - current_length
            if remaining_space > CONTEXT_CONFIG['truncate_threshold']:
                truncated_content = doc.page_content[:remaining_space-50] + "..."
                doc_text = f"[{doc.metadata['article']}] {doc.metadata['source']}\n{truncated_content}"
                formatted.append(doc_text)
            break
        
        formatted.append(doc_text)
        current_length += len(doc_text)
    
    return "\n\n".join(formatted)


def get_prompt_by_type(context: str, question: str, question_type: str) -> str:
    """
    질문 유형에 따른 프롬프트 생성
    
    Args:
        context: 검색된 법률 조문들
        question: 질문 내용
        question_type: "객관식" 또는 "주관식"
        
    Returns:
        해당 유형에 맞는 프롬프트
    """
    if question_type == "객관식":
        template = PROMPTS['mcq_prompt']
    else:
        template = PROMPTS['subjective_prompt']
    
    return template.format(context=context, question=question)


class LocalLLM:
    """
    설정 가능한 로컬 LLM 클래스 (HuggingFace 모델)
    """
    
    def __init__(self):
        """
        LLM 모델 초기화 (settings.yaml에서 모든 설정 로드)
        """
        self.model_name = MODEL_CONFIG['llm_model_name']
        self.model_dir = MODEL_CONFIG['llm_model_dir']
        
        # GPU 사용 가능성 체크
        device_str = MODEL_CONFIG['device']
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = getattr(torch, MODEL_CONFIG['torch_dtype'])
            print(f"GPU 사용: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"데이터 타입: {self.torch_dtype} (GPU)")
        else:
            self.device = torch.device("cpu")
            # CPU에서는 float32 사용
            self.torch_dtype = torch.float32
            print(f"CPU 사용 (GPU 사용 불가 또는 설정)")
            print(f"데이터 타입: {self.torch_dtype} (CPU - float16 미지원)")
        
        
        print(f"LLM 모델: {self.model_name}")
        
        self.load_model()
    
    def load_model(self):
        """
        LLM 모델 로드 (HuggingFace 캐시 시스템 활용, 중복 제거)
        """
        print(f"모델 로드: {self.model_name}")
        print(f"캐시 위치: models/")
        
        try:
            # HuggingFace 표준 캐시 시스템 사용 (중복 저장 없음)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="models"  # models/models--rtzr--ko-gemma-2-9b-it/ 형태
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device.type == "cuda" else str(self.device),
                low_cpu_mem_usage=True,
                cache_dir="models"
            )
            print(f"모델 로드 완료")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
        
        self.model.eval()
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # GPU 메모리 정리 (GPU 사용 시만)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 메모리 - 사용중: {allocated:.2f}GB, 예약: {reserved:.2f}GB")
    
    def generate(self, prompt: str) -> str:
        """
        텍스트 생성 (YAML 설정 사용)
        
        Args:
            prompt: 입력 프롬프트
            
        Returns:
            생성된 텍스트
        """
        # 토크나이징 (YAML 설정 사용)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=GENERATION_CONFIG['max_input_tokens']
        )
        
        # 입력 텐서를 모델과 같은 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성 (YAML 설정 사용)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + GENERATION_CONFIG['max_output_tokens'],
                temperature=GENERATION_CONFIG['temperature'],
                do_sample=GENERATION_CONFIG['do_sample'],
                top_p=GENERATION_CONFIG.get('top_p', 0.9),
                repetition_penalty=GENERATION_CONFIG.get('repetition_penalty', 1.1),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩 (입력 제외하고 생성된 부분만)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # GPU 메모리 정리 (GPU 사용 시만)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text.strip()


def create_law_rag_chain():
    """
    법률 RAG 체인 생성 (Ko-Gemma 사용, YAML 설정 사용)
    
    Returns:
        RAG 체인 함수
    """
    # Retriever 초기화 (YAML 설정 사용)
    retriever = LawFAISSRetriever(db_dir=MODEL_CONFIG['vectorstore_dir'], k=SEARCH_CONFIG['stage1_k'])
    
    # LLM 초기화 (settings.yaml에서 모델 자동 선택)
    llm = LocalLLM()
    
    def rag_pipeline(question: str) -> str:
        """
        RAG 파이프라인 실행 (2단계 검색 + rule based rewriting)
        
        Args:
            question: 사용자 질문
            
        Returns:
            LLM 답변
        """
        # 1. 쿼리 재작성
        if is_multiple_choice(question):
            print("객관식 질문 감지 - 선지번호 제거")
            q_text, choices = extract_question_and_choices(question)
            rewritten_query = rewrite_query_mcq(q_text, choices)
            question_type = "객관식"
            print(f"재작성: '{rewritten_query[:60]}...'")
        else:
            print("주관식 질문 감지 - 원본 그대로 사용")
            rewritten_query = rewrite_query_subjective(question)
            question_type = "주관식"
        
        # 2. 2단계 검색 실행 - LangSmith 추적
        @traceable(name="법률_문서_검색", run_type="retriever")
        def search_documents(query: str, question_type: str):
            print("2단계 검색 실행")
            docs = retriever.search_two_stage(query, k=SEARCH_CONFIG['stage1_k'])
            return {
                "docs": docs,
                "docs_count": len(docs),
                "question_type": question_type
            }
        
        search_result = search_documents(rewritten_query, question_type)
        docs = search_result["docs"]
        
        # 3. 컨텍스트 형식화 (YAML 설정 사용)
        context = format_context(docs, max_length=CONTEXT_CONFIG['max_context_length'])
        
        # 4. 질문 유형별 프롬프트 생성
        prompt = get_prompt_by_type(context, question, question_type)
        
        # 5. LLM 답변 생성 - LangSmith 추적
        @traceable(name="LLM_답변생성", run_type="llm")
        def generate_answer(prompt_text: str, question_type: str):
            print(f"{MODEL_CONFIG['llm_model_name']} 답변 생성 중...")
            answer = llm.generate(prompt_text)
            return {
                "answer": answer,
                "answer_length": len(answer),
                "model": MODEL_CONFIG['llm_model_name'],
                "question_type": question_type
            }
        
        llm_result = generate_answer(prompt, question_type)
        answer = llm_result["answer"]
        
        return answer
    
    return rag_pipeline


@traceable(name="법률_RAG_전체_프로세스", run_type="chain")
def process_test_345():
    """
    task/test.csv에서 TEST_345 문제 처리 (LangSmith 추적 포함)
    """
    print("TEST_345 문제 처리")
    print("=" * 60)
    
    # 1. test.csv 로드
    test_df = pd.read_csv("task/test.csv")
    
    # TEST_000 문제 찾기
    test_000_row = test_df[test_df['ID'] == 'TEST_345'].iloc[0]
    question_id = test_000_row['ID']
    question = test_000_row['Question']
    
    print(f"문제 ID: {question_id}")
    print(f"문제 내용:")
    print("-" * 40)
    print(question)
    print("-" * 40)
    
    # 2. RAG 체인 생성 (YAML 설정 사용)
    print("\n시스템 초기화 중...")
    rag_chain = create_law_rag_chain()
    
    # 3. 문제 처리
    print(f"\n{question_id} 문제 처리 시작:")
    try:
        answer = rag_chain(question)
        
        print(f"\n{MODEL_CONFIG['llm_model_name']} 답변:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
        result = {
            'question_id': question_id,
            'question': question,
            'answer': answer,
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        print(f"오류 발생: {e}")
        
        error_result = {
            'question_id': question_id,
            'question': question,
            'answer': f"처리 실패: {e}",
            'status': 'error',
            'error': str(e)
        }
        
        return error_result


if __name__ == "__main__":
    # TEST_345 문제 처리
    result = process_test_345()
    
    print(f"\n처리 완료: {result['question_id']}")
