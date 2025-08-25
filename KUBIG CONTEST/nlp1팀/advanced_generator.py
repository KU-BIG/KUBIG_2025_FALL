import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import json

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

class AdvancedMovieGenerator:
    """
    고급 영화 추천 생성기
    - Few-shot 예시 포함
    - 체인 오브 쏘트 (Chain of Thought)
    - 고급 프롬프트 엔지니어링
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        고급 영화 추천 생성기 초기화
        
        Args:
            model_name: 사용할 OpenAI 모델명
        """
        self.model_name = model_name
        self.client = None
        self._initialize_client()
        
        # Few-shot 예시들
        self.few_shot_examples = [
            {
                "user_emotion": "이별 후 공허함을 느끼고 있어",
                "emotion_analysis": "슬픔 (0.8), 외로움 (0.6)",
                "movie_context": "1. Eternal Sunshine of the Spotless Mind\n   줄거리: 이별의 아픔을 다루는 영화\n   장르: Drama, Romance, Sci-Fi\n   유사도: 0.85",
                "recommendation": "Eternal Sunshine of the Spotless Mind를 추천합니다. 이별의 아픔을 다루면서도 치유의 메시지를 전하는 영화입니다. 주인공의 감정적 여정을 통해 이별을 받아들이는 방법을 배울 수 있습니다."
            },
            {
                "user_emotion": "스트레스 받아서 웃고 싶어",
                "emotion_analysis": "스트레스 (0.7), 기쁨 (0.3)",
                "movie_context": "1. The Grand Budapest Hotel\n   줄거리: 재미있는 모험과 웃음이 가득한 영화\n   장르: Comedy, Drama\n   유사도: 0.82",
                "recommendation": "The Grand Budapest Hotel을 추천합니다. 재미있는 스토리와 독특한 캐릭터들이 지친 마음을 위로해줄 것입니다. 웃음과 모험이 가득한 이 영화로 스트레스를 해소할 수 있습니다."
            },
            {
                "user_emotion": "새로운 시작을 하고 싶어",
                "emotion_analysis": "희망 (0.8), 기대 (0.6)",
                "movie_context": "1. The Secret Life of Walter Mitty\n   줄거리: 꿈을 향해 나아가는 모험 영화\n   장르: Adventure, Comedy, Drama\n   유사도: 0.88",
                "recommendation": "The Secret Life of Walter Mitty를 추천합니다. 일상에서 벗어나 모험을 시작하는 주인공의 여정을 통해 새로운 시작에 대한 용기를 얻을 수 있습니다."
            }
        ]
    
    def _initialize_client(self):
        """OpenAI 클라이언트를 초기화합니다."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. 모의 응답을 사용합니다.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI 클라이언트 초기화 완료")
    
    def create_chain_of_thought_prompt(self, user_emotion: str, movie_context: str, 
                                     emotion_scores: Dict[str, float]) -> str:
        """
        체인 오브 쏘트 프롬프트를 생성합니다.
        
        Args:
            user_emotion: 사용자 감정
            movie_context: 영화 컨텍스트
            emotion_scores: 감정 분석 결과
            
        Returns:
            체인 오브 쏘트 프롬프트
        """
        # 감정 분석 결과 포맷팅
        emotion_analysis = ", ".join([f"{emotion} ({score:.1f})" for emotion, score in emotion_scores.items() if score > 0.1])
        
        prompt = f"""
당신은 감정 기반 영화 추천 전문가입니다. 단계별로 생각하면서 사용자에게 가장 적합한 영화를 추천해주세요.

## Few-shot 예시들:

{self._format_few_shot_examples()}

## 현재 상황 분석:

사용자 감정: "{user_emotion}"
감정 분석 결과: {emotion_analysis}

검색된 영화들:
{movie_context}

## 단계별 추천 과정:

1단계: 사용자의 감정 상태를 분석하세요.
- 주요 감정: [감정명]
- 감정 강도: [강도]
- 감정의 맥락: [상황 설명]

2단계: 검색된 영화들을 평가하세요.
- 각 영화의 장점과 단점 분석
- 사용자 감정과의 연결점 찾기
- 영화별 적합성 점수 매기기

3단계: 최적의 영화를 선택하세요.
- 가장 적합한 영화: [영화명]
- 선택 이유: [구체적 이유]
- 기대 효과: [사용자가 얻을 수 있는 것]

4단계: 개인화된 추천 메시지를 작성하세요.
- 공감 표현
- 구체적인 추천 이유
- 실용적인 조언

다음 형식으로 답변해주세요:

## 추천 영화

### [영화 제목]
**선택 이유**: [왜 이 영화를 선택했는지 단계별 설명]
**감정적 연결**: [사용자 감정과 영화의 연결점]
**기대 효과**: [이 영화를 통해 얻을 수 있는 것]
**추가 조언**: [실용적인 조언이나 팁]

답변은 친근하고 공감할 수 있는 톤으로 작성해주세요.
"""
        return prompt
    
    def _format_few_shot_examples(self) -> str:
        """Few-shot 예시들을 포맷팅합니다."""
        examples = []
        for i, example in enumerate(self.few_shot_examples, 1):
            examples.append(f"""
예시 {i}:
사용자: "{example['user_emotion']}"
감정 분석: {example['emotion_analysis']}
영화: {example['movie_context']}
추천: {example['recommendation']}
""")
        return "\n".join(examples)
    
    def create_advanced_prompt(self, user_emotion: str, movie_context: str, 
                             emotion_scores: Dict[str, float], search_metadata: Dict = None) -> str:
        """
        고급 프롬프트를 생성합니다.
        
        Args:
            user_emotion: 사용자 감정
            movie_context: 영화 컨텍스트
            emotion_scores: 감정 분석 결과
            search_metadata: 검색 메타데이터
            
        Returns:
            고급 프롬프트
        """
        # 감정 분석 결과
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        emotion_strength = max(emotion_scores.values()) if emotion_scores else 0.0
        
        # 검색 메타데이터
        search_info = ""
        if search_metadata:
            search_info = f"""
검색 메타데이터:
- 검색된 영화 수: {search_metadata.get('total_movies', 0)}
- 평균 유사도 점수: {search_metadata.get('avg_similarity', 0):.3f}
- 장르 분포: {search_metadata.get('genre_distribution', 'N/A')}
"""
        
        prompt = f"""
당신은 감정 기반 영화 추천 전문가입니다. 사용자의 감정을 깊이 이해하고, 검색된 영화들을 분석하여 가장 적합한 영화를 추천해주세요.

## 사용자 정보:
감정: "{user_emotion}"
주요 감정: {primary_emotion} (강도: {emotion_strength:.2f})
감정 분포: {', '.join([f"{emotion}({score:.2f})" for emotion, score in emotion_scores.items() if score > 0.1])}

## 검색된 영화들:
{movie_context}
{search_info}

## 추천 가이드라인:

1. **감정적 연결**: 사용자의 감정과 영화의 주제/분위기가 어떻게 연결되는지 분석
2. **치유/위로 요소**: 영화가 사용자의 감정 상태에 어떤 도움을 줄 수 있는지 설명
3. **실용성**: 영화를 보는 것이 실제로 도움이 될 수 있는 이유
4. **대안 제시**: 필요시 다른 장르나 스타일의 영화도 언급

## 답변 형식:

### 🎬 추천 영화: [영화 제목]

**📊 선택 근거:**
- 감정적 적합성: [구체적 설명]
- 영화의 핵심 메시지: [영화가 전하는 메시지]
- 사용자 상황과의 연결: [왜 이 영화가 적합한지]

**💝 기대 효과:**
- 감정적 변화: [어떤 감정적 변화를 기대할 수 있는지]
- 인사이트: [영화를 통해 얻을 수 있는 깨달음]
- 실용적 조언: [영화 감상 후 할 수 있는 활동]

**🎯 추가 조언:**
- 감상 팁: [영화를 더 잘 감상할 수 있는 방법]
- 후속 활동: [영화 감상 후 할 수 있는 활동]

답변은 따뜻하고 공감할 수 있는 톤으로 작성해주세요.
"""
        return prompt
    
    def generate_chain_of_thought_recommendation(self, user_emotion: str, movie_context: str, 
                                               emotion_scores: Dict[str, float]) -> str:
        """
        체인 오브 쏘트를 사용한 추천을 생성합니다.
        
        Args:
            user_emotion: 사용자 감정
            movie_context: 영화 컨텍스트
            emotion_scores: 감정 분석 결과
            
        Returns:
            생성된 추천
        """
        if self.client is None:
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)
        
        try:
            prompt = self.create_chain_of_thought_prompt(user_emotion, movie_context, emotion_scores)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 감정 기반 영화 추천 전문가입니다. 단계별로 생각하면서 정확하고 공감할 수 있는 추천을 해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            recommendation = response.choices[0].message.content
            logger.info("체인 오브 쏘트 추천 생성 완료")
            return recommendation
            
        except Exception as e:
            logger.error(f"체인 오브 쏘트 추천 생성 중 오류 발생: {e}")
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)
    
    def generate_advanced_recommendation(self, user_emotion: str, movie_context: str, 
                                       emotion_scores: Dict[str, float], 
                                       search_metadata: Dict = None) -> str:
        """
        고급 추천을 생성합니다.
        
        Args:
            user_emotion: 사용자 감정
            movie_context: 영화 컨텍스트
            emotion_scores: 감정 분석 결과
            search_metadata: 검색 메타데이터
            
        Returns:
            생성된 추천
        """
        if self.client is None:
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)
        
        try:
            prompt = self.create_advanced_prompt(user_emotion, movie_context, emotion_scores, search_metadata)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 감정 기반 영화 추천 전문가입니다. 사용자의 감정을 깊이 이해하고 개인화된 추천을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.6
            )
            
            recommendation = response.choices[0].message.content
            logger.info("고급 추천 생성 완료")
            return recommendation
            
        except Exception as e:
            logger.error(f"고급 추천 생성 중 오류 발생: {e}")
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)
    
    def _generate_mock_advanced_recommendation(self, user_emotion: str, movie_context: str, 
                                             emotion_scores: Dict[str, float]) -> str:
        """고급 모의 추천을 생성합니다."""
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        
        mock_recommendations = {
            '슬픔': """
### 🎬 추천 영화: The Perks of Being a Wallflower

**📊 선택 근거:**
- 감정적 적합성: 이별과 상실의 아픔을 다루면서도 치유의 메시지를 전하는 영화입니다.
- 영화의 핵심 메시지: 진정한 우정과 연결의 의미를 통해 치유받을 수 있다는 메시지
- 사용자 상황과의 연결: 어려운 시기를 겪는 주인공의 성장 과정에 공감할 수 있습니다.

**💝 기대 효과:**
- 감정적 변화: 슬픔을 받아들이고 치유받을 수 있는 용기를 얻을 수 있습니다.
- 인사이트: 진정한 우정과 연결의 의미를 깨닫게 됩니다.
- 실용적 조언: 영화 감상 후 자신의 감정을 정리해보세요.

**🎯 추가 조언:**
- 감상 팁: 조용한 환경에서 집중해서 보시면 더 좋습니다.
- 후속 활동: 영화 속 명대사를 메모해보세요.
""",
            '기쁨': """
### 🎬 추천 영화: The Grand Budapest Hotel

**📊 선택 근거:**
- 감정적 적합성: 재미있는 스토리와 독특한 캐릭터들이 지친 마음을 위로해줍니다.
- 영화의 핵심 메시지: 웃음과 모험을 통해 일상의 스트레스를 해소할 수 있다는 메시지
- 사용자 상황과의 연결: 즐거운 시간을 보내며 기분 전환을 할 수 있습니다.

**💝 기대 효과:**
- 감정적 변화: 웃음을 통해 스트레스를 해소하고 기분이 좋아집니다.
- 인사이트: 작은 것들에서도 즐거움을 찾을 수 있다는 것을 배웁니다.
- 실용적 조언: 영화를 보면서 함께 웃어보세요.

**🎯 추가 조언:**
- 감상 팁: 가족이나 친구와 함께 보시면 더 즐거울 수 있습니다.
- 후속 활동: 영화 속 재미있는 장면들을 이야기해보세요.
""",
            '기대': """
### 🎬 추천 영화: The Secret Life of Walter Mitty

**📊 선택 근거:**
- 감정적 적합성: 새로운 시작에 대한 용기와 희망을 주는 영화입니다.
- 영화의 핵심 메시지: 꿈을 향해 나아가는 모험을 통해 새로운 시작을 할 수 있다는 메시지
- 사용자 상황과의 연결: 일상에서 벗어나 모험을 시작하는 주인공의 여정에 공감할 수 있습니다.

**💝 기대 효과:**
- 감정적 변화: 새로운 시작에 대한 두려움을 극복하고 도전할 용기를 얻을 수 있습니다.
- 인사이트: 작은 변화부터 시작해도 괜찮다는 것을 배웁니다.
- 실용적 조언: 영화 감상 후 새로운 목표를 세워보세요.

**🎯 추가 조언:**
- 감상 팁: 영화 속 아름다운 장면들을 사진으로 남겨보세요.
- 후속 활동: 영화에서 영감을 받아 새로운 취미를 시작해보세요.
"""
        }
        
        return mock_recommendations.get(primary_emotion, mock_recommendations['기대'])

if __name__ == "__main__":
    # 테스트
    generator = AdvancedMovieGenerator()
    
    test_emotion = "친구와 다퉈서 속상해"
    test_context = """
1. The Perks of Being a Wallflower
   줄거리: 어려운 시기를 겪는 청소년의 성장 이야기
   장르: Drama, Romance
   유사도 점수: 0.850

2. Inside Out
   줄거리: 감정의 복잡성을 이해하는 애니메이션
   장르: Animation, Adventure, Comedy, Drama, Family
   유사도 점수: 0.820
"""
    test_emotion_scores = {'슬픔': 0.8, '외로움': 0.6, '분노': 0.3}
    
    recommendation = generator.generate_chain_of_thought_recommendation(
        test_emotion, test_context, test_emotion_scores
    )
    print(recommendation) 