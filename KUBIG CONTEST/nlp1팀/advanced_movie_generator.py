import os
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
logger = logging.getLogger(__name__)

class AdvancedMovieGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
        self.few_shot_examples = [
            {
                "user_emotion": "이별 후 공허함을 느끼고 있어",
                "primary_emotion": "슬픔",
                "recommendation": """
**🎬 추천 영화: "라라랜드"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 이별 후 공허함 → 슬픔과 상실감이 주요 감정
- 새로운 시작에 대한 두려움과 희망이 혼재

2단계: 영화 선택 근거
- "라라랜드"는 이별의 아픔을 다루면서도 희망적 메시지 전달
- 음악과 색채로 감정을 표현하는 방식이 위로가 됨
- 꿈과 현실의 갈등을 다루어 공감대 형성

3단계: 구체적 추천 이유
- 현재 상황에서 음악적 위로를 받을 수 있음
- 아름다운 시각적 요소로 감정적 치유 효과
- 새로운 시작에 대한 용기를 얻을 수 있음
                """
            },
            {
                "user_emotion": "스트레스 받아서 웃고 싶어",
                "primary_emotion": "분노",
                "recommendation": """
**🎬 추천 영화: "슈퍼배드"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 스트레스와 분노로 인한 감정적 압박
- 웃음을 통한 스트레스 해소 욕구

2단계: 영화 선택 근거
- "슈퍼배드"는 순수한 코미디로 스트레스 해소에 효과적
- 귀여운 미니언들이 긍정적 에너지 제공
- 복잡한 스토리 없이 즉시 웃음을 유발

3단계: 구체적 추천 이유
- 스트레스 해소를 위한 즉각적인 웃음 효과
- 부담 없는 가벼운 분위기로 마음의 여유 제공
- 긍정적 캐릭터들이 에너지 충전 효과
                """
            },
            {
                "user_emotion": "새로운 시작을 하고 싶어",
                "primary_emotion": "기대",
                "recommendation": """
**🎬 추천 영화: "인터스텔라"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 새로운 시작에 대한 기대와 희망
- 변화와 모험에 대한 열망

2단계: 영화 선택 근거
- "인터스텔라"는 새로운 세계 탐험의 모험을 다룸
- 인간의 무한한 가능성과 도전 정신을 보여줌
- 시각적 스케일과 감동으로 새로운 영감 제공

3단계: 구체적 추천 이유
- 새로운 시작에 대한 용기와 영감을 얻을 수 있음
- 무한한 가능성을 보여주는 스케일감
- 변화와 성장의 메시지가 새로운 시작에 동기 부여
                """
            }
        ]

    def _initialize_client(self):
        """OpenAI 클라이언트를 초기화합니다."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI 클라이언트 초기화 완료")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
                self.client = None
        else:
            logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. 모의 응답을 사용합니다.")
            self.client = None

    def create_chain_of_thought_prompt(self, user_emotion: str, movie_context: str,
                                        emotion_scores: Dict[str, float]) -> str:
        """Chain of Thought 방식의 프롬프트를 생성합니다."""
        
        # Few-shot 예제 포맷팅
        few_shot_examples = self._format_few_shot_examples()
        
        # 주요 감정 도출
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        
        prompt = f"""
당신은 감정 기반 영화 추천 전문가입니다. 사용자의 감정을 분석하고 적절한 영화를 추천해주세요.

{few_shot_examples}

**현재 사용자 감정:**
입력: "{user_emotion}"
주요 감정: {primary_emotion}
감정 점수: {emotion_scores}

**추천할 영화 정보:**
{movie_context}

**Chain of Thought 방식으로 추천 이유를 생성해주세요:**

1단계: 사용자의 감정 분석
- 입력된 텍스트에서 감지된 주요 감정과 세부 감정을 분석하세요

2단계: 영화 선택 근거
- 제시된 영화들 중에서 사용자 감정과 가장 잘 맞는 영화를 선택하고 그 이유를 설명하세요

3단계: 구체적 추천 이유
- 선택한 영화가 사용자의 현재 감정 상태에 어떤 도움이 될지 구체적으로 설명하세요

**응답 형식:**
🎬 추천 영화: "[영화 제목]"

💭 추천 이유:
1단계: [감정 분석]
2단계: [영화 선택 근거]
3단계: [구체적 추천 이유]
        """
        
        return prompt

    def _format_few_shot_examples(self) -> str:
        """Few-shot 예제들을 프롬프트 형식으로 포맷팅합니다."""
        examples = "**Few-shot 예제들:**\n\n"
        
        for i, example in enumerate(self.few_shot_examples, 1):
            examples += f"예제 {i}:\n"
            examples += f"입력: \"{example['user_emotion']}\"\n"
            examples += f"주요 감정: {example['primary_emotion']}\n"
            examples += f"추천: {example['recommendation']}\n\n"
        
        return examples

    def create_advanced_prompt(self, user_emotion: str, movie_context: str,
                                emotion_scores: Dict[str, float], search_metadata: Dict = None) -> str:
        """고급 프롬프트를 생성합니다."""
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        
        prompt = f"""
당신은 감정 기반 영화 추천 전문가입니다. 다음 정보를 바탕으로 개인화된 영화 추천을 제공해주세요.

**사용자 정보:**
- 입력: "{user_emotion}"
- 주요 감정: {primary_emotion}
- 감정 분포: {emotion_scores}

**검색 메타데이터:**
{search_metadata if search_metadata else "메타데이터 없음"}

**추천할 영화 정보:**
{movie_context}

**추천 가이드라인:**
1. 사용자의 감정 상태를 정확히 이해하고 공감하세요
2. 영화의 줄거리, 장르, 분위기를 고려하여 선택하세요
3. 구체적이고 개인화된 추천 이유를 제공하세요
4. 감정적 치유나 위로가 될 수 있는 영화를 우선 고려하세요
5. 사용자가 영화를 통해 얻을 수 있는 인사이트나 메시지를 설명하세요

**응답 형식:**
🎬 추천 영화: [영화 제목]

💭 추천 이유:
[사용자 감정에 대한 이해와 공감]
[선택한 영화의 특징과 매력]
[구체적인 추천 이유와 기대 효과]
        """
        
        return prompt

    def generate_chain_of_thought_recommendation(self, user_emotion: str, movie_context: str,
                                                  emotion_scores: Dict[str, float]) -> str:
        """Chain of Thought 방식으로 추천을 생성합니다."""
        
        if self.client is None:
            return self._generate_mock_chain_of_thought_recommendation(user_emotion, movie_context, emotion_scores)
        
        try:
            prompt = self.create_chain_of_thought_prompt(user_emotion, movie_context, emotion_scores)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 감정 기반 영화 추천 전문가입니다. Chain of Thought 방식으로 단계별 추론을 통해 추천 이유를 설명해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Chain of Thought 추천 생성 실패: {e}")
            return self._generate_mock_chain_of_thought_recommendation(user_emotion, movie_context, emotion_scores)

    def generate_advanced_recommendation(self, user_emotion: str, movie_context: str,
                                         emotion_scores: Dict[str, float],
                                         search_metadata: Dict = None) -> str:
        """고급 추천을 생성합니다."""
        
        if self.client is None:
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)
        
        try:
            prompt = self.create_advanced_prompt(user_emotion, movie_context, emotion_scores, search_metadata)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "당신은 감정 기반 영화 추천 전문가입니다. 개인화된 추천을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"고급 추천 생성 실패: {e}")
            return self._generate_mock_advanced_recommendation(user_emotion, movie_context, emotion_scores)

    def _generate_mock_chain_of_thought_recommendation(self, user_emotion: str, movie_context: str,
                                                        emotion_scores: Dict[str, float]) -> str:
        """Chain of Thought 방식의 모의 추천을 생성합니다."""
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        
        # 감정별 모의 추천
        mock_recommendations = {
            "슬픔": {
                "title": "라라랜드",
                "reason": """
**🎬 추천 영화: "라라랜드"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 슬픔과 상실감이 주요 감정으로 감지됨
- 새로운 시작에 대한 두려움과 희망이 혼재

2단계: 영화 선택 근거
- "라라랜드"는 이별의 아픔을 다루면서도 희망적 메시지 전달
- 음악과 색채로 감정을 표현하는 방식이 위로가 됨
- 꿈과 현실의 갈등을 다루어 공감대 형성

3단계: 구체적 추천 이유
- 현재 상황에서 음악적 위로를 받을 수 있음
- 아름다운 시각적 요소로 감정적 치유 효과
- 새로운 시작에 대한 용기를 얻을 수 있음
                """
            },
            "분노": {
                "title": "슈퍼배드",
                "reason": """
**🎬 추천 영화: "슈퍼배드"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 스트레스와 분노로 인한 감정적 압박
- 웃음을 통한 스트레스 해소 욕구

2단계: 영화 선택 근거
- "슈퍼배드"는 순수한 코미디로 스트레스 해소에 효과적
- 귀여운 미니언들이 긍정적 에너지 제공
- 복잡한 스토리 없이 즉시 웃음을 유발

3단계: 구체적 추천 이유
- 스트레스 해소를 위한 즉각적인 웃음 효과
- 부담 없는 가벼운 분위기로 마음의 여유 제공
- 긍정적 캐릭터들이 에너지 충전 효과
                """
            },
            "기대": {
                "title": "인터스텔라",
                "reason": """
**🎬 추천 영화: "인터스텔라"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 새로운 시작에 대한 기대와 희망
- 변화와 모험에 대한 열망

2단계: 영화 선택 근거
- "인터스텔라"는 새로운 세계 탐험의 모험을 다룸
- 인간의 무한한 가능성과 도전 정신을 보여줌
- 시각적 스케일과 감동으로 새로운 영감 제공

3단계: 구체적 추천 이유
- 새로운 시작에 대한 용기와 영감을 얻을 수 있음
- 무한한 가능성을 보여주는 스케일감
- 변화와 성장의 메시지가 새로운 시작에 동기 부여
                """
            }
        }
        
        # 기본 추천 (감정이 없거나 다른 감정인 경우)
        default_recommendation = {
            "title": "토이스토리",
            "reason": """
**🎬 추천 영화: "토이스토리"**

**💭 추천 이유:**
1단계: 사용자의 감정 분석
- 다양한 감정이 혼재된 상태로 보임
- 위로와 희망이 필요한 상황

2단계: 영화 선택 근거
- "토이스토리"는 순수하고 따뜻한 감동을 제공
- 우정과 가족애의 소중함을 다룸
- 모든 연령대가 즐길 수 있는 보편적 매력

3단계: 구체적 추천 이유
- 마음을 따뜻하게 하는 감동적 스토리
- 일상의 소중함을 일깨워주는 메시지
- 스트레스 해소와 위로를 동시에 제공
                """
        }
        
        recommendation = mock_recommendations.get(primary_emotion, default_recommendation)
        return recommendation["reason"]

    def _generate_mock_advanced_recommendation(self, user_emotion: str, movie_context: str,
                                               emotion_scores: Dict[str, float]) -> str:
        """고급 모의 추천을 생성합니다."""
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "중립"
        
        return f"""
**🎬 고급 RAG 추천 결과**

**💭 개인화된 추천 이유:**

현재 당신의 감정 상태를 분석한 결과, **{primary_emotion}**이 주요 감정으로 감지되었습니다.

**감정 분석:**
- 입력: "{user_emotion}"
- 주요 감정: {primary_emotion} (점수: {emotion_scores.get(primary_emotion, 0):.2f})
- 감정 분포: {emotion_scores}

**추천 영화:**
고급 하이브리드 검색을 통해 당신의 감정에 가장 적합한 영화들을 찾았습니다.

**추천 이유:**
1. **감정적 공감**: 현재 감정 상태와 영화의 분위기가 잘 맞습니다
2. **치유 효과**: 영화를 통해 감정적 위로와 치유를 받을 수 있습니다
3. **새로운 관점**: 영화를 통해 현재 상황을 다른 시각에서 바라볼 수 있습니다

**기대 효과:**
- 감정적 안정감 회복
- 새로운 인사이트와 영감 획득
- 스트레스 해소와 마음의 여유 확보

이 영화가 당신의 현재 감정 상태에 도움이 될 것입니다! 🎬✨
        """ 

    def create_advanced_emotion_prompt(self, user_emotion: str, movie_context: str,
                                      emotion_analysis: Dict[str, Any]) -> str:
        """고급 감정 분석을 바탕으로 한 프롬프트를 생성합니다."""
        
        primary_emotion = emotion_analysis.get('primary_emotion', '중립')
        intensity = emotion_analysis.get('intensity', '보통')
        complex_emotion = emotion_analysis.get('complex_emotion', '단일 감정')
        situations = emotion_analysis.get('situations', {})
        transition_emotions = emotion_analysis.get('transition_emotions', [])
        
        # 감정별 선호도 정보
        preferences = self.emotion_movie_preferences.get(primary_emotion, {})
        
        prompt = f"""
당신은 감정 기반 영화 추천 전문가입니다. 사용자의 복합적인 감정 상태를 분석하고 최적의 영화를 추천해주세요.

**사용자 감정 분석:**
- 입력: "{user_emotion}"
- 주요 감정: {primary_emotion} (강도: {intensity})
- 복합 감정: {complex_emotion}

**상황 분석:**
"""
        
        if situations:
            for situation, emotions in situations.items():
                prompt += f"- {situation}: {', '.join(emotions)}\n"
        else:
            prompt += "- 특별한 상황이 감지되지 않음\n"
        
        if transition_emotions:
            prompt += f"**감정 변화 패턴:** {', '.join(transition_emotions)}\n"
        
        prompt += f"""
**선호 영화 특성:**
- 선호 장르: {', '.join(preferences.get('genres', []))}
- 선호 테마: {', '.join(preferences.get('themes', []))}
- 선호 분위기: {', '.join(preferences.get('mood', []))}

**추천할 영화 정보:**
{movie_context}

**추천 가이드라인:**
1. 사용자의 복합적인 감정 상태를 정확히 이해하세요
2. 상황과 감정 변화를 고려하여 적절한 영화를 선택하세요
3. 감정 강도에 따라 영화의 톤을 조절하세요
4. 치유, 위로, 동기부여 등 사용자가 필요로 하는 요소를 고려하세요
5. 구체적이고 개인화된 추천 이유를 제공하세요

**응답 형식:**
🎬 추천 영화: "[영화 제목]"

💭 추천 이유:
1단계: [감정 분석 - 복합 감정과 상황 고려]
2단계: [영화 선택 근거 - 감정 강도와 선호도 반영]
3단계: [구체적 추천 이유 - 개인화된 메시지]
4단계: [기대 효과 - 감정 변화와 성장]
"""
        
        return prompt

    def generate_advanced_emotion_recommendation(self, user_emotion: str, movie_context: str,
                                               emotion_analysis: Dict[str, Any]) -> str:
        """고급 감정 분석을 바탕으로 한 추천을 생성합니다."""
        try:
            if self.client:
                prompt = self.create_advanced_emotion_prompt(user_emotion, movie_context, emotion_analysis)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "당신은 감정 기반 영화 추천 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7
                )
                
                return response.choices[0].message.content
            else:
                return self._generate_mock_advanced_emotion_recommendation(user_emotion, movie_context, emotion_analysis)
                
        except Exception as e:
            logger.error(f"고급 감정 추천 생성 실패: {e}")
            return self._generate_mock_advanced_emotion_recommendation(user_emotion, movie_context, emotion_analysis)

    def _generate_mock_advanced_emotion_recommendation(self, user_emotion: str, movie_context: str,
                                                     emotion_analysis: Dict[str, Any]) -> str:
        """고급 감정 분석을 바탕으로 한 모의 추천을 생성합니다."""
        primary_emotion = emotion_analysis.get('primary_emotion', '중립')
        intensity = emotion_analysis.get('intensity', '보통')
        complex_emotion = emotion_analysis.get('complex_emotion', '단일 감정')
        situations = emotion_analysis.get('situations', {})
        
        # 감정별 추천 메시지
        emotion_messages = {
            '슬픔': {
                '강함': "깊은 슬픔을 느끼고 계시는군요. 이런 때는 따뜻한 위로와 희망의 메시지가 담긴 영화가 도움이 될 것 같아요.",
                '보통': "슬픈 감정을 느끼고 계시는군요. 마음의 치유와 위로가 필요할 때입니다.",
                '약함': "살짝 슬픈 기분이신가요? 가벼운 위로와 따뜻한 감동이 필요할 것 같아요."
            },
            '스트레스': {
                '강함': "심한 스트레스를 받고 계시는군요. 완전한 해방감과 웃음을 선사하는 영화가 필요할 것 같아요.",
                '보통': "스트레스를 받고 계시는군요. 기분 전환과 동기부여가 필요할 때입니다.",
                '약함': "살짝 스트레스를 받고 계시는군요? 가벼운 웃음과 즐거움이 필요할 것 같아요."
            },
            '외로움': {
                '강함': "깊은 외로움을 느끼고 계시는군요. 따뜻한 연결과 위로의 메시지가 담긴 영화가 필요할 것 같아요.",
                '보통': "외로움을 느끼고 계시는군요. 따뜻한 위로와 연결이 필요할 때입니다.",
                '약함': "살짝 외로우신가요? 따뜻한 감동과 위로가 필요할 것 같아요."
            }
        }
        
        # 기본 메시지
        message = emotion_messages.get(primary_emotion, {}).get(intensity, 
            f"{primary_emotion}을 느끼고 계시는군요. 적절한 영화를 추천해드릴게요.")
        
        # 상황별 메시지 추가
        if situations:
            situation_messages = {
                '직장/업무': "직장에서의 스트레스와 압박감을 느끼고 계시는군요. 해방감과 동기부여가 필요할 것 같아요.",
                '인간관계': "인간관계에서의 복잡한 감정을 느끼고 계시는군요. 이해와 치유의 메시지가 필요할 것 같아요.",
                '가족': "가족과 관련된 따뜻한 감정을 느끼고 계시는군요. 가족애와 따뜻함이 담긴 영화가 좋을 것 같아요."
            }
            
            for situation in situations.keys():
                if situation in situation_messages:
                    message += f"\n\n{situation_messages[situation]}"
        
        return f"""
🎬 추천 영화: "Eternal Sunshine of the Spotless Mind"

💭 추천 이유:
1단계: {message}
2단계: 이 영화는 {primary_emotion}을 다루면서도 희망과 치유의 메시지를 담고 있어요.
3단계: {complex_emotion}을 경험하고 계신 당신에게 깊은 공감과 위로를 제공할 것입니다.
4단계: 영화를 통해 감정의 치유와 성장을 경험하실 수 있을 거예요.
""" 