import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """
    KoBERT 기반 감정분석 모델
    """
    
    def __init__(self, model_name: str = "klue/bert-base"):
        """
        감정분석 모델 초기화
        
        Args:
            model_name: 사용할 모델명
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # 기본 8감정 + 세분화된 감정들
        self.emotion_labels = [
            '기쁨', '신뢰', '두려움', '놀람', '슬픔', '혐오', '분노', '기대',
            '사랑', '외로움', '스트레스', '불안', '희망', '감사', '후회', '열정',
            '평온', '설렘', '답답함', '속상함', '짜증', '우울', '답답해', '속상해'
        ]
        
        # 복합 감정 매핑
        self.complex_emotions = {
            '이별 후 복합감정': ['슬픔', '외로움', '후회', '희망'],
            '새로운 시작의 복합감정': ['기대', '두려움', '설렘', '불안'],
            '스트레스 상황의 복합감정': ['스트레스', '분노', '답답함', '짜증'],
            '성공 후 복합감정': ['기쁨', '감사', '열정', '희망'],
            '실패 후 복합감정': ['슬픔', '후회', '우울', '희망'],
            '사랑의 복합감정': ['사랑', '설렘', '기쁨', '불안'],
            '고독의 복합감정': ['외로움', '슬픔', '평온', '희망']
        }
        
        # 감정 강도 레벨
        self.emotion_intensity = {
            '약함': 0.3,
            '보통': 0.6,
            '강함': 0.9
        }
        
        # 감정별 영화 선호도 매핑
        self.emotion_movie_preferences = {
            '기쁨': {
                'genres': ['코미디', '애니메이션', '뮤지컬', '로맨스'],
                'themes': ['우정', '가족', '성공', '사랑'],
                'mood': ['경쾌한', '즐거운', '따뜻한', '희망찬']
            },
            '슬픔': {
                'genres': ['드라마', '멜로드라마', '로맨스'],
                'themes': ['치유', '위로', '성장', '이별'],
                'mood': ['따뜻한', '감동적인', '희망찬', '치유적인']
            },
            '스트레스': {
                'genres': ['코미디', '액션', '스릴러'],
                'themes': ['해방', '모험', '성공', '도전'],
                'mood': ['경쾌한', '긴장감 있는', '해방감 있는', '동기부여하는']
            },
            '외로움': {
                'genres': ['드라마', '로맨스', '멜로드라마'],
                'themes': ['연결', '사랑', '우정', '치유'],
                'mood': ['따뜻한', '감동적인', '희망찬', '위로하는']
            },
            '사랑': {
                'genres': ['로맨스', '드라마', '코미디'],
                'themes': ['사랑', '설렘', '성장', '희망'],
                'mood': ['로맨틱한', '따뜻한', '설렘있는', '희망찬']
            },
            '기대': {
                'genres': ['드라마', '모험', '판타지'],
                'themes': ['성장', '모험', '변화', '희망'],
                'mood': ['희망찬', '동기부여하는', '모험적인', '성장하는']
            }
        }
        
        # 감정 키워드 확장
        self.emotion_keywords = {
            '기쁨': ['행복', '기쁘', '즐겁', '웃', '신나', '환희', '설렘', '만족', '성취', '축하'],
            '신뢰': ['믿음', '안정', '편안', '따뜻', '위로', '치유', '안심', '신뢰', '평온'],
            '두려움': ['무서', '겁', '불안', '걱정', '긴장', '스트레스', '공포', '불안', '걱정'],
            '놀람': ['놀라', '충격', '예상밖', '갑작스러운', '깜짝', '충격적'],
            '슬픔': ['슬프', '우울', '눈물', '이별', '상실', '고독', '외로움', '절망', '허전', '공허'],
            '혐오': ['싫', '역겨', '불쾌', '짜증', '혐오', '불편'],
            '분노': ['화나', '분노', '열받', '짜증', '스트레스', '열받', '화나', '분노'],
            '기대': ['희망', '꿈', '새로운', '시작', '변화', '모험', '기대', '설렘', '동경'],
            '사랑': ['사랑', '연애', '로맨스', '설렘', '마음', '감정', '연인', '애정'],
            '외로움': ['외로', '고독', '혼자', '허전', '공허', '고립', '고독'],
            '스트레스': ['스트레스', '압박', '부담', '긴장', '피곤', '지치', '힘들'],
            '불안': ['불안', '걱정', '근심', '긴장', '두려움', '불안', '걱정'],
            '희망': ['희망', '꿈', '미래', '기대', '설렘', '동경', '꿈꾸'],
            '감사': ['감사', '고마', '은혜', '축복', '행복', '만족', '감사'],
            '후회': ['후회', '아쉽', '미안', '죄송', '아쉬', '후회'],
            '열정': ['열정', '의지', '의욕', '동기', '목표', '꿈', '열정'],
            '평온': ['평온', '편안', '안정', '고요', '차분', '여유'],
            '설렘': ['설렘', '떨림', '긴장', '기대', '동경'],
            '답답함': ['답답', '답답해', '답답함', '답답한'],
            '속상함': ['속상', '속상해', '속상함', '속상한'],
            '짜증': ['짜증', '짜증나', '짜증내', '짜증스러'],
            '우울': ['우울', '우울해', '우울함', '우울한'],
            '답답해': ['답답해', '답답함', '답답한'],
            '속상해': ['속상해', '속상함', '속상한']
        }
        
        # 상황별 감정 패턴
        self.situation_emotion_patterns = {
            '직장/업무': {
                '스트레스': ['업무', '회사', '직장', '일', '프로젝트', '보고서', '회의', '상사', '동료'],
                '성취감': ['성공', '승진', '인정', '칭찬', '성과', '달성'],
                '불안': ['압박', '기한', '실패', '걱정', '불안', '긴장']
            },
            '인간관계': {
                '사랑': ['연인', '남자친구', '여자친구', '사랑', '연애', '데이트'],
                '우정': ['친구', '동료', '우정', '함께', '지지'],
                '갈등': ['싸움', '다툼', '갈등', '오해', '상처', '이별']
            },
            '가족': {
                '따뜻함': ['가족', '부모', '자식', '형제', '따뜻', '사랑'],
                '부담': ['부담', '기대', '압박', '책임', '의무'],
                '그리움': ['그리움', '보고싶', '떠나', '떨어져']
            },
            '학업/시험': {
                '스트레스': ['시험', '학습', '공부', '과제', '성적', '압박'],
                '성취감': ['합격', '성공', '달성', '성과', '만족'],
                '불안': ['실패', '걱정', '불안', '긴장', '두려움']
            },
            '건강': {
                '걱정': ['건강', '병', '아프', '피곤', '스트레스'],
                '회복': ['치유', '회복', '나아지', '기운'],
                '감사': ['건강', '감사', '소중', '축복']
            },
            '취미/여가': {
                '기쁨': ['취미', '여가', '즐거움', '재미', '관심'],
                '새로움': ['새로운', '도전', '모험', '경험'],
                '평온': ['휴식', '여유', '평온', '편안']
            }
        }
        
        # 감정 변화 패턴
        self.emotion_transition_patterns = {
            '이별 후': ['슬픔', '외로움', '후회', '희망', '성장'],
            '새로운 시작': ['기대', '두려움', '설렘', '불안', '열정'],
            '성공 후': ['기쁨', '감사', '열정', '희망', '만족'],
            '실패 후': ['슬픔', '후회', '우울', '희망', '동기'],
            '스트레스 해소': ['스트레스', '해방', '기쁨', '평온', '만족']
        }
        
        try:
            self._load_model()
            logger.info(f"감정분석 모델 로드 완료: {model_name}")
        except Exception as e:
            logger.warning(f"모델 로드 실패: {e}, 키워드 분석으로 대체")
            self.model = None
    
    def _load_model(self):
        """모델과 토크나이저를 로드합니다."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 감정 분류를 위한 헤드 추가 (8가지 감정)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.emotion_labels),
            problem_type="multi_label_classification"
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        텍스트의 감정을 분석합니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정별 점수 딕셔너리
        """
        if self.model is not None:
            return self._analyze_with_model(text)
        else:
            return self._analyze_with_keywords(text)
    
    def _analyze_with_model(self, text: str) -> Dict[str, float]:
        """모델을 사용한 감정 분석"""
        try:
            # 텍스트 토크나이징
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits)
            
            # 결과 변환
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = float(probabilities[0][i].item())
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"모델 기반 감정분석 실패: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, float]:
        """키워드 기반 감정 분석 (fallback)"""
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.3  # 키워드당 점수
            emotion_scores[emotion] = min(score, 1.0)  # 최대 1.0
        
        return emotion_scores
    
    def get_primary_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        주요 감정을 반환합니다.
        
        Args:
            emotion_scores: 감정별 점수
            
        Returns:
            (주요 감정, 점수)
        """
        if not emotion_scores:
            return "중립", 0.0
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return primary_emotion
    
    def get_emotion_description(self, emotion: str) -> str:
        """
        감정에 대한 설명을 반환합니다.
        
        Args:
            emotion: 감정명
            
        Returns:
            감정 설명
        """
        descriptions = {
            '기쁨': '행복하고 즐거운 감정',
            '신뢰': '안정적이고 믿음직한 감정',
            '두려움': '불안하고 긴장된 감정',
            '놀람': '예상치 못한 상황에 대한 반응',
            '슬픔': '우울하고 슬픈 감정',
            '혐오': '불쾌하고 싫은 감정',
            '분노': '화나고 스트레스 받는 감정',
            '기대': '희망적이고 새로운 시작에 대한 감정'
        }
        
        return descriptions.get(emotion, '중립적인 감정')
    
    def analyze_complex_emotion(self, text: str) -> Dict[str, Any]:
        """복합 감정을 분석합니다."""
        basic_emotions = self.analyze_emotion(text)
        
        # 복합 감정 패턴 매칭
        complex_emotion = self._detect_complex_emotion(text)
        
        # 감정 강도 분석
        intensity = self._analyze_emotion_intensity(text)
        
        # 주요 감정 추출
        primary_emotion = max(basic_emotions.items(), key=lambda x: x[1])[0] if basic_emotions else "중립"
        
        return {
            'basic_emotions': basic_emotions,
            'complex_emotion': complex_emotion,
            'intensity': intensity,
            'primary_emotion': primary_emotion,
            'emotion_summary': self._generate_emotion_summary(basic_emotions, complex_emotion, intensity)
        }

    def _detect_complex_emotion(self, text: str) -> str:
        """복합 감정 패턴을 감지합니다."""
        text_lower = text.lower()
        
        # 복합 감정 패턴 매칭
        for pattern, emotions in self.complex_emotions.items():
            if any(emotion in text_lower for emotion in emotions):
                return pattern
        
        return "단일 감정"

    def _analyze_emotion_intensity(self, text: str) -> str:
        """감정 강도를 분석합니다."""
        intensity_keywords = {
            '강함': ['매우', '너무', '정말', '진짜', '완전', '계속', '오랫동안', '많이'],
            '보통': ['조금', '약간', '좀', '그냥', '보통'],
            '약함': ['살짝', '아주 조금', '가벼운', '살짝']
        }
        
        text_lower = text.lower()
        for intensity, keywords in intensity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return intensity
        
        return "보통"

    def _generate_emotion_summary(self, basic_emotions: Dict[str, float], 
                                complex_emotion: str, intensity: str) -> str:
        """감정 분석 요약을 생성합니다."""
        if not basic_emotions:
            return "감정이 명확하지 않습니다."
        
        primary_emotion = max(basic_emotions.items(), key=lambda x: x[1])[0]
        
        summary = f"주요 감정: {primary_emotion} (강도: {intensity})"
        
        if complex_emotion != "단일 감정":
            summary += f"\n복합 감정: {complex_emotion}"
        
        # 상위 3개 감정 표시
        top_emotions = sorted(basic_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        if len(top_emotions) > 1:
            summary += f"\n감정 조합: {', '.join([f'{emotion}({score:.2f})' for emotion, score in top_emotions])}"
        
        return summary

    def get_movie_recommendation_prompt(self, emotion_analysis: Dict[str, Any]) -> str:
        """감정 분석 결과를 바탕으로 영화 추천 프롬프트를 생성합니다."""
        primary_emotion = emotion_analysis.get('primary_emotion', '중립')
        intensity = emotion_analysis.get('intensity', '보통')
        complex_emotion = emotion_analysis.get('complex_emotion', '단일 감정')
        
        # 감정별 선호도 정보
        preferences = self.emotion_movie_preferences.get(primary_emotion, {})
        
        prompt = f"""
사용자 감정 분석:
- 주요 감정: {primary_emotion}
- 감정 강도: {intensity}
- 복합 감정: {complex_emotion}

선호 영화 특성:
- 선호 장르: {', '.join(preferences.get('genres', []))}
- 선호 테마: {', '.join(preferences.get('themes', []))}
- 선호 분위기: {', '.join(preferences.get('mood', []))}

이 정보를 바탕으로 사용자에게 가장 적합한 영화를 추천해주세요.
"""
        return prompt



    def analyze_situation_emotion(self, text: str) -> Dict[str, Any]:
        """상황 기반 감정 분석을 수행합니다."""
        situation_analysis = {}
        
        # 상황별 감정 패턴 매칭
        for situation, emotion_patterns in self.situation_emotion_patterns.items():
            for emotion, keywords in emotion_patterns.items():
                if any(keyword in text for keyword in keywords):
                    if situation not in situation_analysis:
                        situation_analysis[situation] = []
                    situation_analysis[situation].append(emotion)
        
        # 감정 변화 패턴 매칭
        transition_emotions = []
        for transition, emotions in self.emotion_transition_patterns.items():
            if any(keyword in text for keyword in transition.split()):
                transition_emotions.extend(emotions)
        
        return {
            'situations': situation_analysis,
            'transition_emotions': list(set(transition_emotions)),
            'has_situation': len(situation_analysis) > 0
        }

    def get_comprehensive_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """종합적인 감정 분석을 수행합니다."""
        # 기본 감정 분석
        basic_analysis = self.analyze_complex_emotion(text)
        
        # 상황 기반 감정 분석
        situation_analysis = self.analyze_situation_emotion(text)
        
        # 종합 결과 생성
        comprehensive_result = {
            **basic_analysis,
            **situation_analysis,
            'analysis_summary': self._generate_comprehensive_summary(
                basic_analysis, situation_analysis
            )
        }
        
        return comprehensive_result

    def _generate_comprehensive_summary(self, basic_analysis: Dict[str, Any], 
                                      situation_analysis: Dict[str, Any]) -> str:
        """종합적인 감정 분석 요약을 생성합니다."""
        summary = basic_analysis.get('emotion_summary', '')
        
        # 상황 정보 추가
        if situation_analysis.get('has_situation'):
            situations = situation_analysis.get('situations', {})
            if situations:
                summary += f"\n상황: {', '.join(situations.keys())}"
                
                for situation, emotions in situations.items():
                    summary += f"\n- {situation}: {', '.join(emotions)}"
        
        # 감정 변화 정보 추가
        transition_emotions = situation_analysis.get('transition_emotions', [])
        if transition_emotions:
            summary += f"\n감정 변화: {', '.join(transition_emotions)}"
        
        return summary

if __name__ == "__main__":
    # 테스트
    analyzer = EmotionAnalyzer()
    
    test_texts = [
        "친구와 다퉈서 속상해",
        "새로운 시작을 하고 싶어",
        "스트레스 받아서 웃고 싶어",
        "따뜻한 위로가 필요해"
    ]
    
    for text in test_texts:
        print(f"\n=== '{text}' 감정 분석 ===")
        scores = analyzer.analyze_emotion(text)
        primary, score = analyzer.get_primary_emotion(scores)
        print(f"주요 감정: {primary} ({score:.2f})")
        print(f"전체 점수: {scores}") 