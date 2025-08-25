import streamlit as st
import os
import logging
from typing import List, Dict
from advanced_retriever import AdvancedMovieRetriever
from advanced_movie_generator import AdvancedMovieGenerator
from feedback_system import FeedbackSystem
import utils

# 페이지 설정
st.set_page_config(
    page_title="감정 기반 영화 추천 챗봇",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        color: #ff7f0e;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .movie-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .chat-message {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_components():
    """컴포넌트들을 초기화합니다."""
    # 세션 상태 초기화
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "feedback_system" not in st.session_state:
        st.session_state.feedback_system = FeedbackSystem()
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{hash(st.session_state.get('_session_id', 'default'))}"

def load_models():
    """모델들을 로드합니다."""
    if not st.session_state.data_loaded:
        with st.spinner("고급 RAG 모델을 로딩하고 있어요..."):
            try:
                # Advanced Retriever 초기화
                st.session_state.retriever = AdvancedMovieRetriever()
                st.session_state.retriever.load_data()
                
                # Advanced Generator 초기화
                st.session_state.generator = AdvancedMovieGenerator()
                
                st.session_state.data_loaded = True
                st.success("고급 RAG 모델 로딩 완료! 🎉")
                
            except Exception as e:
                st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
                return False
    
    return st.session_state.data_loaded

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown('<h1 class="main-header">🎬 감정 기반 영화 추천 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            지금 느끼고 계신 감정을 알려주세요. 당신에게 딱 맞는 영화를 추천해드릴게요! 💙
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 컴포넌트 초기화
    initialize_components()
    
    # 사이드바 생성
    top_k = utils.create_sidebar()
    
    # 모델 로드
    if not load_models():
        st.error("모델 로딩에 실패했습니다. 다시 시도해주세요.")
        return
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💭 감정을 표현해보세요</h2>', unsafe_allow_html=True)
        
        # 감정 입력 방식 선택
        input_method = st.radio(
            "입력 방식을 선택하세요:",
            ["🎭 감정 선택", "💬 직접 입력", "🎬 영화 선호도 설정", "😊 해피엔딩 영화 추천"]
        )
        
        user_emotion = ""
        movie_preferences = {}
        
        if input_method == "🎭 감정 선택":
            user_emotion = utils.create_emotion_selector()
            
        elif input_method == "💬 직접 입력":
            user_emotion = st.text_area(
                "지금 어떤 감정을 느끼고 계신가요?",
                placeholder="예: 이별 후 공허함을 느끼고 있어요...",
                height=100
            )
            
        elif input_method == "🎬 영화 선호도 설정":
            # 영화 선호도 설정
            movie_preferences = utils.create_movie_preference_selector()
            
            # 간단한 감정 입력
            user_emotion = st.text_area(
                "간단히 감정이나 상황을 입력하세요 (선택사항):",
                placeholder="예: 기분이 좋아서 해피엔딩 영화를 보고 싶어요",
                height=80
            )
            
        elif input_method == "😊 해피엔딩 영화 추천":
            st.markdown("### 😊 해피엔딩 영화 추천")
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 4px solid #ffc107;">
                <h4>🌟 해피엔딩 영화란?</h4>
                <p>• 사랑과 우정이 승리하는 영화</p>
                <p>• 꿈과 희망을 주는 영화</p>
                <p>• 따뜻하고 감동적인 결말의 영화</p>
                <p>• 기분이 좋아지는 영화</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 해피엔딩 영화 추천 옵션
            happy_ending_options = st.radio(
                "어떤 방식으로 해피엔딩 영화를 찾으시나요?",
                ["🎯 감정 기반 해피엔딩", "🎬 장르별 해피엔딩", "💝 로맨틱 해피엔딩", "👨‍👩‍👧‍👦 가족 해피엔딩"]
            )
            
            user_emotion = ""
            if happy_ending_options == "🎯 감정 기반 해피엔딩":
                user_emotion = st.text_area(
                    "지금 어떤 감정을 느끼고 계신가요?",
                    placeholder="예: 기분이 좋아서 더 행복한 영화를 보고 싶어요",
                    height=100
                )
            elif happy_ending_options == "🎬 장르별 해피엔딩":
                selected_genres = st.multiselect(
                    "선호하는 장르를 선택하세요:",
                    ["Comedy", "Romance", "Family", "Animation", "Adventure", "Drama"],
                    default=["Comedy", "Romance"]
                )
                user_emotion = f"장르: {', '.join(selected_genres)} 해피엔딩"
            elif happy_ending_options == "💝 로맨틱 해피엔딩":
                user_emotion = "로맨틱 해피엔딩 영화"
            elif happy_ending_options == "👨‍👩‍👧‍👦 가족 해피엔딩":
                user_emotion = "가족 해피엔딩 영화"
            
            # 해피엔딩 영화 추천 버튼
            if user_emotion and st.button("😊 해피엔딩 영화 추천받기", type="primary"):
                with st.spinner("해피엔딩 영화를 찾고 있어요... 🌟"):
                    try:
                        # 해피엔딩 영화 추천
                        happy_results = st.session_state.retriever.get_happy_ending_recommendations(
                            query=user_emotion,
                            emotion="기쁨",
                            top_k=top_k
                        )
                        
                        if happy_results:
                            st.markdown("### 🌟 해피엔딩 영화 추천 결과")
                            st.markdown("""
                            <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;">
                                <h4>✨ 해피엔딩 영화를 찾았어요!</h4>
                                <p>이 영화들은 따뜻하고 행복한 결말을 가진 영화들입니다.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 영화 결과 표시
                            for i, movie in enumerate(happy_results):
                                with st.expander(f"🎬 {movie['title']} (해피엔딩 점수: {movie.get('happy_ending_score', 0):.2f})"):
                                    st.markdown(f"**줄거리:** {movie['overview']}")
                                    st.markdown(f"**장르:** {movie['genres']}")
                                    if movie.get('tagline'):
                                        st.markdown(f"**태그라인:** {movie['tagline']}")
                                    if movie.get('keywords'):
                                        st.markdown(f"**키워드:** {movie['keywords']}")
                        else:
                            st.warning("죄송합니다. 조건에 맞는 해피엔딩 영화를 찾지 못했습니다.")
                            
                    except Exception as e:
                        st.error(f"해피엔딩 영화 추천 중 오류가 발생했습니다: {e}")
            
            # 선호도 기반 영화 추천
            if movie_preferences and st.button("🎬 선호도 기반 영화 추천받기", type="primary"):
                with st.spinner("선호도에 맞는 영화를 찾고 있어요..."):
                    try:
                        # 기본 감정 분석 (입력이 있는 경우)
                        if user_emotion:
                            emotion_analysis = st.session_state.retriever.emotion_analyzer.get_comprehensive_emotion_analysis(user_emotion)
                            primary_emotion = emotion_analysis.get('primary_emotion', '중립')
                        else:
                            primary_emotion = '중립'
                        
                        # 선호도 기반 검색
                        search_results = st.session_state.retriever.advanced_search(
                            query=user_emotion or "선호도 기반 추천",
                            emotion=primary_emotion,
                            filters={**movie_preferences, "top_k": top_k}
                        )
                        
                        # 영화 컨텍스트 생성
                        movie_context = ""
                        for i, movie in enumerate(search_results[:3]):
                            movie_context += f"영화 {i+1}: {movie['title']}\n"
                            movie_context += f"줄거리: {movie['overview']}\n"
                            movie_context += f"장르: {movie['genres']}\n"
                            if movie.get('mood'):
                                movie_context += f"분위기: {movie['mood']}\n"
                            if movie.get('ending'):
                                movie_context += f"결말: {movie['ending']}\n"
                            if movie.get('theme'):
                                movie_context += f"테마: {movie['theme']}\n"
                            movie_context += "\n"
                        
                        # 선호도 기반 추천 이유 생성
                        preference_text = f"선택한 선호도: {', '.join([f'{k}: {v}' for k, v in movie_preferences.items()])}"
                        recommendation = f"""
🎬 선호도 기반 영화 추천

{preference_text}

💭 추천 이유:
1단계: 선택하신 선호도에 맞는 영화들을 찾았어요
2단계: 분위기, 결말, 테마를 모두 고려한 최적의 영화를 선별했어요
3단계: 당신이 원하는 분위기와 결말을 가진 영화들을 추천해드려요
4단계: 이 영화들이 당신의 기대에 부합할 것입니다
"""
                        
                        # 결과 표시
                        utils.display_movie_recommendations(search_results, recommendation)
                        
                        st.markdown("---")
                        st.markdown("""
                        <div style="text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;">
                            <h3>🎉 선호도 기반 영화 추천이 완료되었습니다!</h3>
                            <p>선택하신 선호도에 맞는 영화들을 찾아드렸어요! 😊</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"선호도 기반 추천 생성 중 오류가 발생했습니다: {e}")
        
        # 기존 감정 기반 추천 (감정 선택, 직접 입력)
        if user_emotion and input_method in ["🎭 감정 선택", "💬 직접 입력"] and st.button("🎬 영화 추천받기", type="primary"):
            # 해피엔딩 필터 옵션 추가
            col1_filter, col2_filter = st.columns([1, 1])
            with col1_filter:
                happy_ending_only = st.checkbox("😊 해피엔딩 영화만 추천받기", value=False)
            with col2_filter:
                show_happy_score = st.checkbox("🌟 해피엔딩 점수 표시", value=False)
            with st.spinner("고급 RAG 시스템으로 당신의 감정에 맞는 영화를 찾고 있어요..."):
                try:
                    # 고급 감정 분석
                    emotion_analysis = st.session_state.retriever.emotion_analyzer.get_comprehensive_emotion_analysis(user_emotion)
                    
                    # 고급 하이브리드 검색
                    search_filters = {"top_k": top_k}
                    if happy_ending_only:
                        search_filters["happy_ending_only"] = True
                    
                    search_results = st.session_state.retriever.hybrid_search(
                        query=user_emotion,
                        emotion=emotion_analysis.get('primary_emotion', '중립'),
                        top_k=10,
                        user_id=st.session_state.user_id
                    )
                    
                    # 검색 메타데이터 수집
                    search_metadata = {
                        "primary_emotion": emotion_analysis.get('primary_emotion', '중립'),
                        "emotion_score": emotion_analysis.get('primary_emotion_score', 0.0),
                        "total_results": len(search_results),
                        "search_method": "hybrid_search"
                    }
                    
                    # 영화 컨텍스트 생성
                    movie_context = ""
                    for i, movie in enumerate(search_results[:3]):
                        movie_context += f"영화 {i+1}: {movie['title']}\n"
                        movie_context += f"줄거리: {movie['overview']}\n"
                        movie_context += f"장르: {movie['genres']}\n"
                        if movie.get('tagline'):
                            movie_context += f"태그라인: {movie['tagline']}\n"
                        movie_context += "\n"
                    
                    # 감정 분석 결과 표시
                    st.markdown("### 📊 감정 분석 결과")
                    st.markdown(emotion_analysis.get('analysis_summary', '감정 분석 중...'))
                    
                    # 고급 추천 이유 생성 (Chain of Thought)
                    recommendation = st.session_state.generator.generate_advanced_emotion_recommendation(
                        user_emotion, movie_context, emotion_analysis
                    )
                    
                    # 결과 저장
                    emotion_scores = emotion_analysis.get('basic_emotions', {}) if isinstance(emotion_analysis, dict) else {}
                    utils.save_conversation_history(
                        user_emotion, search_results, recommendation, emotion_scores
                    )
                    
                    # 결과 표시
                    st.markdown('<h2 class="sub-header">🎬 고급 RAG 추천 결과</h2>', unsafe_allow_html=True)
                    
                    # 추천 이유 표시
                    st.markdown(recommendation)
                    
                    # 영화 카드들 표시
                    if show_happy_score:
                        # 해피엔딩 점수 계산 및 표시
                        for movie in search_results:
                            movie_text = f"{movie.get('title', '')} {movie.get('overview', '')} {movie.get('keywords', '')}"
                            happy_score = st.session_state.retriever.analyze_happy_ending_probability(movie_text)
                            movie['happy_ending_score'] = happy_score
                    
                    utils.display_movie_recommendations(search_results, "")
                    
                    # 피드백 시스템 추가
                    st.markdown("---")
                    st.markdown("### 💬 추천 결과에 대한 피드백")
                    
                    # 영화별 피드백 수집
                    for i, movie in enumerate(search_results[:5]):  # 상위 5개 영화만
                        with st.expander(f"📝 {movie.get('title', 'Unknown')} 피드백"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                rating = st.selectbox(
                                    f"평점",
                                    options=[5, 4, 3, 2, 1],
                                    index=4,  # 기본값 5점
                                    key=f"rating_{i}"
                                )
                            
                            with col2:
                                feedback_text = st.text_area(
                                    f"추가 의견 (선택사항)",
                                    placeholder="이 영화에 대한 의견을 자유롭게 작성해주세요...",
                                    key=f"feedback_{i}"
                                )
                            
                            if st.button(f"피드백 제출", key=f"submit_{i}"):
                                st.session_state.feedback_system.add_movie_feedback(
                                    user_id=st.session_state.user_id,
                                    movie_id=movie.get('id', 0),
                                    user_emotion=emotion_analysis.get('primary_emotion', '중립'),
                                    movie_title=movie.get('title', 'Unknown'),
                                    rating=rating,
                                    feedback_text=feedback_text
                                )
                                st.success("피드백이 성공적으로 제출되었습니다! 🎉")
                    
                    # 감정 분석 피드백
                    st.markdown("### 🧠 감정 분석 피드백")
                    st.markdown("**현재 감정 분석이 정확한가요?**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 정확해요"):
                            st.success("감정 분석이 정확하다고 평가해주셔서 감사합니다!")
                    
                    with col2:
                        if st.button("❌ 부정확해요"):
                            st.markdown("**실제 감정을 선택해주세요:**")
                            actual_emotion = st.selectbox(
                                "실제 감정",
                                options=['기쁨', '슬픔', '분노', '사랑', '외로움', '스트레스', '불안', '희망', '감사', '후회', '열정', '평온', '설렘'],
                                key="actual_emotion"
                            )
                            
                            emotion_feedback = st.text_area(
                                "추가 설명 (선택사항)",
                                placeholder="왜 이 감정이 더 정확한지 설명해주세요...",
                                key="emotion_feedback"
                            )
                            
                            if st.button("감정 피드백 제출"):
                                st.session_state.feedback_system.add_emotion_feedback(
                                    user_id=st.session_state.user_id,
                                    original_emotion=emotion_analysis.get('primary_emotion', '중립'),
                                    actual_emotion=actual_emotion,
                                    feedback_text=emotion_feedback
                                )
                                st.success("감정 분석 피드백이 제출되었습니다! 🎉")
                    
                except Exception as e:
                    st.error(f"고급 RAG 추천 생성 중 오류가 발생했습니다: {e}")
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 분석 결과</h3>', unsafe_allow_html=True)
        
        # 감정 분석 차트
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            latest_entry = st.session_state.conversation_history[-1]
            emotion_scores = latest_entry.get('emotion_scores', {})
            if isinstance(emotion_scores, dict):
                utils.create_emotion_chart(emotion_scores)
        
        # 대화 기록
        utils.display_conversation_history()
        
        # 피드백 통계
        st.markdown("### 📊 피드백 통계")
        feedback_summary = st.session_state.feedback_system.get_feedback_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 피드백", feedback_summary['total_feedback'])
        with col2:
            st.metric("참여 사용자", feedback_summary['total_users'])
        with col3:
            st.metric("평균 평점", f"{feedback_summary['average_rating']}/5.0")
        
        # 피드백 타입별 통계
        st.markdown("**피드백 타입별 통계:**")
        feedback_types = feedback_summary['feedback_types']
        st.write(f"🎬 영화 추천 피드백: {feedback_types['movie_recommendation']}개")
        st.write(f"🧠 감정 분석 피드백: {feedback_types['emotion_analysis']}개")
    
    # 하단 정보
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>💡 <strong>사용 팁</strong>: 더 구체적으로 감정을 표현할수록 더 정확한 추천을 받을 수 있어요!</p>
        <p>🎯 <strong>예시</strong>: "이별 후 공허함을 느끼고 있어" → "스트레스 받아서 웃고 싶어" → "새로운 시작을 하고 싶어"</p>
    </div>
    """, unsafe_allow_html=True)

def show_about():
    """프로젝트 정보를 표시합니다."""
    st.markdown("""
    ## 📖 고급 RAG 영화 추천 시스템
    
    이 챗봇은 **고급 RAG(Retrieval-Augmented Generation)** 기술을 활용한 감정 기반 영화 추천 시스템입니다.
    
    ### 🔧 고급 기술 스택
    - **임베딩**: sentence-transformers (all-MiniLM-L6-v2)
    - **벡터 검색**: FAISS + 하이브리드 검색
    - **감정 분석**: KoBERT 기반 감정 분류 모델
    - **LLM**: OpenAI GPT-3.5/4 + Chain of Thought
    - **UI**: Streamlit
    
    ### 🎯 고급 RAG 기능
    1. **데이터 확장**: 실제 TMDB API 데이터 활용
    2. **멀티모달 검색**: 벡터 검색 + 키워드 검색 + 장르 가중치
    3. **고급 프롬프트 엔지니어링**: Few-shot + Chain of Thought
    4. **통합된 고급 시스템**: 모든 기능이 통합된 완전한 RAG
    
    ### 🚀 구현된 발전법
    - ✅ **1단계**: 실제 TMDB 데이터 활용 (데이터 확장)
    - ✅ **2단계**: 고급 검색 시스템 (멀티모달/하이브리드 검색)
    - ✅ **3단계**: 고급 프롬프트 엔지니어링 (Few-shot, Chain of Thought)
    - ✅ **4단계**: 통합된 고급 RAG 시스템
    
    ### 🎯 주요 기능
    1. **KoBERT 감정 분석**: 정확한 감정 분류 및 점수화
    2. **하이브리드 검색**: 벡터 유사도 + 키워드 매칭 + 장르 선호도
    3. **Chain of Thought 추천**: 단계별 추론을 통한 상세한 추천 이유
    4. **실시간 TMDB 데이터**: 최신 영화 정보 활용
    
    ### 🚀 향후 계획
    - 일기/일상 입력 기반 주말 영화 추천 시스템
    - 감정 이력 학습 기반 개인화
    - 더 많은 영화 데이터 추가
    - 실시간 영화 정보 업데이트
    """)

if __name__ == "__main__":
    # 탭 생성
    tab1, tab2 = st.tabs(["🎬 영화 추천", "📖 프로젝트 정보"])
    
    with tab1:
        main()
    
    with tab2:
        show_about() 