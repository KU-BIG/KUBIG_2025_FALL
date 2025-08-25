import streamlit as st
import os
import logging
from typing import List, Dict
from advanced_retriever import AdvancedMovieRetriever
from advanced_generator import AdvancedMovieGenerator
from emotion_analyzer import EmotionAnalyzer
import utils

# 페이지 설정
st.set_page_config(
    page_title="고급 감정 기반 영화 추천 챗봇",
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
    .advanced-feature {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .search-stats {
        background-color: #f0f8ff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_components():
    """컴포넌트들을 초기화합니다."""
    if "advanced_retriever" not in st.session_state:
        st.session_state.advanced_retriever = None
    if "advanced_generator" not in st.session_state:
        st.session_state.advanced_generator = None
    if "emotion_analyzer" not in st.session_state:
        st.session_state.emotion_analyzer = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

def load_models():
    """모델들을 로드합니다."""
    if not st.session_state.data_loaded:
        with st.spinner("고급 모델들을 로딩하고 있어요..."):
            try:
                # 고급 Retriever 초기화
                st.session_state.advanced_retriever = AdvancedMovieRetriever()
                st.session_state.advanced_retriever.load_data()
                
                # 고급 Generator 초기화
                st.session_state.advanced_generator = AdvancedMovieGenerator()
                
                # 감정분석기 초기화
                st.session_state.emotion_analyzer = EmotionAnalyzer()
                
                st.session_state.data_loaded = True
                st.success("고급 모델 로딩 완료! 🚀")
                
            except Exception as e:
                st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
                return False
    
    return st.session_state.data_loaded

def create_advanced_sidebar():
    """고급 사이드바를 생성합니다."""
    with st.sidebar:
        st.title("🚀 고급 영화 추천 챗봇")
        st.markdown("---")
        
        st.markdown("### 📊 통계")
        if "conversation_history" in st.session_state:
            total_conversations = len(st.session_state.conversation_history)
            st.metric("총 대화 수", total_conversations)
        
        st.markdown("### ⚙️ 고급 설정")
        
        # 검색 설정
        st.subheader("🔍 검색 설정")
        search_method = st.selectbox(
            "검색 방법",
            ["하이브리드 검색", "벡터 검색", "키워드 검색"]
        )
        
        top_k = st.slider("추천 영화 개수", 1, 10, 5)
        
        # 필터 설정
        st.subheader("🎭 필터 설정")
        selected_genres = st.multiselect(
            "선호 장르",
            ["Drama", "Romance", "Comedy", "Action", "Animation", "Adventure", "Thriller", "Family"]
        )
        
        year_range = st.slider(
            "연도 범위",
            1990, 2024, (2000, 2024)
        )
        
        # 추천 방법 설정
        st.subheader("🧠 추천 방법")
        recommendation_method = st.selectbox(
            "추천 생성 방법",
            ["체인 오브 쏘트", "고급 프롬프트", "기본 추천"]
        )
        
        return {
            'search_method': search_method,
            'top_k': top_k,
            'genres': selected_genres,
            'year_range': year_range,
            'recommendation_method': recommendation_method
        }

def display_search_metadata(search_results: List[Dict]):
    """검색 메타데이터를 표시합니다."""
    if not search_results:
        return
    
    # 검색 통계 계산
    total_movies = len(search_results)
    avg_similarity = sum(movie.get('final_score', 0) for movie in search_results) / total_movies
    
    # 장르 분포 계산
    genre_counts = {}
    for movie in search_results:
        genres = movie.get('genres', '').split(', ')
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    genre_distribution = ', '.join([f"{genre}({count})" for genre, count in genre_counts.items()])
    
    # 메타데이터 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("검색된 영화", total_movies)
    with col2:
        st.metric("평균 점수", f"{avg_similarity:.3f}")
    with col3:
        st.metric("장르 수", len(genre_counts))
    
    st.markdown(f"**장르 분포**: {genre_distribution}")
    
    return {
        'total_movies': total_movies,
        'avg_similarity': avg_similarity,
        'genre_distribution': genre_distribution
    }

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown('<h1 class="main-header">🚀 고급 감정 기반 영화 추천 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            고급 RAG 기술로 더 정확하고 개인화된 영화 추천을 받아보세요! 💙
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 고급 기능 소개
    with st.expander("🚀 고급 기능 소개"):
        st.markdown("""
        ### 🔍 고급 검색 시스템
        - **하이브리드 검색**: 벡터 + 키워드 검색 결합
        - **멀티모달 검색**: 감정 + 장르 + 키워드 통합
        - **필터링 시스템**: 장르, 연도별 필터링
        
        ### 🧠 고급 추천 시스템
        - **체인 오브 쏘트**: 단계별 추론 과정
        - **Few-shot 예시**: 학습된 예시 기반 추천
        - **고급 프롬프트**: 구조화된 추천 메시지
        
        ### 📊 상세 분석
        - **검색 메타데이터**: 검색 통계 및 분석
        - **감정 분석**: KoBERT 기반 정확한 감정 분류
        - **개인화**: 사용자 맞춤형 추천
        """)
    
    # 컴포넌트 초기화
    initialize_components()
    
    # 고급 사이드바 생성
    settings = create_advanced_sidebar()
    
    # 모델 로드
    if not load_models():
        st.error("모델 로딩에 실패했습니다. 다시 시도해주세요.")
        return
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💭 감정을 표현해보세요</h2>', unsafe_allow_html=True)
        
        # 감정 입력
        user_emotion = st.text_area(
            "지금 어떤 감정을 느끼고 계신가요?",
            placeholder="예: 친구와 다퉈서 속상해...",
            height=100
        )
        
        # 추천 버튼
        if user_emotion and st.button("🚀 고급 영화 추천받기", type="primary"):
            with st.spinner("고급 검색과 분석을 수행하고 있어요..."):
                try:
                    # 1. 감정 분석
                    emotion_scores = st.session_state.emotion_analyzer.analyze_emotion(user_emotion)
                    primary_emotion, emotion_score = st.session_state.emotion_analyzer.get_primary_emotion(emotion_scores)
                    
                    # 감정 분석 결과 표시
                    st.markdown("### 📊 감정 분석 결과")
                    st.markdown(f"**주요 감정**: {primary_emotion} ({emotion_score:.2f})")
                    
                    # 감정 분포 차트
                    utils.create_emotion_chart(emotion_scores)
                    
                    # 2. 고급 검색
                    st.markdown("### 🔍 고급 검색 수행")
                    
                    # 필터 설정
                    filters = {}
                    if settings['genres']:
                        filters['genres'] = settings['genres']
                    if settings['year_range']:
                        filters['year_range'] = {
                            'min': settings['year_range'][0],
                            'max': settings['year_range'][1]
                        }
                    
                    # 검색 수행
                    search_results = st.session_state.advanced_retriever.advanced_search(
                        user_emotion, primary_emotion, filters
                    )
                    
                    # 검색 메타데이터 표시
                    search_metadata = display_search_metadata(search_results)
                    
                    # 3. 영화 컨텍스트 구성
                    movie_context = ""
                    for i, movie in enumerate(search_results[:settings['top_k']], 1):
                        movie_context += f"{i}. {movie['title']}\n"
                        movie_context += f"   줄거리: {movie['overview']}\n"
                        movie_context += f"   장르: {movie['genres']}\n"
                        if movie.get('tagline'):
                            movie_context += f"   태그라인: {movie['tagline']}\n"
                        movie_context += f"   최종 점수: {movie.get('final_score', 0):.3f}\n\n"
                    
                    # 4. 고급 추천 생성
                    st.markdown("### 🧠 고급 추천 생성")
                    
                    if settings['recommendation_method'] == "체인 오브 쏘트":
                        recommendation = st.session_state.advanced_generator.generate_chain_of_thought_recommendation(
                            user_emotion, movie_context, emotion_scores
                        )
                    elif settings['recommendation_method'] == "고급 프롬프트":
                        recommendation = st.session_state.advanced_generator.generate_advanced_recommendation(
                            user_emotion, movie_context, emotion_scores, search_metadata
                        )
                    else:
                        # 기본 추천
                        recommendation = st.session_state.advanced_generator._generate_mock_advanced_recommendation(
                            user_emotion, movie_context, emotion_scores
                        )
                    
                    # 결과 표시
                    st.markdown("### 🎬 추천 결과")
                    st.markdown(recommendation)
                    
                    # 영화 카드들 표시
                    utils.display_movie_recommendations(search_results[:settings['top_k']], "")
                    
                    # 결과 저장
                    utils.save_conversation_history(
                        user_emotion, search_results[:settings['top_k']], recommendation, emotion_scores
                    )
                    
                except Exception as e:
                    st.error(f"고급 추천 생성 중 오류가 발생했습니다: {e}")
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 고급 분석</h3>', unsafe_allow_html=True)
        
        # 검색 통계
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            latest_entry = st.session_state.conversation_history[-1]
            
            st.markdown("### 🔍 최근 검색 통계")
            if 'search_metadata' in latest_entry:
                metadata = latest_entry['search_metadata']
                st.metric("검색된 영화", metadata.get('total_movies', 0))
                st.metric("평균 점수", f"{metadata.get('avg_similarity', 0):.3f}")
        
        # 대화 기록
        utils.display_conversation_history()
    
    # 하단 정보
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚀 <strong>고급 기능</strong>: 하이브리드 검색, 체인 오브 쏘트, Few-shot 예시를 활용한 정확한 추천</p>
        <p>🎯 <strong>개인화</strong>: 감정 분석과 필터링을 통한 맞춤형 추천</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 