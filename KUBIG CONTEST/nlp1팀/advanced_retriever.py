import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Optional
import logging
from data_processor import MovieDataProcessor
from emotion_analyzer import EmotionAnalyzer
from review_processor import ReviewProcessor
from feedback_system import FeedbackSystem
import re

logger = logging.getLogger(__name__)

class AdvancedMovieRetriever:
    def __init__(self, data_processor: MovieDataProcessor = None):
        self.data_processor = data_processor or MovieDataProcessor()
        self.movies_df = None
        self.index = None
        self.model = None
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 리뷰 및 피드백 시스템 초기화
        self.review_processor = ReviewProcessor()
        self.feedback_system = FeedbackSystem()
        self.reviews_loaded = False
        
        # 장르 가중치 설정
        self.genre_weights = {
            '액션': 1.2, '어드벤처': 1.1, '애니메이션': 1.0, '코미디': 1.3,
            '범죄': 0.9, '다큐멘터리': 0.8, '드라마': 1.1, '가족': 1.2,
            '판타지': 1.1, '역사': 0.9, '공포': 0.8, '음악': 1.0,
            '미스터리': 1.0, '로맨스': 1.2, 'SF': 1.0, '스릴러': 0.9,
            '전쟁': 0.8, '서부': 0.7
        }
        
        # 감정별 장르 선호도
        self.emotion_genre_preferences = {
            '기쁨': ['코미디', '애니메이션', '가족', '음악'],
            '신뢰': ['드라마', '가족', '다큐멘터리'],
            '두려움': ['스릴러', '공포', '미스터리'],
            '놀람': ['액션', '어드벤처', '판타지'],
            '슬픔': ['드라마', '로맨스', '음악'],
            '혐오': ['액션', '스릴러', '범죄'],
            '분노': ['액션', '스릴러', '코미디'],
            '기대': ['어드벤처', '판타지', 'SF', '로맨스']
        }
        
        # 해피엔딩 관련 키워드
        self.happy_ending_keywords = {
            'positive_outcomes': [
                'happy ending', 'happy', 'joy', 'success', 'victory', 'win', 'triumph',
                'love', 'romance', 'marriage', 'wedding', 'together', 'reunion',
                'friendship', 'family', 'reconciliation', 'forgiveness', 'redemption',
                'hope', 'dreams', 'fulfillment', 'achievement', 'accomplishment',
                'healing', 'recovery', 'transformation', 'growth', 'change for better',
                'second chance', 'new beginning', 'fresh start', 'opportunity',
                'inspiration', 'motivation', 'encouragement', 'support'
            ],
            'positive_emotions': [
                'joyful', 'cheerful', 'delighted', 'pleased', 'satisfied', 'content',
                'grateful', 'blessed', 'fortunate', 'lucky', 'excited', 'thrilled',
                'ecstatic', 'elated', 'euphoric', 'blissful', 'peaceful', 'calm',
                'relaxed', 'comfortable', 'secure', 'safe', 'protected', 'loved'
            ],
            'positive_actions': [
                'succeed', 'achieve', 'accomplish', 'overcome', 'conquer', 'defeat',
                'resolve', 'solve', 'fix', 'heal', 'recover', 'improve', 'progress',
                'develop', 'grow', 'learn', 'understand', 'realize', 'discover',
                'find', 'meet', 'connect', 'bond', 'unite', 'reunite', 'forgive',
                'accept', 'embrace', 'celebrate', 'enjoy', 'appreciate', 'value'
            ],
            'positive_relationships': [
                'love', 'romance', 'relationship', 'marriage', 'family', 'friendship',
                'bond', 'connection', 'partnership', 'teamwork', 'collaboration',
                'support', 'care', 'nurture', 'protect', 'guide', 'mentor',
                'inspire', 'encourage', 'motivate', 'help', 'assist', 'serve'
            ]
        }
        
        # 해피엔딩 확률이 높은 장르들
        self.happy_ending_genres = [
            'Comedy', 'Romance', 'Family', 'Animation', 'Musical', 'Adventure'
        ]

    def load_data(self, load_path: str = "movie_data"):
        """데이터를 로드합니다."""
        try:
            # 영화 데이터 로드
            try:
                # 먼저 직접 CSV 파일 로드 시도
                import pandas as pd
                self.movies_df = pd.read_csv(f"{load_path}/movies.csv")
                logger.info(f"movies.csv 파일로 로드 완료: {len(self.movies_df)}개 영화")
            except Exception as e:
                logger.error(f"영화 데이터 로드 실패: {e}")
                try:
                    # data_processor로 로드 시도
                    self.movies_df = self.data_processor.load_processed_data(load_path)
                    logger.info("data_processor로 영화 데이터 로드 완료")
                except Exception as e2:
                    logger.error(f"data_processor 로드도 실패: {e2}")
                    self.movies_df = None
            
            # FAISS 인덱스 로드 (파일명 수정)
            try:
                self.index = self.data_processor.load_index(load_path)
            except:
                # 다른 파일명으로 시도
                try:
                    import faiss
                    self.index = faiss.read_index(f"{load_path}/movie_index.faiss")
                    logger.info("movie_index.faiss 파일로 로드 완료")
                except Exception as e:
                    logger.error(f"FAISS 인덱스 로드 실패: {e}")
                    self.index = None
            
            # 임베딩 모델 로드
            try:
                # 먼저 data_processor에서 모델 가져오기 시도
                if hasattr(self.data_processor, 'model') and self.data_processor.model is not None:
                    self.model = self.data_processor.model
                    logger.info("data_processor에서 모델 로드 완료")
                else:
                    # 저장된 모델 로드 시도
                    try:
                        from sentence_transformers import SentenceTransformer
                        self.model = SentenceTransformer(f"{load_path}/embedding_model")
                        logger.info("저장된 모델 로드 완료")
                    except:
                        # 기본 모델 로드
                        from sentence_transformers import SentenceTransformer
                        self.model = SentenceTransformer('all-MiniLM-L6-v2')
                        logger.info("기본 모델 로드 완료")
            except Exception as e:
                logger.error(f"임베딩 모델 로드 실패: {e}")
                self.model = None
            
            # 리뷰 데이터 로드 시도 (선택적)
            try:
                self.reviews_loaded = self.review_processor.load_review_data(f"{load_path}/reviews_data.pkl")
                if self.reviews_loaded:
                    logger.info("리뷰 데이터 로드 완료")
                else:
                    logger.info("리뷰 데이터가 없습니다. 기본 모드로 실행합니다.")
                    self.reviews_loaded = False
            except Exception as e:
                logger.info("리뷰 데이터 없이 기본 모드로 실행합니다.")
                self.reviews_loaded = False
            
            logger.info("고급 영화 검색 데이터 로드 완료")
            return True
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return False
    
    def _build_review_data(self):
        """리뷰 데이터를 구축합니다."""
        try:
            # ratings.txt 파일이 있는지 확인
            import os
            if not os.path.exists('ratings.txt'):
                logger.info("ratings.txt 파일이 없습니다.")
                return
            
            # 리뷰 데이터 로드
            reviews_df = self.review_processor.load_ratings_data()
            if reviews_df.empty:
                logger.warning("리뷰 데이터를 로드할 수 없습니다.")
                return
            
            # 영화와 매칭
            movie_reviews = self.review_processor.match_reviews_to_movies(reviews_df, self.movies_df)
            
            # 리뷰 인덱스 구축
            self.review_processor.build_review_index(movie_reviews)
            
            # 저장
            self.review_processor.save_review_data('movie_data/reviews_data.pkl')
            self.reviews_loaded = True
            
            logger.info("리뷰 데이터 구축 완료")
            
        except Exception as e:
            logger.error(f"리뷰 데이터 구축 실패: {e}")
            self.reviews_loaded = False

    def analyze_happy_ending_probability(self, movie_text: str) -> float:
        """영화의 해피엔딩 확률을 분석합니다."""
        movie_text_lower = movie_text.lower()
        score = 0.0
        total_keywords = 0
        
        # 각 카테고리별 키워드 검사
        for category, keywords in self.happy_ending_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword.lower() in movie_text_lower:
                    category_score += 1
                    total_keywords += 1
            
            # 카테고리별 가중치 적용
            if category == 'positive_outcomes':
                score += category_score * 2.0  # 가장 중요한 지표
            elif category == 'positive_emotions':
                score += category_score * 1.5
            elif category == 'positive_actions':
                score += category_score * 1.3
            elif category == 'positive_relationships':
                score += category_score * 1.8
        
        # 정규화 (0-1 사이 값으로)
        if total_keywords > 0:
            normalized_score = min(score / (total_keywords * 0.5), 1.0)
        else:
            normalized_score = 0.0
            
        return normalized_score

    def is_happy_ending_movie(self, movie: Dict) -> bool:
        """영화가 해피엔딩인지 판단합니다."""
        # 영화 텍스트 결합
        movie_text = f"{movie.get('title', '')} {movie.get('overview', '')} {movie.get('keywords', '')}"
        
        # 해피엔딩 확률 계산
        happy_ending_score = self.analyze_happy_ending_probability(movie_text)
        
        # 장르 기반 보정
        genres = movie.get('genres', '').split(',')
        genre_bonus = 0.0
        
        for genre in genres:
            genre = genre.strip()
            if genre in self.happy_ending_genres:
                genre_bonus += 0.1
        
        # 최종 점수 계산
        final_score = happy_ending_score + genre_bonus
        
        # 임계값 (0.3 이상이면 해피엔딩으로 판단)
        return final_score >= 0.3

    def filter_happy_ending_movies(self, movies: List[Dict]) -> List[Dict]:
        """해피엔딩 영화만 필터링합니다."""
        happy_ending_movies = []
        
        for movie in movies:
            if self.is_happy_ending_movie(movie):
                # 해피엔딩 점수 추가
                movie_text = f"{movie.get('title', '')} {movie.get('overview', '')} {movie.get('keywords', '')}"
                happy_score = self.analyze_happy_ending_probability(movie_text)
                movie['happy_ending_score'] = happy_score
                happy_ending_movies.append(movie)
        
        # 해피엔딩 점수로 정렬
        happy_ending_movies.sort(key=lambda x: x.get('happy_ending_score', 0), reverse=True)
        
        return happy_ending_movies

    def get_happy_ending_recommendations(self, query: str = "", emotion: str = "기쁨", top_k: int = 5) -> List[Dict]:
        """해피엔딩 영화를 추천합니다."""
        try:
            if query:
                # 쿼리가 있는 경우 하이브리드 검색 후 해피엔딩 필터링
                results = self.hybrid_search(query, emotion, top_k * 3)
                happy_results = self.filter_happy_ending_movies(results)
            else:
                # 쿼리가 없는 경우 전체 영화에서 해피엔딩 영화만 추출
                all_movies = []
                for idx, movie in self.movies_df.iterrows():
                    movie_dict = movie.to_dict()
                    all_movies.append(movie_dict)
                
                happy_results = self.filter_happy_ending_movies(all_movies)
            
            logger.info(f"해피엔딩 영화 {len(happy_results)}개 추천 완료")
            return happy_results[:top_k]
            
        except Exception as e:
            logger.error(f"해피엔딩 영화 추천 실패: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 감정 및 장르 키워드를 추출합니다."""
        keywords = []
        
        # 감정 키워드
        emotion_keywords = {
            '기쁨': ['행복', '기쁘', '즐겁', '웃', '신나', '환희', '설렘'],
            '신뢰': ['믿음', '안정', '편안', '따뜻', '위로', '치유'],
            '두려움': ['무서', '겁', '불안', '걱정', '긴장', '스트레스'],
            '놀람': ['놀라', '충격', '예상밖', '갑작스러운'],
            '슬픔': ['슬프', '우울', '눈물', '이별', '상실', '고독', '외로움'],
            '혐오': ['싫', '역겨', '불쾌', '짜증'],
            '분노': ['화나', '분노', '열받', '짜증', '스트레스'],
            '기대': ['희망', '꿈', '새로운', '시작', '변화', '모험']
        }
        
        # 장르 키워드
        genre_keywords = {
            '액션': ['액션', '싸움', '전투', '폭력'],
            '코미디': ['코미디', '웃음', '재미', '유머'],
            '드라마': ['드라마', '감동', '인간', '감정'],
            '로맨스': ['로맨스', '사랑', '연애', '로맨틱'],
            '스릴러': ['스릴러', '긴장', '서스펜스', '미스터리'],
            '판타지': ['판타지', '마법', '상상', '신비'],
            'SF': ['SF', '과학', '미래', '우주'],
            '애니메이션': ['애니메이션', '만화', '캐릭터']
        }
        
        text_lower = text.lower()
        
        # 감정 키워드 추출
        for emotion, words in emotion_keywords.items():
            for word in words:
                if word in text_lower:
                    keywords.append(emotion)
                    break
        
        # 장르 키워드 추출
        for genre, words in genre_keywords.items():
            for word in words:
                if word in text_lower:
                    keywords.append(genre)
                    break
        
        return list(set(keywords))

    def calculate_genre_score(self, movie_genres: str, emotion: str) -> float:
        """영화의 장르와 감정에 따른 점수를 계산합니다."""
        if not movie_genres or emotion not in self.emotion_genre_preferences:
            return 1.0
        
        movie_genre_list = [g.strip() for g in movie_genres.split(',')]
        preferred_genres = self.emotion_genre_preferences[emotion]
        
        score = 1.0
        for genre in movie_genre_list:
            if genre in preferred_genres:
                score += 0.3
            if genre in self.genre_weights:
                score *= self.genre_weights[genre]
        
        return min(score, 2.0)  # 최대 2배까지 가중치 적용

    def hybrid_search(self, query: str, emotion: str, top_k: int = 5, user_id: str = "default") -> List[Dict]:
        """하이브리드 검색을 수행합니다 (리뷰 및 피드백 통합)."""
        try:
            # 더 많은 후보를 검색하도록 수정
            search_candidates = max(top_k * 3, 15)  # 최소 15개, 최대 top_k * 3개
            
            # 1. 벡터 검색
            vector_results = self._vector_search(query, search_candidates)
            
            # 2. 키워드 검색
            keyword_results = self._keyword_search(query, search_candidates)
            
            # 3. 리뷰 기반 검색 (리뷰 데이터가 있는 경우)
            review_results = []
            if self.reviews_loaded:
                review_results = self._review_search(query, search_candidates)
            
            # 4. 결과 결합 및 재점수화
            combined_results = self._combine_results_with_reviews(
                vector_results, keyword_results, review_results, emotion, top_k, user_id
            )
            
            logger.info(f"'{query}'에 대한 {len(combined_results)}개 영화 하이브리드 검색 완료")
            return combined_results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            return []

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """벡터 검색을 수행합니다."""
        try:
            # 모델이 없으면 키워드 검색으로 대체
            if self.model is None:
                logger.warning("임베딩 모델이 없습니다. 키워드 검색으로 대체합니다.")
                return self._keyword_search(query, top_k)
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])
            
            # FAISS 검색
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.movies_df):
                    movie = self.movies_df.iloc[idx].to_dict()
                    movie['vector_score'] = float(score)
                    movie['search_rank'] = i + 1
                    results.append(movie)
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            # 벡터 검색 실패 시 키워드 검색으로 대체
            logger.info("키워드 검색으로 대체합니다.")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """키워드 검색을 수행합니다."""
        try:
            keywords = self.extract_keywords(query)
            if not keywords:
                # 키워드가 없으면 모든 영화를 반환 (점수는 낮게)
                results = []
                for idx, movie in self.movies_df.iterrows():
                    movie_dict = movie.to_dict()
                    movie_dict['keyword_score'] = 0.1  # 기본 점수
                    movie_dict['search_rank'] = idx + 1
                    results.append(movie_dict)
                return results[:top_k]
            
            results = []
            for idx, movie in self.movies_df.iterrows():
                movie_text = f"{movie['title']} {movie['overview']} {movie['genres']}".lower()
                
                keyword_score = 0
                for keyword in keywords:
                    if keyword.lower() in movie_text:
                        keyword_score += 1
                
                # 키워드가 없어도 기본 점수 부여
                if keyword_score == 0:
                    keyword_score = 0.1
                
                movie_dict = movie.to_dict()
                movie_dict['keyword_score'] = keyword_score
                movie_dict['search_rank'] = len(results) + 1
                results.append(movie_dict)
            
            # 키워드 점수로 정렬
            results.sort(key=lambda x: x['keyword_score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"키워드 검색 실패: {e}")
            return []

    def _combine_results(self, vector_results: List[Dict], keyword_results: List[Dict],
                         emotion: str, top_k: int) -> List[Dict]:
        """검색 결과를 결합하고 재점수화합니다."""
        combined = {}
        
        # 벡터 검색 결과 처리
        for movie in vector_results:
            movie_id = movie.get('title', '')
            if movie_id not in combined:
                combined[movie_id] = movie.copy()
                combined[movie_id]['final_score'] = movie.get('vector_score', 0) * 0.6
            else:
                combined[movie_id]['final_score'] += movie.get('vector_score', 0) * 0.6
        
        # 키워드 검색 결과 처리
        for movie in keyword_results:
            movie_id = movie.get('title', '')
            if movie_id not in combined:
                combined[movie_id] = movie.copy()
                combined[movie_id]['final_score'] = movie.get('keyword_score', 0) * 0.4
            else:
                combined[movie_id]['final_score'] += movie.get('keyword_score', 0) * 0.4
        
        # 장르 가중치 적용
        for movie_id, movie in combined.items():
            genre_score = self.calculate_genre_score(movie.get('genres', ''), emotion)
            movie['final_score'] *= genre_score
        
        # 최종 점수로 정렬
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:top_k]
    
    def _review_search(self, query: str, top_k: int) -> List[Dict]:
        """리뷰 기반 검색을 수행합니다."""
        try:
            if not self.reviews_loaded:
                return []
            
            # 유사한 리뷰 검색
            similar_reviews = self.review_processor.search_similar_reviews(query, top_k)
            
            # 리뷰에서 영화 ID 추출
            movie_scores = {}
            for review in similar_reviews:
                movie_id = review['movie_id']
                score = review['score']
                
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = []
                movie_scores[movie_id].append(score)
            
            # 영화별 평균 점수 계산
            results = []
            for movie_id, scores in movie_scores.items():
                avg_score = np.mean(scores)
                
                # 영화 정보 찾기
                movie_info = self.movies_df[self.movies_df['id'] == movie_id]
                if not movie_info.empty:
                    movie = movie_info.iloc[0].to_dict()
                    movie['review_score'] = avg_score
                    movie['search_rank'] = len(results) + 1
                    results.append(movie)
            
            return results
            
        except Exception as e:
            logger.error(f"리뷰 검색 실패: {e}")
            return []
    
    def _combine_results_with_reviews(self, vector_results: List[Dict], keyword_results: List[Dict],
                                    review_results: List[Dict], emotion: str, top_k: int, user_id: str) -> List[Dict]:
        """검색 결과를 결합하고 재점수화합니다 (리뷰 및 피드백 통합)."""
        combined = {}
        
        # 벡터 검색 결과 처리 (가중치: 0.4)
        for movie in vector_results:
            movie_id = movie.get('id', movie.get('title', ''))
            if movie_id not in combined:
                combined[movie_id] = movie.copy()
                combined[movie_id]['final_score'] = movie.get('vector_score', 0) * 0.4
            else:
                combined[movie_id]['final_score'] += movie.get('vector_score', 0) * 0.4
        
        # 키워드 검색 결과 처리 (가중치: 0.3)
        for movie in keyword_results:
            movie_id = movie.get('id', movie.get('title', ''))
            if movie_id not in combined:
                combined[movie_id] = movie.copy()
                combined[movie_id]['final_score'] = movie.get('keyword_score', 0) * 0.3
            else:
                combined[movie_id]['final_score'] += movie.get('keyword_score', 0) * 0.3
        
        # 리뷰 검색 결과 처리 (가중치: 0.2)
        for movie in review_results:
            movie_id = movie.get('id', movie.get('title', ''))
            if movie_id not in combined:
                combined[movie_id] = movie.copy()
                combined[movie_id]['final_score'] = movie.get('review_score', 0) * 0.2
            else:
                combined[movie_id]['final_score'] += movie.get('review_score', 0) * 0.2
        
        # 장르 가중치 적용
        for movie_id, movie in combined.items():
            genre_score = self.calculate_genre_score(movie.get('genres', ''), emotion)
            movie['final_score'] *= genre_score
            
            # 피드백 기반 부스트 점수 (가중치: 0.1)
            if user_id != "default":
                feedback_boost = self.feedback_system.get_recommendation_boost(
                    user_id, movie.get('id', 0), emotion
                )
                movie['final_score'] += feedback_boost * 0.1
                movie['feedback_boost'] = feedback_boost
        
        # 최종 점수로 정렬
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 최소한 top_k개가 되도록 보장
        if len(final_results) < top_k:
            # 부족한 만큼 키워드 검색 결과에서 추가
            remaining = top_k - len(final_results)
            for movie in keyword_results:
                movie_id = movie.get('id', movie.get('title', ''))
                if movie_id not in [m.get('id', m.get('title', '')) for m in final_results]:
                    final_results.append(movie)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        return final_results[:top_k]

    def filter_by_genre(self, movies: List[Dict], target_genres: List[str]) -> List[Dict]:
        """장르별로 영화를 필터링합니다."""
        if not target_genres:
            return movies
        
        filtered = []
        for movie in movies:
            movie_genres = [g.strip() for g in movie.get('genres', '').split(',')]
            if any(genre in movie_genres for genre in target_genres):
                filtered.append(movie)
        
        return filtered

    def filter_by_year(self, movies: List[Dict], min_year: int = None, max_year: int = None) -> List[Dict]:
        """연도별로 영화를 필터링합니다."""
        if min_year is None and max_year is None:
            return movies
        
        filtered = []
        for movie in movies:
            release_date = movie.get('release_date', '')
            if release_date:
                try:
                    year = int(release_date.split('-')[0])
                    if min_year and year < min_year:
                        continue
                    if max_year and year > max_year:
                        continue
                    filtered.append(movie)
                except:
                    continue
            else:
                filtered.append(movie)
        
        return filtered

    def filter_by_mood(self, movies: List[Dict], target_moods: List[str]) -> List[Dict]:
        """분위기별로 영화를 필터링합니다."""
        if not target_moods:
            return movies
        
        filtered = []
        for movie in movies:
            movie_mood = movie.get('mood', '').lower()
            if any(mood.lower() in movie_mood for mood in target_moods):
                filtered.append(movie)
        
        return filtered

    def filter_by_ending(self, movies: List[Dict], target_ending: str) -> List[Dict]:
        """결말별로 영화를 필터링합니다."""
        if not target_ending:
            return movies
        
        filtered = []
        for movie in movies:
            movie_ending = movie.get('ending', '').lower()
            if target_ending.lower() in movie_ending:
                filtered.append(movie)
        
        return filtered

    def filter_by_theme(self, movies: List[Dict], target_themes: List[str]) -> List[Dict]:
        """테마별로 영화를 필터링합니다."""
        if not target_themes:
            return movies
        
        filtered = []
        for movie in movies:
            movie_theme = movie.get('theme', '').lower()
            if any(theme.lower() in movie_theme for theme in target_themes):
                filtered.append(movie)
        
        return filtered

    def filter_by_tone(self, movies: List[Dict], target_tones: List[str]) -> List[Dict]:
        """톤별로 영화를 필터링합니다."""
        if not target_tones:
            return movies
        
        filtered = []
        for movie in movies:
            movie_tone = movie.get('tone', '').lower()
            if any(tone.lower() in movie_tone for tone in target_tones):
                filtered.append(movie)
        
        return filtered

    def advanced_search(self, query: str, emotion: str, filters: Dict = None) -> List[Dict]:
        """고급 검색을 수행합니다."""
        try:
            # 기본 하이브리드 검색
            results = self.hybrid_search(query, emotion, filters.get('top_k', 5) if filters else 5)
            
            # 필터 적용
            if filters:
                # 기존 필터들
                if 'genres' in filters:
                    results = self.filter_by_genre(results, filters['genres'])
                
                if 'min_year' in filters or 'max_year' in filters:
                    results = self.filter_by_year(results, 
                                               filters.get('min_year'), 
                                               filters.get('max_year'))
                
                # 새로운 필터들
                if 'moods' in filters:
                    results = self.filter_by_mood(results, filters['moods'])
                
                if 'ending' in filters:
                    results = self.filter_by_ending(results, filters['ending'])
                
                if 'themes' in filters:
                    results = self.filter_by_theme(results, filters['themes'])
                
                if 'tones' in filters:
                    results = self.filter_by_tone(results, filters['tones'])
                
                # 해피엔딩 필터 추가
                if filters.get('happy_ending_only', False):
                    results = self.filter_happy_ending_movies(results)
            
            return results[:5]  # 최대 5개 반환
            
        except Exception as e:
            logger.error(f"고급 검색 실패: {e}")
            return []

    def analyze_emotion(self, query: str) -> Dict[str, float]:
        """감정을 분석합니다."""
        return self.emotion_analyzer.analyze_emotion(query)

    def get_primary_emotion(self, query: str) -> Tuple[str, float]:
        """주요 감정을 도출합니다."""
        emotion_scores = self.analyze_emotion(query)
        return self.emotion_analyzer.get_primary_emotion(emotion_scores)

    def get_emotion_recommendation_prompt(self, query: str) -> str:
        """감정 기반 추천 프롬프트를 생성합니다."""
        emotion_scores = self.analyze_emotion(query)
        return self.emotion_analyzer.get_movie_recommendation_prompt(emotion_scores) 