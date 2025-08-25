import pandas as pd
import numpy as np
import faiss
import logging
import os
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieDataProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        영화 데이터 처리 클래스
        
        Args:
            model_name: 사용할 임베딩 모델명
        """
        self.model_name = model_name
        self.model = None
        self.movies_df = None
        self.index = None
        
    def download_tmdb_data(self) -> pd.DataFrame:
        """TMDB API를 통해 실제 영화 데이터를 다운로드합니다."""
        TMDB_API_KEY = os.getenv("TMDB_API_KEY")
        
        if not TMDB_API_KEY:
            logger.warning("TMDB_API_KEY가 설정되지 않았습니다. 샘플 데이터를 사용합니다.")
            return self._create_sample_data()
        
        try:
            movies = []
            
            # 인기 영화 데이터 수집 (여러 페이지)
            for page in range(1, 6):  # 5페이지 = 100개 영화
                url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=ko-KR&page={page}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for movie in data['results']:
                        # 각 영화의 상세 정보 가져오기
                        detail_url = f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={TMDB_API_KEY}&language=ko-KR&append_to_response=keywords"
                        detail_response = requests.get(detail_url)
                        
                        if detail_response.status_code == 200:
                            detail = detail_response.json()
                            
                            # 키워드 정보 가져오기
                            keywords_url = f"https://api.themoviedb.org/3/movie/{movie['id']}/keywords?api_key={TMDB_API_KEY}"
                            keywords_response = requests.get(keywords_url)
                            keywords = []
                            if keywords_response.status_code == 200:
                                keywords_data = keywords_response.json()
                                keywords = [kw['name'] for kw in keywords_data.get('keywords', [])]
                            
                            # 분위기와 결말 추정 (키워드 기반)
                            mood, ending, theme, tone = self._analyze_movie_metadata(
                                detail.get('overview', ''),
                                keywords,
                                detail.get('genres', []),
                                detail.get('vote_average', 0)
                            )
                            
                            movie_data = {
                                'title': detail.get('title', ''),
                                'overview': detail.get('overview', ''),
                                'genres': ', '.join([g['name'] for g in detail.get('genres', [])]),
                                'tagline': detail.get('tagline', ''),
                                'release_date': detail.get('release_date', ''),
                                'vote_average': detail.get('vote_average', 0),
                                'popularity': detail.get('popularity', 0),
                                'mood': mood,
                                'ending': ending,
                                'theme': theme,
                                'tone': tone,
                                'keywords': ', '.join(keywords)
                            }
                            
                            movies.append(movie_data)
                            
                            # API 호출 제한을 위한 딜레이
                            time.sleep(0.1)
                
                logger.info(f"TMDB API에서 {len(movies)}개 영화 데이터 수집 완료")
                return pd.DataFrame(movies)
                
        except Exception as e:
            logger.error(f"TMDB 데이터 다운로드 실패: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """샘플 영화 데이터를 생성합니다."""
        movies = [
            {
                'id': 38,
                'title': 'Eternal Sunshine of the Spotless Mind',
                'overview': '사랑하는 사람의 기억을 지우려는 남자의 이야기. 이별의 아픔과 치유를 다룬 영화.',
                'genres': '드라마, 로맨스, SF',
                'tagline': '사랑의 기억을 지울 수 있을까?',
                'release_date': '2004-03-19',
                'mood': '치유적, 감동적, 희망적',
                'ending': '해피엔딩',
                'theme': '사랑, 치유, 성장',
                'tone': '따뜻한, 감동적인'
            },
            {
                'id': 313369,
                'title': 'The Secret Life of Walter Mitty',
                'overview': '평범한 직장인이 모험을 떠나 자신의 꿈을 찾는 이야기. 동기부여와 성장을 다룬 영화.',
                'genres': '코미디, 드라마, 모험',
                'tagline': '당신의 꿈을 찾아 떠나는 여행',
                'release_date': '2013-12-25',
                'mood': '동기부여하는, 경쾌한, 희망적',
                'ending': '해피엔딩',
                'theme': '성장, 모험, 자아발견',
                'tone': '경쾌한, 동기부여하는'
            },
            {
                'id': 313369,
                'title': 'La La Land',
                'overview': '꿈을 향해 나아가는 두 젊은이의 사랑과 성공을 다룬 뮤지컬 영화.',
                'genres': '뮤지컬, 로맨스, 드라마',
                'tagline': '꿈과 사랑이 만나는 곳',
                'release_date': '2016-12-09',
                'mood': '로맨틱한, 경쾌한, 감동적',
                'ending': '해피엔딩',
                'theme': '사랑, 꿈, 성공',
                'tone': '로맨틱한, 경쾌한'
            },
            {
                'id': 120467,
                'title': 'The Grand Budapest Hotel',
                'overview': '호텔에서 일어나는 코믹한 모험을 다룬 영화. 웃음과 따뜻함이 가득한 이야기.',
                'genres': '코미디, 모험, 드라마',
                'tagline': '세상에서 가장 특별한 호텔',
                'release_date': '2014-03-07',
                'mood': '경쾌한, 재미있는, 따뜻한',
                'ending': '해피엔딩',
                'theme': '우정, 모험, 따뜻함',
                'tone': '경쾌한, 재미있는'
            },
            {
                'id': 301528,
                'title': '500 Days of Summer',
                'overview': '사랑에 대한 현실적인 이야기. 이별과 성장을 다룬 로맨틱 코미디.',
                'genres': '로맨스, 코미디, 드라마',
                'tagline': '사랑은 예상대로 되지 않는다',
                'release_date': '2009-07-17',
                'mood': '로맨틱한, 감동적, 성장하는',
                'ending': '희망적',
                'theme': '사랑, 성장, 현실',
                'tone': '로맨틱한, 감동적인'
            },
            {
                'title': 'The Intern',
                'overview': '은퇴한 남자가 젊은 CEO의 인턴이 되어 서로를 돕는 따뜻한 이야기.',
                'genres': '코미디, 드라마',
                'tagline': '나이는 숫자일 뿐',
                'release_date': '2015-09-25',
                'mood': '따뜻한, 재미있는, 감동적',
                'ending': '해피엔딩',
                'theme': '우정, 성장, 따뜻함',
                'tone': '따뜻한, 재미있는'
            },
            {
                'title': 'About Time',
                'overview': '시간을 되돌릴 수 있는 능력을 가진 남자의 사랑과 가족에 대한 이야기.',
                'genres': '로맨스, 드라마, 판타지',
                'tagline': '시간을 되돌릴 수 있다면',
                'release_date': '2013-09-04',
                'mood': '따뜻한, 감동적, 희망적',
                'ending': '해피엔딩',
                'theme': '사랑, 가족, 시간',
                'tone': '따뜻한, 감동적인'
            },
            {
                'title': 'The Devil Wears Prada',
                'overview': '패션 잡지에서 일하는 여성의 성장과 도전을 다룬 영화.',
                'genres': '코미디, 드라마',
                'tagline': '패션의 세계로의 도전',
                'release_date': '2006-06-30',
                'mood': '동기부여하는, 재미있는, 성장하는',
                'ending': '해피엔딩',
                'theme': '성장, 도전, 성공',
                'tone': '동기부여하는, 재미있는'
            },
            {
                'title': 'The Notebook',
                'overview': '오랫동안 이어져 온 사랑의 이야기. 로맨틱하고 감동적인 멜로드라마.',
                'genres': '로맨스, 드라마, 멜로드라마',
                'tagline': '영원한 사랑의 이야기',
                'release_date': '2004-06-25',
                'mood': '로맨틱한, 감동적, 따뜻한',
                'ending': '해피엔딩',
                'theme': '사랑, 영원, 감동',
                'tone': '로맨틱한, 감동적인'
            },
            {
                'title': 'The Pursuit of Happyness',
                'overview': '가난한 아버지가 아들과 함께 성공을 향해 나아가는 실화 기반 영화.',
                'genres': '드라마, 전기',
                'tagline': '행복을 향한 여정',
                'release_date': '2006-12-15',
                'mood': '동기부여하는, 감동적, 희망적',
                'ending': '해피엔딩',
                'theme': '성공, 가족, 희망',
                'tone': '동기부여하는, 감동적인'
            }
        ]
        
        return pd.DataFrame(movies)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        영화 데이터를 전처리합니다.
        
        Args:
            df: 원본 영화 데이터
            
        Returns:
            전처리된 데이터
        """
        # NaN 값 처리
        df = df.fillna('')
        
        # 필요한 컬럼만 선택 (ID 포함)
        required_columns = ['id', 'title', 'overview', 'genres', 'tagline']
        df = df[required_columns].copy()
        
        # 텍스트 필드 결합 (검색용)
        df['combined_text'] = df['title'] + ' ' + df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']
        
        # 텍스트 정리
        df['combined_text'] = df['combined_text'].str.strip()
        
        logger.info(f"데이터 전처리 완료: {len(df)}개 영화")
        return df
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        텍스트를 임베딩으로 변환합니다.
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 배열
        """
        logger.info(f"모델 로딩 중: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("모델 로딩 완료")
        
        # 임베딩 생성
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"임베딩 생성 완료: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        FAISS 인덱스를 생성합니다.
        
        Args:
            embeddings: 임베딩 배열
            
        Returns:
            FAISS 인덱스
        """
        # 정규화
        faiss.normalize_L2(embeddings)
        
        # FAISS 인덱스 생성 (코사인 유사도)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS 인덱스 생성 완료: {index.ntotal}개 벡터")
        return index
    
    def process_and_save(self, save_path: str = "movie_data"):
        """
        데이터를 처리하고 저장합니다.
        
        Args:
            save_path: 저장 경로
        """
        # 데이터 다운로드
        self.movies_df = self.download_tmdb_data()
        
        # 전처리
        self.movies_df = self.preprocess_data(self.movies_df)
        
        # 임베딩 생성
        embeddings = self.create_embeddings(self.movies_df['combined_text'].tolist())
        
        # FAISS 인덱스 생성
        self.index = self.build_faiss_index(embeddings)
        
        # 저장
        os.makedirs(save_path, exist_ok=True)
        
        # DataFrame 저장
        self.movies_df.to_csv(os.path.join(save_path, 'movies.csv'), index=False)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, os.path.join(save_path, 'faiss_index.bin'))
        
        # 모델 저장 (선택사항)
        self.model.save(os.path.join(save_path, 'embedding_model'))
        
        logger.info(f"데이터 저장 완료: {save_path}")
    
    def load_processed_data(self, load_path: str = "movie_data"):
        """
        저장된 데이터를 로드합니다.
        
        Args:
            load_path: 로드 경로
        """
        # DataFrame 로드
        self.movies_df = pd.read_csv(os.path.join(load_path, 'movies.csv'))
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(os.path.join(load_path, 'faiss_index.bin'))
        
        # 모델 로드
        self.model = SentenceTransformer(os.path.join(load_path, 'embedding_model'))
        
        logger.info("저장된 데이터 로드 완료")

    def _analyze_movie_metadata(self, overview: str, keywords: List[str], genres: List[Dict], rating: float) -> Tuple[str, str, str, str]:
        """영화 메타데이터를 분석하여 분위기, 결말, 테마, 톤을 추정합니다."""
        
        # 키워드와 개요를 결합한 텍스트
        text = f"{overview} {' '.join(keywords)}".lower()
        genre_names = [g['name'].lower() for g in genres]
        
        # 분위기 분석
        mood_keywords = {
            '따뜻한': ['따뜻', '가족', '사랑', '우정', '치유', '위로'],
            '경쾌한': ['모험', '액션', '스릴', '재미', '웃음', '코미디'],
            '로맨틱한': ['사랑', '로맨스', '연애', '설렘', '로맨틱'],
            '감동적인': ['감동', '감동적', '울음', '눈물', '희망'],
            '동기부여하는': ['성공', '성장', '도전', '꿈', '희망', '동기'],
            '재미있는': ['재미', '웃음', '코미디', '유머', '즐거운'],
            '치유적인': ['치유', '위로', '회복', '희망', '따뜻'],
            '희망적인': ['희망', '성공', '성장', '꿈', '미래']
        }
        
        detected_moods = []
        for mood, keywords_list in mood_keywords.items():
            if any(keyword in text for keyword in keywords_list):
                detected_moods.append(mood)
        
        mood = ', '.join(detected_moods[:2]) if detected_moods else '일반적'
        
        # 결말 추정 (키워드와 평점 기반)
        ending_keywords = {
            '해피엔딩': ['성공', '희망', '사랑', '가족', '성장'],
            '희망적': ['희망', '미래', '성장', '성공'],
            '현실적': ['현실', '일상', '성장', '이별'],
            '열린 결말': ['모호', '열린', '상상', '미래']
        }
        
        ending = '해피엔딩'  # 기본값
        for ending_type, keywords_list in ending_keywords.items():
            if any(keyword in text for keyword in keywords_list):
                ending = ending_type
                break
        
        # 평점이 낮으면 현실적 결말로 조정
        if rating < 6.0:
            ending = '현실적'
        
        # 테마 분석
        theme_keywords = {
            '사랑': ['사랑', '로맨스', '연애', '설렘'],
            '성장': ['성장', '성공', '도전', '꿈'],
            '우정': ['우정', '친구', '동료', '함께'],
            '가족': ['가족', '부모', '자식', '따뜻'],
            '성공': ['성공', '도전', '성장', '꿈'],
            '치유': ['치유', '위로', '회복', '희망'],
            '모험': ['모험', '여행', '도전', '새로운'],
            '희망': ['희망', '꿈', '미래', '성공']
        }
        
        detected_themes = []
        for theme, keywords_list in theme_keywords.items():
            if any(keyword in text for keyword in keywords_list):
                detected_themes.append(theme)
        
        theme = ', '.join(detected_themes[:3]) if detected_themes else '일반'
        
        # 톤 분석
        tone_keywords = {
            '따뜻한': ['따뜻', '가족', '사랑', '우정'],
            '경쾌한': ['모험', '액션', '스릴', '재미'],
            '로맨틱한': ['사랑', '로맨스', '설렘'],
            '감동적인': ['감동', '울음', '희망'],
            '동기부여하는': ['성공', '성장', '도전'],
            '재미있는': ['재미', '웃음', '코미디']
        }
        
        detected_tones = []
        for tone, keywords_list in tone_keywords.items():
            if any(keyword in text for keyword in keywords_list):
                detected_tones.append(tone)
        
        tone = ', '.join(detected_tones[:2]) if detected_tones else '일반적'
        
        return mood, ending, theme, tone

if __name__ == "__main__":
    # 테스트
    processor = MovieDataProcessor()
    processor.process_and_save() 