import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """사용자 피드백을 수집하고 학습하는 시스템"""
    
    def __init__(self, feedback_file: str = 'user_feedback.json'):
        self.feedback_file = feedback_file
        self.feedback_data = self.load_feedback_data()
        
    def load_feedback_data(self) -> Dict:
        """피드백 데이터를 로드합니다."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"피드백 데이터 로드 실패: {e}")
        
        return {
            'feedback_history': [],
            'user_preferences': {},
            'movie_ratings': {},
            'emotion_patterns': {}
        }
    
    def save_feedback_data(self) -> None:
        """피드백 데이터를 저장합니다."""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            logger.info("피드백 데이터 저장 완료")
        except Exception as e:
            logger.error(f"피드백 데이터 저장 실패: {e}")
    
    def add_movie_feedback(self, user_id: str, movie_id: int, 
                          user_emotion: str, movie_title: str,
                          rating: int, feedback_text: str = "") -> None:
        """영화 추천에 대한 피드백을 추가합니다."""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'movie_id': movie_id,
            'movie_title': movie_title,
            'user_emotion': user_emotion,
            'rating': rating,  # 1-5점
            'feedback_text': feedback_text,
            'feedback_type': 'movie_recommendation'
        }
        
        self.feedback_data['feedback_history'].append(feedback_entry)
        
        # 사용자 선호도 업데이트
        self._update_user_preferences(user_id, movie_id, rating, user_emotion)
        
        # 영화 평점 업데이트
        self._update_movie_ratings(movie_id, rating, user_emotion)
        
        # 감정 패턴 업데이트
        self._update_emotion_patterns(user_emotion, movie_id, rating)
        
        self.save_feedback_data()
        logger.info(f"피드백 추가 완료: {movie_title} (평점: {rating})")
    
    def add_emotion_feedback(self, user_id: str, original_emotion: str,
                           actual_emotion: str, feedback_text: str = "") -> None:
        """감정 분석에 대한 피드백을 추가합니다."""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'original_emotion': original_emotion,
            'actual_emotion': actual_emotion,
            'feedback_text': feedback_text,
            'feedback_type': 'emotion_analysis'
        }
        
        self.feedback_data['feedback_history'].append(feedback_entry)
        
        # 감정 분석 정확도 패턴 업데이트
        self._update_emotion_accuracy_patterns(original_emotion, actual_emotion)
        
        self.save_feedback_data()
        logger.info(f"감정 피드백 추가 완료: {original_emotion} → {actual_emotion}")
    
    def _update_user_preferences(self, user_id: str, movie_id: int, 
                               rating: int, user_emotion: str) -> None:
        """사용자 선호도를 업데이트합니다."""
        if user_id not in self.feedback_data['user_preferences']:
            self.feedback_data['user_preferences'][user_id] = {
                'preferred_genres': {},
                'preferred_emotions': {},
                'movie_ratings': {},
                'total_feedback': 0
            }
        
        user_prefs = self.feedback_data['user_preferences'][user_id]
        user_prefs['total_feedback'] += 1
        
        # 영화 평점 저장
        if movie_id not in user_prefs['movie_ratings']:
            user_prefs['movie_ratings'][movie_id] = []
        user_prefs['movie_ratings'][movie_id].append({
            'rating': rating,
            'emotion': user_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        # 감정별 선호도 업데이트
        if user_emotion not in user_prefs['preferred_emotions']:
            user_prefs['preferred_emotions'][user_emotion] = {
                'total_ratings': 0,
                'avg_rating': 0.0,
                'high_rated_movies': []  # 4점 이상 영화들
            }
        
        emotion_prefs = user_prefs['preferred_emotions'][user_emotion]
        emotion_prefs['total_ratings'] += 1
        
        # 평균 평점 계산
        total_rating = emotion_prefs['avg_rating'] * (emotion_prefs['total_ratings'] - 1) + rating
        emotion_prefs['avg_rating'] = total_rating / emotion_prefs['total_ratings']
        
        # 높은 평점 영화 저장
        if rating >= 4:
            emotion_prefs['high_rated_movies'].append(movie_id)
    
    def _update_movie_ratings(self, movie_id: int, rating: int, user_emotion: str) -> None:
        """영화 평점을 업데이트합니다."""
        if movie_id not in self.feedback_data['movie_ratings']:
            self.feedback_data['movie_ratings'][movie_id] = {
                'total_ratings': 0,
                'avg_rating': 0.0,
                'emotion_ratings': {},
                'recent_ratings': []
            }
        
        movie_ratings = self.feedback_data['movie_ratings'][movie_id]
        movie_ratings['total_ratings'] += 1
        
        # 평균 평점 계산
        total_rating = movie_ratings['avg_rating'] * (movie_ratings['total_ratings'] - 1) + rating
        movie_ratings['avg_rating'] = total_rating / movie_ratings['total_ratings']
        
        # 감정별 평점
        if user_emotion not in movie_ratings['emotion_ratings']:
            movie_ratings['emotion_ratings'][user_emotion] = {
                'count': 0,
                'avg_rating': 0.0
            }
        
        emotion_rating = movie_ratings['emotion_ratings'][user_emotion]
        emotion_rating['count'] += 1
        
        total_emotion_rating = emotion_rating['avg_rating'] * (emotion_rating['count'] - 1) + rating
        emotion_rating['avg_rating'] = total_emotion_rating / emotion_rating['count']
        
        # 최근 평점 저장 (최대 10개)
        movie_ratings['recent_ratings'].append({
            'rating': rating,
            'emotion': user_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(movie_ratings['recent_ratings']) > 10:
            movie_ratings['recent_ratings'] = movie_ratings['recent_ratings'][-10:]
    
    def _update_emotion_patterns(self, user_emotion: str, movie_id: int, rating: int) -> None:
        """감정 패턴을 업데이트합니다."""
        if user_emotion not in self.feedback_data['emotion_patterns']:
            self.feedback_data['emotion_patterns'][user_emotion] = {
                'high_rated_movies': [],
                'low_rated_movies': [],
                'total_feedback': 0,
                'avg_rating': 0.0
            }
        
        emotion_pattern = self.feedback_data['emotion_patterns'][user_emotion]
        emotion_pattern['total_feedback'] += 1
        
        # 평균 평점 계산
        total_rating = emotion_pattern['avg_rating'] * (emotion_pattern['total_feedback'] - 1) + rating
        emotion_pattern['avg_rating'] = total_rating / emotion_pattern['total_feedback']
        
        # 높은/낮은 평점 영화 분류
        if rating >= 4:
            if movie_id not in emotion_pattern['high_rated_movies']:
                emotion_pattern['high_rated_movies'].append(movie_id)
        elif rating <= 2:
            if movie_id not in emotion_pattern['low_rated_movies']:
                emotion_pattern['low_rated_movies'].append(movie_id)
    
    def _update_emotion_accuracy_patterns(self, original_emotion: str, actual_emotion: str) -> None:
        """감정 분석 정확도 패턴을 업데이트합니다."""
        if 'emotion_accuracy' not in self.feedback_data:
            self.feedback_data['emotion_accuracy'] = {}
        
        if original_emotion not in self.feedback_data['emotion_accuracy']:
            self.feedback_data['emotion_accuracy'][original_emotion] = {
                'corrections': {},
                'total_feedback': 0
            }
        
        accuracy_data = self.feedback_data['emotion_accuracy'][original_emotion]
        accuracy_data['total_feedback'] += 1
        
        if actual_emotion not in accuracy_data['corrections']:
            accuracy_data['corrections'][actual_emotion] = 0
        accuracy_data['corrections'][actual_emotion] += 1
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """사용자 선호도를 반환합니다."""
        return self.feedback_data['user_preferences'].get(user_id, {})
    
    def get_movie_rating(self, movie_id: int) -> Dict:
        """영화 평점 정보를 반환합니다."""
        return self.feedback_data['movie_ratings'].get(movie_id, {})
    
    def get_emotion_patterns(self) -> Dict:
        """감정 패턴을 반환합니다."""
        return self.feedback_data['emotion_patterns']
    
    def get_emotion_accuracy(self) -> Dict:
        """감정 분석 정확도를 반환합니다."""
        return self.feedback_data.get('emotion_accuracy', {})
    
    def get_recommendation_boost(self, user_id: str, movie_id: int, 
                               user_emotion: str) -> float:
        """피드백 기반 추천 부스트 점수를 계산합니다."""
        boost = 0.0
        
        # 1. 사용자 개인 선호도
        user_prefs = self.get_user_preferences(user_id)
        if user_emotion in user_prefs.get('preferred_emotions', {}):
            emotion_prefs = user_prefs['preferred_emotions'][user_emotion]
            if movie_id in emotion_prefs.get('high_rated_movies', []):
                boost += 0.3  # 이 감정에서 높은 평점을 준 영화
        
        # 2. 영화의 감정별 평점
        movie_rating = self.get_movie_rating(movie_id)
        if user_emotion in movie_rating.get('emotion_ratings', {}):
            emotion_rating = movie_rating['emotion_ratings'][user_emotion]
            avg_rating = emotion_rating.get('avg_rating', 0.0)
            if avg_rating >= 4.0:
                boost += 0.2  # 이 감정에서 높은 평점을 받은 영화
        
        # 3. 감정 패턴
        emotion_patterns = self.get_emotion_patterns()
        if user_emotion in emotion_patterns:
            pattern = emotion_patterns[user_emotion]
            if movie_id in pattern.get('high_rated_movies', []):
                boost += 0.1  # 이 감정에서 전반적으로 높은 평점을 받은 영화
        
        return min(boost, 0.5)  # 최대 0.5점 부스트
    
    def get_feedback_summary(self) -> Dict:
        """피드백 요약을 반환합니다."""
        total_feedback = len(self.feedback_data['feedback_history'])
        total_users = len(self.feedback_data['user_preferences'])
        total_movies = len(self.feedback_data['movie_ratings'])
        
        # 평균 평점 계산
        all_ratings = []
        for feedback in self.feedback_data['feedback_history']:
            if feedback.get('feedback_type') == 'movie_recommendation':
                all_ratings.append(feedback.get('rating', 0))
        
        avg_rating = np.mean(all_ratings) if all_ratings else 0.0
        
        return {
            'total_feedback': total_feedback,
            'total_users': total_users,
            'total_movies': total_movies,
            'average_rating': round(avg_rating, 2),
            'feedback_types': {
                'movie_recommendation': len([f for f in self.feedback_data['feedback_history'] 
                                          if f.get('feedback_type') == 'movie_recommendation']),
                'emotion_analysis': len([f for f in self.feedback_data['feedback_history'] 
                                       if f.get('feedback_type') == 'emotion_analysis'])
            }
        }

if __name__ == "__main__":
    # 테스트
    feedback_system = FeedbackSystem()
    
    # 샘플 피드백 추가
    feedback_system.add_movie_feedback(
        user_id="user1",
        movie_id=19995,  # Avatar
        user_emotion="기쁨",
        movie_title="Avatar",
        rating=5,
        feedback_text="정말 재미있었어요!"
    )
    
    feedback_system.add_emotion_feedback(
        user_id="user1",
        original_emotion="슬픔",
        actual_emotion="외로움",
        feedback_text="외로움이 더 정확해요"
    )
    
    # 요약 출력
    summary = feedback_system.get_feedback_summary()
    print("피드백 요약:", summary) 