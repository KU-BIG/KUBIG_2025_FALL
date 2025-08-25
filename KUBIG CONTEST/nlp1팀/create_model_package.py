import os
import json
import shutil
from pathlib import Path

def create_model_package():
    """감정분석 모델 패키지를 생성합니다."""
    
    # 1. 패키지 디렉토리 생성
    package_dir = Path("./emotion_analyzer_package")
    package_dir.mkdir(exist_ok=True)
    
    # 2. 필요한 파일들 복사
    files_to_copy = [
        "emotion_analyzer.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, package_dir / file)
    
    # 3. 모델 설정 파일 생성
    model_config = {
        "model_name": "klue/bert-base",
        "emotion_labels": [
            '기쁨', '신뢰', '두려움', '놀람', '슬픔', '혐오', '분노', '기대',
            '사랑', '외로움', '스트레스', '불안', '희망', '감사', '후회', '열정',
            '평온', '설렘', '답답함', '속상함', '짜증', '우울', '답답해', '속상해'
        ],
        "num_labels": 24,
        "problem_type": "multi_label_classification",
        "max_length": 512
    }
    
    with open(package_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    # 4. 감정 키워드 데이터 저장
    emotion_keywords = {
        '기쁨': ['행복', '기쁘', '즐겁', '웃', '신나', '환희', '설렘', '만족', '성취', '축하'],
        '슬픔': ['슬프', '우울', '눈물', '이별', '상실', '고독', '외로움', '절망', '허전', '공허'],
        '스트레스': ['스트레스', '압박', '부담', '긴장', '피곤', '지치', '힘들'],
        '외로움': ['외로', '고독', '혼자', '허전', '공허', '고립', '고독'],
        '사랑': ['사랑', '연애', '로맨스', '설렘', '마음', '감정', '연인', '애정'],
        '기대': ['희망', '꿈', '새로운', '시작', '변화', '모험', '기대', '설렘', '동경']
    }
    
    with open(package_dir / "emotion_keywords.json", "w", encoding="utf-8") as f:
        json.dump(emotion_keywords, f, ensure_ascii=False, indent=2)
    
    # 5. 사용법 가이드 생성
    usage_guide = """# 감정 분석 모델 사용법

## 설치

```bash
pip install -r requirements.txt
```

## 기본 사용법

```python
from emotion_analyzer import EmotionAnalyzer

# 모델 초기화
analyzer = EmotionAnalyzer()

# 감정 분석
text = "이별 후 공허함을 느끼고 있어"
emotion_scores = analyzer.analyze_emotion(text)
primary_emotion, score = analyzer.get_primary_emotion(emotion_scores)

print(f"주요 감정: {primary_emotion} ({score:.2f})")
print(f"전체 점수: {emotion_scores}")
```

## 고급 사용법

```python
# 복합 감정 분석
complex_analysis = analyzer.analyze_complex_emotion(text)
print(f"복합 감정: {complex_analysis['complex_emotion']}")

# 상황 기반 분석
situation_analysis = analyzer.analyze_situation_emotion(text)
print(f"상황: {situation_analysis['situations']}")

# 종합 분석
comprehensive = analyzer.get_comprehensive_emotion_analysis(text)
print(f"분석 요약: {comprehensive['analysis_summary']}")
```

## 모델 정보

- 기반 모델: KLUE-BERT
- 감정 카테고리: 24가지
- 분류 방식: 다중 레이블 분류
- 최대 입력 길이: 512 토큰
"""
    
    with open(package_dir / "USAGE.md", "w", encoding="utf-8") as f:
        f.write(usage_guide)
    
    # 6. 압축 파일 생성
    shutil.make_archive("emotion_analyzer_package", "zip", package_dir)
    
    print("모델 패키지가 생성되었습니다!")
    print(f"패키지 위치: {package_dir}")
    print("압축 파일: emotion_analyzer_package.zip")
    print("\n공유 방법:")
    print("1. emotion_analyzer_package.zip 파일을 공유")
    print("2. 압축 해제 후 README.md와 USAGE.md 참조")

if __name__ == "__main__":
    create_model_package()
