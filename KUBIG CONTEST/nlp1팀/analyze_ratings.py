import pandas as pd
import numpy as np

def analyze_ratings_file():
    """ratings.txt 파일을 분석합니다."""
    print("=== ratings.txt 파일 분석 ===")
    
    # 파일 읽기 시도
    try:
        # 먼저 몇 줄만 읽어서 구조 파악
        with open('ratings.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
        
        print("첫 10줄:")
        for i, line in enumerate(lines):
            print(f"{i+1}: {line.strip()}")
        
        # 컬럼 수 확인
        if lines:
            columns = lines[0].strip().split('\t')
            print(f"\n컬럼 수: {len(columns)}")
            print(f"컬럼명: {columns}")
        
        # 전체 파일 분석
        print("\n=== 전체 파일 분석 ===")
        
        # 탭으로 구분된 파일로 읽기 시도
        try:
            df = pd.read_csv('ratings.txt', sep='\t', encoding='utf-8')
            print(f"총 행 수: {len(df)}")
            print(f"컬럼: {list(df.columns)}")
            print("\n데이터 타입:")
            print(df.dtypes)
            print("\n첫 5행:")
            print(df.head())
            print("\n기본 통계:")
            print(df.describe())
            
            # label 컬럼이 있다면 분포 확인
            if 'label' in df.columns:
                print(f"\nlabel 분포:")
                print(df['label'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"pandas로 읽기 실패: {e}")
            
            # 다른 구분자로 시도
            try:
                df = pd.read_csv('ratings.txt', sep=',', encoding='utf-8')
                print(f"쉼표로 구분하여 읽기 성공")
                print(f"총 행 수: {len(df)}")
                print(f"컬럼: {list(df.columns)}")
                return df
            except Exception as e2:
                print(f"쉼표로도 읽기 실패: {e2}")
                
                # 수동으로 파싱
                print("\n수동 파싱 시도...")
                parsed_lines = []
                with open('ratings.txt', 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 100:  # 처음 100줄만
                            break
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            parsed_lines.append(parts)
                
                if parsed_lines:
                    print(f"파싱된 행 수: {len(parsed_lines)}")
                    print("첫 5개 파싱된 행:")
                    for i, parts in enumerate(parsed_lines[:5]):
                        print(f"{i+1}: {parts}")
                
                return None
                
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return None

def compare_with_movies():
    """기존 영화 데이터와 비교합니다."""
    print("\n=== 기존 영화 데이터와 비교 ===")
    
    try:
        # TMDB 영화 데이터 읽기
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        print(f"TMDB 영화 수: {len(movies_df)}")
        print(f"TMDB 컬럼: {list(movies_df.columns)}")
        
        # ID 범위 확인
        if 'id' in movies_df.columns:
            print(f"TMDB ID 범위: {movies_df['id'].min()} ~ {movies_df['id'].max()}")
        
        return movies_df
        
    except Exception as e:
        print(f"TMDB 데이터 읽기 실패: {e}")
        return None

if __name__ == "__main__":
    ratings_df = analyze_ratings_file()
    movies_df = compare_with_movies()
    
    print("\n=== 분석 완료 ===") 