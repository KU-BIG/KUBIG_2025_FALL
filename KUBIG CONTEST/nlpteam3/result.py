import pandas as pd
import re
from datetime import datetime
from retriever import is_multiple_choice

def extract_answer_for_multiple_choice(answer_text: str) -> int:
    """
    객관식 문제의 답변에서 정답 선지 숫자를 추출하는 함수
    
    Args:
        answer_text: 모델의 답변 텍스트
        
    Returns:
        정답 선지 숫자 (1-5), 찾지 못하면 0
    """
    lines = answer_text.strip().split('\n')
    
    # 1. '정답' 키워드가 나온 첫 번째 줄 찾기
    answer_line = None
    for line in lines:
        if '정답' in line:
            answer_line = line
            break
    
    # 1.1 '정답' 키워드가 없다면, 전체 답변에서 첫 번째 숫자 찾기
    if answer_line is None:
        all_text = answer_text
        # 전체 텍스트에서 첫 번째 숫자 찾기
        first_digit_match = re.search(r'\d', all_text)
        if first_digit_match:
            digit = int(first_digit_match.group())
            if 1 <= digit <= 5:
                return digit
        # 1.2 숫자가 없으면 0 반환
        return 0
    
    # 2. '{숫자}번' 또는 '{숫자}.' 패턴 찾기
    pattern1 = re.findall(r'(\d)번', answer_line)  # {숫자}번
    pattern2 = re.findall(r'(\d)\.', answer_line)  # {숫자}.
    
    # 패턴이 발견된 숫자들 수집
    found_numbers = []
    found_numbers.extend([int(x) for x in pattern1 if x.isdigit() and 1 <= int(x) <= 5])
    found_numbers.extend([int(x) for x in pattern2 if x.isdigit() and 1 <= int(x) <= 5])
    
    # 2.2 중복해서 등장한다면 0 반환
    if len(found_numbers) > 1:
        return 0
    
    # 패턴이 하나 발견된 경우
    if len(found_numbers) == 1:
        return found_numbers[0]
    
    # 2.1 패턴이 없다면, 해당 줄의 첫 번째 숫자 찾기
    first_digit_match = re.search(r'\d', answer_line)
    if first_digit_match:
        digit = int(first_digit_match.group())
        if 1 <= digit <= 5:
            return digit
    
    return 0

def extract_answer_for_subjective(answer_text: str) -> str:
    """
    주관식 문제의 답변에서 '##'로 시작하는 줄을 제거한 나머지 부분을 반환
    
    Args:
        answer_text: 모델의 답변 텍스트
        
    Returns:
        정제된 답변 텍스트
    """
    lines = answer_text.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        # '##'로 시작하는 줄은 제외
        if not line.strip().startswith('##'):
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()

def main():
    # 파일 읽기
    print("파일들을 읽는 중...")
    test_df = pd.read_csv('task/test.csv')
    sample_df = pd.read_csv('task/sample_submission.csv')
    submission_df = pd.read_csv('output/submission1.csv')
    
    print(f"총 {len(test_df)}개의 문제를 처리합니다.")
    
    # 결과를 저장할 딕셔너리
    processed_answers = {}
    
    # 각 문제에 대해 처리
    for idx, row in test_df.iterrows():
        test_id = row['ID']
        question = row['Question']
        
        # submission1에서 해당 ID의 답변 찾기
        answer_row = submission_df[submission_df['ID'] == test_id]
        if answer_row.empty:
            print(f"경고: {test_id}에 대한 답변을 찾을 수 없습니다.")
            processed_answers[test_id] = "0"
            continue
            
        answer_text = answer_row.iloc[0]['Answer']
        
        # 객관식/주관식 구분
        if is_multiple_choice(question):
            # 객관식 처리
            extracted_answer = extract_answer_for_multiple_choice(answer_text)
            processed_answers[test_id] = str(extracted_answer)
            if idx < 10:  # 처음 몇 개만 출력
                print(f"{test_id} (객관식): {extracted_answer}")
        else:
            # 주관식 처리
            extracted_answer = extract_answer_for_subjective(answer_text)
            processed_answers[test_id] = extracted_answer
            if idx < 10:  # 처음 몇 개만 출력
                print(f"{test_id} (주관식): {extracted_answer[:50]}..." if len(extracted_answer) > 50 else f"{test_id} (주관식): {extracted_answer}")
    
    # sample_submission.csv 기반으로 새로운 답변 파일 생성
    result_df = sample_df.copy()
    
    # Answer 열 업데이트
    for idx, row in result_df.iterrows():
        test_id = row['ID']
        if test_id in processed_answers:
            result_df.at[idx, 'Answer'] = processed_answers[test_id]
    
    # 현재 날짜와 시간으로 파일명 생성
    now = datetime.now()
    filename = f"output/submission{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}.csv"
    
    # 결과 저장
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n결과가 {filename}에 저장되었습니다.")
    print(f"총 {len(processed_answers)}개의 답변이 처리되었습니다.")
    
    # 객관식과 주관식 개수 확인
    objective_count = 0
    subjective_count = 0
    
    for _, row in test_df.iterrows():
        question = row['Question']
        if is_multiple_choice(question):
            objective_count += 1
        else:
            subjective_count += 1
    
    print(f"객관식 문제: {objective_count}개")
    print(f"주관식 문제: {subjective_count}개")

if __name__ == "__main__":
    main()
