# Image Style Transfer

## 1. Team
- Beyond Frame: 21기 박세진, 21기 엄희문, 22기 박서희

## 2. Introduction
- Motivation:  
  그날의 기분에 따라 사람들은 같은 장면을 봐도 서로 다르게 기억한다.  
  "일기나 여행 후기들을 바탕으로 그 기억에 적합한 Style의 Image 로 변화시켜보자"
  <br/>*Image Style Transfer* - Text Prompt에 따라 Image의 Style을 변환한다.
- CLIPStyler:  
  Zero-Shot Learning이기 때문에 Large dataset을 활용할 필요가 없어 Contest에 적합하고, 다양한 변화가 가능하다.

## 3. Method
- Method 1:
  <br/>긴 Text를 처리하기 위해 Image Style과 관련한 Style bank feature 생성 -> 학습
  <br/>기존 Loss function 그대로 사용
  <img width="1256" height="580" alt="스크린샷 2025-08-25 224351" src="https://github.com/user-attachments/assets/e3e6359a-8075-4f12-9f45-042534a1fe27" />

- Method 2:
  <br/>긴 Text에서 시각적 표현에 적합한 단어 리스트 추출
  <br/>Image Style에 해당하는 Text Prompt 대신 추출한 키워드를 넣고 CLIPStyler 사용
  <img width="1549" height="528" alt="스크린샷 2025-08-25 224626" src="https://github.com/user-attachments/assets/9780b739-4eea-46b2-9b43-e86f3bea1320" />

- Method 3:
  <br/>긴 Text를 처리하기 위해 Text를 문장/절 단위로 분할하고 Lexicon과 형용사 어미로 가볍게 가중 후 CLIP-ancho로 score 계산 -> Top-K 문장 선택 후 가중치 정규화 -> 텍스트 방향 생성
  <br/>나머지 방법은 논문의 방식과 동일
  <img width="1331" height="542" alt="스크린샷 2025-08-25 225844" src="https://github.com/user-attachments/assets/95e4a63f-3ba3-42b1-8f75-f5e6843586aa" />

## 4. Experiments
- Results: Method 1 <br/>
  <img width="558" height="257" alt="스크린샷 2025-08-25 230036" src="https://github.com/user-attachments/assets/42100e7f-9e52-4c95-8569-2cf7091a71a9" />

- Results: Method 2 <br/>
  <img width="585" height="259" alt="스크린샷 2025-08-25 230240" src="https://github.com/user-attachments/assets/09cdd769-4ce3-4dcb-b715-52797bf43c90" />

- Results: Method 3 <br/>
  <img width="555" height="263" alt="스크린샷 2025-08-25 230358" src="https://github.com/user-attachments/assets/61d3fa8e-fb5e-4d45-b19a-8e1494cbadb2" />



## 5. Reference

Kwon, Gihyun, and Jong Chul Ye. "Clipstyler: Image style transfer with a single text condition" Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

https://github.com/cyclomon/CLIPstyler
