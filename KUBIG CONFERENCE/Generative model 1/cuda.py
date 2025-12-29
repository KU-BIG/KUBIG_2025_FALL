import torch
import time

if torch.cuda.is_available():
    # GPU에서 텐서 생성
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    # 연산 테스트
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()  # GPU 연산 완료 대기
    end_time = time.time()
    
    print(f"GPU 연산 성공! 시간: {end_time - start_time:.4f}초")
    print(f"결과 shape: {z.shape}")
else:
    print("CUDA를 사용할 수 없습니다.")