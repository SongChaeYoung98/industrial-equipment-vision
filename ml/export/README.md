# 모델 변환

학습용 모델과 운영 추론용 모델을 분리합니다.

권장 흐름:

```text
PyTorch checkpoint
  → ONNX
  → CPU/edge benchmark
  → 필요 시 FP16 또는 INT8 양자화
  → versioned artifact 저장
```

운영 서버에는 학습 프레임워크 전체를 넣지 않고, 가능한 경우 경량 추론 runtime만 설치합니다.

