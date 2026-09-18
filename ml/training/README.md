# 학습

초기 모델은 처음부터 학습하지 않고 사전 학습 모델을 이용한 전이학습을 기본으로 합니다.

학습 스크립트가 수행해야 하는 일:

1. manifest 로드
2. upload/session 기준 split 검증
3. class imbalance 확인
4. 모델 학습
5. best checkpoint 저장
6. 학습 설정과 데이터 버전 기록

최종 모델에는 `model_version`, `label_version`, `preprocessing_version`을 함께 기록합니다.

