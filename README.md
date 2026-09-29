# check-lab-ai

Portable YOLO detector training from a copied dataset: [detector training guide](docs/detector-training.md).

`check-lab-python-back`과 분리된 AI 프로젝트입니다.

이 저장소는 다음 역할을 담당합니다.

- 업로드 이미지 데이터셋 정제 및 manifest 생성
- 사람이 검수한 라벨을 이용한 모델 학습
- 모델 평가와 경량 포맷 변환
- 이미지 분류 및 임베딩 기반 유사 사례 검색 API

기존 백엔드에는 학습 코드나 대형 모델 파일을 넣지 않습니다. 백엔드는 추론 API를 호출하고 결과를 저장하는 역할만 맡습니다.

## 구조

```text
app/                     운영 추론 API
  api/                   API route
  core/                  설정
  schemas/               요청·응답 모델
  services/              추론 서비스
ml/                      오프라인 데이터·학습 파이프라인
  data/                  manifest 예시와 데이터 규칙
  preprocessing/         전처리
  training/              학습
  evaluation/            평가
  export/                ONNX/경량 변환
configs/                 라벨·모델 설정
tests/                   API 테스트
```

## 실행

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-inference.txt
uvicorn app.main:app --reload --port 8100
```

모델이 아직 없으면 `/v1/analyze`는 `503 model_not_ready`를 반환합니다. 이것은 모델이 없는 상태를 정상적으로 표현하기 위한 초기 동작입니다.

## 서버와의 연결 원칙

운영 백엔드는 로컬 절대 경로 대신 공유 스토리지 URI 또는 signed URL을 AI API에 전달해야 합니다.

```text
POST /v1/analyze
POST /v1/search/similar
GET  /health
GET  /ready
```

모델 버전, 전처리 버전, 입력 해시를 결과에 남겨야 재현과 롤백이 가능합니다.

## 학습 데이터 원칙

- `upload_id` 또는 촬영 세션 단위로 train/validation/test를 분리합니다.
- AI가 예측한 라벨은 사람 검수 전까지 정답 데이터로 승격하지 않습니다.
- 원본 데이터와 파생 이미지(일반·열화상·UV·프레임)를 구분합니다.
- 실제 데이터와 개인정보는 Git에 넣지 않습니다.
- `unknown`과 `unusable`을 별도 라벨로 관리합니다.
