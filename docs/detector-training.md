# 다른 컴퓨터에서 Pilot 탐지 모델 재학습

이 저장소는 원천 이미지와 모델 가중치를 Git에 넣지 않습니다. 학습에 필요한 데이터는
`yolo_final_v1` 폴더 하나입니다. 원천 `raw/`, SAM 마스크, Stage 6 검수 폴더는
재학습에 필요하지 않습니다.

데이터셋 폴더에 다음 항목을 **함께** 복사하세요.

```text
yolo_final_v1/
  data.yaml
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  summary.json             # 선택: 원본 검수 결과와 개수 대조
```

현재 데이터셋은 28개 클래스, train 5,236장, val 1,334장입니다. `data.yaml`의
`path: .`은 저장용 상대 표기입니다. 학습 명령이 복사된 폴더의 실제 위치를 감지해
YOLOv5용 설정 파일을 `ml/data/runs/detector/configs/`에 생성합니다. 데이터셋을
어느 드라이브나 디렉터리로 옮겨도 소스 파일을 고칠 필요가 없습니다.

## 준비

저장소를 클론한 다음 Python 3.12 가상환경을 만드세요. GPU 사용 시 먼저 해당
OS·GPU에 맞는 PyTorch CUDA 빌드를 설치하세요. `torch.cuda.is_available()`가
`True`인지 확인한 다음 나머지 패키지를 설치합니다. CPU만 있으면 학습은 가능하지만
매우 오래 걸립니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-detector.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Linux에서는 `python3.12 -m venv .venv`를 사용하고 아래 명령의
`.\.venv\Scripts\python.exe`를 `.venv/bin/python`으로 바꾸세요. CUDA 휠은
[PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 기기에 맞는
명령을 선택하세요. `requirements-detector.txt`는 NumPy 1.x와 OpenCV 4.x를
유지해 검증된 YOLOv5 환경과 맞춥니다.

## 검사와 학습

아래의 `--dataset`에는 **복사한 `yolo_final_v1` 폴더**를 지정합니다. 명령은
저장소 루트에서 실행하세요. `check` 명령은 이미지·라벨 대응과 28개 클래스,
중첩 정답 수를 검사합니다. 기본 런타임 경로에 공식 YOLOv5가 없으면 검증된
리비전 `402e17ddf820996f51a191cbb798376e1144f069`을 자동으로 내려받습니다.
사전학습 가중치 `yolov5s.pt`도 첫 학습 때 내려받으므로 최초 실행에는 인터넷이
필요합니다.

```powershell
.\.venv\Scripts\python.exe tools/train_detector.py check --dataset "D:\data\yolo_final_v1"
.\.venv\Scripts\python.exe tools/train_detector.py train --dataset "D:\data\yolo_final_v1" --allow-primary-only --run-name pilot_p2
```

`--dataset`을 생략하면 `ml/data/processed/yolo_final_v1`을 사용합니다.
`--device auto`가 기본이며 CUDA가 보이면 GPU 0, 없으면 CPU를 사용합니다.
검증된 RTX 3050 8GB 파일럿 설정은 512px, batch 2, 20 epochs, workers 2입니다.
실험 결과는 `ml/data/runs/detector/pilot_p2/`에 저장됩니다. 별도의 원본 이미지,
Stage 6 폴더, 저장된 임베딩이나 이전 체크포인트는 필요하지 않습니다.

현재 데이터셋에는 이미지당 주 피사체 박스가 하나뿐입니다. `--allow-primary-only`는
그 사실을 알고 파일럿을 학습한다는 명시적 옵션입니다. 설비·부품 다중 박스를
검수해 새 데이터셋을 만들면 이 옵션 없이 학습할 수 있습니다.

학습 후 검증 및 예측 예시:

```powershell
.\.venv\Scripts\python.exe tools/train_detector.py validate --dataset "D:\data\yolo_final_v1" --weights "ml/data/runs/detector/pilot_p2/weights/best.pt"
.\.venv\Scripts\python.exe tools/train_detector.py predict --dataset "D:\data\yolo_final_v1" --weights "ml/data/runs/detector/pilot_p2/weights/best.pt" --source "D:\data\yolo_final_v1\images\val"
```

Git에는 학습 코드와 설정만 커밋합니다. 데이터셋과 런타임 YOLOv5, 가중치,
로그는 `.gitignore` 대상입니다. 데이터셋을 다른 기기로 별도로 복사해야 합니다.
