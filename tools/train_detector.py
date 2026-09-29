"""Run audited YOLOv5 P2 detector training, validation, or inference.

Training is deliberately an explicit subcommand. The official anchor-based
ultralytics/yolov5 checkout is downloaded to an ignored runtime directory on
first use. A copied YOLO dataset is localized at runtime, so machine-specific
source paths are never required. AutoAnchor, multi-scale and P2/4 remain on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "ml/data/processed/yolo_final_v1"
REPO = ROOT / "ml/data/processed/yolov5_runtime"
CONFIG = ROOT / "tools/configs/yolov5s-p2.yaml"
RUNS = ROOT / "ml/data/runs/detector"
YOLOV5_URL = "https://github.com/ultralytics/yolov5.git"
YOLOV5_REVISION = "402e17ddf820996f51a191cbb798376e1144f069"


def audit_dataset(dataset: Path) -> dict:
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or sorted(names) != list(range(28)):
        raise ValueError("Expected contiguous class IDs for 28 confirmed labels")
    if config.get("nc") != 28 or any(not isinstance(name, str) for name in names.values()):
        raise ValueError("data.yaml must declare all 28 class names")
    if config.get("train") != "images/train" or config.get("val") != "images/val":
        raise ValueError("Expected portable images/train and images/val layout")
    counts = Counter()
    image_counts = Counter()
    nested_pairs = 0
    for split in ("train", "val"):
        images = dataset / "images" / split
        labels = dataset / "labels" / split
        image_files = [path for path in images.iterdir()
                       if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in labels.glob("*.txt")}
        if len(image_stems) != len(image_files) or image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {split}: {len(image_stems ^ label_stems)}")
        if not image_files:
            raise ValueError(f"No images found in {images}")
        for label_file in labels.glob("*.txt"):
            rows = [line.split() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            seen = set()
            boxes = []
            for row in rows:
                if len(row) != 5 or not row[0].isdigit() or not 0 <= int(row[0]) < 28:
                    raise ValueError(f"Invalid YOLO row in {label_file}: {row}")
                coords = tuple(map(float, row[1:]))
                x, y, w, h = coords
                tolerance = 1e-7  # nine-decimal YOLO serialization can exceed an edge by <1e-9
                if not (0 < w <= 1 and 0 < h <= 1
                        and -tolerance <= x-w/2 < x+w/2 <= 1+tolerance
                        and -tolerance <= y-h/2 < y+h/2 <= 1+tolerance):
                    raise ValueError(f"Out-of-range box in {label_file}: {row}")
                key = tuple(row)
                if key in seen:
                    raise ValueError(f"Exact duplicate label in {label_file}")
                seen.add(key)
                boxes.append((int(row[0]), x-w/2, y-h/2, x+w/2, y+h/2))
            for parent in boxes:
                for child in boxes:
                    if (parent is not child and parent[0] != child[0]
                            and parent[1] <= child[1] and parent[2] <= child[2]
                            and child[3] <= parent[3] and child[4] <= parent[4]
                            and (parent[3]-parent[1]) * (parent[4]-parent[2])
                            > (child[3]-child[1]) * (child[4]-child[2])):
                        nested_pairs += 1
            counts[split] += len(rows)
        image_counts[split] = len(image_files)
    summary_path = dataset / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("class_count") != 28 or summary.get("object_count") != sum(counts.values()):
            raise ValueError("YOLO data differs from optional review summary")
    return {"images_by_split": dict(image_counts), "objects_by_split": dict(counts),
            "classes": len(names), "nested_box_pairs": nested_pairs}


def local_data_yaml(dataset: Path, runs: Path = RUNS) -> Path:
    """Bind a copied dataset to its current location for YOLOv5's path rules."""
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    config["path"] = dataset.resolve().as_posix()
    key = hashlib.sha256(str(dataset.resolve()).encode("utf-8")).hexdigest()[:12]
    path = runs / "configs" / f"dataset_{key}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def ensure_yolov5(repo: Path) -> None:
    """Install the pinned YOLOv5 source only into the local ignored runtime path."""
    if repo.is_dir():
        return
    if repo.resolve() != REPO.resolve():
        raise FileNotFoundError(f"Custom --yolov5-repo does not exist: {repo}")
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", YOLOV5_URL, str(repo)], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != YOLOV5_REVISION:
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", YOLOV5_REVISION], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != YOLOV5_REVISION:
        raise RuntimeError(f"YOLOv5 revision mismatch: {revision}")


def audit_yolov5(repo: Path) -> Path:
    train = repo / "train.py"
    val = repo / "val.py"
    detect = repo / "detect.py"
    p2 = repo / "models/hub/yolov5-p2.yaml"
    for file in (train, val, detect, p2):
        if not file.is_file():
            raise FileNotFoundError(f"Missing official YOLOv5 file: {file}")
    train_text = train.read_text(encoding="utf-8")
    val_text = val.read_text(encoding="utf-8")
    detect_text = detect.read_text(encoding="utf-8")
    p2_text = p2.read_text(encoding="utf-8")
    if "check_anchors(dataset" not in train_text or "if not opt.noautoanchor" not in train_text:
        raise ValueError("YOLOv5 AutoAnchor entry point changed; inspect before training")
    if '"--multi-scale"' not in train_text:
        raise ValueError("YOLOv5 multi-scale flag missing")
    if "P2/4" not in p2_text or "[[21, 24, 27, 30], 1, Detect" not in p2_text:
        raise ValueError("Expected official P2/P3/P4/P5 detection architecture")
    if "agnostic=single_cls" not in val_text:
        raise ValueError("YOLOv5 validation NMS behavior changed; inspect before use")
    if "agnostic_nms=False" not in detect_text or "agnostic_nms, max_det=" not in detect_text:
        raise ValueError("YOLOv5 inference NMS behavior changed; inspect before use")
    # The official loader removes byte-identical label rows only. Distinct
    # parent/child boxes have different coordinates and stay independent.
    loader = (repo / "utils/dataloaders.py").read_text(encoding="utf-8")
    if "np.unique(lb, axis=0, return_index=True)" not in loader:
        raise ValueError("YOLOv5 label deduplication changed; inspect nested GT handling")
    compact = CONFIG.read_text(encoding="utf-8")
    if "P2/4" not in compact or "[[21, 24, 27, 30], 1, Detect" not in compact:
        raise ValueError("Compact P2 configuration lost its four detection heads")
    return CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "validate", "predict"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--yolov5-repo", type=Path, default=REPO)
    parser.add_argument("--weights", default="yolov5s.pt",
                        help="Train: transferable YOLOv5 weights; validate/predict: trained best.pt")
    parser.add_argument("--source", type=Path, help="Image or folder for predict")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto", help="Use CUDA GPU 0 if available, otherwise CPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-name", default="yolov5_p2")
    parser.add_argument("--conf-thres", type=float, default=0.15,
                        help="Prediction confidence; validation always uses 0.001 for recall measurement")
    parser.add_argument("--allow-primary-only", action="store_true",
                        help="Explicitly permit pilot training when no nested ground-truth boxes exist")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    audit = audit_dataset(dataset)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    repo = args.yolov5_repo.resolve()
    ensure_yolov5(repo)
    p2 = audit_yolov5(repo)
    if repo == REPO.resolve():
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if revision != YOLOV5_REVISION:
            raise RuntimeError(f"Local YOLOv5 checkout is not the pinned revision: {revision}")
    data_yaml = local_data_yaml(dataset)
    if args.action == "check":
        subprocess.run([sys.executable, str(repo / "train.py"), "--help"], cwd=repo,
                       stdout=subprocess.DEVNULL, check=True)
        subprocess.run([
            sys.executable, "-c",
            "import sys; from utils.general import check_dataset; check_dataset(sys.argv[1], autodownload=False)",
            str(data_yaml),
        ], cwd=repo, check=True)
        print("Portable data, pinned YOLOv5 dependencies, AutoAnchor, P2 and NMS checked.")
        return
    if args.device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Resolved device: {device}", flush=True)
    common = ["--data", str(data_yaml), "--imgsz", str(args.imgsz),
              "--device", device]
    if args.action == "train":
        if audit["nested_box_pairs"] == 0 and not args.allow_primary_only:
            raise ValueError("No nested ground-truth boxes exist. Add reviewed component annotations "
                             "and rebuild the dataset, or explicitly run a primary-only pilot.")
        if args.epochs <= 0 or args.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        command = [sys.executable, str(repo / "train.py"), *common,
                   "--cfg", str(p2), "--weights", args.weights,
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--workers", str(args.workers), "--multi-scale",
                   "--project", str(RUNS), "--name", args.run_name]
        # Intentionally no --noautoanchor or --single-cls.  AutoAnchor runs
        # on the complete YOLO train set, including small boxes.
    elif args.action == "validate":
        if not Path(args.weights).is_file():
            raise FileNotFoundError("Use a trained detector checkpoint for validation")
        command = [sys.executable, str(repo / "val.py"), *common,
                   "--weights", str(Path(args.weights).resolve()), "--task", "val", "--conf-thres", "0.001",
                   "--iou-thres", "0.65", "--max-det", "1000", "--verbose",
                   "--project", str(RUNS / "validation")]
        # val.py invokes NMS with agnostic=single_cls; --single-cls is absent.
    else:
        if not Path(args.weights).is_file() or not args.source or not args.source.exists():
            raise FileNotFoundError("Use a trained checkpoint and existing --source")
        if not 0 < args.conf_thres < 1:
            raise ValueError("Prediction confidence must be between 0 and 1")
        command = [sys.executable, str(repo / "detect.py"),
                   "--weights", str(Path(args.weights).resolve()), "--source", str(args.source.resolve()),
                   "--data", str(data_yaml), "--imgsz", str(args.imgsz),
                   "--device", device, "--conf-thres", str(args.conf_thres),
                   "--iou-thres", "0.65", "--max-det", "1000",
                   "--save-txt", "--save-conf",
                   "--project", str(RUNS / "predictions")]
        # detect.py default agnostic_nms=False; never add --agnostic-nms.
    print("Command:", subprocess.list2cmdline(command), flush=True)
    if args.action != "train":
        subprocess.run(command, cwd=repo, check=True)
        return
    status_path = RUNS / "pilot_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "started_at": datetime.now().astimezone().isoformat(),
              "run_name": args.run_name, "epochs_requested": args.epochs,
              "command": command}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        subprocess.run(command, cwd=repo, check=True)
    except BaseException as error:
        status.update(state="failed", error=str(error),
                      ended_at=datetime.now().astimezone().isoformat())
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise
    status.update(state="completed", ended_at=datetime.now().astimezone().isoformat())
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
