"""End-to-end local inference: SAM primary proposal -> masked crop -> classifier."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from tools.run_full_pipeline import clahe_rgb, center_score, remove_nested, score_candidates
from tools.train_classifier import build_model

CHECKPOINT = ROOT / "models/sam2/sam2.1_hiera_small.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CLASSIFIER = ROOT / "ml/data/runs/supervised_classifier_v1/classifier_last.pt"
DEFAULT_OUTPUT = ROOT / "ml/data/processed/inference_results"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def input_files(path: Path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    raise SystemExit(f"Input path does not exist: {path}")


def classifier_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def candidates_for(image, generator):
    height, width = image.shape[:2]
    generated = generator.generate(clahe_rgb(image))
    candidates = []
    for raw in generated:
        mask = raw["segmentation"].astype(bool)
        x, y, w, h = map(int, raw["bbox"])
        mask_ratio = float(mask.mean())
        box_ratio = float(w * h / (width * height))
        if mask_ratio < .01:
            continue
        candidates.append({"segmentation": mask, "bbox": [x, y, w, h],
                           "mask_area_ratio": mask_ratio, "box_area_ratio": box_ratio,
                           "full_background_suspect": box_ratio >= .90})
    candidates = remove_nested(candidates)
    if not candidates:
        return None
    score_candidates(image, candidates)
    return max(candidates, key=lambda candidate: candidate["final_score"])


def masked_crop(image, candidate):
    x, y, w, h = candidate["bbox"]
    crop = image[y:y + h, x:x + w].copy()
    mask = candidate["segmentation"][y:y + h, x:x + w]
    crop[~mask] = 0
    return crop


def draw_result(image, candidate, label, confidence):
    output = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(output)
    x, y, w, h = candidate["bbox"]
    draw.rectangle((x, y, x + w, y + h), outline=(40, 255, 130), width=max(3, image.shape[1] // 320))
    text = f"{label} ({confidence * 100:.1f}%)"
    try:
        font = ImageFont.truetype("arial.ttf", max(18, image.shape[1] // 35))
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((x, max(0, y - 35)), text, font=font)
    draw.rectangle(box, fill=(40, 255, 130))
    draw.text((box[0] + 4, box[1] + 2), text, fill=(0, 0, 0), font=font)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER)
    parser.add_argument("--sam-checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    files = input_files(args.input)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No supported images found")
    if not args.classifier.is_file():
        raise SystemExit(f"Missing classifier checkpoint: {args.classifier}")
    if not args.sam_checkpoint.is_file():
        raise SystemExit(f"Missing SAM checkpoint: {args.sam_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_checkpoint = torch.load(args.classifier, map_location=device, weights_only=True)
    classes = classifier_checkpoint["classes"]
    classifier = build_model(len(classes), pretrained=False).to(device).eval()
    classifier.load_state_dict(classifier_checkpoint["model"])
    sam = build_sam2(SAM_CONFIG, str(args.sam_checkpoint), device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        sam, points_per_side=16, points_per_batch=16, pred_iou_thresh=.80,
        stability_score_thresh=.90, box_nms_thresh=.70, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=2000,
        output_mode="binary_mask",
    )
    transform = classifier_transform()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, path in enumerate(files, 1):
            try:
                image = np.asarray(Image.open(path).convert("RGB"))
                candidate = candidates_for(image, generator)
                if candidate is None:
                    records.append({"source": str(path), "status": "no_candidate"})
                    continue
                crop = masked_crop(image, candidate)
                tensor = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                probabilities = torch.softmax(classifier(tensor), dim=1)[0]
                confidence, class_index = probabilities.max(0)
                label = classes[int(class_index)]
                output_name = f"{index:04d}__{path.stem}__{label}.png"
                draw_result(image, candidate, label, float(confidence)).save(args.output_dir / output_name)
                records.append({"source": str(path), "status": "ok", "output": output_name,
                                "label": label, "confidence": float(confidence),
                                "bbox": candidate["bbox"], "final_score": candidate["final_score"],
                                "center_score": candidate["center_score"],
                                "size_score": candidate["size_score"],
                                "sharpness_score": candidate["sharpness_score"],
                                "background_removed": True})
                print(f"{index}/{len(files)} {path.name} -> {label} {float(confidence):.4f}", flush=True)
            except Exception as error:
                records.append({"source": str(path), "status": "failed",
                                "error": f"{type(error).__name__}: {error}"})
                print(f"{index}/{len(files)} FAILED {path.name}: {error}", flush=True)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "classifier": str(args.classifier), "classes": classes,
        "device": str(device), "count": len(files), "records": records,
        "note": "Inference masks background; classifier training used unmasked Stage 6 crops.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "processed": len(files),
                      "ok": sum(row["status"] == "ok" for row in records),
                      "failed": sum(row["status"] == "failed" for row in records),
                      "no_candidate": sum(row["status"] == "no_candidate" for row in records)}, indent=2))


if __name__ == "__main__":
    main()
