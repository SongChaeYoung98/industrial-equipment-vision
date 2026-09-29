"""Audit cross-class container/component overlaps in YOLOv5 predictions.

Input is detect.py output with --save-txt --save-conf.  This finds candidate
pseudo-labels; visual review remains necessary because no component GT exists.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "ml/data/processed/yolo_final_v1"
PARENTS = {
    "control panel", "distribution panel", "enclosure or panel", "electricity device",
    "gas meter", "heater", "meter box", "motor", "solar module", "tank", "utility pole",
}
COMPONENTS = {
    "board", "bolts", "chain wheel", "fan", "flange", "gas ball valve",
    "gas pressure regulator", "gauges", "insulator", "light", "nameplate",
    "piping", "socket", "switching devices", "valve handwheel", "wire",
}


def corners(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def child_covered(parent: list[float], child: list[float]) -> float:
    px1, py1, px2, py2 = corners(parent)
    cx1, cy1, cx2, cy2 = corners(child)
    overlap = max(0.0, min(px2, cx2) - max(px1, cx1)) * max(0.0, min(py2, cy2) - max(py1, cy1))
    area = (cx2 - cx1) * (cy2 - cy1)
    return overlap / area if area > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="YOLOv5 detect.py output directory")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-child-coverage", type=float, default=0.85)
    args = parser.parse_args()
    if not (0 < args.min_confidence < 1 and 0 < args.min_child_coverage <= 1):
        raise ValueError("Thresholds must be between 0 and 1")
    names = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    label_dir = args.predictions / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"No YOLOv5 label directory: {label_dir}")
    examples = []
    parent_counts = Counter()
    component_counts = Counter()
    images_with_detections = 0
    for label_file in sorted(label_dir.glob("*.txt")):
        detections = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Expected class, xywh and confidence in {label_file}: {line}")
            class_id = int(parts[0])
            score = float(parts[5])
            if score >= args.min_confidence:
                detections.append({"label": names[class_id],
                                   "xywh": list(map(float, parts[1:5])),
                                   "confidence": score})
        if detections:
            images_with_detections += 1
        pairs = []
        for parent in detections:
            if parent["label"] not in PARENTS:
                continue
            for child in detections:
                if child is parent or child["label"] not in COMPONENTS:
                    continue
                p_area = parent["xywh"][2] * parent["xywh"][3]
                c_area = child["xywh"][2] * child["xywh"][3]
                coverage = child_covered(parent["xywh"], child["xywh"])
                if p_area > c_area and coverage >= args.min_child_coverage:
                    pairs.append({"parent": parent, "component": child,
                                  "child_coverage": round(coverage, 4)})
                    parent_counts[parent["label"]] += 1
                    component_counts[child["label"]] += 1
        if pairs:
            image_file = args.predictions / f"{label_file.stem}.png"
            examples.append({"image": str(image_file), "prediction_labels": str(label_file),
                             "pairs": pairs})
    examples.sort(key=lambda item: max(
        min(pair["parent"]["confidence"], pair["component"]["confidence"])
        for pair in item["pairs"]), reverse=True)
    manifest = args.dataset / "manifest.jsonl"
    by_stem = {}
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                by_stem[Path(item["image_name"]).stem] = item
    for example in examples:
        source = by_stem.get(Path(example["image"]).stem)
        if source:
            example["source_sha256"] = source.get("sha256")
            example["capture_group"] = source.get("capture_group")
    hashes = {example["source_sha256"] for example in examples if example.get("source_sha256")}
    result = {
        "prediction_label_files": len(list(label_dir.glob("*.txt"))),
        "images_with_detections_at_threshold": images_with_detections,
        "images_with_candidate_nested_pairs": len(examples),
        "candidate_nested_pair_count": sum(len(item["pairs"]) for item in examples),
        "unique_source_hashes_with_candidates": len(hashes),
        "min_confidence": args.min_confidence,
        "min_child_coverage": args.min_child_coverage,
        "parent_counts": dict(parent_counts),
        "component_counts": dict(component_counts),
        "examples": examples,
        "interpretation": "Predicted pairs are pseudo-label candidates, not verified ground truth.",
    }
    output = args.predictions / "nested_probe.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, indent=2))
    print(f"Detailed pairs: {output}")


if __name__ == "__main__":
    main()
