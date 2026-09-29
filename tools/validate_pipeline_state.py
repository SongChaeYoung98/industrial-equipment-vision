"""Validate the local discovery pipeline artifacts before model training.

This is an audit only: it never edits raw, derived, or processed media.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


RAW = ROOT / "ml/data/raw/2026_09_16"
FULL = ROOT / "ml/data/processed/full_pipeline"
STAGE6 = ROOT / "ml/data/processed/stage6_auto_clusters_v2"
SPLITS = ROOT / "ml/data/processed/dataset_split_v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(value: str | None) -> Path:
    path = Path(value or "")
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    records = read_json(FULL / "final_records.json")["records"]
    embedding_sources = read_json(FULL / "embedding_rows.json")
    embeddings = np.load(FULL / "embeddings.npy", mmap_mode="r")
    stage6 = read_json(STAGE6 / "metadata.json")
    split_lines = (SPLITS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    split_records = [json.loads(line) for line in split_lines if line.strip()]

    status_counts = Counter(row.get("status") for row in records)
    source_exists = sum(Path(row["source"]).is_file() for row in records)
    ok_rows = [row for row in records if row.get("status") == "ok"]
    mask_exists = sum(repo_path(row.get("mask_path")).is_file() for row in ok_rows)
    crop_exists = sum((FULL / "primary" / row["primary_crop"]).is_file()
                      for row in ok_rows)

    labels = Counter(item.get("auto_label", "unknown") for item in stage6["items"])
    split_sources = {row["source"] for row in split_records}
    stage6_sources = {item["source"] for item in stage6["items"]}
    embedding_source_set = set(embedding_sources)

    component_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    for row in split_records:
        component_splits[row["duplicate_component"]].add(row["split"])
        if row.get("exact_hash"):
            hash_splits[row["exact_hash"]].add(row["split"])

    report = {
        "raw_counts": {
            "images": len(list((RAW / "image").glob("*.png"))),
            "videos": len(list((RAW / "video").glob("*.avi"))),
            "frames": len(list((ROOT / "ml/data/derived/2026_09_16/frames").glob("*.png"))),
        },
        "full_pipeline": {
            "records": len(records),
            "status_counts": dict(status_counts),
            "source_files_present": source_exists,
            "ok_masks_present": mask_exists,
            "ok_primary_crops_present": crop_exists,
        },
        "embeddings": {
            "rows": len(embedding_sources),
            "shape": list(embeddings.shape),
            "all_sources_in_records": embedding_source_set <= {row["source"] for row in records},
        },
        "stage6": {
            "items": len(stage6["items"]),
            "dropped_count": stage6.get("dropped_count"),
            "label_counts": dict(labels),
            "items_match_embedding_sources": stage6_sources <= embedding_source_set,
        },
        "dataset_split": {
            "records": len(split_records),
            "split_counts": dict(Counter(row["split"] for row in split_records)),
            "all_sources_in_stage6": split_sources <= stage6_sources,
            "component_leaks": sum(len(splits) > 1 for splits in component_splits.values()),
            "exact_hash_leaks": sum(len(splits) > 1 for splits in hash_splits.values()),
        },
    }
    report["checks"] = {
        "embedding_alignment": embeddings.shape[0] == len(embedding_sources),
        "stage6_alignment": stage6_sources <= embedding_source_set,
        "split_alignment": len(split_records) == len(stage6["items"]),
        "no_component_leak": report["dataset_split"]["component_leaks"] == 0,
        "no_exact_hash_leak": report["dataset_split"]["exact_hash_leaks"] == 0,
    }
    report["valid"] = all(report["checks"].values())
    output = SPLITS / "pipeline_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
