"""Synchronize Explorer edits in Stage 6 auto-label folders.

Folder edits are treated as review feedback, not automatic ground truth:
renamed folders become ``pending_confirmation`` labels. Only labels supplied
through ``--confirm-label`` are written as verified labels for training.
Missing crop files are marked ``dropped``; no source or raw file is deleted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE6 = ROOT / "ml/data/processed/stage6_auto_clusters_v2"
DEFAULT_META = STAGE6 / "metadata.json"
DEFAULT_OUT = ROOT / "ml/data/processed/stage6_feedback_v1"
RESERVED = {"unknown"}


def normal_name(value: str) -> str:
    return value.strip().lower()


def index_folder_files(root: Path):
    locations = defaultdict(list)
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        for path in folder.iterdir():
            if path.is_file():
                locations[path.name].append((folder.name, path))
    return locations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=STAGE6)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--confirm-label", action="append", default=[],
                        help="Mark a reviewed folder label as verified; repeatable")
    args = parser.parse_args()
    if not args.metadata.is_file():
        raise SystemExit(f"Missing metadata: {args.metadata}")
    if not args.root.is_dir():
        raise SystemExit(f"Missing Stage 6 folder: {args.root}")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    locations = index_folder_files(args.root)
    confirmed = {normal_name(label) for label in args.confirm_label}
    records = []
    label_counts = Counter()
    status_counts = Counter()
    for item in metadata.get("items", []):
        crop = item.get("primary_crop")
        matches = locations.get(crop, [])
        if not matches:
            status = "dropped"
            label = None
        elif len(matches) > 1:
            raise SystemExit(f"Crop appears in multiple folders: {crop}")
        else:
            folder, _ = matches[0]
            label = folder
            is_reserved = normal_name(folder) in RESERVED or normal_name(folder).startswith("cluster_")
            status = "verified" if normal_name(folder) in confirmed else (
                "pending_confirmation" if not is_reserved else "pending_review"
            )
            label_counts[folder] += 1
        status_counts[status] += 1
        records.append({
            "source": item.get("source"),
            "primary_crop": crop,
            "previous_auto_label": item.get("auto_label"),
            "review_label": label,
            "status": status,
            "anchor_similarity": item.get("anchor_similarity"),
        })

    verified_labels = sorted({
        row["review_label"] for row in records
        if row["status"] == "verified" and row["review_label"]
    })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_items": len(records),
        "status_counts": dict(status_counts),
        "folder_counts": dict(label_counts),
        "verified_labels": verified_labels,
        "note": "Folder renames are pending_confirmation until explicitly confirmed.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "verified_labels.json").write_text(
        json.dumps({"labels": verified_labels}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
