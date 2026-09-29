"""Prepare a human/expert review report for candidate Stage 6 labels.

This script does not promote pseudo-labels to verified labels. It only creates
a compact report and a template that a reviewer may edit explicitly.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE6 = ROOT / "ml/data/processed/stage6_auto_clusters_v2/metadata.json"
SPLITS = ROOT / "ml/data/processed/dataset_split_v1/manifest.jsonl"
OUT = ROOT / "ml/data/processed/label_review"


def main() -> None:
    metadata = json.loads(STAGE6.read_text(encoding="utf-8"))
    split_by_source = {}
    with SPLITS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split_by_source[row["source"]] = row["split"]

    grouped = defaultdict(list)
    for item in metadata["items"]:
        label = item.get("auto_label", "unknown")
        if label == "unknown" or label.startswith("cluster_"):
            continue
        grouped[label].append(item)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "Candidate pseudo-label review only; no label is verified by this file.",
        "source_metadata": str(STAGE6.relative_to(ROOT)).replace("\\", "/"),
        "candidate_labels": [],
    }
    for label in sorted(grouped):
        items = grouped[label]
        scores = [float(item["anchor_similarity"]) for item in items
                  if item.get("anchor_similarity") is not None]
        examples = sorted(items, key=lambda item: float(item.get("anchor_similarity") or 0), reverse=True)[:12]
        report["candidate_labels"].append({
            "candidate_label": label,
            "count": len(items),
            "split_counts": dict(Counter(split_by_source.get(item["source"], "missing") for item in items)),
            "similarity": {
                "min": min(scores) if scores else None,
                "mean": sum(scores) / len(scores) if scores else None,
                "max": max(scores) if scores else None,
            },
            "review_examples": [
                {
                    "source": item["source"],
                    "primary_crop": item.get("primary_crop"),
                    "split": split_by_source.get(item["source"]),
                    "anchor_similarity": item.get("anchor_similarity"),
                }
                for item in examples
            ],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    template = {
        "labels": [],
        "instructions": (
            "After expert/user review, add only labels confirmed as equipment classes. "
            "Do not add unknown or cluster_* unless explicitly reviewed."
        ),
    }
    (OUT / "verified_labels.template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "candidate_labels": [entry["candidate_label"] for entry in report["candidate_labels"]],
        "report": str((OUT / "candidate_report.json").relative_to(ROOT)).replace("\\", "/"),
        "template": str((OUT / "verified_labels.template.json").relative_to(ROOT)).replace("\\", "/"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
