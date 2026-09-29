"""Train/evaluate a classifier only from explicitly verified labels.

The Stage 6 labels are pseudo-labels. This command deliberately refuses to
train on ``unknown`` or ``cluster_*`` unless a human explicitly places a label
in the verification file. The default feature input is the saved DINOv2
patch-average embedding; the split is the leakage-aware split already built by
``build_dataset_splits.py``.

Verification file format:
    {"labels": ["motor", "gauges_or_panels", "switchboard_or_swichgear"]}

Run only after reviewing and creating
``ml/data/processed/verified_labels.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FULL = ROOT / "ml/data/processed/full_pipeline"
SPLIT_MANIFEST = ROOT / "ml/data/processed/dataset_split_v1/manifest.jsonl"
DEFAULT_LABELS = ROOT / "ml/data/processed/verified_labels.json"
DEFAULT_FEEDBACK = ROOT / "ml/data/processed/stage6_feedback_v1/records.jsonl"
OUT = ROOT / "ml/data/processed/verified_classifier_v1"


def load_verified_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(
            f"Missing verification file: {path}\n"
            "Create it after human review; pseudo-labels are not accepted automatically."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    labels = value.get("labels") if isinstance(value, dict) else value
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
        raise SystemExit("Verification file must contain a non-empty JSON list under 'labels'.")
    if len(set(labels)) != len(labels):
        raise SystemExit("Verification file contains duplicate labels.")
    return labels


def load_feedback(path: Path | None):
    if path is None or not path.is_file():
        return {}
    feedback = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "verified" and row.get("review_label"):
                feedback[row["source"]] = row["review_label"]
    return feedback


def load_split_rows(labels: set[str], feedback: dict[str, str]):
    rows = []
    with SPLIT_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reviewed_label = feedback.get(row["source"])
            if reviewed_label:
                row["label"] = reviewed_label
            if row.get("label") in labels and (not feedback or reviewed_label):
                rows.append(row)
    if not rows:
        raise SystemExit("No records match the explicitly verified labels.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK,
                        help="Stage 6 feedback records; only status=verified rows are used")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args()

    labels = load_verified_labels(args.labels)
    feedback = load_feedback(args.feedback)
    rows = load_split_rows(set(labels), feedback)
    embedding_rows = json.loads((FULL / "embedding_rows.json").read_text(encoding="utf-8"))
    index_by_source = {source: index for index, source in enumerate(embedding_rows)}
    matrix = np.load(FULL / "dino_embeddings.npy", mmap_mode="r")

    usable = [row for row in rows if row["source"] in index_by_source]
    if len(usable) != len(rows):
        raise SystemExit(f"{len(rows) - len(usable)} selected records have no DINO embedding.")
    train_rows = [row for row in usable if row["split"] == "train"]
    validation_rows = [row for row in usable if row["split"] == "validation"]
    test_rows = [row for row in usable if row["split"] == "test"]
    counts = {
        split: dict(Counter(row["label"] for row in group))
        for split, group in (("train", train_rows), ("validation", validation_rows), ("test", test_rows))
    }
    missing_train = [label for label in labels if counts["train"].get(label, 0) < 2]
    missing_test = [label for label in labels if counts["test"].get(label, 0) < 1]
    if missing_train or missing_test:
        raise SystemExit(json.dumps({"insufficient_train": missing_train, "missing_test": missing_test}, indent=2))

    encoder = LabelEncoder().fit(labels)

    def xy(group):
        x = np.asarray([matrix[index_by_source[row["source"]]] for row in group], dtype="float32")
        return normalize(x), encoder.transform([row["label"] for row in group])

    x_train, y_train = xy(train_rows)
    x_validation, y_validation = xy(validation_rows) if validation_rows else (None, None)
    x_test, y_test = xy(test_rows)
    model = LogisticRegression(max_iter=args.max_iter, class_weight="balanced", multi_class="auto")
    model.fit(x_train, y_train)

    def evaluate(name, x, y):
        predicted = model.predict(x)
        return {
            "count": int(len(y)),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y, predicted, labels=np.arange(len(labels)), target_names=labels,
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y, predicted, labels=np.arange(len(labels))).tolist(),
        }

    metrics = {
        "features": "dino_embeddings.npy (DINOv2 patch-average, L2-normalized at fit)",
        "verified_labels": labels,
        "feedback_records": str(args.feedback) if feedback else None,
        "counts": counts,
        "validation": evaluate("validation", x_validation, y_validation) if validation_rows else None,
        "test": evaluate("test", x_test, y_test),
        "warning": "Metrics are valid only for the explicitly human-verified labels supplied to this run.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "classifier.joblib")
    (args.output / "label_encoder.json").write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
