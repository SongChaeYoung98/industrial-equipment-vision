"""Freeze reviewed Stage 6 folders into a full-image YOLO detection dataset.

The reviewed crop identifies the primary object's class.  Its saved SAM box
is transferred to the unchanged source image.  Raw/derived sources and review
folders are read-only; missing crops are recorded as dropped in a new snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STAGE6 = ROOT / "ml/data/processed/stage6_auto_clusters_v2"
FULL_RECORDS = ROOT / "ml/data/processed/full_pipeline/final_records.json"
OUTPUT = ROOT / "ml/data/processed/yolo_final_v1"
RESERVED = {"unknown", "mixed", "recluster"}


def capture_group(source: Path) -> str:
    stem = source.stem
    return stem.split("__frame_", 1)[0] if "__frame_" in stem else stem.split("__", 1)[0]


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def load_review():
    metadata = json.loads((STAGE6 / "metadata.json").read_text(encoding="utf-8"))
    locations = {}
    folders = sorted(folder for folder in STAGE6.iterdir() if folder.is_dir())
    if len(folders) != 28:
        raise ValueError(f"Expected 28 reviewed folders; found {len(folders)}")
    for folder in folders:
        if folder.name.lower() in RESERVED or folder.name.lower().startswith("cluster_"):
            raise ValueError(f"Unreviewed folder present: {folder.name}")
        for file in folder.iterdir():
            if not file.is_file():
                continue
            if file.suffix.lower() != ".png":
                raise ValueError(f"Unexpected file in review folder: {file}")
            if file.name in locations:
                raise ValueError(f"Crop appears in two folders: {file.name}")
            locations[file.name] = (folder.name, file)
    items = metadata["items"]
    if len({item["primary_crop"] for item in items}) != len(items):
        raise ValueError("Stage 6 metadata has repeated crop names")
    known = {item["primary_crop"] for item in items}
    extra = set(locations) - known
    if extra:
        raise ValueError(f"{len(extra)} review files have no Stage 6 metadata; e.g. {sorted(extra)[0]}")
    return items, locations, [folder.name for folder in folders]


def source_path(value: str) -> Path:
    source = Path(value).resolve(strict=True)
    allowed = (ROOT / "ml/data/raw", ROOT / "ml/data/derived")
    if not any(source.is_relative_to(root.resolve()) for root in allowed):
        raise ValueError(f"Source outside local raw/derived data: {source}")
    if source.suffix.lower() != ".png":
        raise ValueError(f"Expected source PNG, got: {source}")
    return source


def make_split(rows: list[dict]) -> None:
    """Keep a capture/video and byte-identical sources on the same side."""
    groups = {row["capture_group"] for row in rows}
    parent = {group: group for group in groups}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    by_hash = defaultdict(set)
    for row in rows:
        by_hash[row["sha256"]].add(row["capture_group"])
    for linked in by_hash.values():
        first = min(linked)
        for other in linked:
            union(first, other)

    units = defaultdict(list)
    for row in rows:
        row["split_group"] = find(row["capture_group"])
        units[row["split_group"]].append(row)
    assignments = {}
    for group in units:
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 1000
        assignments[group] = "train" if bucket < 800 else "val"

    # Tiny classes need at least one validation group when two independent
    # capture groups exist.  Move an entire group, never an individual frame.
    labels = sorted({row["label"] for row in rows})
    for label in labels:
        train_groups = [g for g, members in units.items()
                        if assignments[g] == "train" and any(r["label"] == label for r in members)]
        val_groups = [g for g, members in units.items()
                      if assignments[g] == "val" and any(r["label"] == label for r in members)]
        all_groups = train_groups + val_groups
        if len(all_groups) < 2:
            continue
        if not val_groups:
            candidate = min(train_groups, key=lambda g: (len(units[g]), g))
            assignments[candidate] = "val"
        elif not train_groups:
            candidate = min(val_groups, key=lambda g: (len(units[g]), g))
            assignments[candidate] = "train"
    for row in rows:
        row["split"] = assignments[row["split_group"]]
    if any(len({r["split"] for r in members}) != 1 for members in units.values()):
        raise AssertionError("Capture/duplicate group leaked across splits")


def snapshot_and_rows(items, locations, full_records):
    by_crop = {row.get("primary_crop"): row for row in full_records if row.get("status") == "ok"}
    feedback = []
    rows = []
    for item in items:
        crop = item["primary_crop"]
        found = locations.get(crop)
        label = found[0] if found else None
        feedback.append({"source": item["source"], "primary_crop": crop,
                         "previous_auto_label": item.get("auto_label"),
                         "review_label": label,
                         "status": "verified" if found else "dropped"})
        if not found:
            continue
        full = by_crop.get(crop)
        if not full or full.get("source") != item["source"]:
            raise ValueError(f"No matching SAM record for reviewed crop: {crop}")
        source = source_path(item["source"])
        with Image.open(source) as im:
            im.verify()
        with Image.open(source) as im:
            width, height = im.size
        box = full.get("primary_box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Missing SAM box: {crop}")
        x, y, w, h = box
        if not all(isinstance(v, int) for v in box) or x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"Invalid SAM box: {crop} {box}")
        if x + w > width or y + h > height:
            raise ValueError(f"Out-of-image SAM box: {crop} {box} vs {width}x{height}")
        image_name = crop.removesuffix("__primary.png") + ".png"
        rows.append({"source": source.relative_to(ROOT).as_posix(),
                     "review_crop": found[1].relative_to(ROOT).as_posix(),
                     "image_name": image_name, "label": label,
                     "bbox_xywh": box, "image_size": [width, height],
                     "annotations": [{"label": label, "bbox_xywh": box, "origin": "reviewed_primary_sam"}],
                     "capture_group": capture_group(source), "sha256": digest(source)})
        if len(rows) % 500 == 0:
            print(f"Preflight {len(rows)} reviewed images checked", flush=True)
    if len({r["image_name"] for r in rows}) != len(rows):
        raise ValueError("Generated image names collide")
    if len({r["source"] for r in rows}) != len(rows):
        raise ValueError("Same source occurs more than once; review multi-box mapping manually")
    return feedback, rows


def add_reviewed_component_boxes(rows: list[dict], labels: list[str], path: Path | None) -> None:
    """Append human-reviewed extra boxes; keep parent and child independently."""
    if path is None:
        return
    by_source = {row["source"]: row for row in rows}
    allowed_labels = set(labels)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            source = source_path(str(item["source"])).relative_to(ROOT).as_posix()
            if source not in by_source:
                raise ValueError(f"Extra box source is not a verified image, line {line_number}: {source}")
            label = item["label"]
            if label not in allowed_labels:
                raise ValueError(f"Unknown extra-box label on line {line_number}: {label}")
            box = item["bbox_xywh"]
            row = by_source[source]
            iw, ih = row["image_size"]
            if (not isinstance(box, list) or len(box) != 4 or
                not all(isinstance(v, int) for v in box)):
                raise ValueError(f"Invalid extra box on line {line_number}: {box}")
            x, y, w, h = box
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
                raise ValueError(f"Out-of-image extra box on line {line_number}: {box}")
            if any(a["label"] == label and a["bbox_xywh"] == box for a in row["annotations"]):
                raise ValueError(f"Exact duplicate extra box on line {line_number}")
            row["annotations"].append({"label": label, "bbox_xywh": box,
                                       "origin": "human_reviewed_extra"})


def strict_contains(outer: list[int], inner: list[int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ox <= ix and oy <= iy and ix + iw <= ox + ow and iy + ih <= oy + oh
            and (ow * oh > iw * ih))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--extra-annotations", type=Path,
                        help='Optional reviewed JSONL: {"source": path, "label": name, "bbox_xywh": [x,y,w,h]}')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output already exists; use a new --output to preserve previous snapshot: {output}")
    items, locations, labels = load_review()
    full = json.loads(FULL_RECORDS.read_text(encoding="utf-8"))["records"]
    feedback, rows = snapshot_and_rows(items, locations, full)
    add_reviewed_component_boxes(rows, labels, args.extra_annotations)
    make_split(rows)
    label_ids = {name: i for i, name in enumerate(labels)}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        split = row["split"]
        target = output / "images" / split / row["image_name"]
        shutil.copy2(ROOT / row["source"], target)
        iw, ih = row["image_size"]
        label_file = output / "labels" / split / f"{Path(row['image_name']).stem}.txt"
        label_lines = []
        for ann in row["annotations"]:
            x, y, w, h = ann["bbox_xywh"]
            xc, yc = (x + w / 2) / iw, (y + h / 2) / ih
            yw, yh = w / iw, h / ih
            label_lines.append(f"{label_ids[ann['label']]} {xc:.9f} {yc:.9f} {yw:.9f} {yh:.9f}")
        label_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        if index % 500 == 0:
            print(f"Copied {index}/{len(rows)} full images and labels", flush=True)
    for row in rows:
        row["label_id"] = label_ids[row["label"]]
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "feedback_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in feedback:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # JSON is a YAML 1.2 subset; PyYAML and YOLOv5 accept these scalar/list forms.
    # The training launcher resolves this directory after the dataset is moved.
    # YOLOv5 itself interprets relative `path` from its checkout, so the
    # launcher generates an absolute runtime YAML before calling train.py.
    yaml_lines = ["path: .", "train: images/train", "val: images/val",
                  f"nc: {len(labels)}", "names:"]
    yaml_lines.extend(f"  {i}: {json.dumps(name, ensure_ascii=False)}" for i, name in enumerate(labels))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    class_counts = {name: {"train": 0, "val": 0, "total": 0} for name in labels}
    for row in rows:
        for ann in row["annotations"]:
            count = class_counts[ann["label"]]
            count[row["split"]] += 1
            count["total"] += 1
    nested_pairs = sum(
        strict_contains(a["bbox_xywh"], b["bbox_xywh"])
        for row in rows for a in row["annotations"] for b in row["annotations"]
        if a is not b and a["label"] != b["label"]
    )
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "class_count": len(labels), "review_metadata_count": len(items),
        "verified_count": len(rows), "dropped_count": len(feedback) - len(rows),
        "source_image_count": len(rows), "object_count": sum(len(r["annotations"]) for r in rows),
        "class_counts": class_counts,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "capture_duplicate_group_count": len({row["split_group"] for row in rows}),
        "images_with_multiple_boxes": sum(len(r["annotations"]) > 1 for r in rows),
        "nested_box_pairs": nested_pairs,
        "annotation_limit": ("Only primary SAM boxes are present; reviewed extra boxes are needed for "
                             "component-within-container supervision. Unannotated objects are not negatives."),
        "sources_modified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
