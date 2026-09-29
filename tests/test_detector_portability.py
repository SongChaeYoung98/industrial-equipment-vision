"""A copied YOLO folder is enough to configure detector training."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PIL import Image

from tools.train_detector import audit_dataset, local_data_yaml


def test_copied_dataset_ignores_original_machine_path(tmp_path: Path) -> None:
    original = tmp_path / "old_location" / "yolo_final_v1"
    names = {index: f"class_{index}" for index in range(28)}
    original.mkdir(parents=True)
    (original / "data.yaml").write_text(yaml.safe_dump({
        "path": "C:/some/other/computer/yolo_final_v1",
        "train": "images/train", "val": "images/val", "nc": 28, "names": names,
    }), encoding="utf-8")
    for split in ("train", "val"):
        image_dir = original / "images" / split
        label_dir = original / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), "white").save(image_dir / f"{split}.png")
        labels = "0 0.5 0.5 0.8 0.8\n"
        if split == "train":
            labels += "1 0.5 0.5 0.2 0.2\n"
        (label_dir / f"{split}.txt").write_text(labels, encoding="utf-8")

    moved = tmp_path / "new_location" / "yolo_final_v1"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original), str(moved))
    audit = audit_dataset(moved)
    assert audit["images_by_split"] == {"train": 1, "val": 1}
    assert audit["objects_by_split"] == {"train": 2, "val": 1}
    assert audit["nested_box_pairs"] == 1

    runtime = local_data_yaml(moved, runs=tmp_path / "runs")
    config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert Path(config["path"]).resolve() == moved.resolve()
    assert (Path(config["path"]) / config["train"]).is_dir()
    assert (Path(config["path"]) / config["val"]).is_dir()
