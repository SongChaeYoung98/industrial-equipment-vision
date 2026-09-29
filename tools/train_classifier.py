"""Train a supervised equipment classifier from reviewed crop folders.

Only explicit folder names are accepted as classes. ``unknown`` and
``cluster_*`` are always excluded. This script prepares a ResNet50 classifier
but does not run training until explicitly invoked.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFile
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "ml/data/processed/stage6_auto_clusters_v2"
DEFAULT_RUN = ROOT / "ml/data/runs/supervised_classifier_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_EQUIPMENT_LABELS = {"office", "background", "no_equipment"}
ImageFile.LOAD_TRUNCATED_IMAGES = False


def is_explicit_label(name: str) -> bool:
    lowered = name.lower()
    return (lowered != "unknown" and not lowered.startswith("cluster_")
            and lowered not in NON_EQUIPMENT_LABELS)


def capture_group(filename: str) -> str:
    stem = Path(filename).stem
    if "__frame_" in stem:
        return stem.split("__frame_", 1)[0]
    parts = stem.split("__")
    return parts[1] if len(parts) > 1 else stem


def collect(data_root: Path):
    items = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir() or not is_explicit_label(folder.name):
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append({"path": str(path), "label": folder.name, "group": capture_group(path.name)})
    return items


class Crops(Dataset):
    def __init__(self, items, indices, transform):
        self.items = items
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.items[self.indices[index]]
        with Image.open(item["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), item["label"]


def build_model(class_count: int, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("val-ratio must be between 0 and 1")
    items = collect(args.data_root)
    labels = sorted({item["label"] for item in items})
    if len(labels) < 2:
        raise SystemExit("At least two explicit label folders are required")
    if not items:
        raise SystemExit("No explicitly labeled crop images found")
    label_to_index = {label: index for index, label in enumerate(labels)}
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=20260921)
    train_indices, val_indices = next(splitter.split(items, groups=[item["group"] for item in items]))
    train_counts = Counter(items[index]["label"] for index in train_indices)
    val_counts = Counter(items[index]["label"] for index in val_indices)
    missing = [label for label in labels if not train_counts[label] or not val_counts[label]]
    if missing:
        raise SystemExit(f"Group split has no train or validation examples for: {missing}")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    train_loader = DataLoader(Crops(items, train_indices, transform_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(Crops(items, val_indices, transform_val), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Inverse-frequency weights prevent the 26-image solar class from being
    # ignored in favor of the 1,620-image gauges class.
    total_train = len(train_indices)
    class_weights = torch.tensor(
        [total_train / (len(labels) * train_counts[label]) for label in labels],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "classes.json").write_text(json.dumps({"classes": labels}, indent=2), encoding="utf-8")
    settings = vars(args) | {"labels": labels, "train_count": len(train_indices),
                              "validation_count": len(val_indices), "device": str(device),
                              "class_weights": dict(zip(labels, class_weights.cpu().tolist()))}
    (args.run / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    history = []
    best_val_accuracy = -1.0
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0; train_correct = 0; train_total = 0
        for images, names in train_loader:
            targets = torch.tensor([label_to_index[name] for name in names], device=device)
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * len(targets)
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_total += len(targets)
        model.eval(); val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.inference_mode():
            for images, names in val_loader:
                targets = torch.tensor([label_to_index[name] for name in names], device=device)
                images = images.to(device, non_blocking=True)
                logits = model(images); val_loss += criterion(logits, targets).item() * len(targets)
                val_correct += (logits.argmax(1) == targets).sum().item(); val_total += len(targets)
        row = {"epoch": epoch, "train_loss": train_loss / train_total,
               "train_accuracy": train_correct / train_total,
               "val_loss": val_loss / val_total, "val_accuracy": val_correct / val_total}
        history.append(row); print(json.dumps(row), flush=True)
        torch.save({"model": model.state_dict(), "classes": labels, "epoch": epoch},
                   args.run / "classifier_last.pt")
        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "classes": labels, "epoch": epoch,
                        "val_accuracy": best_val_accuracy}, args.run / "classifier_best.pt")
        (args.run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"training_completed": True, "epochs": args.epochs,
                      "best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy}), flush=True)


if __name__ == "__main__":
    main()
