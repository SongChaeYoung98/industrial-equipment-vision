"""Sort snapshot-root media into image and video directories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SNAPSHOT_ROOT = Path(
    r"C:\Users\sooji\repository\check-lab-ai\ml\data\raw\2026_09_16"
)
DESTINATIONS = {".png": "image", ".avi": "video"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not SNAPSHOT_ROOT.is_dir():
        raise SystemExit(f"snapshot root not found: {SNAPSHOT_ROOT}")

    files = [path for path in SNAPSHOT_ROOT.iterdir() if path.is_file()]
    unexpected = [path for path in files if path.suffix.lower() not in DESTINATIONS]
    if unexpected:
        raise SystemExit(
            "unexpected root files found; refusing to move anything:\n"
            + "\n".join(str(path) for path in unexpected[:20])
        )

    moves = [(path, SNAPSHOT_ROOT / DESTINATIONS[path.suffix.lower()] / path.name) for path in files]
    destinations = [destination for _, destination in moves]
    if len(destinations) != len(set(destinations)):
        raise SystemExit("destination filename collision detected")
    existing = [destination for destination in destinations if destination.exists()]
    if existing:
        raise SystemExit(
            "destination already contains files; refusing to overwrite:\n"
            + "\n".join(str(path) for path in existing[:20])
        )

    print(f"snapshot_root={SNAPSHOT_ROOT}")
    print(f"image_files={sum(1 for path in files if path.suffix.lower() == '.png')}")
    print(f"video_files={sum(1 for path in files if path.suffix.lower() == '.avi')}")
    if not args.apply:
        print("dry_run=true (use --apply to move files)")
        return 0

    for directory in {destination.parent for _, destination in moves}:
        directory.mkdir(exist_ok=True)
    for source, destination in moves:
        shutil.move(str(source), str(destination))
    print("applied=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
