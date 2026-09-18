"""Move snapshot PNG/AVI files directly under the snapshot root.

The upload directory name is prefixed to every filename because many uploads
contain identically named media files.  The operation is dry-run by default;
use --apply to perform the moves.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


SNAPSHOT_ROOT = Path(
    r"C:\Users\sooji\repository\check-lab-ai\ml\data\raw\2026_09_16"
)
MEDIA_EXTENSIONS = {".png", ".avi"}
SOURCE_DIR_NAMES = {"ud_origin", "ud_unzip"}
SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def upload_id_for(path: Path) -> str:
    """Get the upload directory identifier for both old and current layouts."""
    parts = path.parts
    for index, part in enumerate(parts):
        if part in SOURCE_DIR_NAMES and index > 0:
            return parts[index - 1]
    if path.parent != SNAPSHOT_ROOT:
        return path.parent.name
    return "root"


def safe_upload_id(value: str) -> str:
    cleaned = SAFE_ID.sub("_", value).strip("._")
    return cleaned or "upload"


def media_files() -> list[Path]:
    return sorted(
        path
        for path in SNAPSHOT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def unexpected_files() -> list[Path]:
    return sorted(
        path
        for path in SNAPSHOT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() not in MEDIA_EXTENSIONS
    )


def plan_moves(files: list[Path]) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for source in files:
        if source.parent == SNAPSHOT_ROOT:
            continue
        destination = SNAPSHOT_ROOT / (
            f"{safe_upload_id(upload_id_for(source))}__{source.name}"
        )
        if destination in destinations or destination.exists():
            raise RuntimeError(f"destination collision or existing file: {destination}")
        destinations.add(destination)
        moves.append((source, destination))
    return moves


def remove_empty_directories() -> int:
    removed = 0
    directories = sorted(
        (path for path in SNAPSHOT_ROOT.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not SNAPSHOT_ROOT.is_dir():
        raise SystemExit(f"snapshot root not found: {SNAPSHOT_ROOT}")

    unexpected = unexpected_files()
    if unexpected:
        raise SystemExit(
            "unexpected non-media files found; refusing to move/delete anything:\n"
            + "\n".join(str(path) for path in unexpected[:20])
        )

    files = media_files()
    moves = plan_moves(files)
    print(f"snapshot_root={SNAPSHOT_ROOT}")
    print(f"media_files={len(files)}")
    print(f"files_to_move={len(moves)}")
    print(f"already_at_root={len(files) - len(moves)}")

    if not args.apply:
        print("dry_run=true (use --apply to move files)")
        return 0

    for source, destination in moves:
        shutil.move(os.fspath(source), os.fspath(destination))

    removed = remove_empty_directories()
    print(f"empty_directories_removed={removed}")
    print("applied=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
