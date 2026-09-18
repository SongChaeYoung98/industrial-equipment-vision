"""Move retained PNG/AVI files out of ud_origin and ud_unzip.

Only the local ``ml/data/raw/2026_09_16`` snapshot is touched.  The default is
a dry-run; actual deletion and moves require ``--apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path


SNAPSHOT_NAME = "2026_09_16"
MEDIA_EXTENSIONS = {".png", ".avi"}
U_IMAGE_PATTERN = re.compile(r"(^|_)U($|_)", re.IGNORECASE)
SOURCE_DIRECTORY_NAMES = {"ud_origin", "ud_unzip"}


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_root(repo_root: Path) -> Path:
    expected = (repo_root / "ml" / "data" / "raw" / SNAPSHOT_NAME).resolve()
    root = expected.resolve(strict=True)
    raw_root = (repo_root / "ml" / "data" / "raw").resolve()
    if root != expected or not is_within(root, raw_root) or not root.is_dir():
        raise ValueError(f"refusing unexpected snapshot root: {root}")
    return root


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_directory(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name in SOURCE_DIRECTORY_NAMES:
            return parent
    return None


def collect_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = resolve_root(repo_root)
    files = collect_files(root)

    delete_paths: list[Path] = []
    move_pairs: list[tuple[Path, Path]] = []
    duplicate_paths: list[Path] = []
    already_flattened: list[Path] = []
    seen_destinations: set[Path] = set()

    for path in files:
        suffix = path.suffix.lower()
        source_dir = source_directory(path)
        is_u_png = suffix == ".png" and U_IMAGE_PATTERN.search(path.stem) is not None
        if path.stat().st_size == 0 or is_u_png or suffix not in MEDIA_EXTENSIONS:
            delete_paths.append(path)
            continue

        if source_dir is None:
            if suffix in MEDIA_EXTENSIONS:
                already_flattened.append(path)
                continue
            raise ValueError(f"media file is not under ud_origin/ud_unzip: {path}")
        destination = source_dir.parent / path.name
        if destination in seen_destinations:
            raise ValueError(f"destination name collision: {destination}")
        seen_destinations.add(destination)
        if destination.exists():
            if destination.is_dir():
                raise ValueError(f"destination is a directory: {destination}")
            if sha256_path(path) != sha256_path(destination):
                raise ValueError(f"destination collision has different bytes: {destination}")
            duplicate_paths.append(path)
        else:
            move_pairs.append((path, destination))

    empty_directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )

    print(f"root={root}")
    print(f"total_files={len(files)}")
    print(f"move_files={len(move_pairs)}")
    print(f"delete_files={len(delete_paths)}")
    print(f"duplicate_source_files={len(duplicate_paths)}")
    print(f"already_flattened_files={len(already_flattened)}")
    print(f"empty_directories_to_remove={len(empty_directories)}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")

    if not args.apply:
        print("No files were changed. Re-run with --apply to perform this exact cleanup.")
        return 0

    for source, destination in move_pairs:
        if destination.exists():
            raise FileExistsError(f"destination appeared during run: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    for path in duplicate_paths + delete_paths:
        if path.exists():
            path.unlink()

    removed_directories = 0
    for directory in empty_directories:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
            removed_directories += 1

    print(f"moved_files={len(move_pairs)}")
    print(f"deleted_files={len(delete_paths) + len(duplicate_paths)}")
    print(f"removed_empty_directories={removed_directories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
