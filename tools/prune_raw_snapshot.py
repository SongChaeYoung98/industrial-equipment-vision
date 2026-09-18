"""Remove everything except ud_origin and ud_unzip trees from one snapshot.

Default mode is a read-only dry-run.  Actual deletion requires ``--apply``.
The root is deliberately constrained to this repository's
``ml/data/raw/2026_09_16`` directory so a typo cannot target an arbitrary path.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


KEEP_DIRECTORY_NAMES = frozenset({"ud_origin", "ud_unzip"})
SNAPSHOT_NAME = "2026_09_16"


@dataclass(frozen=True)
class CleanupPlan:
    root: Path
    keep_directories: tuple[Path, ...]
    delete_entries: tuple[Path, ...]


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_snapshot(repo_root: Path, supplied: Path | None) -> Path:
    expected_root = (repo_root / "ml" / "data" / "raw" / SNAPSHOT_NAME).resolve()
    root = (supplied or expected_root).expanduser().resolve(strict=True)

    raw_root = (repo_root / "ml" / "data" / "raw").resolve()
    if root != expected_root or not is_within(root, raw_root):
        raise ValueError(f"refusing unexpected snapshot root: {root}")
    if not root.is_dir():
        raise ValueError(f"snapshot root is not a directory: {root}")
    return root


def collect_keep_directories(root: Path) -> tuple[Path, ...]:
    keep_directories: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing to process symlink: {path}")
        if path.is_dir() and path.name in KEEP_DIRECTORY_NAMES:
            keep_directories.append(path)
    if not keep_directories:
        raise ValueError("no ud_origin or ud_unzip directories found; refusing cleanup")
    return tuple(sorted(keep_directories))
2

def build_plan(root: Path) -> CleanupPlan:
    keep_directories = collect_keep_directories(root)
    protected_ancestors: set[Path] = {root}
    for keep_directory in keep_directories:
        current = keep_directory
        while current != root:
            protected_ancestors.add(current)
            current = current.parent

    delete_entries: list[Path] = []

    def inspect_directory(directory: Path) -> None:
        for child in directory.iterdir():
            if child.is_symlink():
                raise ValueError(f"refusing to process symlink: {child}")
            if child in protected_ancestors:
                if child.is_dir() and child.name not in KEEP_DIRECTORY_NAMES:
                    inspect_directory(child)
                continue
            delete_entries.append(child)

    inspect_directory(root)
    return CleanupPlan(
        root=root,
        keep_directories=keep_directories,
        delete_entries=tuple(delete_entries),
    )


def count_tree(path: Path) -> tuple[int, int, int]:
    files = 0
    directories = 0
    bytes_total = 0
    if path.is_file():
        return 1, 0, path.stat().st_size
    for current, dirnames, filenames in __import__("os").walk(path, followlinks=False):
        directories += len(dirnames)
        files += len(filenames)
        for filename in filenames:
            bytes_total += (Path(current) / filename).stat().st_size
    return files, directories, bytes_total


def execute(plan: CleanupPlan) -> tuple[int, int, int]:
    deleted_files = 0
    deleted_directories = 0
    deleted_bytes = 0
    for entry in plan.delete_entries:
        if not is_within(entry.resolve(), plan.root) or entry == plan.root:
            raise RuntimeError(f"refusing deletion outside snapshot root: {entry}")
        files, directories, bytes_total = count_tree(entry)
        if entry.is_dir():
            shutil.rmtree(entry)
            deleted_directories += directories + 1
        else:
            entry.unlink()
        deleted_files += files
        deleted_bytes += bytes_total
    return deleted_files, deleted_directories, deleted_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform deletion; without this flag only a dry-run is performed",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = resolve_snapshot(repo_root, args.root)
    plan = build_plan(root)

    delete_files = 0
    delete_directories = 0
    delete_bytes = 0
    for entry in plan.delete_entries:
        files, directories, bytes_total = count_tree(entry)
        delete_files += files
        delete_directories += directories + int(entry.is_dir())
        delete_bytes += bytes_total

    print(f"root={root}")
    print(f"preserved_directories={len(plan.keep_directories)}")
    print(f"delete_entries={len(plan.delete_entries)}")
    print(f"delete_files={delete_files}")
    print(f"delete_directories={delete_directories}")
    print(f"delete_bytes={delete_bytes}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")

    if not args.apply:
        print("No files were deleted. Re-run with --apply to perform this exact cleanup.")
        return 0

    deleted_files, deleted_directories, deleted_bytes = execute(plan)
    print(
        "deleted_files="
        f"{deleted_files} deleted_directories={deleted_directories} "
        f"deleted_bytes={deleted_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
