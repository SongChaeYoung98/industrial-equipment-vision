"""Extract and prune a local UD snapshot using check-lab-python-back rules.

Reference-only behavior copied from the backend:
* ``.avi/.mp4/.wav`` means video.
* otherwise ``.png/.csv`` means image.

This script changes only the local AI snapshot.  It never imports or writes to
check-lab-python-back.  It is a dry-run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


SNAPSHOT_NAME = "2026_09_16"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".wav"}
IMAGE_EXTENSIONS = {".png", ".csv"}
KEEP_FOR_IMAGE = ".png"
KEEP_FOR_VIDEO = ".avi"


@dataclass
class UploadPlan:
    upload_root: Path
    origin_dir: Path
    unzip_dir: Path
    kind: str
    archives: list[Path]
    archive_members: dict[Path, list[str]]
    archive_errors: list[str]


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_root(repo_root: Path, supplied: Path | None) -> Path:
    expected = (repo_root / "ml" / "data" / "raw" / SNAPSHOT_NAME).resolve()
    root = (supplied or expected).expanduser().resolve(strict=True)
    raw_root = (repo_root / "ml" / "data" / "raw").resolve()
    if root != expected or not is_within(root, raw_root) or not root.is_dir():
        raise ValueError(f"refusing unexpected snapshot root: {root}")
    return root


def reject_symlinks(root: Path) -> None:
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            raise ValueError(f"refusing symlink: {current_path}")
        for name in [*dirnames, *filenames]:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"refusing symlink: {path}")


def determine_backend_type(extensions: set[str]) -> str:
    if extensions & VIDEO_EXTENSIONS:
        return "video"
    if extensions & IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def read_archive_members(archive: Path) -> tuple[list[str], str | None]:
    try:
        with zipfile.ZipFile(archive) as handle:
            return [info.filename for info in handle.infolist() if not info.is_dir()], None
    except (OSError, zipfile.BadZipFile) as exc:
        return [], f"{archive}: {exc}"


def collect_upload_plans(root: Path) -> list[UploadPlan]:
    upload_roots = set()
    for path in root.rglob("*"):
        if path.is_dir() and path.name in {"ud_origin", "ud_unzip"}:
            upload_roots.add(path.parent)

    plans: list[UploadPlan] = []
    for upload_root in sorted(upload_roots):
        origin_dir = upload_root / "ud_origin"
        unzip_dir = upload_root / "ud_unzip"
        archives = sorted(origin_dir.glob("*.ud")) if origin_dir.is_dir() else []
        archive_members: dict[Path, list[str]] = {}
        archive_errors: list[str] = []
        extensions: set[str] = set()

        for archive in archives:
            members, error = read_archive_members(archive)
            archive_members[archive] = members
            extensions.update(Path(member).suffix.lower() for member in members)
            if error:
                archive_errors.append(error)

        if unzip_dir.is_dir():
            extensions.update(
                path.suffix.lower()
                for path in unzip_dir.rglob("*")
                if path.is_file()
            )
        if origin_dir.is_dir():
            extensions.update(
                path.suffix.lower()
                for path in origin_dir.rglob("*")
                if path.is_file()
            )

        plans.append(
            UploadPlan(
                upload_root=upload_root,
                origin_dir=origin_dir,
                unzip_dir=unzip_dir,
                kind=determine_backend_type(extensions),
                archives=archives,
                archive_members=archive_members,
                archive_errors=archive_errors,
            )
        )
    return plans


def selected_members(plan: UploadPlan) -> list[tuple[Path, str]]:
    wanted = KEEP_FOR_IMAGE if plan.kind == "image" else KEEP_FOR_VIDEO
    result: list[tuple[Path, str]] = []
    for archive, members in plan.archive_members.items():
        for member in members:
            if Path(member).suffix.lower() == wanted:
                result.append((archive, member))
    return result


def safe_member_name(member: str) -> str:
    path = Path(member)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"unsafe archive member path: {member}")
    return path.name


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def extract_selected_members(plan: UploadPlan) -> int:
    target_dir = plan.origin_dir if plan.kind == "image" else plan.unzip_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for archive, member in selected_members(plan):
        destination = target_dir / safe_member_name(member)
        with zipfile.ZipFile(archive) as handle:
            info = handle.getinfo(member)
            temporary = destination.with_name(destination.name + ".part")
            if destination.exists():
                with handle.open(info) as source:
                    incoming = sha256_stream(source)
                if incoming != sha256_path(destination):
                    raise RuntimeError(f"local target collision with different bytes: {destination}")
                continue
            if temporary.exists():
                raise FileExistsError(f"refusing existing partial file: {temporary}")
            with handle.open(info) as source, temporary.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.rename(temporary, destination)
            extracted += 1
    return extracted


def count_tree(path: Path) -> tuple[int, int, int]:
    files = 0
    directories = 0
    bytes_total = 0
    if path.is_file():
        return 1, 0, path.stat().st_size
    for current, dirnames, filenames in os.walk(path, followlinks=False):
        directories += len(dirnames)
        files += len(filenames)
        for filename in filenames:
            bytes_total += (Path(current) / filename).stat().st_size
    return files, directories, bytes_total


def delete_children_except(plan: UploadPlan) -> tuple[int, int, int]:
    target_dir = plan.origin_dir if plan.kind == "image" else plan.unzip_dir
    keep_suffix = KEEP_FOR_IMAGE if plan.kind == "image" else KEEP_FOR_VIDEO
    deleted_files = 0
    deleted_directories = 0
    deleted_bytes = 0

    if not plan.upload_root.is_dir():
        return 0, 0, 0

    for child in list(plan.upload_root.iterdir()):
        if child == target_dir and child.is_dir():
            for target_child in list(child.iterdir()):
                if target_child.is_file() and target_child.suffix.lower() == keep_suffix:
                    continue
                files, directories, bytes_total = count_tree(target_child)
                if target_child.is_dir():
                    shutil.rmtree(target_child)
                    deleted_directories += directories + 1
                else:
                    target_child.unlink()
                deleted_files += files
                deleted_bytes += bytes_total
            continue

        files, directories, bytes_total = count_tree(child)
        if child.is_dir():
            shutil.rmtree(child)
            deleted_directories += directories + 1
        else:
            child.unlink()
        deleted_files += files
        deleted_bytes += bytes_total

    return deleted_files, deleted_directories, deleted_bytes


def delete_all_children(plan: UploadPlan) -> tuple[int, int, int]:
    deleted_files = 0
    deleted_directories = 0
    deleted_bytes = 0
    if not plan.upload_root.is_dir():
        return 0, 0, 0
    for child in list(plan.upload_root.iterdir()):
        files, directories, bytes_total = count_tree(child)
        if child.is_dir():
            shutil.rmtree(child)
            deleted_directories += directories + 1
        else:
            child.unlink()
        deleted_files += files
        deleted_bytes += bytes_total
    return deleted_files, deleted_directories, deleted_bytes


def remove_empty_directories(root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if not any(directory.iterdir()):
            directory.rmdir()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = resolve_root(repo_root, args.root)
    reject_symlinks(root)
    plans = collect_upload_plans(root)
    unknown = [plan for plan in plans if plan.kind == "unknown"]

    counts = {"image": 0, "video": 0}
    extract_counts = {"image": 0, "video": 0}
    for plan in plans:
        if plan.kind == "unknown":
            continue
        counts[plan.kind] += 1
        extract_counts[plan.kind] += len(selected_members(plan))

    print(f"root={root}")
    print(f"image_uploads={counts['image']}")
    print(f"video_uploads={counts['video']}")
    print(f"image_png_members_to_extract={extract_counts['image']}")
    print(f"video_avi_members_to_extract={extract_counts['video']}")
    print(f"bad_ud_archives={sum(bool(plan.archive_errors) for plan in plans)}")
    print(f"unknown_uploads_to_delete={len(unknown)}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")

    if not args.apply:
        print("No files were changed. Re-run with --apply to extract and prune this snapshot.")
        return 0

    extracted = 0
    for plan in plans:
        if plan.kind == "unknown":
            continue
        extracted += extract_selected_members(plan)

    deleted_files = deleted_directories = deleted_bytes = 0
    for plan in plans:
        if plan.kind == "unknown":
            files, directories, bytes_total = delete_all_children(plan)
        else:
            files, directories, bytes_total = delete_children_except(plan)
        deleted_files += files
        deleted_directories += directories
        deleted_bytes += bytes_total

    # Remove any snapshot-level item that is not an upload tree.
    upload_roots = {plan.upload_root for plan in plans}
    for child in list(root.iterdir()):
        if any(child == plan.upload_root or is_within(plan.upload_root, child) for plan in plans):
            continue
        files, directories, bytes_total = count_tree(child)
        if child.is_dir():
            shutil.rmtree(child)
            deleted_directories += directories + 1
        else:
            child.unlink()
        deleted_files += files
        deleted_bytes += bytes_total

    removed_empty_directories = remove_empty_directories(root)

    print(f"extracted_files={extracted}")
    print(f"deleted_files={deleted_files}")
    print(f"deleted_directories={deleted_directories}")
    print(f"deleted_bytes={deleted_bytes}")
    print(f"removed_empty_directories={removed_empty_directories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
