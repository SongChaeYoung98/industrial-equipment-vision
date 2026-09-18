"""Copy UD sensor images into a local, immutable training snapshot.

The source is treated as read-only.  This script deliberately has no source
delete, move, rename, or overwrite operation.  It only accepts files below a
directory named ``ud_unzip`` and writes below this repository's
``ml/data/raw`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SENSOR_SUFFIXES = {
    "_t": "thermal",
    "_u": "ultraviolet",
    "_v": "vision",
}


def resolve_existing_directory(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory: {resolved}")
    return resolved


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_sensor_image(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for suffix, asset_type in SENSOR_SUFFIXES.items():
        if stem.endswith(suffix):
            return asset_type
    return "sensor_image"


def is_original_sensor_image(path: Path) -> bool:
    """Only accept the canonical ``*_T``, ``*_U``, and ``*_V`` images.

    Names such as ``*_T_markup``, ``*_U_particle``, ``*_U_offset``, and
    ``*_thumb`` are generated/preview assets even when they live in ud_unzip.
    """

    return path.suffix.lower() in IMAGE_SUFFIXES and path.stem.lower().endswith(
        ("_t", "_u", "_v")
    )


def iter_source_images(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirnames, filenames in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        # A junction/symlink must never redirect the scan outside source_root.
        dirnames[:] = [
            name
            for name in dirnames
            if not (current_path / name).is_symlink()
        ]
        if current_path.is_symlink() or "ud_unzip" not in current_path.parts:
            continue
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not is_original_sensor_image(path):
                continue
            files.append(path)
    return sorted(files)


def upload_id_from_relative_path(relative_path: Path) -> str:
    parts = relative_path.parts
    try:
        unzip_index = parts.index("ud_unzip")
    except ValueError as exc:
        raise ValueError(f"source image is outside an ud_unzip directory: {relative_path}") from exc
    if unzip_index < 3 or len(parts) <= unzip_index + 1:
        raise ValueError(
            "expected source layout {user_id}/{project_id}/{upload_id}/ud_unzip/<image>: "
            f"{relative_path}"
        )
    return parts[unzip_index - 1]


def copy_one_source_file(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing local file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse existing partial file: {temporary}")

    before = source.stat()
    try:
        # copyfile reads source and writes only the local temporary destination.
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed while being copied: {source}")
        checksum = sha256_file(temporary)
        # os.rename on Windows refuses to replace an existing destination.
        os.rename(temporary, destination)
        return checksum
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_root = (repo_root / "ml" / "data" / "raw").resolve()
    source_root = resolve_existing_directory(args.source, name="source")
    destination_root = args.destination.expanduser().resolve()

    if not is_within(destination_root, raw_root) or destination_root == raw_root:
        raise ValueError(f"destination must be a new directory below {raw_root}")
    if is_within(source_root, destination_root) or is_within(destination_root, source_root):
        raise ValueError("source and destination must be disjoint")
    if destination_root.exists():
        raise FileExistsError(f"destination already exists; choose a new snapshot: {destination_root}")

    source_files = iter_source_images(source_root)
    plan: list[tuple[Path, Path, str]] = []
    seen_destinations: set[Path] = set()
    for source in source_files:
        relative = source.relative_to(source_root)
        upload_id = upload_id_from_relative_path(relative)
        destination = destination_root / "uploads" / upload_id / source.name
        if destination in seen_destinations:
            raise ValueError(f"destination collision detected: {destination}")
        seen_destinations.add(destination)
        plan.append((source, destination, upload_id))

    total_bytes = sum(source.stat().st_size for source, _, _ in plan)
    print(json.dumps({"files": len(plan), "bytes": total_bytes, "dry_run": args.dry_run}))
    if args.dry_run:
        return 0

    destination_root.mkdir(parents=True)
    manifest_path = destination_root / "manifest.jsonl"
    manifest_temporary = destination_root / "manifest.jsonl.part"
    records: list[dict[str, object]] = []
    try:
        for source, destination, upload_id in plan:
            checksum = copy_one_source_file(source, destination)
            records.append(
                {
                    "sample_id": f"{upload_id}:{source.stem}",
                    "upload_id": upload_id,
                    "asset_type": classify_sensor_image(source.name),
                    "image_uri": destination.relative_to(destination_root).as_posix(),
                    "label": None,
                    "reviewed": False,
                    "checksum": f"sha256:{checksum}",
                    "source_kind": "ud_unzip_sensor_image",
                }
            )
        with manifest_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.rename(manifest_temporary, manifest_path)
    except Exception:
        if manifest_temporary.exists():
            manifest_temporary.unlink()
        raise

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_read_only": True,
        "source_root": "redacted",
        "destination": destination_root.relative_to(repo_root).as_posix(),
        "files": len(records),
        "bytes": total_bytes,
        "manifest": manifest_path.relative_to(repo_root).as_posix(),
    }
    (destination_root / "import-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
