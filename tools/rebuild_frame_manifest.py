"""Rebuild frame provenance from preserved representative PNG filenames.

The operation is limited to the derived frame manifest. It never creates,
deletes, or modifies raw images/videos or the frame pixels themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FRAME_RE = re.compile(r"^(?P<video>.+)__frame_(?P<frame>\d+)\.png$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("ml/data/derived/2026_09_16/frames"))
    parser.add_argument("--videos", type=Path, default=Path("ml/data/raw/2026_09_16/video"))
    args = parser.parse_args()
    frame_root = args.frames.resolve()
    video_root = args.videos.resolve()
    if not frame_root.is_dir():
        raise SystemExit(f"Missing frame directory: {frame_root}")

    records = []
    skipped = []
    for path in sorted(frame_root.glob("*.png")):
        match = FRAME_RE.match(path.name)
        if not match:
            skipped.append(path.name)
            continue
        video_name = f"{match.group('video')}.avi"
        payload = path.read_bytes()
        source = video_root / video_name
        records.append({
            "source": f"video/{video_name}",
            "upload_id": video_name.split("__", 1)[0],
            "requested_frame": int(match.group("frame")),
            "actual_frame": int(match.group("frame")),
            "fps": None,
            "status": "extracted",
            "path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_exists": source.is_file(),
        })
    output = frame_root / "manifest.json"
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "manifest": str(output),
        "frames": len(records),
        "unrecognized_files": skipped,
        "missing_source_videos": sum(not row["source_exists"] for row in records),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
