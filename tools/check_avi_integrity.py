"""Find AVI files that FFmpeg cannot fully decode.

The default mode only reports results. Use --delete after reviewing the
reported files to remove files for which FFmpeg returns a decoding error.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError as exc:  # pragma: no cover
    raise SystemExit("imageio-ffmpeg is required: pip install imageio-ffmpeg") from exc


VIDEO_DIR = Path(
    r"C:\Users\sooji\repository\check-lab-ai\ml\data\raw\2026_09_16\video"
)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def inspect_video(path: Path, timeout_seconds: int) -> tuple[bool, str]:
    command = [
        FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout_after_{timeout_seconds}s"

    if result.returncode == 0:
        return True, "ok"
    diagnostics = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    reason = diagnostics[-1] if diagnostics else f"ffmpeg_exit_{result.returncode}"
    return False, reason[:240]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete AVI files for which FFmpeg reports a decode failure",
    )
    parser.add_argument("--folder", type=Path, default=VIDEO_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"video directory not found: {args.folder}")

    videos = sorted(path for path in args.folder.iterdir() if path.suffix.lower() == ".avi")
    unreadable: list[Path] = []
    for index, path in enumerate(videos, start=1):
        valid, reason = inspect_video(path, args.timeout)
        state = "OK" if valid else "UNREADABLE_OR_CORRUPT"
        print(f"[{index}/{len(videos)}] {state} {path.name} ({reason})")
        if not valid:
            unreadable.append(path)

    print(f"total={len(videos)}")
    print(f"unreadable_or_corrupt={len(unreadable)}")
    if args.delete:
        for path in unreadable:
            path.unlink()
        print(f"deleted={len(unreadable)}")
    else:
        print("deleted=0 (use --delete after reviewing the report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
