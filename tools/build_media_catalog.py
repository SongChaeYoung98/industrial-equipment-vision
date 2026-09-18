"""Read-only raw-media audit. Store derived metadata in a separate SQLite DB."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(path):
    result = {}
    if path.suffix.lower() == '.png':
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {'status': 'decode_failed'}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = small[:, 1:] > small[:, :-1]
        result.update(width=frame.shape[1], height=frame.shape[0],
                      brightness=float(gray.mean()),
                      sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                      dhash=f'{int("".join("1" if b else "0" for b in bits.flat), 2):016x}')
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return {'status': 'open_failed'}
            result.update(width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          fps=capture.get(cv2.CAP_PROP_FPS),
                          frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
    result['status'] = 'metadata_ok'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('ml/data/raw/2026_09_16'))
    parser.add_argument('--output', type=Path, default=Path('ml/data/derived/2026_09_16/catalog.sqlite'))
    args = parser.parse_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(root):
        raise SystemExit('Output must be outside the raw snapshot')
    files = sorted(p for folder in ('image', 'video') for p in (root / folder).iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.avi'})
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise SystemExit('Linked media outside the snapshot is not supported')
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as db:
        db.execute('CREATE TABLE IF NOT EXISTS media (path TEXT PRIMARY KEY, upload_id TEXT, kind TEXT, bytes INTEGER, mtime_ns INTEGER, sha256 TEXT, metadata TEXT)')
        for index, path in enumerate(files, 1):
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            old = db.execute('SELECT bytes, mtime_ns FROM media WHERE path=?', (rel,)).fetchone()
            if old != (stat.st_size, stat.st_mtime_ns):
                try:
                    metadata = inspect(path)
                except Exception as error:
                    metadata = {'status': 'inspection_error', 'reason': str(error)}
                db.execute('INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?,?)',
                           (rel, path.name.split('__')[0], path.parent.name,
                            stat.st_size, stat.st_mtime_ns, digest(path), json.dumps(metadata)))
            if index % 500 == 0:
                db.commit()
                print(f'{index}/{len(files)} catalogued', flush=True)
        # Remove stale catalog entries only; never remove source media.
        live = {p.relative_to(root).as_posix() for p in files}
        for (rel,) in db.execute('SELECT path FROM media').fetchall():
            if rel not in live:
                db.execute('DELETE FROM media WHERE path=?', (rel,))
        db.commit()
        print('counts:', db.execute('SELECT kind, COUNT(*) FROM media GROUP BY kind').fetchall())
        print('exact_duplicate_excess:', db.execute('SELECT COUNT(*)-COUNT(DISTINCT sha256) FROM media').fetchone()[0])
        metadata = [json.loads(row[0]) for row in db.execute('SELECT metadata FROM media')]
        summary = {
            'counts': dict(db.execute('SELECT kind, COUNT(*) FROM media GROUP BY kind')),
            'statuses': dict(Counter(m['status'] for m in metadata)),
            'dimensions': dict(Counter(f'{m.get("width")}x{m.get("height")}' for m in metadata)),
            'exact_duplicate_excess': db.execute('SELECT COUNT(*)-COUNT(DISTINCT sha256) FROM media').fetchone()[0],
        }
        output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('catalog:', output)


if __name__ == '__main__':
    main()
