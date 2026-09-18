"""Extract bounded representative frames with explicit source provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('ml/data/raw/2026_09_16/video'))
    parser.add_argument('--output', type=Path, default=Path('ml/data/derived/2026_09_16/frames'))
    parser.add_argument('--per-video', type=int, default=5)
    args = parser.parse_args()
    if args.per_video < 1:
        raise SystemExit('per-video must be positive')
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(root.parent):
        raise SystemExit('Output must be outside the raw snapshot')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, source in enumerate(sorted(root.glob('*.avi')), 1):
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise SystemExit('Unexpected linked source')
        cap = cv2.VideoCapture(str(source))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            targets = sorted({int((count - 1) * (j + .5) / args.per_video)
                              for j in range(args.per_video)}) if count > 0 else []
            for target in targets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                record = {'source': 'video/' + source.name,
                          'upload_id': source.name.split('__')[0],
                          'requested_frame': target, 'fps': fps}
                if ok:
                    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    name = f'{source.stem}__frame_{target:08d}.png'
                    success, encoded = cv2.imencode('.png', frame)
                    if not success:
                        raise RuntimeError(f'PNG encoding failed: {source}')
                    encoded.tofile(output / name)
                    record.update(status='extracted', path=name, actual_frame=actual,
                                  sha256=hashlib.sha256(encoded.tobytes()).hexdigest())
                else:
                    record.update(status='decode_failed')
                records.append(record)
        finally:
            cap.release()
        if i % 100 == 0:
            print(f'{i} videos sampled', flush=True)
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'frames={sum(r["status"] == "extracted" for r in records)} failures={sum(r["status"] != "extracted" for r in records)}')


if __name__ == '__main__':
    main()
