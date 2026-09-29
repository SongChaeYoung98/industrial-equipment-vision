"""Permanently remove all dataset-specific thermal assets.

For this dataset, normal frames are 640x480. Thermal frames are the
non-standard 640x512 or 320x256 frames. The script removes their source AVIs,
derived frames, processed copies, and corresponding rows/vectors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ml.training.visual_baseline import ROOT

DATA = ROOT / 'ml/data'
RAW_IMAGE = DATA / 'raw/2026_09_16/image'
RAW_VIDEO = DATA / 'raw/2026_09_16/video'
FRAMES = DATA / 'derived/2026_09_16/frames'
PROCESSED = DATA / 'processed'
EXCLUDED = {'manifest.example.jsonl'}


def unlink(path, counts, bucket):
    if path.is_file():
        path.unlink()
        counts[bucket] = counts.get(bucket, 0) + 1


def discover():
    frame_paths = []
    video_stems = set()
    for path in FRAMES.glob('*.png'):
        try:
            size = Image.open(path).size
        except OSError:
            continue
        if size != (640, 480):
            frame_paths.append(path)
            video_stems.add(path.stem.rsplit('__frame_', 1)[0])
    raw_images = []
    for path in RAW_IMAGE.glob('*.png'):
        try:
            size = Image.open(path).size
        except OSError:
            size = None
        if path.name.endswith('_T.png') or size != (640, 480):
            raw_images.append(path)
    raw_videos = [path for path in RAW_VIDEO.glob('*.avi') if path.stem in video_stems]
    return frame_paths, raw_images, raw_videos, video_stems


def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def thermal_dict(value, video_stems):
    if not isinstance(value, dict):
        return False
    if str(value.get('category', '')).casefold() == 'thermal':
        return True
    for key in ('source', 'source_path', 'path', 'image_uri'):
        raw = value.get(key)
        if not isinstance(raw, str):
            continue
        name = Path(raw.replace('\\', '/')).name
        stem = Path(name).stem
        if stem in video_stems or any(stem.startswith(v + '__frame_') for v in video_stems):
            return True
    return False


def walk(value, video_stems):
    if isinstance(value, dict):
        if thermal_dict(value, video_stems):
            yield value
        for child in value.values():
            yield from walk(child, video_stems)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child, video_stems)


def clean_json(value, video_stems):
    if isinstance(value, list):
        result = []
        for child in value:
            cleaned = clean_json(child, video_stems)
            if cleaned is not None:
                result.append(cleaned)
        return result
    if isinstance(value, dict):
        if thermal_dict(value, video_stems):
            return None
        result = {}
        for key, child in value.items():
            if key == 'categories' and isinstance(child, dict) and 'thermal' in child:
                result[key] = {k: v for k, v in child.items() if k != 'thermal'}
                if isinstance(value.get('count'), int):
                    result['count'] = max(0, value['count'] - int(child['thermal']))
            else:
                cleaned = clean_json(child, video_stems)
                if cleaned is not None:
                    result[key] = cleaned
        return result
    return value


def rewrite(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def filter_embedding_pair(directory, rows_name, vector_names, video_stems, counts):
    rows_path = directory / rows_name
    existing_vectors = [directory / name for name in vector_names if (directory / name).exists()]
    if not rows_path.exists() or not existing_vectors:
        return
    rows = json.loads(rows_path.read_text(encoding='utf-8'))
    keep = []
    for index, source in enumerate(rows):
        name = Path(source.replace('\\', '/')).name
        stem = Path(name).stem
        if stem not in video_stems and not any(stem.startswith(v + '__frame_') for v in video_stems):
            keep.append(index)
    for vectors_path in existing_vectors:
        vectors = np.load(vectors_path)
        np.save(vectors_path, vectors[keep])
    rewrite(rows_path, [rows[i] for i in keep])
    counts['embedding_rows_removed'] = counts.get('embedding_rows_removed', 0) + len(rows) - len(keep)


def main():
    frame_paths, raw_images, raw_videos, video_stems = discover()
    counts = {}
    for path in frame_paths:
        unlink(path, counts, 'thermal_frames')
    for path in raw_images:
        unlink(path, counts, 'thermal_raw_images')
    for path in raw_videos:
        unlink(path, counts, 'thermal_raw_videos')

    thermal_names = set()
    json_paths = [p for p in DATA.rglob('*.json') if p.name not in EXCLUDED]
    for path in json_paths:
        data = load_json(path)
        if data is None:
            continue
        for record in walk(data, video_stems):
            for key in ('primary_crop', 'crop_path', 'mask_path', 'path'):
                if isinstance(record.get(key), str):
                    thermal_names.add(Path(record[key].replace('\\', '/')).name)

    for path in PROCESSED.rglob('*'):
        if path.is_file() and path.name in thermal_names:
            unlink(path, counts, 'thermal_processed_files')

    full = PROCESSED / 'full_pipeline'
    filter_embedding_pair(full, 'embedding_rows.json', ['embeddings.npy', 'dino_embeddings.npy'], video_stems, counts)
    stage5 = PROCESSED / 'stage5_unknown_clusters'
    stage5_meta = stage5 / 'metadata.json'
    stage5_vectors = stage5 / 'unknown_embeddings.npy'
    if stage5_meta.exists() and stage5_vectors.exists():
        meta = load_json(stage5_meta)
        vectors = np.load(stage5_vectors)
        keep = []
        for index, item in enumerate(meta.get('items', [])):
            name = Path(str(item.get('source', '')).replace('\\', '/')).stem
            if name not in video_stems and not any(name.startswith(v + '__frame_') for v in video_stems):
                keep.append(index)
        np.save(stage5_vectors, vectors[keep])
        meta['items'] = [meta['items'][i] for i in keep]
        meta['input_count'] = len(meta['items'])
        rewrite(stage5_meta, meta)
        counts['stage5_embedding_rows_removed'] = len(vectors) - len(keep)

    json_updated = 0
    for path in json_paths:
        data = load_json(path)
        if data is None:
            continue
        cleaned = clean_json(data, video_stems)
        if cleaned != data:
            rewrite(path, cleaned)
            json_updated += 1

    for path in DATA.rglob('*.jsonl'):
        if path.name in EXCLUDED:
            continue
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
            kept = []
            changed = False
            for line in lines:
                if not line.strip():
                    continue
                value = json.loads(line)
                cleaned = clean_json(value, video_stems)
                if cleaned is None:
                    changed = True
                else:
                    kept.append(json.dumps(cleaned, ensure_ascii=False))
                    changed = changed or cleaned != value
            if changed:
                path.write_text('\n'.join(kept) + ('\n' if kept else ''), encoding='utf-8')
                json_updated += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

    print(json.dumps({'thermal_video_ids': len(video_stems), 'thermal_frames': len(frame_paths),
                      'thermal_raw_images': len(raw_images), 'thermal_raw_videos': len(raw_videos),
                      'json_files_updated': json_updated, 'deleted': counts,
                      'total_deleted_files': sum(v for k, v in counts.items() if 'rows' not in k)}, indent=2))


if __name__ == '__main__':
    main()
