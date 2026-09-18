"""Permanently purge thermal assets identified by project metadata.

This is intentionally narrow: it deletes only `_T.png` assets, thermal video
IDs recorded in the approved sample manifest, their derived frames, and
processed artifacts whose metadata category is thermal. It also removes those
records from JSON metadata while preserving the surrounding schema.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ml.training.visual_baseline import ROOT

DATA = ROOT / 'ml/data'
RAW = DATA / 'raw'
DERIVED = DATA / 'derived'
PROCESSED = DATA / 'processed'
SAMPLE_MANIFEST = PROCESSED / 'stage1_sample/sample_manifest.json'
EXCLUDED_JSON = {'manifest.example.jsonl'}
KNOWN_THERMAL_VIDEO_STEMS = {
    '462fc5b1-6314-4c7f-8e2b-8c5a72d7fa66__20260319_151916',
    '4b1afadb-f813-451e-9b6d-8cfd13f446e5__20251113_182145',
    '52eeab84-a803-450d-8c59-0ab6740ca2ba__20251106_125653',
    '55634f6e-351a-4ec6-98aa-6a4e8ab221a9__20251106_110705',
    '600785c8-6027-4288-9b93-5b0038928195__20260113_105645',
    '6b7a4b5e-2f95-4e2d-8f9d-5973e143def3__20260319_151936',
    '6ff8df3e-1c2a-46cb-a87b-49c556a5b940__20251113_174853',
    '87c308f7-0c00-40b0-91a9-670cb49faaf2__20250818_134856',
    '8c5ace9c-6411-4c79-bd35-dfb1f1a11e99__20251106_113413',
    '99470cfb-aecf-4db6-8074-d8d521cefc84__20251106_125446',
    'dc3fa32c-22e4-41cd-a7ae-8c24d7b009f7__20251106_125424',
    'e10f8d57-d865-4807-ad60-7120557b7e2a__20251113_174911',
    'f4f6513d-57b5-4f2a-8ef2-641ca7bbc6c4__20251106_102135',
}


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def thermal_assets_from_manifest():
    data = load_json(SAMPLE_MANIFEST)
    if not isinstance(data, dict):
        raise RuntimeError(f'Cannot read sample manifest: {SAMPLE_MANIFEST}')
    thermal = [r for r in data.get('records', []) if isinstance(r, dict)
               and str(r.get('category', '')).casefold() == 'thermal']
    image_sources = {str(r['source']).replace('/', '\\') for r in thermal
                     if r.get('kind') == 'image' and r.get('source')}
    video_sources = {str(r['source']).replace('/', '\\') for r in thermal
                     if r.get('kind') == 'frame' and r.get('source')}
    video_stems = {Path(s).stem for s in video_sources} | KNOWN_THERMAL_VIDEO_STEMS
    return image_sources, video_sources, video_stems


def collect_processed_names():
    """Collect crop/mask basenames from any thermal metadata records."""
    names = set()
    for path in DATA.rglob('*.json'):
        if path.name in EXCLUDED_JSON:
            continue
        data = load_json(path)
        if data is None:
            continue
        for record in walk_records(data):
            if str(record.get('category', '')).casefold() != 'thermal':
                continue
            for key in ('primary_crop', 'crop_path', 'mask_path', 'path'):
                value = record.get(key)
                if isinstance(value, str):
                    names.add(Path(value.replace('\\', '/')).name)
    return names


def walk_records(value: Any):
    if isinstance(value, dict):
        if 'category' in value:
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def remove_thermal_records(value: Any):
    """Remove thermal records and clean aggregate category counts."""
    if isinstance(value, list):
        result = []
        for child in value:
            cleaned = remove_thermal_records(child)
            if cleaned is not None:
                result.append(cleaned)
        return result
    if isinstance(value, dict):
        if str(value.get('category', '')).casefold() == 'thermal':
            return None
        if str(value.get('modality', '')).casefold() == 'thermal':
            return None
        result = {}
        for key, child in value.items():
            if key == 'categories' and isinstance(child, dict) and 'thermal' in child:
                result[key] = {k: v for k, v in child.items() if k != 'thermal'}
                if isinstance(value.get('count'), int):
                    result['count'] = max(0, value['count'] - int(child['thermal']))
            else:
                cleaned = remove_thermal_records(child)
                if cleaned is not None:
                    result[key] = cleaned
        return result
    return value


def delete_file(path: Path, bucket: str, counts: dict[str, int]):
    if path.is_file():
        path.unlink()
        counts[bucket] = counts.get(bucket, 0) + 1


def main():
    image_sources, video_sources, video_stems = thermal_assets_from_manifest()
    processed_names = collect_processed_names()
    counts: dict[str, int] = {}

    # Raw thermal images: all files using the repository's thermal suffix plus
    # exact manifest paths.
    for path in RAW.rglob('*_T.png'):
        delete_file(path, 'raw_images', counts)
    for source in image_sources:
        delete_file(RAW / source, 'raw_images', counts)

    # Raw videos and every derived frame belonging to the identified videos.
    for source in video_sources:
        delete_file(RAW / source, 'raw_videos', counts)
    for path in RAW.rglob('*.avi'):
        if path.stem in video_stems:
            delete_file(path, 'raw_videos', counts)
    for path in DERIVED.rglob('*'):
        if not path.is_file():
            continue
        normalized = path.name
        if any(normalized.startswith(stem) for stem in video_stems):
            delete_file(path, 'derived_video_frames', counts)

    # Stage 1/2/3 crops, masks, and copied review images referenced by thermal
    # records. The basename match handles renamed parent folders safely.
    for path in PROCESSED.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.npy'}:
            continue
        if path.name in processed_names:
            delete_file(path, 'processed_artifacts', counts)

    # Pilot grids and embedding runs contain thermal tiles/vectors and cannot
    # be made non-thermal by removing only individual crop files.
    for path in (PROCESSED / 'stage1_sample').glob('*grid*.jpg'):
        delete_file(path, 'thermal_mixed_grids', counts)
    for run_name in ('stage2_3_grayscale_dinov2', 'stage2_3_grayscale_dinov2_v3'):
        run = PROCESSED / 'stage1_sample' / run_name
        if run.is_dir():
            for path in [p for p in run.rglob('*') if p.is_file()]:
                delete_file(path, 'thermal_mixed_embedding_runs', counts)
            shutil.rmtree(run)

    # Pop thermal records from JSON metadata/manifests. Write atomically so a
    # failed write cannot leave a truncated metadata file.
    json_updated = 0
    for path in DATA.rglob('*.json'):
        if path.name in EXCLUDED_JSON or path == SAMPLE_MANIFEST:
            continue
        data = load_json(path)
        if data is None:
            continue
        cleaned = remove_thermal_records(data)
        if cleaned != data:
            temp = path.with_suffix(path.suffix + '.tmp')
            temp.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8')
            os.replace(temp, path)
            json_updated += 1

    # The sample manifest is itself the source of the thermal identity list.
    manifest = load_json(SAMPLE_MANIFEST)
    cleaned_manifest = remove_thermal_records(manifest)
    if cleaned_manifest != manifest:
        temp = SAMPLE_MANIFEST.with_suffix('.json.tmp')
        temp.write_text(json.dumps(cleaned_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(temp, SAMPLE_MANIFEST)
        json_updated += 1

    print(json.dumps({'deleted': counts, 'total_deleted_files': sum(counts.values()),
                      'json_files_updated': json_updated,
                      'thermal_video_ids': len(video_stems),
                      'thermal_processed_names': len(processed_names)}, indent=2))


if __name__ == '__main__':
    main()
