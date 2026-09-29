"""Sync Explorer feedback from stage5_unknown_clusters and merge new anchors.

Delete unwanted files and rename reviewed folders before running. Renamed
folders become temporary labels; default cluster_N folders are not labels.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from ml.training.visual_baseline import ROOT

STAGE5 = ROOT / 'ml/data/processed/stage5_unknown_clusters'
META = STAGE5 / 'metadata.json'
EMBEDDINGS = STAGE5 / 'unknown_embeddings.npy'
BASE_ANCHORS = ROOT / 'ml/data/processed/full_pipeline/anchors.json'
FULL_FINAL = ROOT / 'ml/data/processed/full_pipeline/final_records.json'
FULL_EMBEDDINGS = ROOT / 'ml/data/processed/full_pipeline/embeddings.npy'
FULL_EMBEDDING_ROWS = ROOT / 'ml/data/processed/full_pipeline/embedding_rows.json'
DEFAULT_FOLDERS = {f'cluster_{i}' for i in range(20)}


def main():
    base = json.loads(META.read_text(encoding='utf-8'))
    vectors = np.load(EMBEDDINGS)
    items = base['items']
    if len(items) != len(vectors):
        raise RuntimeError('Stage 5 metadata and embedding rows differ')
    by_name = {item['primary_crop']: index for index, item in enumerate(items)}
    locations = {}
    for folder in STAGE5.iterdir():
        if not folder.is_dir() or folder.name.startswith('sync_'):
            continue
        for path in folder.iterdir():
            if path.is_file() and path.name in by_name:
                if path.name in locations:
                    raise RuntimeError(f'Duplicate reviewed file: {path.name}')
                locations[path.name] = folder.name
    reviewed = []
    groups = defaultdict(list)
    for index, item in enumerate(items):
        updated = dict(item)
        folder = locations.get(item['primary_crop'])
        if folder is None:
            updated['status'] = 'dropped'
            updated['temp_label'] = None
        else:
            updated['status'] = 'retained'
            updated['temp_label'] = None if folder in DEFAULT_FOLDERS else folder
            if updated['temp_label']:
                groups[updated['temp_label']].append(index)
        reviewed.append(updated)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = STAGE5 / f'sync_{timestamp}'
    output.mkdir()
    (output / 'metadata.json').write_text(json.dumps({
        'input_count': len(items), 'retained': sum(x['status'] == 'retained' for x in reviewed),
        'dropped': sum(x['status'] == 'dropped' for x in reviewed), 'items': reviewed
    }, indent=2), encoding='utf-8')

    anchors = json.loads(BASE_ANCHORS.read_text(encoding='utf-8'))
    anchor_vectors = {label: np.asarray(vector, dtype='float32')
                      for label, vector in anchors.get('vectors', {}).items()}
    if not anchor_vectors:
        # Backward-compatible reconstruction for the first pipeline version,
        # whose anchors.json stored labels but not the vectors themselves.
        full = json.loads(FULL_FINAL.read_text(encoding='utf-8'))
        full_sources = json.loads(FULL_EMBEDDING_ROWS.read_text(encoding='utf-8'))
        full_vectors = np.load(FULL_EMBEDDINGS)
        source_to_vector = {source: full_vectors[index] for index, source in enumerate(full_sources)}
        grouped_base = defaultdict(list)
        for item in full['records']:
            label = item.get('auto_label')
            if label and label != 'unknown' and item.get('source') in source_to_vector:
                grouped_base[label].append(source_to_vector[item['source']])
        for label, values in grouped_base.items():
            mean = np.mean(values, axis=0)
            anchor_vectors[label] = mean / max(float(np.linalg.norm(mean)), 1e-12)
    for label, indices in groups.items():
        mean = vectors[indices].mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
        anchor_vectors[label] = mean
    merged = {'labels': sorted(anchor_vectors), 'threshold': anchors.get('threshold', .55),
              'vectors': {label: vector.tolist() for label, vector in anchor_vectors.items()},
              'new_labels': sorted(groups)}
    (output / 'anchors_merged.json').write_text(json.dumps(merged, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'retained': sum(x['status'] == 'retained' for x in reviewed),
                      'dropped': sum(x['status'] == 'dropped' for x in reviewed),
                      'new_labels': sorted(groups)}, indent=2))


if __name__ == '__main__':
    main()
