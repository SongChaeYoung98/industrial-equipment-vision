"""Synchronize human review performed directly in stage3_clusters.

Workflow:
1. Open ml/data/processed/stage3_clusters in Explorer.
2. Delete unwanted crop files and rename folders freely.
3. Run this script. It writes synced_metadata.json without deleting files.

Folders named exactly ``mixed`` or ``recluster`` are kept without a label and
are reclustered separately with the requested K (default 4).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from ml.training.visual_baseline import ROOT

STAGE3 = ROOT / 'ml/data/processed/stage3_clusters'
BASE_META = ROOT / 'ml/data/processed/stage1_sample/metadata_v3.json'
PCA_EMBEDDINGS = ROOT / 'ml/data/processed/stage1_sample/stage2_3_grayscale_dinov2_v3/pca32_embeddings.npy'
MIXED_NAMES = {'mixed', 'recluster'}


def file_index(stage3: Path):
    found = defaultdict(list)
    unknown = []
    for folder in sorted(p for p in stage3.iterdir() if p.is_dir() and not p.name.startswith('sync_')):
        # Nested files are allowed so Explorer-created subfolders do not silently
        # lose reviewed images; their immediate top-level folder is the label.
        for path in folder.rglob('*'):
            if not path.is_file():
                continue
            found[path.name].append((folder.name, path))
    return found, unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3-dir', type=Path, default=STAGE3)
    parser.add_argument('--mixed-k', type=int, default=4)
    args = parser.parse_args()
    stage3 = args.stage3_dir.resolve()
    base = json.loads(BASE_META.read_text(encoding='utf-8'))
    items = base['items']
    embeddings = np.load(PCA_EMBEDDINGS)
    if len(items) != len(embeddings):
        raise RuntimeError('metadata_v3 items and PCA embedding rows differ')
    if not stage3.is_dir():
        raise FileNotFoundError(stage3)

    found, _ = file_index(stage3)
    by_name = {item['primary_crop']: i for i, item in enumerate(items)}
    duplicate_names = sorted(name for name, locations in found.items() if len(locations) > 1)
    if duplicate_names:
        raise RuntimeError('Duplicate crop files found in multiple folders: ' + ', '.join(duplicate_names))

    records = []
    mixed_indices = []
    unknown_files = []
    retained = 0
    dropped = 0
    for index, item in enumerate(items):
        name = item['primary_crop']
        locations = found.get(name, [])
        updated = dict(item)
        updated['previous_cluster'] = item['cluster']
        if not locations:
            updated['status'] = 'dropped'
            updated['temp_label'] = None
            updated['drop_reason'] = 'deleted_from_review_folder'
            dropped += 1
        else:
            folder_name, path = locations[0]
            updated['review_folder'] = folder_name
            updated['review_path'] = str(path)
            if folder_name.casefold() in MIXED_NAMES:
                updated['status'] = 'needs_reclustering'
                updated['temp_label'] = None
                mixed_indices.append(index)
            else:
                updated['status'] = 'retained'
                updated['temp_label'] = folder_name
            retained += 1
        records.append(updated)

    for name in found:
        if name not in by_name:
            unknown_files.extend(str(path) for _, path in found[name])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sync_output = stage3 / f'sync_{timestamp}'
    sync_output.mkdir(parents=False, exist_ok=False)
    sync_metadata = {'source_metadata': str(BASE_META), 'stage3_directory': str(stage3),
                     'created_at': timestamp, 'mixed_folder_names': sorted(MIXED_NAMES),
                     'summary': {'input_items': len(items), 'retained': retained,
                                 'dropped': dropped, 'mixed_for_reclustering': len(mixed_indices),
                                 'unknown_files': len(unknown_files)},
                     'unknown_files': unknown_files, 'items': records}
    (sync_output / 'synced_metadata.json').write_text(json.dumps(sync_metadata, indent=2), encoding='utf-8')

    recluster_output = None
    if mixed_indices:
        actual_k = min(args.mixed_k, len(mixed_indices))
        labels = KMeans(n_clusters=actual_k, n_init=20, random_state=42).fit_predict(
            embeddings[mixed_indices])
        recluster_output = sync_output / 'mixed_recluster'
        recluster_output.mkdir()
        grouped = defaultdict(list)
        for source_index, label in zip(mixed_indices, labels):
            grouped[int(label)].append(source_index)
        for label, source_indices in sorted(grouped.items()):
            cluster_dir = recluster_output / f'cluster_{label}'
            cluster_dir.mkdir()
            for source_index in source_indices:
                item = records[source_index]
                shutil.copy2(Path(item['review_path']), cluster_dir / item['primary_crop'])
        recluster_metadata = []
        for source_index, label in zip(mixed_indices, labels):
            item = dict(records[source_index])
            item['recluster_cluster'] = int(label)
            recluster_metadata.append(item)
        (recluster_output / 'metadata.json').write_text(
            json.dumps({'input_count': len(mixed_indices), 'k': actual_k,
                        'items': recluster_metadata}, indent=2), encoding='utf-8')

    print(json.dumps({'sync_metadata': str(sync_output / 'synced_metadata.json'),
                      'retained': retained, 'dropped': dropped,
                      'mixed_for_reclustering': len(mixed_indices),
                      'mixed_recluster_output': str(recluster_output) if recluster_output else None,
                      'unknown_files': len(unknown_files)}, indent=2))


if __name__ == '__main__':
    main()
