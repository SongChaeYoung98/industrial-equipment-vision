"""Fast Active Learning round for the unknown predictions.

Uses saved PCA embeddings only; no SAM or DINO inference is performed here.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

FULL = ROOT / 'ml/data/processed/full_pipeline'
OUT = ROOT / 'ml/data/processed/stage5_unknown_clusters'
FINAL = FULL / 'final_records.json'
EMBEDDINGS = FULL / 'embeddings.npy'
EMBEDDING_ROWS = FULL / 'embedding_rows.json'
K = 20


def main():
    if OUT.exists():
        raise RuntimeError(f'Output already exists; refusing to overwrite reviewed folder: {OUT}')
    final = json.loads(FINAL.read_text(encoding='utf-8'))
    sources = json.loads(EMBEDDING_ROWS.read_text(encoding='utf-8'))
    matrix = np.load(EMBEDDINGS)
    if len(sources) != len(matrix):
        raise RuntimeError('embedding_rows.json and embeddings.npy differ')
    by_source = {source: index for index, source in enumerate(sources)}
    unknown = [row for row in final['records']
               if row.get('auto_label') == 'unknown' and row.get('source') in by_source]
    if len(unknown) < K:
        raise RuntimeError(f'Only {len(unknown)} unknown vectors available')
    unknown_matrix = np.stack([matrix[by_source[row['source']]] for row in unknown])
    labels = KMeans(n_clusters=K, n_init=20, random_state=42).fit_predict(unknown_matrix)
    OUT.mkdir(parents=True)
    for cluster in range(K):
        (OUT / f'cluster_{cluster}').mkdir()
    primary_dir = FULL / 'primary'
    metadata = []
    for row, cluster in zip(unknown, labels):
        destination = OUT / f'cluster_{int(cluster)}' / row['primary_crop']
        shutil.copy2(primary_dir / row['primary_crop'], destination)
        metadata.append({'source': row['source'], 'source_name': row.get('source_name'),
                         'primary_crop': row['primary_crop'], 'original_label': 'unknown',
                         'cluster': int(cluster), 'embedding_index': by_source[row['source']]})
    (OUT / 'metadata.json').write_text(json.dumps({
        'input_count': len(unknown), 'k': K, 'clusters': dict(Counter(labels.tolist())),
        'embedding_source': str(EMBEDDINGS.relative_to(ROOT)), 'items': metadata
    }, indent=2), encoding='utf-8')
    np.save(OUT / 'unknown_embeddings.npy', unknown_matrix)
    print(json.dumps({'input_count': len(unknown), 'k': K,
                      'cluster_counts': dict(Counter(labels.tolist())),
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
