"""Second pseudo-labeling pass using merged human-reviewed anchor vectors."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

FULL = ROOT / 'ml/data/processed/full_pipeline'
STAGE5 = ROOT / 'ml/data/processed/stage5_unknown_clusters'
FINAL = FULL / 'final_records.json'
EMBEDDINGS = FULL / 'embeddings.npy'
EMBEDDING_ROWS = FULL / 'embedding_rows.json'
PRIMARY = FULL / 'primary'
OUT = ROOT / 'ml/data/processed/stage6_auto_clusters_v2'
THRESHOLD = .55


def latest_sync():
    candidates = [p for p in STAGE5.iterdir() if p.is_dir() and (p / 'anchors_merged.json').exists()
                  and (p / 'metadata.json').exists()]
    if not candidates:
        raise FileNotFoundError('No Stage 5 synchronized feedback folder found')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    if OUT.exists():
        raise RuntimeError(f'Refusing to overwrite existing output: {OUT}')
    sync = latest_sync()
    merged = json.loads((sync / 'anchors_merged.json').read_text(encoding='utf-8'))
    stage5_meta = json.loads((sync / 'metadata.json').read_text(encoding='utf-8'))
    dropped = {item['source'] for item in stage5_meta['items'] if item.get('status') == 'dropped'}
    anchors = {label: np.asarray(vector, dtype='float32')
               for label, vector in merged['vectors'].items()}
    if not anchors:
        raise RuntimeError('Merged anchors contain no vectors')
    labels = sorted(anchors)
    anchor_matrix = normalize(np.stack([anchors[label] for label in labels]))
    rows = json.loads(EMBEDDING_ROWS.read_text(encoding='utf-8'))
    matrix = np.load(EMBEDDINGS)
    final = json.loads(FINAL.read_text(encoding='utf-8'))
    primary_by_source = {item['source']: item['primary_crop'] for item in final['records']
                         if item.get('status') == 'ok' and item.get('primary_crop')}
    if len(rows) != len(matrix):
        raise RuntimeError('Embedding rows and matrix differ')
    OUT.mkdir(parents=True)
    for label in labels + ['unknown']:
        (OUT / label).mkdir()
    results = []
    counts = Counter()
    for source, vector in zip(rows, matrix):
        if source in dropped:
            continue
        similarities = anchor_matrix @ vector
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        label = labels[best] if score >= THRESHOLD else 'unknown'
        crop_name = primary_by_source.get(source)
        if crop_name is None:
            label = 'unknown'
        if crop_name:
            shutil.copy2(PRIMARY / crop_name, OUT / label / crop_name)
        results.append({'source': source, 'primary_crop': crop_name, 'auto_label': label,
                        'anchor_similarity': score, 'dropped_by_stage5': False})
        counts[label] += 1
    metadata = {'stage': 'stage6_auto_labeling_v2', 'source_sync': str(sync.relative_to(ROOT)),
                'threshold': THRESHOLD, 'anchor_count': len(labels), 'anchors': labels,
                'input_embedding_count': len(rows), 'dropped_count': len(dropped),
                'classified_count': len(results), 'counts': dict(counts), 'items': results,
                'created_at': datetime.now().isoformat(timespec='seconds')}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'anchors': labels, 'dropped': len(dropped),
                      'classified': len(results), 'counts': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
