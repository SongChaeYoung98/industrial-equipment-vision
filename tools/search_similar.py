"""Search indexed equipment examples by cosine similarity.

This queries the already computed PCA-32 vectors; it does not claim physical
device identity and only searches assets present in the indexed dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

FULL = ROOT / 'ml/data/processed/full_pipeline'
STAGE6 = ROOT / 'ml/data/processed/stage6_auto_clusters_v2/metadata.json'


def main():
    parser = argparse.ArgumentParser()
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument('--source', help='Exact source path or filename from the indexed dataset')
    query.add_argument('--crop', help='Exact primary crop filename')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    rows = json.loads((FULL / 'embedding_rows.json').read_text(encoding='utf-8'))
    matrix = np.load(FULL / 'embeddings.npy').astype('float32')
    metadata = json.loads(STAGE6.read_text(encoding='utf-8'))
    by_source = {item['source']: item for item in metadata['items']}
    candidates = [source for source in rows if source in by_source]
    if args.source:
        matches = [source for source in candidates if source == args.source or Path(source).name == args.source]
    else:
        matches = [source for source in candidates if by_source[source].get('primary_crop') == args.crop]
    if len(matches) != 1:
        raise SystemExit(f'Expected one query asset, found {len(matches)}')
    query_index = rows.index(matches[0])
    scores = matrix @ matrix[query_index]
    ranked = np.argsort(-scores)
    results = []
    for index in ranked:
        source = rows[int(index)]
        if source == matches[0]:
            continue
        item = by_source[source]
        results.append({'source': source, 'source_name': item.get('source_name'),
                        'primary_crop': item.get('primary_crop'),
                        'label': item.get('auto_label', 'unknown'),
                        'similarity': float(scores[index])})
        if len(results) >= args.top_k:
            break
    print(json.dumps({'query': matches[0], 'top_k': args.top_k, 'results': results},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
