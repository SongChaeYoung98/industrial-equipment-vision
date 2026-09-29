"""Build leakage-aware train/validation/test manifests from saved embeddings.

The split unit is the capture/upload group (the filename prefix before the
timestamp). All representative frames from one video stay in one split.
Exact catalog hashes are recorded when available for duplicate auditing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

FULL = ROOT / 'ml/data/processed/full_pipeline'
STAGE6 = ROOT / 'ml/data/processed/stage6_auto_clusters_v2/metadata.json'
CATALOG = ROOT / 'ml/data/derived/2026_09_16/catalog.sqlite'
OUT = ROOT / 'ml/data/processed/dataset_split_v1'


def source_group(source: str) -> str:
    name = Path(source.replace('\\', '/')).stem
    if '__frame_' in name:
        return name.split('__frame_', 1)[0]
    return name.split('__', 1)[0]


def split_for(group: str) -> str:
    value = int(hashlib.sha256(group.encode('utf-8')).hexdigest()[:8], 16) % 1000
    return 'train' if value < 800 else 'validation' if value < 900 else 'test'


def union_find(groups, duplicate_groups):
    parent = {group: group for group in groups}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for linked in duplicate_groups.values():
        for other in linked[1:]:
            union(linked[0], other)
    return {group: find(group) for group in groups}


def catalog_hashes():
    if not CATALOG.is_file():
        return {}
    with sqlite3.connect(CATALOG) as db:
        return {path: digest for path, digest in db.execute('SELECT path, sha256 FROM media') if digest}


def main():
    final = json.loads((FULL / 'final_records.json').read_text(encoding='utf-8'))
    rows = json.loads((FULL / 'embedding_rows.json').read_text(encoding='utf-8'))
    stage6 = json.loads(STAGE6.read_text(encoding='utf-8')) if STAGE6.is_file() else {'items': []}
    stage6_by_source = {item['source']: item for item in stage6['items']}
    final_by_source = {item['source']: item for item in final['records']}
    hashes = catalog_hashes()
    records = []
    for source in rows:
        if source not in stage6_by_source or source not in final_by_source:
            continue
        item = final_by_source[source]
        group = source_group(source)
        relative = None
        try:
            relative = Path(source).relative_to(ROOT / 'ml/data/raw/2026_09_16').as_posix()
        except ValueError:
            pass
        records.append({
            'source': source,
            'source_name': item.get('source_name'),
            'kind': item.get('kind'),
            'primary_crop': item.get('primary_crop'),
            'label': stage6_by_source[source].get('auto_label', 'unknown'),
            'similarity': stage6_by_source[source].get('anchor_similarity'),
            'capture_group': group,
            'exact_hash': hashes.get(relative),
            'split': split_for(group),
        })
    if not records:
        raise RuntimeError('No records available for split')
    duplicate_groups = defaultdict(list)
    for record in records:
        if record['exact_hash']:
            duplicate_groups[record['exact_hash']].append(record['capture_group'])
    components = union_find({r['capture_group'] for r in records}, duplicate_groups)
    for record in records:
        record['duplicate_component'] = components[record['capture_group']]
        record['split'] = split_for(record['duplicate_component'])
    groups = defaultdict(set)
    for record in records:
        groups[record['duplicate_component']].add(record['split'])
    if any(len(splits) != 1 for splits in groups.values()):
        raise AssertionError('Capture group leaked across splits')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'manifest.jsonl').open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    duplicate_hashes = defaultdict(list)
    for record in records:
        if record['exact_hash']:
            duplicate_hashes[record['exact_hash']].append(record['source'])
    summary = {
        'record_count': len(records),
        'capture_group_count': len(groups),
        'split_counts': dict(Counter(record['split'] for record in records)),
        'label_counts': dict(Counter(record['label'] for record in records)),
        'groups_by_split': {split: sum(split in values for values in groups.values())
                            for split in ('train', 'validation', 'test')},
        'exact_duplicate_group_count': sum(len(paths) > 1 for paths in duplicate_hashes.values()),
        'exact_duplicate_record_count': sum(len(paths) for paths in duplicate_hashes.values() if len(paths) > 1),
        'split_policy': 'union capture groups linked by exact SHA-256, then sha256(component) modulo 1000: train<800, validation<900, test>=900',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
