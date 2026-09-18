"""Apply human cluster feedback and recluster the retained Stage 2-3 pilot data.

Example:
  python tools/apply_stage_feedback.py --feedback configs/stage_feedback.json

Feedback is cluster-level and intentionally does not delete source files:
{
  "cluster_1": {"action": "drop", "reason": "hands/office"},
  "cluster_7": {"action": "label", "label": "Control_Panel"}
}
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from ml.training.visual_baseline import ROOT

BASE = ROOT / 'ml/data/processed/stage1_sample'
META = BASE / 'metadata_v3.json'
EMBEDDINGS = BASE / 'stage2_3_grayscale_dinov2_v3/pca32_embeddings.npy'
OUT_ROOT = ROOT / 'ml/data/processed/stage3_clusters'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', required=True, type=Path)
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()
    feedback = json.loads(args.feedback.read_text(encoding='utf-8'))
    base = json.loads(META.read_text(encoding='utf-8'))
    embeddings = np.load(EMBEDDINGS)
    items = base['items']
    if len(items) != len(embeddings):
        raise RuntimeError('Metadata and embedding row counts differ')

    drop_clusters = set()
    labels = {}
    for cluster_name, decision in feedback.items():
        cluster_id = int(cluster_name.split('_')[-1])
        action = decision.get('action', '').lower()
        if action == 'drop':
            drop_clusters.add(cluster_id)
        elif action == 'label':
            if not decision.get('label'):
                raise ValueError(f'Missing label for {cluster_name}')
            labels[cluster_id] = decision['label']
        elif action in ('keep', ''):
            continue
        else:
            raise ValueError(f'Unsupported action for {cluster_name}: {action}')

    retained_indices = [i for i, item in enumerate(items) if item['cluster'] not in drop_clusters]
    if len(retained_indices) < 2:
        raise RuntimeError('Feedback dropped too much data to recluster')
    actual_k = min(args.k, len(retained_indices))
    reclustered = KMeans(n_clusters=actual_k, n_init=20, random_state=42).fit_predict(
        embeddings[retained_indices]
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = OUT_ROOT / f'feedback_round_{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    result_items = []
    for new_cluster, source_index in zip(reclustered, retained_indices):
        item = dict(items[source_index])
        old_cluster = item['cluster']
        item['previous_cluster'] = old_cluster
        item['cluster'] = int(new_cluster)
        if old_cluster in labels:
            item['human_label'] = labels[old_cluster]
        result_items.append(item)
    result = {'feedback': feedback, 'dropped_original_clusters': sorted(drop_clusters),
              'input_count': len(items), 'retained_count': len(result_items),
              'recluster_k': actual_k, 'items': result_items}
    (output / 'metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'input_count': len(items),
                      'retained_count': len(result_items), 'recluster_k': actual_k}, indent=2))


if __name__ == '__main__':
    main()
