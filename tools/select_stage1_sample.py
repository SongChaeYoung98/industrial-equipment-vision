"""Create the approved 5x20 Stage 1 SAM sample manifest only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.training.visual_baseline import ROOT, candidates


OUT = ROOT / 'ml/data/processed/stage1_sample'
BASELINE = ROOT / 'ml/data/runs/visual_baseline_v1'


def main():
    records = candidates()
    baseline = json.loads((BASELINE / 'samples.json').read_text(encoding='utf-8'))
    assignments = np.load(BASELINE / 'assignments.npy')
    by_group = {}
    for record, group in zip(baseline, assignments):
        by_group.setdefault(int(group), []).append(record)
    by_source = {}
    for record in records:
        by_source.setdefault(record['source'], []).append(record)

    chosen = []
    used = set()

    def take(group, count, category, known_no_equipment=False, subcategory=None):
        pool = [r for r in by_group[group] if r['id'] not in used]
        if len(pool) < count:
            raise RuntimeError(f'group {group} has only {len(pool)} available for {category}')
        for record in pool[:count]:
            used.add(record['id'])
            chosen.append(dict(record, category=category,
                               subcategory=subcategory or category,
                               known_no_equipment=known_no_equipment,
                               selection_basis=f'baseline_group_{group}'))

    # These source groups were manually inspected before this manifest was made.
    take(1, 20, 'close_equipment_pipe', subcategory='close_single_equipment')
    take(15, 20, 'wide_field', subcategory='wide_multi_equipment_proxy')
    take(8, 20, 'control_panel', subcategory='control_or_distribution_panel')
    take(17, 10, 'no_equipment', True, 'human_hand')
    take(19, 10, 'no_equipment', True, 'office_scene')

    thermal_seed = [r for r in by_group[20] if r['id'] not in used]
    for record in thermal_seed:
        if len([r for r in chosen if r['category'] == 'thermal']) >= 20:
            break
        used.add(record['id'])
        chosen.append(dict(record, category='thermal', subcategory='thermal_image',
                           known_no_equipment=False, selection_basis='baseline_group_20'))
    # Fill remaining thermal slots with other frames from the same known thermal videos.
    thermal_sources = {r['source'] for r in thermal_seed}
    for source in sorted(thermal_sources):
        for record in by_source.get(source, []):
            if len([r for r in chosen if r['category'] == 'thermal']) >= 20:
                break
            if record['id'] in used:
                continue
            used.add(record['id'])
            chosen.append(dict(record, category='thermal', subcategory='thermal_image',
                               known_no_equipment=False, selection_basis='same_known_thermal_video'))
        if len([r for r in chosen if r['category'] == 'thermal']) >= 20:
            break
    if len([r for r in chosen if r['category'] == 'thermal']) < 20:
        raise RuntimeError('Could not construct 20 thermal samples')

    chosen = chosen[:100]
    counts = {}
    for record in chosen:
        counts[record['category']] = counts.get(record['category'], 0) + 1
    if counts != {'close_equipment_pipe': 20, 'wide_field': 20, 'control_panel': 20,
                  'no_equipment': 20, 'thermal': 20}:
        raise RuntimeError(f'wrong sample composition: {counts}')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sample_manifest.json').write_text(json.dumps({
        'approved_by_user': True,
        'sample_size': len(chosen),
        'categories': counts,
        'selection_method': 'manual semantic baseline-group selection; thermal filled from same known thermal videos',
        'records': chosen,
    }, indent=2), encoding='utf-8')
    print(json.dumps(counts, indent=2))
    for category in sorted(counts):
        print(category, [r['source'] for r in chosen if r['category'] == category][:3])


if __name__ == '__main__':
    main()
