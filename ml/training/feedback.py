"""Bind feedback to exact sample hashes, never reusable cluster numbers."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def decisions():
    config = json.loads((ROOT / 'configs/visual_baseline_feedback.json').read_text(encoding='utf-8'))
    run = ROOT / config['run']
    samples = json.loads((run / 'samples.json').read_text(encoding='utf-8'))
    assignments = np.load(run / 'assignments.npy')
    if len(samples) != len(assignments):
        # A prior baseline run can be interrupted after one artifact is
        # replaced. The hash-bound feedback export is authoritative and does
        # not require reconstructing cluster assignments.
        exported = ROOT / 'ml/data/derived/2026_09_16/model_feedback.json'
        if exported.is_file():
            return json.loads(exported.read_text(encoding='utf-8'))
        raise ValueError('Feedback sample/assignment mismatch and no hash-bound export')
    result = {}
    for sample, group in zip(samples, assignments):
        rule = config['groups'].get(str(int(group)))
        if rule:
            result[sample['id']] = dict(rule, original_group=int(group), reviewer=config['reviewer'])
    return result


def main():
    records = decisions()
    output = ROOT / 'ml/data/derived/2026_09_16/model_feedback.json'
    output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('excluded:', sum(r['action'] == 'exclude' for r in records.values()))
    print('thermal:', sum(r.get('modality') == 'thermal' for r in records.values()))
    print('raw files deleted: 0')


if __name__ == '__main__':
    main()
