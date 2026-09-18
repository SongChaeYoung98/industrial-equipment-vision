"""Full local pipeline: SAM proposals -> masked grayscale DINOv2 -> anchors.

The source trees are read-only. Results and resumable checkpoints live under
ml/data/processed/full_pipeline. Run with ``--resume`` after interruption.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from ml.training.visual_baseline import ROOT

RAW = ROOT / 'ml/data/raw/2026_09_16/image'
FRAMES = ROOT / 'ml/data/derived/2026_09_16/frames'
GOLDEN = ROOT / 'ml/data/processed/stage3_clusters'
OUT = ROOT / 'ml/data/processed/full_pipeline'
MANIFEST = OUT / 'manifest.jsonl'
RECORDS = OUT / 'records.jsonl'
EMBEDDINGS = OUT / 'embeddings.npy'
ANCHORS = OUT / 'anchors.json'
AUTO = ROOT / 'ml/data/processed/stage4_auto_clusters'
CHECKPOINT = ROOT / 'models/sam2/sam2.1_hiera_small.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
DINOV2 = 'facebook/dinov2-small'


def clahe_rgb(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def center_score(box, width, height):
    x, y, w, h = box
    distance = ((x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2) ** .5
    maximum = (width ** 2 / 4 + height ** 2 / 4) ** .5
    return max(0.0, 1.0 - distance / maximum)


def remove_nested(candidates):
    areas = [int(c['segmentation'].sum()) for c in candidates]
    result = []
    for i, child in enumerate(candidates):
        child_area = max(areas[i], 1)
        nested = any(i != j and areas[j] > child_area * 1.10 and
                     np.logical_and(child['segmentation'], parent['segmentation']).sum() / child_area >= .98
                     for j, parent in enumerate(candidates))
        if not nested:
            result.append(child)
    return result


def build_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(RAW.glob('*.png')):
        records.append({'source': str(path), 'source_name': path.name, 'kind': 'image'})
    for path in sorted(FRAMES.glob('*')):
        if path.is_file():
            records.append({'source': str(path), 'source_name': path.name, 'kind': 'frame'})
    with MANIFEST.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    return records


def load_done():
    done = {}
    if RECORDS.exists():
        with RECORDS.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[row['source']] = row
    return done


def score_candidates(image, candidates):
    height, width = image.shape[:2]
    sharpness = []
    for c in candidates:
        x, y, w, h = map(int, c['bbox'])
        crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
        crop = crop[c['segmentation'][y:y + h, x:x + w]]
        sharpness.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 8 else 0.0)
    lo, hi = min(sharpness), max(sharpness)
    height, width = image.shape[:2]
    for c, raw in zip(candidates, sharpness):
        x, y, w, h = c['bbox']
        c['center_score'] = center_score(c['bbox'], width, height)
        c['size_score'] = min(c['box_area_ratio'] / .50, 1.0)
        c['sharpness_raw'] = raw
        c['sharpness_score'] = .5 if hi == lo else (raw - lo) / (hi - lo)
        c['final_score'] = .45 * c['center_score'] + .35 * c['size_score'] + .20 * c['sharpness_score']


def run_sam(records, device):
    model = build_sam2(CONFIG, str(CHECKPOINT), device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=16, points_per_batch=16, pred_iou_thresh=.80,
        stability_score_thresh=.90, box_nms_thresh=.70, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=2000,
        output_mode='binary_mask')
    done = load_done()
    primary_dir = OUT / 'primary'
    masks_dir = OUT / 'masks'
    primary_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    mode = 'a' if RECORDS.exists() else 'w'
    with RECORDS.open(mode, encoding='utf-8') as handle:
        for index, record in enumerate(records, 1):
            if record['source'] in done:
                continue
            path = Path(record['source'])
            try:
                image = np.asarray(Image.open(path).convert('RGB'))
                # Force full decode here; truncated PNGs can open successfully
                # and fail only when pixel data is read.
                image.shape
            except (OSError, ValueError) as error:
                if path.is_file():
                    path.unlink()
                row = dict(record, status='corrupt_deleted', primary_crop=None,
                           primary_box=None, error=str(error))
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                print(f'SKIP corrupt_deleted {index}/{len(records)} {path.name}', flush=True)
                continue
            try:
                height, width = image.shape[:2]
                generated = generator.generate(clahe_rgb(image))
                candidates = []
                for raw in generated:
                    mask = raw['segmentation'].astype(bool)
                    x, y, w, h = map(int, raw['bbox'])
                    mask_ratio = float(mask.mean())
                    box_ratio = float(w * h / (width * height))
                    if mask_ratio < .01:
                        continue
                    candidates.append({'segmentation': mask, 'bbox': [x, y, w, h],
                                       'mask_area_ratio': mask_ratio, 'box_area_ratio': box_ratio,
                                       'full_background_suspect': box_ratio >= .90})
                candidates = remove_nested(candidates)
            except Exception as error:
                row = dict(record, status='sam_failed', primary_crop=None,
                           primary_box=None, error=f'{type(error).__name__}: {error}')
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                torch.cuda.empty_cache()
                print(f'SKIP sam_failed {index}/{len(records)} {path.name}: {error}', flush=True)
                continue
            row = dict(record, status='no_candidate', primary_crop=None, primary_box=None)
            if candidates:
                try:
                    score_candidates(image, candidates)
                    primary = sorted(candidates, key=lambda c: c['final_score'], reverse=True)[0]
                    stem = f'{index:05d}__{path.stem}'
                    crop_name = f'{stem}__primary.png'
                    mask_name = f'{stem}__mask.png'
                    x, y, w, h = primary['bbox']
                    Image.fromarray(image[y:y + h, x:x + w]).save(primary_dir / crop_name)
                    Image.fromarray((primary['segmentation'] * 255).astype(np.uint8)).save(masks_dir / mask_name)
                    row.update(status='ok', primary_crop=crop_name, primary_box=primary['bbox'],
                               mask_path=str((masks_dir / mask_name).relative_to(ROOT)),
                               final_score=primary['final_score'])
                except Exception as error:
                    row.update(status='sam_failed', error=f'{type(error).__name__}: {error}')
                    torch.cuda.empty_cache()
                    print(f'SKIP sam_failed {index}/{len(records)} {path.name}: {error}', flush=True)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            if index % 25 == 0:
                print(f'SAM {index}/{len(records)} processed', flush=True)
    del model, generator
    torch.cuda.empty_cache()


def load_rows():
    with RECORDS.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dino(rows, device):
    processor = AutoImageProcessor.from_pretrained(DINOV2)
    model = AutoModel.from_pretrained(DINOV2).to(device).eval()
    vectors = []
    valid_rows = []
    batch_images, batch_rows = [], []
    def flush():
        if not batch_images:
            return
        with torch.inference_mode():
            inputs = processor(images=batch_images, return_tensors='pt').to(device)
            tokens = model(**inputs).last_hidden_state[:, 1:, :].mean(dim=1)
            vectors.append(torch.nn.functional.normalize(tokens, dim=1).cpu().numpy())
        valid_rows.extend(batch_rows)
        batch_images.clear()
        batch_rows.clear()
    for index, row in enumerate(rows, 1):
        if row['status'] != 'ok':
            continue
        image = np.asarray(Image.open(ROOT / row['mask_path']).convert('L'))
        crop = np.asarray(Image.open(OUT / 'primary' / row['primary_crop']).convert('RGB'))
        # The stored mask is full-frame; use its bbox to crop the same object.
        x, y, w, h = row['primary_box']
        mask = image[y:y + h, x:x + w] > 0
        crop[~mask] = 0
        gray = Image.fromarray(crop).convert('L').convert('RGB')
        batch_images.append(gray)
        batch_rows.append(row)
        if len(batch_images) == 32:
            flush()
        if index % 250 == 0:
            print(f'DINO {index}/{len(rows)} queued', flush=True)
    flush()
    dino_matrix = normalize(np.concatenate(vectors).astype('float32'))
    pca = PCA(n_components=32, random_state=42)
    matrix = normalize(pca.fit_transform(dino_matrix).astype('float32'))
    np.save(OUT / 'dino_embeddings.npy', dino_matrix)
    np.save(EMBEDDINGS, matrix)
    (OUT / 'pca.json').write_text(json.dumps({
        'components': 32,
        'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())
    }, indent=2), encoding='utf-8')
    with (OUT / 'embedding_rows.json').open('w', encoding='utf-8') as handle:
        json.dump([r['source'] for r in valid_rows], handle, indent=2)
    return valid_rows, matrix


def golden_labels():
    labels = {}
    for folder in GOLDEN.iterdir():
        if not folder.is_dir() or folder.name.startswith('sync_'):
            continue
        for file in folder.rglob('*'):
            if file.is_file():
                labels[file.name] = folder.name
    return labels


def classify(all_rows, valid_rows, matrix, threshold):
    by_source = {row['source']: (index, row) for index, row in enumerate(valid_rows)}
    # Golden crop -> original source mapping from the pilot metadata.
    pilot = json.loads((ROOT / 'ml/data/processed/stage1_sample/metadata_v3.json').read_text(encoding='utf-8'))
    golden = golden_labels()
    source_labels = {}
    for item in pilot.get('items', []):
        if item.get('primary_crop') in golden:
            source_labels[Path(item['source_path']).name] = golden[item['primary_crop']]
    anchors = {}
    for label in sorted(set(source_labels.values())):
        indices = [i for i, row in enumerate(valid_rows) if Path(row['source']).name in source_labels
                   and source_labels[Path(row['source']).name] == label]
        if indices:
            anchors[label] = normalize(matrix[indices].mean(axis=0, keepdims=True))[0]
    ANCHORS.write_text(json.dumps({'labels': sorted(anchors), 'threshold': threshold,
                                   'golden_count': len(source_labels)}, indent=2), encoding='utf-8')
    for path in AUTO.glob('*'):
        if path.is_dir():
            shutil.rmtree(path)
    AUTO.mkdir(parents=True, exist_ok=True)
    anchor_names = sorted(anchors)
    anchor_matrix = np.stack([anchors[name] for name in anchor_names]) if anchor_names else np.empty((0, matrix.shape[1]))
    counts = defaultdict(int)
    vector_by_source = {row['source']: matrix[index] for index, row in enumerate(rows)}
    for row in all_rows:
        if row['status'] != 'ok' or row['source'] not in vector_by_source or not anchor_names:
            label, score = 'unknown', None
        else:
            similarities = anchor_matrix @ vector_by_source[row['source']]
            best = int(np.argmax(similarities))
            score = float(similarities[best])
            label = anchor_names[best] if score >= threshold else 'unknown'
        row['auto_label'] = label
        row['anchor_similarity'] = score
        destination = AUTO / label
        destination.mkdir(parents=True, exist_ok=True)
        if row['status'] == 'ok':
            shutil.copy2(OUT / 'primary' / row['primary_crop'], destination / row['primary_crop'])
        counts[label] += 1
    (OUT / 'final_records.json').write_text(json.dumps({'threshold': threshold, 'anchors': anchor_names,
                                                        'records': rows}, indent=2), encoding='utf-8')
    print(json.dumps({'anchors': anchor_names, 'counts': dict(counts)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threshold', type=float, default=.55)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full pipeline')
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    records = build_manifest()
    device = 'cuda'
    start = time.time()
    run_sam(records, device)
    rows = load_rows()
    valid_rows, matrix = run_dino(rows, device)
    classify(rows, valid_rows, matrix, args.threshold)
    print(f'completed in {(time.time() - start) / 3600:.2f} hours', flush=True)


if __name__ == '__main__':
    main()
