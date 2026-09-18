"""Run approved Stage 1 SAM object proposals on exactly 100 manifest records."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from ml.training.visual_baseline import ROOT


MANIFEST = ROOT / 'ml/data/processed/stage1_sample/sample_manifest.json'
OUTPUT = ROOT / 'ml/data/processed'
PRIMARY = OUTPUT / 'primary_v2'
BACKGROUND = OUTPUT / 'background_v2'
MASKS = OUTPUT / 'stage1_sample/masks_v2'
METADATA = OUTPUT / 'stage1_sample/metadata_v2.json'
GRID = OUTPUT / 'stage1_sample/primary_grid_v2.jpg'
CHECKPOINT = ROOT / 'models/sam2/sam2.1_hiera_small.pt'
CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'


def source_path(record):
    if record['kind'] == 'image':
        return ROOT / 'ml/data/raw/2026_09_16' / record['source']
    return ROOT / 'ml/data/derived/2026_09_16/frames' / Path(record['path']).name


def center_score(box, width, height):
    x, y, w, h = box
    dx = x + w / 2 - width / 2
    dy = y + h / 2 - height / 2
    max_distance = ((width / 2) ** 2 + (height / 2) ** 2) ** .5
    return max(0.0, 1.0 - ((dx * dx + dy * dy) ** .5) / max_distance)


def sharpness_score(mask, image, boxes):
    raw = []
    for candidate in boxes:
        x, y, w, h = map(int, candidate['bbox'])
        crop = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
        crop = crop[candidate['segmentation'][y:y + h, x:x + w]]
        raw.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 8 else 0.0)
    low, high = min(raw), max(raw)
    for candidate, value in zip(boxes, raw):
        candidate['sharpness_raw'] = value
        candidate['sharpness_score'] = 0.5 if high == low else (value - low) / (high - low)


def make_grid(items):
    cards = []
    for item in items:
        image = Image.open(PRIMARY / item['primary_crop']).convert('RGB')
        image = ImageOps.contain(image, (240, 180))
        card = Image.new('RGB', (260, 220), 'white')
        card.paste(image, ((260 - image.width) // 2, 4))
        draw = ImageDraw.Draw(card)
        prefix = '[NO-EQUIPMENT] ' if item['known_no_equipment'] else ''
        draw.text((4, 188), prefix + item['category'], fill='black')
        draw.text((4, 202), f"score={item['final_score']:.3f}", fill='black')
        cards.append(card)
    columns = 5
    grid = Image.new('RGB', (columns * 260, ((len(cards) + columns - 1) // columns) * 220), '#dddddd')
    for index, card in enumerate(cards):
        grid.paste(card, ((index % columns) * 260, (index // columns) * 220))
    grid.save(GRID, quality=92)


def clahe_rgb(image):
    """Make edges less dependent on thermal/illumination contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def remove_nested_candidates(candidates):
    """Prefer a whole-object mask over a sub-part fully inside it."""
    areas = [int(c['segmentation'].sum()) for c in candidates]
    keep = []
    for i, child in enumerate(candidates):
        child_area = max(areas[i], 1)
        discard = False
        for j, parent in enumerate(candidates):
            if i == j or areas[j] <= child_area * 1.10:
                continue
            overlap = np.logical_and(child['segmentation'], parent['segmentation']).sum()
            if overlap / child_area >= 0.98:
                discard = True
                break
        if not discard:
            keep.append(child)
    return keep


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    records = manifest['records']
    if len(records) != 100 or sorted({r['category'] for r in records}) != sorted([
        'close_equipment_pipe', 'wide_field', 'control_panel', 'thermal', 'no_equipment'
    ]):
        raise RuntimeError('Manifest is not the approved 5x20 sample')
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError('SAM sample requires CUDA; refusing CPU fallback')
    for path in (PRIMARY, BACKGROUND, MASKS):
        path.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    model = build_sam2(CONFIG, str(CHECKPOINT), device=device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=16, points_per_batch=16, pred_iou_thresh=.80,
        stability_score_thresh=.90, box_nms_thresh=.70, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=2000,
        output_mode='binary_mask',
    )
    metadata = {'thresholds': {'min_mask_area_ratio': .01, 'full_background_ratio': .90},
                'weights': {'center': .45, 'size': .35, 'sharpness': .20},
                'postprocessing': {'clahe': {'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
                                   'min_mask_region_area': 2000,
                                   'nested_mask_containment': .98,
                                   'nested_mask_parent_area_margin': 1.10},
                'model': {'name': 'SAM 2.1 Hiera Small', 'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
                          'config': CONFIG, 'device': torch.cuda.get_device_name(0)},
                'sample_manifest': str(MANIFEST.relative_to(ROOT)), 'records': []}
    primary_items = []
    for index, record in enumerate(records, 1):
        path = source_path(record)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = np.asarray(Image.open(path).convert('RGB'))
        height, width = image.shape[:2]
        sam_image = clahe_rgb(image)
        generated = generator.generate(sam_image)
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
                               'full_background_suspect': box_ratio >= .90,
                               'predicted_iou': float(raw.get('predicted_iou', 0)),
                               'stability_score': float(raw.get('stability_score', 0))})
        candidates = remove_nested_candidates(candidates)
        if not candidates:
            metadata['records'].append(dict(record, status='no_candidate', source_path=str(path)))
            continue
        for candidate in candidates:
            candidate['center_score'] = center_score(candidate['bbox'], width, height)
            candidate['size_score'] = min(candidate['box_area_ratio'] / .50, 1.0)
        sharpness_score(None, image, candidates)
        for candidate in candidates:
            candidate['final_score'] = (.45 * candidate['center_score']
                                        + .35 * candidate['size_score']
                                        + .20 * candidate['sharpness_score'])
        candidates.sort(key=lambda item: item['final_score'], reverse=True)
        primary = candidates[0]
        source_stem = Path(record['source']).stem
        prefix = f'{index:03d}__{source_stem}'
        item = dict(record, status='ok', source_path=str(path), width=width, height=height,
                    candidate_count=len(candidates), primary_candidate_index=0,
                    primary_crop=f'{prefix}__primary_01.png', candidates=[])
        for candidate_index, candidate in enumerate(candidates, 1):
            x, y, w, h = candidate['bbox']
            role = 'primary' if candidate_index == 1 else 'background'
            crop_name = f'{prefix}__{role}_{candidate_index:02d}.png'
            mask_name = f'{prefix}__candidate_{candidate_index:02d}.png'
            Image.fromarray(image[y:y + h, x:x + w]).save((PRIMARY if role == 'primary' else BACKGROUND) / crop_name)
            Image.fromarray((candidate['segmentation'] * 255).astype(np.uint8)).save(MASKS / mask_name)
            serial = {key: value for key, value in candidate.items() if key != 'segmentation'}
            serial.update(role=role, crop_path=str((Path('ml/data/processed') / role / crop_name).as_posix()),
                          mask_path=str((Path('ml/data/processed/stage1_sample/masks') / mask_name).as_posix()))
            item['candidates'].append(serial)
        item['final_score'] = primary['final_score']
        item['primary_box'] = primary['bbox']
        item['primary_scores'] = {key: primary[key] for key in ('center_score', 'size_score', 'sharpness_score',
                                                                  'sharpness_raw', 'box_area_ratio', 'mask_area_ratio',
                                                                  'full_background_suspect')}
        metadata['records'].append(item)
        primary_items.append(item)
        if index % 10 == 0:
            print(f'{index}/100 processed', flush=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_grid(primary_items)
    summary = {category: sum(r.get('category') == category for r in metadata['records'])
               for category in sorted({r['category'] for r in records})}
    print(json.dumps({'summary': summary, 'records_with_candidates': len(primary_items),
                      'metadata': str(METADATA), 'grid': str(GRID)}, indent=2))


if __name__ == '__main__':
    main()
