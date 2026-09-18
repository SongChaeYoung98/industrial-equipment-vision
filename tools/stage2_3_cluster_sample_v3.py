"""Stage 2-3 v3 pilot: SAM foreground masking, grayscale DINOv2 patch pooling, PCA, K-means."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import AutoImageProcessor, AutoModel

from ml.training.visual_baseline import ROOT

META = ROOT / 'ml/data/processed/stage1_sample/metadata_v2.json'
OUT = ROOT / 'ml/data/processed/stage1_sample'
RUN = OUT / 'stage2_3_grayscale_dinov2_v3'
PRIMARY = ROOT / 'ml/data/processed/primary_v2'
MODEL_NAME = 'facebook/dinov2-small'


def masked_gray(record):
    source = Path(record['source_path'])
    image = np.asarray(Image.open(source).convert('RGB'))
    x, y, w, h = map(int, record['primary_box'])
    mask_path = ROOT / record['candidates'][0]['mask_path']
    mask = np.asarray(Image.open(mask_path).convert('L')) > 0
    crop = image[y:y + h, x:x + w].copy()
    crop_mask = mask[y:y + h, x:x + w]
    crop[~crop_mask] = 0
    gray = np.asarray(Image.fromarray(crop).convert('L'))
    return Image.fromarray(gray).convert('RGB')


def make_grid(rows):
    cards = []
    for row in rows:
        image = Image.open(PRIMARY / row['primary_crop']).convert('RGB')
        image = ImageOps.contain(image, (240, 180))
        card = Image.new('RGB', (280, 224), 'white')
        card.paste(image, ((280 - image.width) // 2, 4))
        draw = ImageDraw.Draw(card)
        label = f"C{row['cluster']} | {row['category']}"
        if row['known_no_equipment']:
            label = '[NO-EQUIPMENT] ' + label
        draw.text((4, 188), label, fill='black')
        draw.text((4, 203), f"score={row['stage1_final_score']:.3f}", fill='black')
        cards.append(card)
    columns = 5
    grid = Image.new('RGB', (columns * 280, ((len(cards) + columns - 1) // columns) * 224), '#dddddd')
    for index, card in enumerate(cards):
        grid.paste(card, ((index % columns) * 280, (index // columns) * 224))
    grid.save(OUT / 'cluster_grid_v3.jpg', quality=92)


def main():
    meta = json.loads(META.read_text(encoding='utf-8'))
    rows = []
    for record in meta['records']:
        if record.get('status') != 'ok':
            continue
        primary = next(c for c in record['candidates'] if c['role'] == 'primary')
        rows.append({'source_path': record['source_path'], 'primary_box': record['primary_box'],
                     'candidates': record['candidates'], 'primary_crop': record['primary_crop'],
                     'category': record['category'], 'known_no_equipment': record['known_no_equipment'],
                     'stage1_final_score': record['final_score'], 'mask_path': primary['mask_path']})
    if len(rows) != 98:
        raise RuntimeError(f'Expected 98 Stage 1 v2 crops, got {len(rows)}')
    RUN.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = [masked_gray(row) for row in rows[start:start + 16]]
            inputs = processor(images=batch, return_tensors='pt').to(device)
            output = model(**inputs).last_hidden_state
            patch_avg = output[:, 1:, :].mean(dim=1)
            patch_avg = torch.nn.functional.normalize(patch_avg, dim=1)
            vectors.append(patch_avg.cpu().numpy())
    embeddings = np.concatenate(vectors).astype('float32')
    np.save(RUN / 'patch_average_embeddings.npy', embeddings)
    pca = PCA(n_components=32, random_state=42)
    reduced = pca.fit_transform(embeddings).astype('float32')
    np.save(RUN / 'pca32_embeddings.npy', reduced)
    scores = {}
    labels_by_k = {}
    for k in range(5, 9):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(reduced)
        labels_by_k[k] = labels
        scores[str(k)] = float(silhouette_score(reduced, labels, metric='euclidean'))
    selected_k = int(max(scores, key=scores.get))
    for row, label in zip(rows, labels_by_k[selected_k]):
        row['cluster'] = int(label)
    summary = {}
    for row in rows:
        key = str(row['cluster'])
        summary.setdefault(key, {'count': 0, 'categories': {}, 'known_no_equipment_count': 0})
        summary[key]['count'] += 1
        summary[key]['categories'][row['category']] = summary[key]['categories'].get(row['category'], 0) + 1
        summary[key]['known_no_equipment_count'] += int(row['known_no_equipment'])
    result = {'stage': 'stage2_3_pilot_v3', 'input_count': len(rows), 'requested_input_count': 98,
              'model': MODEL_NAME, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else device,
              'pipeline': ['SAM foreground mask', 'black background', 'grayscale', 'DINOv2 patch token average',
                           'PCA 32 dimensions', 'KMeans K=5..8'],
              'k_silhouette_scores': scores, 'selected_k': selected_k,
              'pca': {'components': 32, 'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum())},
              'clusters': summary, 'items': rows}
    (OUT / 'metadata_v3.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    make_grid(rows)
    print(json.dumps({'input_count': len(rows), 'selected_k': selected_k, 'silhouette': scores,
                      'pca_variance': float(pca.explained_variance_ratio_.sum()),
                      'metadata': str(OUT / 'metadata_v3.json'),
                      'grid': str(OUT / 'cluster_grid_v3.jpg')}, indent=2))


if __name__ == '__main__':
    main()
