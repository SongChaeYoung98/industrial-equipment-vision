"""Stage 2-3 pilot: grayscale DINOv2 embeddings and K selection on Stage 1 v2 crops."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoImageProcessor, AutoModel

from ml.training.visual_baseline import ROOT

META = ROOT / 'ml/data/processed/stage1_sample/metadata_v2.json'
OUT = ROOT / 'ml/data/processed/stage1_sample'
RUN = OUT / 'stage2_3_grayscale_dinov2'
MODEL_NAME = 'facebook/dinov2-small'


def make_grid(rows, selected_k):
    cards = []
    for row in rows:
        image = Image.open(ROOT / 'ml/data/processed/primary_v2' / row['primary_crop']).convert('RGB')
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
    grid.save(OUT / 'cluster_grid.jpg', quality=92)


def main():
    meta = json.loads(META.read_text(encoding='utf-8'))
    rows = []
    for record in meta['records']:
        if record.get('status') != 'ok':
            continue
        rows.append({
            'source_path': record['source_path'],
            'primary_crop': record['primary_crop'],
            'category': record['category'],
            'known_no_equipment': record['known_no_equipment'],
            'stage1_final_score': record['final_score'],
        })
    if len(rows) < 8:
        raise RuntimeError('Too few Stage 1 crops')
    RUN.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = []
            for row in rows[start:start + 16]:
                image = Image.open(ROOT / 'ml/data/processed/primary_v2' / row['primary_crop']).convert('L').convert('RGB')
                batch.append(image)
            inputs = processor(images=batch, return_tensors='pt').to(device)
            output = model(**inputs).last_hidden_state[:, 0]
            output = torch.nn.functional.normalize(output, dim=1)
            vectors.append(output.cpu().numpy())
    matrix = np.concatenate(vectors).astype('float32')
    np.save(RUN / 'embeddings.npy', matrix)
    scores = {}
    for k in range(5, 9):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(matrix)
        scores[str(k)] = float(silhouette_score(matrix, labels, metric='cosine'))
    selected_k = int(max(scores, key=scores.get))
    labels = KMeans(n_clusters=selected_k, n_init=20, random_state=42).fit_predict(matrix)
    for row, label in zip(rows, labels):
        row['cluster'] = int(label)
    summary = {}
    for row in rows:
        key = str(row['cluster'])
        summary.setdefault(key, {'count': 0, 'categories': {}, 'known_no_equipment_count': 0})
        summary[key]['count'] += 1
        summary[key]['categories'][row['category']] = summary[key]['categories'].get(row['category'], 0) + 1
        summary[key]['known_no_equipment_count'] += int(row['known_no_equipment'])
    result = {'stage': 'stage2_3_pilot', 'input_count': len(rows), 'requested_input_count': 100,
              'model': MODEL_NAME, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else device,
              'input_preprocessing': 'grayscale converted to 3-channel before DINOv2',
              'k_silhouette_scores': scores, 'selected_k': selected_k, 'clusters': summary,
              'items': rows}
    (OUT / 'stage2_3_metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    make_grid(rows, selected_k)
    print(json.dumps({'input_count': len(rows), 'selected_k': selected_k,
                      'silhouette': scores, 'metadata': str(OUT / 'stage2_3_metadata.json'),
                      'grid': str(OUT / 'cluster_grid.jpg')}, indent=2))


if __name__ == '__main__':
    main()
