"""Local pretrained feature + unsupervised clustering experiment; no web app.

Run: python -m ml.training.visual_baseline --limit 1024 --clusters 24
Predict: python -m ml.training.visual_baseline --query image.png [--box x y w h]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ml.training.feedback import decisions

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'ml/data/raw/2026_09_16'
DERIVED = ROOT / 'ml/data/derived/2026_09_16'
DEFAULT_RUN = ROOT / 'ml/data/runs/visual_baseline_v1'
VERSION = 'resnet18-imagenet1k-v1-rgb-letterbox224-imagenetnorm-l2'


def candidates():
    unique = {}
    def add(sha, path, source, kind):
        record = unique.setdefault(sha, dict(id=sha, path=path.relative_to(ROOT).as_posix(),
                                           source=source, kind=kind, uploads=[]))
        upload = Path(source).name.split('__')[0]
        if upload not in record['uploads']:
            record['uploads'].append(upload)
    with sqlite3.connect(f'{(DERIVED / "catalog.sqlite").as_uri()}?mode=ro', uri=True) as db:
        for path, sha, metadata in db.execute('SELECT path,sha256,metadata FROM media WHERE kind="image" ORDER BY path'):
            if json.loads(metadata)['status'] == 'metadata_ok':
                add(sha, RAW / path, path, 'image')
    for frame in json.loads((DERIVED / 'frames/manifest.json').read_text(encoding='utf-8')):
        if frame['status'] == 'extracted':
            add(frame['sha256'], DERIVED / 'frames' / frame['path'], frame['source'], 'frame')
    feedback = decisions()
    # Known thermal examples are preserved for a separate modality experiment.
    return [record for key, record in unique.items() if key not in feedback]


def preprocess(image):
    image = ImageOps.pad(image.convert('RGB'), (224, 224), method=Image.Resampling.BILINEAR,
                         color=(124, 116, 104))
    tensor = transforms.functional.to_tensor(image)
    return transforms.functional.normalize(tensor, [.485, .456, .406], [.229, .224, .225])


def encoder(pretrained):
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / 'models/torch-cache'))
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = torch.nn.Identity()
    return net.eval()


def features(net, tensors):
    with torch.inference_mode():
        vector = net(torch.stack(tensors))
        return torch.nn.functional.normalize(vector, dim=1).numpy()


def sheet(records, path, labels):
    canvas = Image.new('RGB', (5 * 192, ((len(records) + 4) // 5) * 164), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (record, label) in enumerate(zip(records, labels)):
        with Image.open(ROOT / record['path']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (190, 140))
        x, y = (i % 5) * 192, (i // 5) * 164
        canvas.paste(thumb, (x, y))
        draw.text((x + 3, y + 142), label, fill='black')
    canvas.save(path)


def query(args):
    config = json.loads((args.run / 'experiment.json').read_text(encoding='utf-8'))
    if config['feature_version'] != VERSION:
        raise ValueError('Incompatible preprocessing version')
    net = encoder(False)
    net.load_state_dict(torch.load(args.run / 'encoder.pt', map_location='cpu', weights_only=True))
    with Image.open(args.query) as image:
        if args.box:
            x, y, w, h = args.box
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > image.width or y+h > image.height:
                raise ValueError('Invalid crop')
            image = image.crop((x, y, x+w, y+h))
        embedding = features(net, [preprocess(image)])[0]
    stored = np.load(args.run / 'features.npy')
    centers = np.load(args.run / 'centers.npy')
    records = json.loads((args.run / 'samples.json').read_text(encoding='utf-8'))
    feedback = decisions()
    scores = stored @ embedding
    order = np.argsort(-scores)
    query_sha = hashlib.sha256(args.query.read_bytes()).hexdigest()
    matches = [int(i) for i in order if records[i]['id'] != query_sha
               and records[i]['id'] not in feedback][:10]
    group = int(np.argmin(((centers - embedding) ** 2).sum(axis=1)))
    result = dict(provisional_group=f'group_{group:03d}', equipment_name=None,
                  warning='Unverified visual group; cosine similarity is not class probability.',
                  neighbors=[dict(path=records[i]['path'], cosine=float(scores[i])) for i in matches])
    if args.run == DEFAULT_RUN.resolve() and group in (17, 19, 20):
        result['provisional_group'] = None
        result['warning'] = 'Nearest original group was rejected or is a modality, not an equipment class.'
    if query_sha in feedback:
        result['review_decision'] = feedback[query_sha]
        result['provisional_group'] = None
        result['neighbors'] = []
    print(json.dumps(result, indent=2))


def train(args):
    if args.run.exists():
        raise ValueError('Run folder exists; choose a new --run to preserve prior results')
    records = candidates()
    rng = np.random.default_rng(20260917)
    records = [records[i] for i in rng.permutation(len(records))[:args.limit]]
    if not 2 <= args.clusters < len(records):
        raise ValueError('Require 2 <= clusters < sample count')
    args.run.mkdir(parents=True)
    net = encoder(True)
    embeddings = []
    for start in range(0, len(records), 32):
        tensors = []
        for record in records[start:start+32]:
            with Image.open(ROOT / record['path']) as im:
                tensors.append(preprocess(im))
        embeddings.append(features(net, tensors))
        print(f'embedded {min(start+32,len(records))}/{len(records)}', flush=True)
    vectors = np.concatenate(embeddings)
    model = KMeans(n_clusters=args.clusters, random_state=20260917, n_init=10).fit(vectors)
    np.save(args.run / 'features.npy', vectors)
    np.save(args.run / 'centers.npy', model.cluster_centers_)
    np.save(args.run / 'assignments.npy', model.labels_)
    torch.save(net.state_dict(), args.run / 'encoder.pt')
    (args.run / 'samples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    report = dict(feature_version=VERSION, seed=20260917, samples=len(records),
                  source_counts=dict(Counter(r['kind'] for r in records)), clusters=args.clusters,
                  cluster_sizes=dict(Counter(map(str,model.labels_))),
                  silhouette=float(silhouette_score(vectors, model.labels_, sample_size=min(1000,len(records)), random_state=42)),
                  supervised_training=False, encoder_finetuned=False, equipment_accuracy=None,
                  note='Unsupervised training-set geometry only; no validated equipment labels. Whole-image baseline; multi-object localization not implemented.',
                  torch_version=torch.__version__)
    (args.run / 'experiment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for group in range(args.clusters):
        indices = np.flatnonzero(model.labels_ == group)
        indices = indices[np.argsort(((vectors[indices]-model.cluster_centers_[group])**2).sum(axis=1))][:15]
        sheet([records[i] for i in indices], args.run / f'cluster_{group:03d}.jpg',
              [f'group {group:03d} / sample {i}' for i in indices])
    neighbors = []
    cards, captions = [], []
    for i in range(min(8,len(records))):
        score = vectors @ vectors[i]
        allowed = [int(j) for j in np.argsort(-score) if j != i and not set(records[i]['uploads']).intersection(records[j]['uploads'])][:4]
        neighbors.append(dict(query=i, matches=[dict(index=j,cosine=float(score[j])) for j in allowed]))
        cards.extend([records[i]]+[records[j] for j in allowed])
        captions.extend([f'QUERY {i}']+[f'{j} cosine={score[j]:.3f}' for j in allowed])
    sheet(cards, args.run / 'neighbors.jpg', captions)
    (args.run / 'neighbors.json').write_text(json.dumps(neighbors, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=1024)
    parser.add_argument('--clusters', type=int, default=24)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--query', type=Path)
    parser.add_argument('--box', type=int, nargs=4)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.run.is_relative_to(RAW.resolve()):
        raise ValueError('Run artifacts must be outside raw data')
    if args.query:
        query(args)
    else:
        if args.limit < 3:
            raise ValueError('limit must be at least 3')
        train(args)


if __name__ == '__main__':
    main()
