"""Configurable deep prototype training; KMeans is deliberately absent.

The learnable prototype head and backbone are optimized together. This is an
offline experiment and never marks its checkpoint as the final model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from ml.training.visual_baseline import ROOT, candidates, preprocess, sheet


class EquipmentViews(Dataset):
    def __init__(self, records):
        self.records = records
        self.view = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.35, .35, .55, .12),
            transforms.RandomGrayscale(.45),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(ROOT / self.records[index]['path']) as image:
            image = image.convert('RGB')
        return self.view(image), self.view(image), index


def make_backbone(name):
    if name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        width = 512
    elif name == 'resnet50':
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        width = 2048
    else:
        raise ValueError('backbone must be resnet18 or resnet50')
    net.fc = nn.Identity()
    return net, width


class DeepPrototypeModel(nn.Module):
    def __init__(self, backbone_name, embedding=256, prototypes=64):
        super().__init__()
        self.backbone, width = make_backbone(backbone_name)
        self.projector = nn.Sequential(nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2),
                                       nn.GELU(), nn.Linear(width // 2, embedding))
        self.prototypes = nn.utils.weight_norm(nn.Linear(embedding, prototypes, bias=False))

    def forward(self, x):
        features = self.backbone(x)
        embeddings = nn.functional.normalize(self.projector(features), dim=1)
        logits = self.prototypes(embeddings)
        return embeddings, logits


@torch.no_grad()
def sinkhorn(logits, iterations=3, epsilon=.07):
    q = torch.exp((logits / epsilon).float()).T
    q /= q.sum()
    for _ in range(iterations):
        q /= q.sum(dim=1, keepdim=True)
        q /= q.shape[0]
        q /= q.sum(dim=0, keepdim=True)
        q /= q.shape[1]
    return (q * q.shape[1]).T


def swav_loss(logits_a, logits_b, temperature=.1):
    assignments_a = sinkhorn(logits_a.detach())
    assignments_b = sinkhorn(logits_b.detach())
    return (-assignments_a * nn.functional.log_softmax(logits_b / temperature, dim=1)).sum(1).mean() / 2 \
        + (-assignments_b * nn.functional.log_softmax(logits_a / temperature, dim=1)).sum(1).mean() / 2


@torch.no_grad()
def encode(model, records, device):
    model.eval()
    values = []
    for start in range(0, len(records), 64):
        images = []
        for record in records[start:start + 64]:
            with Image.open(ROOT / record['path']) as image:
                images.append(preprocess(image.convert('RGB')))
        values.append(model(torch.stack(images).to(device, non_blocking=True))[0].cpu().numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'ml/data/runs/deep_prototype_round_01')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--prototypes', type=int, default=64)
    parser.add_argument('--embedding', type=int, default=256)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true',
                        help='Resume the latest checkpoint in --run')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; CPU fallback is disabled')
    if args.run.exists() and not args.resume:
        raise SystemExit(f'Run exists; use a new run path: {args.run}')
    if args.epochs < 1 or args.batch_size < 2 or args.grad_accum < 1:
        raise ValueError('epochs, batch-size and grad-accum must be positive')
    seed = 20260917
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    records = candidates()
    rng = np.random.default_rng(seed)
    records = [records[i] for i in rng.permutation(len(records))]
    split = max(1, int(len(records) * .1))
    validation, training = records[:split], records[split:]
    args.run.mkdir(parents=True, exist_ok=True)
    model = DeepPrototypeModel(args.backbone, args.embedding, args.prototypes).to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    loader = DataLoader(EquipmentViews(training), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=3 if args.workers else None)
    settings = dict(status='training', final_model=False, method='learnable SwAV-style prototypes',
                    backbone=args.backbone, embedding=args.embedding, prototypes=args.prototypes,
                    epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                    workers=args.workers, lr=args.lr, temperature=.1, seed=seed,
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    torch_version=torch.__version__, training_samples=len(training),
                    validation_samples=len(validation), kmeans_used=False)
    history_path = args.run / 'history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if args.resume and history_path.exists() else []
    start_epoch = 1
    if args.resume:
        checkpoints = sorted(args.run.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            raise SystemExit(f'No checkpoint found to resume in {args.run}')
        checkpoint = torch.load(checkpoints[-1], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch']) + 1
        print(f'resuming from epoch {start_epoch} using {checkpoints[-1].name}', flush=True)
    settings['status'] = 'training'
    (args.run / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    started = time.monotonic(); history = []
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.
        epoch_started = time.monotonic()
        for step, (a, b, _) in enumerate(loader, 1):
            a = a.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            b = b.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                _, logits_a = model(a)
                _, logits_b = model(b)
                loss = swav_loss(logits_a, logits_b) / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += loss.item() * args.grad_accum
            if step % 50 == 0:
                rate = step * args.batch_size / (time.monotonic() - epoch_started)
                print(f'epoch={epoch}/{args.epochs} step={step}/{len(loader)} loss={total/step:.4f} samples_per_second={rate:.1f}', flush=True)
        row = dict(epoch=epoch, loss=total / len(loader), epoch_seconds=time.monotonic()-epoch_started)
        history.append(row)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'final_model': False},
                   args.run / f'checkpoint_epoch_{epoch:02d}.pt')
        (args.run / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    vectors = encode(model, records, device)
    # Assignment is generated by the learned prototype weights, never KMeans.
    with torch.inference_mode():
        assignments = model.prototypes(torch.from_numpy(vectors).to(device)).argmax(1).cpu().numpy()
    np.save(args.run / 'features.npy', vectors); np.save(args.run / 'assignments.npy', assignments)
    (args.run / 'samples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    for group in range(args.prototypes):
        members = np.flatnonzero(assignments == group)
        if len(members):
            selected = members[:15]
            sheet([records[i] for i in selected], args.run / f'prototype_{group:03d}.jpg',
                  [f'prototype {group:03d} #{i}' for i in selected])
    settings.update(status='awaiting_user_review', final_model=False,
                    training_seconds=time.monotonic()-started,
                    gradient_steps=args.epochs * len(loader) // args.grad_accum,
                    prototype_sizes=dict(Counter(map(str, assignments))))
    (args.run / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == '__main__':
    main()
