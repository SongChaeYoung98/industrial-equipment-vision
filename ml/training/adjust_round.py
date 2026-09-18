"""One approval-gated GPU neural fine-tuning round. Never publishes a final model."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
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
from torchvision import transforms
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from ml.training.visual_baseline import ROOT, candidates, encoder, preprocess, sheet


class Pairs(Dataset):
    def __init__(self, records):
        self.records = records
        self.appearance = transforms.Compose([
            transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.7, hue=.15),
            transforms.RandomGrayscale(p=.5),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(ROOT / self.records[index]['path']) as image:
            image = image.convert('RGB')
        return preprocess(self.appearance(image)), preprocess(self.appearance(image))


def worker_init(worker_id):
    # Each loading process prepares one batch; avoid CPU thread oversubscription.
    torch.set_num_threads(1)


class SingleImages(Dataset):
    def __init__(self, records, gray=False):
        self.records, self.gray = records, gray

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(ROOT / self.records[index]['path']) as image:
            if self.gray:
                image = ImageOps.grayscale(image).convert('RGB')
            return preprocess(image)


def grouped_split(records):
    # Connect aliases across uploads, video frames and original capture filenames.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, r in enumerate(records):
        filename = Path(r['source']).name.split('__', 1)[-1]
        keys = ['upload:' + u for u in r['uploads']] + ['capture:' + filename]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    components = {}
    for i in range(len(records)):
        components.setdefault(find(i), []).append(i)
    validation = set()
    for indices in components.values():
        identity = min(records[i]['id'] for i in indices)
        if int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 10 == 0:
            validation.update(indices)
    training = [i for i in range(len(records)) if i not in validation]
    if len(training) < 32 or len(validation) < 32:
        raise ValueError('Not enough independent groups for training/validation')
    return training, sorted(validation), len(components)


def vicreg(a, b):
    invariance = (a-b).square().mean()
    variance = (torch.relu(1-(a.var(0)+1e-4).sqrt()).mean()
                + torch.relu(1-(b.var(0)+1e-4).sqrt()).mean()) / 2
    covariance = a.new_zeros(())
    for z in (a, b):
        z = z-z.mean(0)
        c = z.T @ z / (len(z)-1)
        covariance = covariance + (c.square().sum()-c.diagonal().square().sum()) / z.shape[1]
    return 25*invariance+25*variance+covariance, (invariance, variance, covariance)


def encode_records(net, records, gray=False):
    net.eval()
    vectors = []
    with torch.inference_mode():
        loader = DataLoader(SingleImages(records, gray), batch_size=64, num_workers=4,
                            pin_memory=True, worker_init_fn=worker_init)
        for images in loader:
            features = nn.functional.normalize(net(images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)), dim=1)
            vectors.append(features.cpu().numpy())
    return np.concatenate(vectors)


def diagnose(net, records):
    a, b = encode_records(net, records), encode_records(net, records, gray=True)
    similarity = a @ b.T
    unrelated = a @ a.T
    unrelated_mean = (unrelated.sum()-np.trace(unrelated)) / (len(a)*(len(a)-1))
    return dict(grayscale_same_image_cosine=float(np.trace(similarity)/len(a)),
                grayscale_self_retrieval_top1=float(np.mean(similarity.argmax(1)==np.arange(len(a)))),
                unrelated_cosine=float(unrelated_mean), embedding_std=float(a.std(0).mean()),
                note='Augmentation diagnostics only, not equipment recognition accuracy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'ml/data/runs/adjust_round_01')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--clusters', type=int, default=24)
    parser.add_argument('--resume', type=Path, help='Resume an epoch checkpoint from this run')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for this approved round; no silent CPU fallback')
    if args.epochs < 1 or args.clusters < 2 or args.batch_size < 2 or args.workers < 0:
        raise ValueError('Positive epochs and at least two clusters required')
    args.run = args.run.resolve()
    if args.run.is_relative_to(ROOT / 'ml/data/raw') or (args.run.exists() and not args.resume):
        raise ValueError('Use a new run directory outside raw data')
    args.run.mkdir(parents=True, exist_ok=bool(args.resume))
    if args.resume and args.resume.resolve().parent != args.run:
        raise ValueError('Resume checkpoint must belong to this run')
    torch.backends.cudnn.benchmark = True
    seed = 20260917
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    records = (json.loads((args.run/'samples.json').read_text(encoding='utf-8')) if args.resume else candidates())
    training, validation, components = grouped_split(records)
    for i, r in enumerate(records):
        r['split'] = 'validation' if i in set(validation) else 'train'
    (args.run / 'samples.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    net = encoder(True).cuda().to(memory_format=torch.channels_last)
    initial = {k: v.detach().cpu().clone() for k,v in net.named_parameters()}
    projector = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,256)).cuda()
    optimizer = torch.optim.AdamW(list(net.parameters())+list(projector.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    start_epoch, previous_seconds, completed_steps = 1, 0., 0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda', weights_only=True)
        net.load_state_dict(checkpoint['encoder'])
        projector.load_state_dict(checkpoint['projector'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'rng_cpu' in checkpoint:
            torch.set_rng_state(checkpoint['rng_cpu'].cpu())
            torch.cuda.set_rng_state(checkpoint['rng_cuda'].cpu())
        history = json.loads((args.run/'history.json').read_text())[:checkpoint['epoch']]
        previous_seconds = history[-1]['elapsed_seconds']
        completed_steps = checkpoint.get('gradient_steps', checkpoint['epoch'] * (len(training)//32))
    rng = np.random.default_rng(seed)
    evaluation = [records[i] for i in rng.permutation(validation)[:128]]
    config = (json.loads((args.run/'experiment.json').read_text(encoding='utf-8')) if args.resume else dict(status='training', final_model=False, method='VICReg-style appearance consistency',
                  backbone='ResNet18 ImageNet initialization; all layers trainable', epochs=args.epochs,
                  batch_size=32, lr=1e-4, seed=seed, gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, cuda=torch.version.cuda,
                  source_counts=dict(Counter(r['kind'] for r in records)), samples=len(records),
                  train=len(training), validation=len(validation), connected_groups=components,
                  unknown_thermal_may_remain=True, before=diagnose(net,evaluation)))
    config.update(status='training', epochs=args.epochs)
    config.setdefault('runtime_changes', []).append(dict(start_epoch=start_epoch, batch_size=args.batch_size,
                  workers=args.workers, mixed_precision='float16; loss float32', channels_last=True,
                  resume_checkpoint=str(args.resume) if args.resume else None))
    (args.run / 'experiment.json').write_text(json.dumps(config,indent=2), encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)
    loader = DataLoader(Pairs([records[i] for i in training]), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                        worker_init_fn=worker_init, persistent_workers=args.workers > 0,
                        **({'prefetch_factor': 3} if args.workers else {}))
    started = time.monotonic()
    for epoch in range(start_epoch,args.epochs+1):
        net.train(); projector.train()
        sums = np.zeros(4)
        epoch_started = time.monotonic()
        data_wait = 0.
        ready = time.monotonic()
        for step,(a,b) in enumerate(loader,1):
            data_wait += time.monotonic()-ready
            optimizer.zero_grad(set_to_none=True)
            images = torch.cat([a.cuda(non_blocking=True),b.cuda(non_blocking=True)]).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                z = projector(net(images))
            loss,parts = vicreg(*z.float().chunk(2))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stopping round')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(projector.parameters()),5)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            completed_steps += int(scaler.get_scale() >= old_scale)
            sums += [loss.item(),*(p.item() for p in parts)]
            if step % 20 == 0:
                rate = step*args.batch_size/(time.monotonic()-epoch_started)
                print(f'epoch={epoch}/{args.epochs} step={step}/{len(loader)} loss={sums[0]/step:.4f} samples_per_second={rate:.1f}',flush=True)
            ready = time.monotonic()
        row = dict(epoch=epoch, loss=sums[0]/len(loader), invariance=sums[1]/len(loader),
                   variance=sums[2]/len(loader), covariance=sums[3]/len(loader), elapsed_seconds=previous_seconds+time.monotonic()-started,
                   batch_size=args.batch_size, workers=args.workers, data_wait_seconds=data_wait,
                   epoch_seconds=time.monotonic()-epoch_started, gradient_steps=completed_steps)
        history.append(row)
        torch.save(dict(encoder=net.state_dict(), projector=projector.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch, final_model=False, scaler=scaler.state_dict(), gradient_steps=completed_steps,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), args.run / f'checkpoint_epoch_{epoch:02d}.pt')
        (args.run / 'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    config['after'] = diagnose(net,evaluation)
    delta = sum(float((v.detach().cpu()-initial[k]).square().sum()) for k,v in net.named_parameters()) ** .5
    if delta == 0:
        raise RuntimeError('Encoder parameters did not change')
    config.update(encoder_weight_l2_change=delta, training_seconds=previous_seconds+time.monotonic()-started,
                  gradient_steps=completed_steps, history=history)
    vectors = encode_records(net, records)
    # Fit only train split. Validation receives nearest train centroid assignment.
    with threadpool_limits(limits=4):
        clustering = KMeans(n_clusters=args.clusters, random_state=seed,n_init=10).fit(vectors[training])
        labels = clustering.predict(vectors)
    np.save(args.run/'features.npy',vectors); np.save(args.run/'centers.npy',clustering.cluster_centers_)
    np.save(args.run/'assignments.npy',labels)
    for group in range(args.clusters):
        members = np.flatnonzero(labels==group)
        if not len(members):
            continue
        distances = ((vectors[members]-clustering.cluster_centers_[group])**2).sum(1)
        ordered = members[np.argsort(distances)]
        selected = list(dict.fromkeys([*ordered[:5],*ordered[-5:],*rng.choice(members,min(5,len(members)),replace=False)]))
        sheet([records[i] for i in selected],args.run/f'group_{group:03d}.jpg',
              [f'R1 G{group:02d} #{i} {records[i]["split"]}' for i in selected])
    cards, captions = [], []
    for i in rng.permutation(validation)[:8]:
        order = np.argsort(-(vectors @ vectors[i]))
        nearby = [j for j in order if j in set(training)
                  and not set(records[j]['uploads']).intersection(records[i]['uploads'])][:4]
        cards.extend([records[i]]+[records[j] for j in nearby])
        captions.extend([f'VALIDATION #{i}']+[f'TRAIN #{j}' for j in nearby])
    sheet(cards,args.run/'validation_neighbors.jpg',captions)
    config.update(status='awaiting_user_review', clusters=args.clusters,
                  cluster_sizes=dict(Counter(map(str,labels))), final_model=False)
    (args.run/'experiment.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)


if __name__ == '__main__':
    main()
