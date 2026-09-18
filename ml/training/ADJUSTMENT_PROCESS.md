# User-approved iterative neural training

Round 1 was authorized by the user on 2026-09-17. Further rounds require their
review and approval. Final model export/publication also requires final approval.

## What is trained

The ResNet18 neural backbone starts from ImageNet pretrained weights. All of its
layers and a new 512→512→256 projection head are optimized with AdamW. The
projection head is used for the self-supervised objective; normalized backbone
features are used for equipment similarity. The loss has invariance, variance
and covariance terms (25, 25, 1), based on VICReg:
https://arxiv.org/abs/2105.04906

Both views retain the whole image. Independently sampled color, brightness and
grayscale transformations encourage appearance invariance without assuming that
two unrelated crops of a multi-device photo show the same equipment. This round
does not train an object detector and does not prove shape recognition.

KMeans only organizes the learned neural features into 24 review groups. It is
not the learning mechanism for the visual representation. Group numbers are
specific to this run and must not be confused with the earlier baseline groups.

## Data and validation

The existing catalog combines original images and extracted video frames.
Byte-identical samples are merged; user-rejected hashes are omitted. Known
thermal samples are held separately. Other thermal/non-equipment samples may
remain until reviewed. No rejection labels are generalized to unseen images.

Upload aliases and original capture filenames connect related samples before
the train/validation split. This is conservative metadata grouping, not proof
that every near-duplicate physical scene is isolated. Validation samples are
not used in gradient optimization or centroid fitting.

Before and after training, 128 fixed validation examples are measured for
grayscale stability, exact-image retrieval and embedding diversity. These are
diagnostics, not equipment-name accuracy. True shape and device-type validation
requires human-reviewed positive/negative equipment pairs and object regions.

## Artifacts to review

`ml/data/runs/adjust_round_01/` contains:

- `samples.json`: source mapping and split.
- `history.json`: epoch losses and elapsed time.
- `checkpoint_epoch_*.pt`: experimental network/optimizer states, not a final model.
- `experiment.json`: GPU, gradient steps, actual encoder weight delta,
  before/after diagnostics, explicit final_model=false.
- `group_*.jpg`: representative, boundary and random examples from each group.
- `validation_neighbors.jpg`: held-out queries with training-set matches.
- `features.npy`, `centers.npy`, `assignments.npy`: provisional grouping results.

After the round, the user reviews groups and requests exclusions, splits, merges
or corrections. Their response is attached to sample identities and the exact
run. The next adjustment starts only after approval. Unverified group membership
must not silently become a ground-truth equipment class.

The final deployment model is produced only after final user confirmation and
validation of the intended detection/classification behavior. No server/UI work
is part of this adjustment loop.

## GPU utilization adjustment within round 1

The original runtime loaded/augmented images synchronously (zero workers).
After the first epoch checkpoint, resume the same network and optimizer using
six persistent loader workers with prefetching, pinned transfer buffers,
channels-last convolution inputs and float16 automatic mixed precision. Compute
the variance/covariance loss in float32 and apply gradient scaling/clipping.
Batch size increases from 32 pairs to 64 pairs; this affects batch statistics
and is explicitly recorded as a runtime change. Epoch count and approval gate
remain unchanged. GPU utilization varies during loading, saving and evaluation;
throughput and finite-loss/gradient checks matter alongside utilization.

Observed during this run: the original epoch 1 completed in 303.375 seconds.
The epoch-1 encoder and optimizer checkpoint was verified before restarting;
partial unsaved epoch-2 steps from the original process were discarded and
epoch 2 restarted from that checkpoint. After loader startup and cuDNN warmup,
eight consecutive one-second NVIDIA measurements read GPU utilization
100, 100, 99, 100, 100, 100, 100, 100 percent at approximately 3,380 MiB total
device memory use (includes desktop/display allocations). This demonstrates
compute saturation at that interval, not a guarantee of 100% throughout I/O,
checkpoint saves or clustering.

```powershell
.venv-model\Scripts\python.exe -m ml.training.adjust_round --epochs 3 --resume ml/data/runs/adjust_round_01/checkpoint_epoch_01.pt --batch-size 64 --workers 6
```
