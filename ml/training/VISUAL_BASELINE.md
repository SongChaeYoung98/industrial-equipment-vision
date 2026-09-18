# Offline visual equipment baseline

This experiment uses TorchVision's ResNet18 IMAGENET1K_V1 feature extractor
and fits KMeans on local deduplicated samples. The fitted centroids are a
provisional grouping model. The encoder is pretrained, not fine-tuned on
equipment labels. Groups do not have verified equipment names.

Reference: https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html

Run from the repository root:

```powershell
.venv-model\Scripts\python.exe -m ml.training.visual_baseline --limit 1024 --clusters 24
.venv-model\Scripts\python.exe -m ml.training.visual_baseline --query "path\image.png"
.venv-model\Scripts\python.exe -m ml.training.visual_baseline --query "path\image.png" --box 20 30 200 200
```

Artifacts are in `ml/data/runs/visual_baseline_v1`. Use another `--run` path
for a new experiment. `encoder.pt` and `centers.npy` are the reusable model;
`features.npy` and `samples.json` form the exemplar search dataset.
`neighbors.jpg` and `cluster_*.jpg` show actual results without a server.

Preprocessing is RGB conversion, aspect-preserving pad to 224x224 with bilinear
resizing, ImageNet mean/std normalization, then L2 normalization of 512 features.
Weights are downloaded from the official PyTorch endpoint to the local model
cache. Inputs are not uploaded. Seed, preprocessing identity, versions and
sample paths are recorded in the experiment artifacts.

The data is unlabeled. Silhouette measures cluster geometry on training samples,
not device-recognition accuracy. Repeated scenes across uploads and background
similarity can inflate apparent quality. The neighbor report excludes known
same-upload matches and byte-identical candidates; this does not prove physical
device independence. The optional query crop is compared to whole-image indexed
features, so it remains a diagnostic rather than a trained object detector.

Next experiments must check representative query matches, background shortcuts,
multi-device scenes, and independently confirmed equipment labels before
fine-tuning/evaluating an equipment classifier or detector.
