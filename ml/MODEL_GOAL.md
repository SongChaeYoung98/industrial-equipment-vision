# Current goal — updated by user direction

## Controlling instruction: iterative deep-learning adjustment

Use the locally collected image/video data to actually optimize neural network
weights and produce provisional clusters. Present each round's method, losses,
weight-change evidence and visual results to the user. Wait for explicit user
approval/corrections before starting the next adjustment round. The user has
authorized round 1 now. Only the user's final confirmation authorizes release
as the final model. Do not equate frozen library feature extraction plus KMeans
with the requested deep-learning training. No new API/UI work.

Round 1: GPU self-supervised fine-tuning of a pretrained visual backbone using
paired appearance perturbations and variance/covariance regularization; retain
full-image content to avoid teaching that disjoint devices are the same object.
Use the source-image and extracted-video-frame catalog, excluding reviewed
non-equipment hashes and holding known thermal samples for separate review.
This does not yet solve multiple-object localization or physical-device identity.
Color invariance and embedding collapse checks are diagnostics, not equipment
accuracy. Label uncertainty and other thermal images remain to be reviewed.

## Architecture correction requested by the user

KMeans is not a model-training method for this project. It may remain only as a
diagnostic comparison outside the model. The next round must use a learnable
prototype head whose assignments and backbone weights are optimized together
with a SwAV/DeepCluster-style self-supervised objective. Backbone depth,
embedding width, prototype count, augmentations, batch size, learning rate,
temperature and training length must be explicit configuration and recorded.
The prototype assignment is provisional equipment grouping, never an equipment
name. Use the resulting groups for user review, then train supervised detection
and classification after confirmed object boxes/classes exist.

Build and test a local AI model for visually distinguishing equipment from
images and video frames. Focus on model experiments, not server endpoints or
review UI. This direction supersedes UI/API deliverables in the earlier plan.

1. Run a reproducible pretrained-image-feature baseline on deduplicated local
   images and representative frames. Fit provisional equipment clusters and
   save a reusable model artifact, embeddings and visual examples.
2. Inspect actual nearest-neighbor/cluster results for equipment similarity,
   background shortcuts and repeated captures. Do not call similarity a
   probability of equipment identity or call clusters verified device classes.
3. Improve the representation and add object localization/crops where actual
   failure examples justify it; verify images containing multiple devices.
4. Obtain independently verified equipment labels and group related captures
   before train/validation/test splitting; fine-tune and evaluate equipment
   detection/classification with unknown/abstention handling.

Preserve raw data. Work only in check-lab-ai locally. No production/backend
changes. No additional server or web features. Expert-confirmed labels are
required for trustworthy equipment names; provisional groups remain unnamed.

The product goal tool does not provide objective editing or resume actions.
This file records the user's revised objective without falsely marking the
previous unfinished objective complete.

## User corrections after first baseline

- The first experiment did not train the neural encoder: it ran frozen ResNet18
  inference and KMeans. Treat it as a diagnostic baseline, not a trained equipment model.
- Exclude baseline-v1 groups 17 (hands) and 19 (office scenes) from model inputs
  and search. Bind decisions to sample hashes; future group numbers are unrelated.
- Group 20 is thermal imagery of multiple equipment types; preserve it as a
  separate modality, not a device-type label.
- Next neural training must address shape sensitivity and color/background
  shortcuts. Compare fixed-encoder versus fine-tuned performance on held-out
  equipment examples, color variants and cross-modality cases. Grayscale alone
  does not establish shape recognition, especially for false-color thermal images.
- Inspect available GPU before selecting runtime. RTX 3050 8 GB is available;
  the first experiment used CPU-only PyTorch. Measure actual steps/second and
  report epochs, loss, weight changes and evaluation instead of promising a duration.
