# First model experiment: 2026-09-17

Run: `ml/data/runs/visual_baseline_v1`

- 1,024 byte-deduplicated samples: 832 still images and 192 video frames.
- Frozen ResNet18 ImageNet features; KMeans fitted to 24 temporary groups.
- 512-dimensional normalized vectors; saved encoder weights, centroids,
  assignments, sample provenance and visual reports.
- Training silhouette: 0.0456. This is weak separation in feature space and
  is not an equipment recognition accuracy measurement.
- Reloaded the saved encoder and centers and successfully queried an existing
  source PNG through the offline prediction command.

Visual inspection of `neighbors.jpg` (eight queries, four matches each):

- Query 0 returns nearly identical cabinet scenes across different upload IDs.
  Upload exclusion alone does not remove repeated physical captures.
- Query 5 retrieves circular handles and yellow piping with visually related
  structure. This supports further experiments but does not confirm device names.
- Queries 2 and 4 retrieve cabinet/interior wiring scenes with different layouts;
  scene-level similarity is not precise component identity.
- Query 3 matches fence/wire geometry and then electrical wiring: clear evidence
  that background and line patterns can dominate the embedding.
- Query 6 includes a severely blurred example among close matches; successful
  decoding does not imply a usable equipment image.

Next model work: compare stronger self-supervised visual features on the same
fixed samples, evaluate object crops versus full scenes, and establish a small
independently reviewed similarity set. Do not enlarge the server/UI scope.
Named-equipment supervised fine-tuning is pending verified labels.
