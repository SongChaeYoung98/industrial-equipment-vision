# Equipment discovery workflow

The current user-approved scope is in `ml/MODEL_GOAL.md`: focus on offline
model experiments. The UI/API gates below are historical and deferred, not
current deliverables. Run the first model experiment with
`.venv-model/Scripts/python -m ml.training.visual_baseline`.

## Current commands

Run from the repository root:

```powershell
python tools/build_media_catalog.py
python tools/extract_review_frames.py
```

Both commands read `ml/data/raw/2026_09_16` and write only to
`ml/data/derived/2026_09_16`. The catalog is resumable using source size and
mtime; a changed file is reinspected. SHA-256 identifies byte-identical sources.
The SQLite `metadata` column contains image brightness, Laplacian sharpness,
dimensions and a difference hash, or video dimensions, FPS and frame count.
These values are measurements, not equipment labels or quality verdicts.
Video metadata inspection is not an integrity test.

Frames are sampled at up to five evenly spaced positions per video. Their
manifest links every frame to its source upload and requested/actual frame
position. This bounds the initial review workload; it does not guarantee every
device or event in a video is represented. OpenCV may conceal decoder errors;
successful frame extraction is not proof of visual quality.

## Remaining implementation gates

1. Review representative imagery, near duplicates and codec warnings; create
   quality review flags instead of deleting source files.
2. Provide a local review UI supporting image/frame selection and editable
   object rectangles. Add automatic region proposals with explicit review status.
3. Use a versioned pretrained visual encoder on object crops. Store model and
   preprocessing identity with embeddings. Validate retrieval against manually
   checked equipment pairs, including background/overlay confounders.
4. Build a similarity index and temporary clusters; expose similar examples,
   group reassignment, merge/split decisions and unknown/unusable decisions.
5. Record confirmed equipment names separately from provisional group IDs,
   including reviewer and label provenance. Expert knowledge or a verified
   equipment inventory is required for trustworthy device names.
6. Group source video frames, uploads and duplicates before dataset splitting.
   Review physical-device/site grouping as metadata becomes available; upload
   IDs alone cannot prevent all leakage.
7. Train detection/classification models using reviewed labels, evaluate on
   independent groups, record per-class errors and abstention behavior, and
   connect validated artifacts to inference.

No named equipment taxonomy or trained detector/classifier is available yet.
The backend and production server are outside the write scope of this workflow.
