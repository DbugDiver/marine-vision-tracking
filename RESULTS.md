# Results

## Setup

- **Models / weights:** `yolo11n.pt` (Ultralytics COCO-pretrained, class
  `boat`), `sam2_b.pt` (Ultralytics SAM2), CoTracker3 offline
  (`facebookresearch/co-tracker` via torch.hub, `scaled_offline.pth`).
- **Library versions:** pinned in `requirements.txt` (ultralytics 8.4.131,
  torch 2.11.0+cu128, opencv-python 5.0.0.93, numpy 2.4.6).
- **Hardware:** NVIDIA GPU, 16 GB VRAM, Windows 11, Python 3.12. Input
  1280x720 @ 25 fps; detection at `imgsz=960`, `conf=0.25`.
- **Footage:** "Team sailing match-racing start at NHYC" by Don Ramey Logan,
  Wikimedia Commons, **CC BY-SA 4.0** — fetched by
  `scripts/download_sample.py` (first 6 s skipped: source title card). All
  derived media here (demo video, GIF, stills) inherits CC BY-SA 4.0 with
  that attribution.

## Performance

Measured by `scripts/benchmark.py` over the same 500 frames (20 s), one
warm-up detection excluded, ~9.5 detected boats per frame (12-13 vessels in
scene):

| config | re-detect every N | FPS | notes |
|---|---|---|---|
| detection only | - | 79.0 | 9.5 boats/frame |
| detect+SAM+LK | 1 | 3.6 | 32 ids |
| detect+SAM+LK | 5 | 14.0 | 16 ids |
| detect+SAM+LK | 10 | 20.5 | 15 ids |
| detect+SAM+CoTracker3 | 24 | 36.0 | 15 ids |
| detect+SAM+CoTracker3 | 48 | 39.9 | 14 ids |

Readings, not spin:

- **The neural backend wins on BOTH axes.** 39.9 fps with 14 distinct ids vs
  LK's best of 20.5 fps / 15 ids. The reason is structural: YOLO+SAM run once
  per 48 frames instead of once per 5-10, and CoTracker3's temporal window
  holds points through glare flicker that resets LK.
- **N=1 is the worst of everything** (3.6 fps AND 32 ids): re-associating
  noisy per-frame detections churns identities faster than coasting does.
  "More detection" is not "more stable".
- ~12-13 real vessels and 14 ids at the best config = roughly one or two
  id re-mints over 20 s, both on small boats dropping below the confidence
  floor for longer than the carry-over can bridge.

## What worked / what failed

- **Worked:** mask-gated point seeding — points start on hulls and sails, not
  wake; visually obvious in the demo.
- **Worked:** carrying surviving points across tracker re-inits as new
  queries. Largest single reducer of id churn.
- **Worked:** percentile (8/92) boxes from the point cloud; min/max boxes
  ballooned on single stray points and jittered labels.
- **Failed / open:** small distant boats flicker at the detector's confidence
  floor; beyond max-age the id re-mints. More or fine-tuned training data is
  the lever, not threshold-tuning (lowering conf admits wave crests).
- **Failed / open:** no global motion compensation — trajectories are
  image-space; a panning camera writes its own motion into every trail.
- **Open:** long overlaps at crossings can swap ids; association keeps short
  merges only.

## Commands to reproduce

```bash
python scripts/download_sample.py
python scripts/benchmark.py --input data/samples/sample_open.mp4 --seconds 20 --skip-seconds 6
python scripts/run_video_neural.py --input data/samples/sample_open.mp4 --output media/neural_run.mp4 --side-by-side
python scripts/make_demo.py --input data/samples/sample_open.mp4 --output media/demo.mp4
pytest tests/ -q
```
