# Marine Vision Tracking: Work Plan

This repo stays **private** until the checklist below is done. Then I'll write the public README from `RESULTS.md` and flip it public.

## 1. Clean-room rules (non-negotiable)

- **Code:** write everything from scratch. Don't copy, paraphrase or consult code, notebooks, configs, tuned parameters or internal docs from Tario Marine Technologies or any other employer.
- **Libraries:** use only public open-source libraries and public model weights (Ultralytics YOLO, SAM/SAM2, OpenCV, and CoTracker if used). Follow their licenses. Ultralytics is AGPL-3.0, so this repo should be AGPL-3.0 or otherwise compatible.
- **Footage:**
  - **Preferred:** video you shot yourself, from a shoreline, pier, ferry or boat, with no identifiable faces in focus.
  - **Allowed:** a public dataset **only if its license allows redistributing derived clips and GIFs**. Check before you use it. Don't commit dataset files; add a download script instead.
  - **Employer footage:** one short clip is used **with explicit permission from the employer** (keep the written permission on file; credit the source in RESULTS.md). No telemetry, logs, or anything beyond that clip.
- **Framing:** no mention of any employer in the code or README. This is an independent project.

## 2. Target structure

```
marine_tracking/
  detect.py        YOLO wrapper: frame -> boxes, classes, scores
  segment.py       SAM wrapper: frame + boxes -> masks
  track.py         point tracking (start with OpenCV Lucas-Kanade inside masks;
                   optional CoTracker) + ID association across frames
  pipeline.py      glue: runs detect -> segment -> track per frame, handles re-detection
  visualize.py     draws boxes, masks, IDs, trajectories
scripts/
  run_video.py     CLI: python scripts/run_video.py --input clip.mp4 --output out.mp4
  download_sample.py  (only if using a public dataset)
tests/
  test_track.py    unit tests for association logic on synthetic data
data/samples/      one short (<10 s, <10 MB) sample clip you own
media/             demo GIF + before/after stills
```

## 3. Engineering points worth doing well (the interview talking points)

1. **Re-detection cadence.** Running YOLO + SAM every frame is slow. Detect every N frames and track points in between. Measure the FPS vs accuracy trade-off for N = 1, 5, 10.
2. **Small, distant targets.** Try tiled or sliced inference, or a higher input resolution, and measure the effect.
3. **Glare and waves.** Points on water texture drift. Keep only points inside the SAM mask and drop points whose motion disagrees with the object's median motion.
4. **Camera motion.** Optionally estimate global motion (homography from background features) and subtract it so trajectories are world-stable.
5. **ID association.** Match new detections to existing tracks with IoU or centroid distance and a max-age rule. Unit-test it.

## 4. Deliverables: send me `RESULTS.md` + media

Create `RESULTS.md` in the repo root:

```markdown
## Setup
- Models + weights used (e.g. yolo11n.pt, sam2_b.pt), library versions
- Hardware (CPU/GPU model) and input resolution
- Footage source + license (own footage / dataset name + license link)

## Performance
| Config | Re-detect every N frames | FPS | Notes (ID switches, lost tracks) |
|---|---|---|---|
| detection only |  |  |  |
| detect + segment + track | 1 |  |  |
| detect + segment + track | 5 |  |  |
| detect + segment + track | 10 |  |  |

## What worked / what failed
- 3-5 honest bullets (glare, tiny targets, occlusion by waves, etc.)

## Commands to reproduce
- exact commands
```

Also add:
- `media/demo.gif`: **15–20 s, side by side (raw | tracked)**, under 10 MB, about 480–640 px wide.
- `media/stills/`: 2–3 PNG frames showing masks, IDs and trajectories.
- `tests/`: runs with `pytest`.
- A pinned `requirements.txt` (`pip freeze` of the packages actually used).

When it's pushed, tell me. I'll write the public README (pipeline diagram, GIF, results table) and make it public.

## Resume line (use once the results exist)

> Built a marine object-tracking pipeline (YOLO detection → SAM segmentation → mask-constrained optical-flow point tracking) handling glare, wave clutter and small distant targets; X FPS on <hardware> with re-detection every N frames.
