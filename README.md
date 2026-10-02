# Marine Vision Tracking

This project tracks boats in open-water video. YOLO11 finds the boats, SAM2 masks them, and CoTracker3 follows points on each hull, so every boat keeps the same ID from frame to frame. It runs at about 40 FPS on a 16 GB GPU.

<p align="center">
  <img src="media/demo.gif" alt="Raw footage next to tracked output: each boat keeps its ID, point cloud and trail" width="860">
</p>

<p align="center"><sub>Left: raw footage. Right: tracked output. <a href="media/demo.mp4">Full demo video (29 s)</a>. Footage credit is at the bottom.</sub></p>

## The problem

Water is a rough environment for computer vision:

- Glare and wave crests look like objects.
- Points that are supposed to stay on a boat slide off onto the moving water.
- A lot of the boats are small and far away, right at the edge of what the detector can see.
- The camera is usually moving too.

I wanted to keep one ID per boat without running the heavy models on every single frame.

## How it works

<p align="center">
  <img src="docs/pipeline.png" alt="Pipeline: YOLO detection, SAM segmentation, points seeded inside masks, CoTracker3 or Lucas-Kanade tracking, ID association, re-detection every N frames" width="420">
</p>

1. **Detect.** YOLO11 finds boats in the frame.
2. **Segment.** SAM2 makes a mask for each box. A box drawn around a boat on water is mostly water, so the mask is what separates the hull and sails from the wake.
3. **Seed points inside the mask.** Points start on the boat, not the water next to it.
4. **Track the points.** There are two backends:
   - **CoTracker3:** looks at a window of frames at once.
   - **Lucas-Kanade optical flow:** frame to frame, with a filter that drops points moving differently from the rest of the boat.
5. **Assign IDs.** IoU matching first, then a centroid-distance fallback, then old tracks expire after a max age. This part is plain numpy and has unit tests.
6. **Re-detect every N frames,** or sooner if a track runs low on points. Points that survive are carried into the next segment, so a boat the detector misses for a moment keeps its ID.

More detail on each decision is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Results

All runs use the same 500 frames (20 s at 1280x720) on a 16 GB NVIDIA GPU, measured with `scripts/benchmark.py`. There are about 12-13 actual boats in the clip, so fewer IDs is better.

| Setup | Re-detect every N frames | FPS | IDs |
|---|---:|---:|---:|
| Detection only | - | 79.0 | - |
| Detect + SAM + Lucas-Kanade | 1 | 3.6 | 32 |
| Detect + SAM + Lucas-Kanade | 5 | 14.0 | 16 |
| Detect + SAM + Lucas-Kanade | 10 | 20.5 | 15 |
| Detect + SAM + CoTracker3 | 24 | 36.0 | 15 |
| Detect + SAM + CoTracker3 | 48 | 39.9 | 14 |

Takeaways:

- **CoTracker3 came out ahead on both speed and stability.**
  - Speed: the heavy models only run once every 48 frames.
  - Stability: its multi-frame window holds points through glare that keeps resetting Lucas-Kanade.
- **Detecting every frame was the worst option on both counts.** Re-matching noisy detections on every frame keeps creating new IDs.
- **14 IDs for about 13 boats** means roughly one ID got re-created in 20 seconds. It was a small boat that stayed below the detection threshold too long.

Full setup and numbers are in [RESULTS.md](RESULTS.md).

<table>
  <tr>
    <td width="33%"><img src="media/stills/01_yolo_detection.png" alt="YOLO detection boxes"></td>
    <td width="33%"><img src="media/stills/02_sam_masks.png" alt="SAM2 masks per boat"></td>
    <td width="33%"><img src="media/stills/03_point_tracking.png" alt="CoTracker3 point tracking with IDs and trails"></td>
  </tr>
  <tr>
    <td align="center"><sub>1. YOLO detection</sub></td>
    <td align="center"><sub>2. SAM2 masks</sub></td>
    <td align="center"><sub>3. Point tracking with IDs and trails</sub></td>
  </tr>
</table>

## Running it

```bash
git clone https://github.com/DbugDiver/marine-vision-tracking.git
cd marine-vision-tracking
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # pinned, CUDA 12.8 torch build
python scripts/download_sample.py                      # sample clip, needs ffmpeg
python scripts/run_video_neural.py --input data/samples/sample_open.mp4 \
       --output outputs/tracked.mp4 --side-by-side
```

Other scripts:
- `scripts/run_video.py`: the Lucas-Kanade version.
- `scripts/benchmark.py`: reproduces the table above.
- `scripts/make_demo.py`: builds the demo video.

CoTracker3 isn't on pip. It downloads through `torch.hub` the first time you run it.

## Tests

```bash
pytest tests/ -q
```

15 tests on synthetic data. They need no models or video and finish in under a second. They cover IoU, matching, track expiry and the motion filter.

## Layout

```
marine_tracking/
  detect.py           YOLO wrapper
  segment.py          SAM2 wrapper, falls back to the box if a mask fails
  track.py            LK tracking, motion filter, matching, expiry
  pipeline.py         frame loop for the LK version
  neural_pipeline.py  CoTracker3 version, IDs carried per point
  visualize.py        drawing: masks, points, IDs, trails
scripts/              run_video, run_video_neural, benchmark, make_demo, download_sample
tests/                unit tests
docs/                 pipeline diagram
```

## What's not solved yet

- **Small boats far away.** They flicker around the detection threshold. Lowering the threshold picks up wave crests instead, so the real fix is fine-tuning on marine footage.
- **Camera motion.** Trails are in image coordinates, so a panning camera shows up in every trail. Next step is a background homography to cancel it out.
- **Boats crossing.** Long overlaps can swap IDs. Short ones are handled.

## Credits and licenses

- **Footage:** "Team sailing match-racing start at NHYC" by Don Ramey Logan, from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Team_sailing_match-racing_start_at_NHYC_by_Don_Ramey_Logan.webm), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  - The demo video, GIF and stills in `media/` are made from it and shared under the same license.
  - The clip itself isn't in the repo. `scripts/download_sample.py` fetches it.
- **Models:**
  - [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 and SAM2 wrappers (AGPL-3.0).
  - [SAM 2](https://github.com/facebookresearch/sam2) (Apache-2.0).
  - [CoTracker3](https://github.com/facebookresearch/co-tracker) (CC BY-NC 4.0, non-commercial).
- **Code:** [AGPL-3.0](LICENSE), to match Ultralytics.
