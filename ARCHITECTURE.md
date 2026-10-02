# Architecture

The pipeline detects, segments and tracks boats in water video. Water makes every stage harder:

- glare and wave crests get picked up as objects;
- tracked points slide off onto the moving water;
- targets are small and far away;
- the camera moves.

## Pipeline

<p align="center">
  <img src="docs/pipeline.png" alt="Pipeline diagram" width="420">
</p>

Diagram source: [`docs/pipeline.mmd`](docs/pipeline.mmd)

## Modules

| file | what it does |
|---|---|
| `marine_tracking/detect.py` | YOLO wrapper. Loads the model lazily and returns plain-numpy `Detection`s |
| `marine_tracking/segment.py` | SAM wrapper. If a mask fails, it uses the filled box instead |
| `marine_tracking/track.py` | LK point tracking, median-motion gate, IoU/centroid association, expiry. Pure functions with unit tests |
| `marine_tracking/pipeline.py` | Per-frame glue for the LK backend. Handles re-detection cadence, plus early re-detect when a track runs low on points |
| `marine_tracking/neural_pipeline.py` | Segment-based glue for CoTracker3. Each query point carries a track ID |
| `marine_tracking/visualize.py` | Draws masks, points, IDs and trails. Can draw without boxes |
| `scripts/run_video.py` | CLI for the LK backend |
| `scripts/run_video_neural.py` | CLI for the CoTracker3 backend |
| `scripts/make_demo.py` | Builds the staged demo video, with the follow-zoom virtual camera |
| `scripts/benchmark.py` | Produces the numbers in RESULTS.md |
| `scripts/download_sample.py` | Downloads the CC BY-SA sample clip (attribution is in the file header) |

## Design decisions

### How often to re-detect

Running YOLO and SAM every frame gives the best detections but the lowest FPS. So detection runs every N frames, and in between, objects coast on their tracked points:

- **LK backend:** each box shifts by the median motion of its points.
- **CoTracker3 backend:** each box is rebuilt from its points every frame.

If a track drops below a minimum number of points, it triggers a re-detect early.

### Points only inside the mask

A detection box on water is mostly water. Points are seeded only inside the SAM mask, so they start on the boat and not on the wake next to it.

### Median-motion gate (LK)

A point stuck on a wave crest moves with the water, not the boat. If a point's motion is too far from the median motion of its object, it gets dropped.

### Identity belongs to points (CoTracker3)

Every CoTracker3 query point belongs to exactly one track ID. The box comes from the points (8th/92nd percentile, so one stray point can't stretch it), not the other way round.

### Keeping points across re-inits

CoTracker3 queries are fixed when the tracker starts, so the tracker gets re-initialized at every re-detection. Sometimes the detector misses a boat at that exact frame. When that happens, the boat's surviving points are passed in as queries again, so it keeps its ID through the gap. This cut new IDs more than any other change.

### Association logic is pure and tested

Matching runs in three steps:

1. Greedy best-IoU matching.
2. A centroid-distance pass for anything left over. This catches re-detections whose box drifted too far for IoU to match.
3. A max-age rule that expires old tracks.

It's all numpy with no models, so the tests run in milliseconds.

### Degrade instead of dropping

- If SAM fails on a box, the filled box is used as the mask, so the object is still tracked.
- If a track is occluded, it coasts and expires after a counter runs out. It doesn't disappear the first frame a detection blinks out.

## Known limitations

- **Small far-away boats** flicker around the confidence threshold. If one is gone longer than max-age, it comes back with a new ID. Carry-over and cadence help, but don't fix it.
- **No camera-motion compensation yet.** Trajectories are in image space, not world space.
- **CoTracker3 memory grows with segment length**, because the whole segment sits on the GPU. 48 frames at 720p fits easily in 16 GB.
- **Boats that overlap while crossing** can merge into one detection. Short merges keep their IDs; long ones can swap them.
