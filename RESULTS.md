# Results

## Setup

- Models: `yolo11n.pt` (Ultralytics, COCO-pretrained, class `boat`), `sam2_b.pt` (Ultralytics SAM2), and CoTracker3 offline (`facebookresearch/co-tracker` via torch.hub, `scaled_offline.pth`).
- Library versions are pinned in `requirements.txt`: ultralytics 8.4.131, torch 2.11.0+cu128, opencv-python 5.0.0.93, numpy 2.4.6.
- Hardware: NVIDIA GPU with 16 GB VRAM, Windows 11, Python 3.12.
- Input: 1280x720 at 25 fps. Detection runs at `imgsz=960`, `conf=0.25`.
- Footage: "Team sailing match-racing start at NHYC" by Don Ramey Logan, Wikimedia Commons, CC BY-SA 4.0.
  - Downloaded with `scripts/download_sample.py`.
  - The first 6 s are skipped because they're the source's title card.
  - The demo video, GIF and stills are derived from it, so they're CC BY-SA 4.0 too, with the same attribution.

## Performance

All numbers come from `scripts/benchmark.py`, run on the same 500 frames (20 s), with the warm-up detection excluded. There were about 9.5 boats detected per frame, out of 12-13 vessels in the scene.

| config | re-detect every N | FPS | IDs |
|---|---|---|---|
| detection only | - | 79.0 | - |
| detect + SAM + LK | 1 | 3.6 | 32 |
| detect + SAM + LK | 5 | 14.0 | 16 |
| detect + SAM + LK | 10 | 20.5 | 15 |
| detect + SAM + CoTracker3 | 24 | 36.0 | 15 |
| detect + SAM + CoTracker3 | 48 | 39.9 | 14 |

What the numbers say:

- **CoTracker3 is both faster and more stable than LK here.**
  - Speed: YOLO and SAM only run once every 48 frames, instead of every 5 or 10.
  - Stability: CoTracker3 looks at a window of frames, so it holds points through the glare flicker that keeps resetting LK.
- **Detecting every frame (N=1) is the worst setting on both counts.** It ran at 3.6 fps and produced 32 IDs. Re-matching noisy detections every frame creates new IDs faster than just coasting on tracked points.
- **14 IDs for 12-13 boats** means one or two IDs got re-created over 20 s. Both were small boats that stayed under the confidence threshold longer than the tracker could bridge.

## What worked and what didn't

Worked:
- **Seeding points only inside the SAM mask.** Points start on the hull and sails instead of the wake, and you can see it in the demo.
- **Carrying surviving points over when the tracker re-initializes.** This did more to cut down on new IDs than anything else.
- **Building boxes from the 8th/92nd percentile of the point cloud.** Min/max boxes blew up whenever a single point strayed, and the labels jittered.

Still open:
- **Small far-away boats** flicker around the confidence threshold. Once they're gone longer than max-age, they get a new ID. Lowering the threshold lets wave crests in, so better or fine-tuned training data is the real fix.
- **No camera-motion compensation.** Trajectories are in image space, so when the camera pans, every trail picks up that motion.
- **Long overlaps when boats cross** can swap IDs. Association only handles short merges right now.

## Reproduce

```bash
python scripts/download_sample.py
python scripts/benchmark.py --input data/samples/sample_open.mp4 --seconds 20 --skip-seconds 6
python scripts/run_video_neural.py --input data/samples/sample_open.mp4 --output media/neural_run.mp4 --side-by-side
python scripts/make_demo.py --input data/samples/sample_open.mp4 --output media/demo.mp4
pytest tests/ -q
```
