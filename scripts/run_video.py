#!/usr/bin/env python3
"""CLI: run the full pipeline over a video and write an annotated copy.

    python scripts/run_video.py --input clip.mp4 --output out.mp4
    python scripts/run_video.py --input clip.mp4 --output out.mp4 \
        --redetect-every 10 --side-by-side --conf 0.25

Prints a timing summary at the end - those numbers go in RESULTS.md verbatim,
never rounded up.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marine_tracking.detect import Detector            # noqa: E402
from marine_tracking.pipeline import Pipeline, PipelineConfig  # noqa: E402
from marine_tracking.segment import Segmenter          # noqa: E402
from marine_tracking.visualize import draw_tracks, side_by_side  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--sam-weights", default="sam2_b.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="boat",
                    help="comma-separated COCO names; 'all' disables filtering")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--redetect-every", type=int, default=5)
    ap.add_argument("--max-age", type=int, default=15)
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--device", default=None, help="e.g. 0 for CUDA, cpu")
    a = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        print(f"cannot open {a.input}")
        return 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    classes = None if a.classes == "all" else tuple(
        s.strip() for s in a.classes.split(","))
    pipe = Pipeline(
        Detector(a.weights, conf=a.conf, classes=classes, imgsz=a.imgsz,
                 device=a.device),
        Segmenter(a.sam_weights, device=a.device),
        PipelineConfig(redetect_every=a.redetect_every, max_age=a.max_age))

    writer = None
    t0 = time.time()
    frames = 0
    ids_seen: set[int] = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe.step(frame)
        ids_seen.update(t.track_id for t in res.tracks)
        vis = draw_tracks(frame, res.tracks,
                          res.masks if res.redetected else None)
        out_frame = side_by_side(frame, vis) if a.side_by_side else vis
        if writer is None:
            writer = cv2.VideoWriter(
                a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                (out_frame.shape[1], out_frame.shape[0]))
        writer.write(out_frame)
        frames += 1
        if frames % 50 == 0:
            el = time.time() - t0
            print(f"  {frames}/{n_frames}  {frames / el:.1f} fps")

    cap.release()
    if writer:
        writer.release()
    el = time.time() - t0
    print(f"\n  frames {frames}   wall {el:.1f}s   {frames / el:.2f} fps   "
          f"redetect_every={a.redetect_every}   distinct ids {len(ids_seen)}")
    print(f"  wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
