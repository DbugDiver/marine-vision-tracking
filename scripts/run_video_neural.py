#!/usr/bin/env python3
"""CLI: YOLO -> SAM -> CoTracker3 over a video, box-free rendering.

    python scripts/run_video_neural.py --input data/samples/sample_open.mp4 \
        --output media/neural_run.mp4 --side-by-side --max-seconds 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marine_tracking.detect import Detector                      # noqa: E402
from marine_tracking.neural_pipeline import (NeuralConfig,       # noqa: E402
                                             NeuralPipeline)
from marine_tracking.segment import Segmenter                    # noqa: E402
from marine_tracking.visualize import draw_tracks, side_by_side  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--sam-weights", default="sam2_b.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="boat")
    ap.add_argument("--segment-frames", type=int, default=24)
    ap.add_argument("--points-per-obj", type=int, default=24)
    ap.add_argument("--max-seconds", type=float, default=0, help="0 = all")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        print(f"cannot open {a.input}")
        return 1
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = int(a.max_seconds * fps_in) if a.max_seconds else 10 ** 9

    classes = None if a.classes == "all" else tuple(
        s.strip() for s in a.classes.split(","))
    pipe = NeuralPipeline(
        Detector(a.weights, conf=a.conf, classes=classes, device=a.device),
        Segmenter(a.sam_weights, device=a.device),
        NeuralConfig(redetect_every=a.segment_frames,
                     points_per_obj=a.points_per_obj),
        device=a.device)

    writer = None
    t0 = time.time()
    frames_done = 0
    ids = set()
    buf: list = []
    index = 0
    while frames_done < limit:
        ok, frame = cap.read()
        if not ok and not buf:
            break
        if ok:
            buf.append(frame)
        if len(buf) == a.segment_frames or (not ok and buf):
            seg = pipe.run_segment(buf, index)
            for f_i, (fr, trks) in enumerate(zip(seg.frames,
                                                 seg.tracks_per_frame)):
                ids.update(t.track_id for t in trks)
                vis = draw_tracks(fr, trks,
                                  seg.masks if f_i == 0 else None,
                                  boxes=False)
                out = side_by_side(fr, vis) if a.side_by_side else vis
                if writer is None:
                    writer = cv2.VideoWriter(
                        a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in,
                        (out.shape[1], out.shape[0]))
                writer.write(out)
                frames_done += 1
            index += len(buf)
            buf = []
            el = time.time() - t0
            print(f"  {frames_done} frames  {frames_done / el:.1f} fps")
        if not ok:
            break

    cap.release()
    if writer:
        writer.release()
    el = time.time() - t0
    print(f"\n  frames {frames_done}   wall {el:.1f}s   "
          f"{frames_done / el:.2f} fps   segment={a.segment_frames}   "
          f"distinct ids {len(ids)}")
    print(f"  wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
