#!/usr/bin/env python3
"""Measure the configs RESULTS.md reports. Prints a markdown table row per run.

    python scripts/benchmark.py --input data/samples/sample_open.mp4 \
        --seconds 20 --skip-seconds 6
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
from marine_tracking.pipeline import Pipeline, PipelineConfig    # noqa: E402
from marine_tracking.segment import Segmenter                    # noqa: E402


def load_frames(path, skip_s, seconds):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(skip_s * fps))
    frames = []
    while len(frames) < int(seconds * fps):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--skip-seconds", type=float, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    frames, fps_in = load_frames(a.input, a.skip_seconds, a.seconds)
    print(f"{len(frames)} frames @ {fps_in:.0f} fps input, imgsz {a.imgsz}, "
          f"conf {a.conf}\n")
    rows = []

    def det():
        return Detector(conf=a.conf, classes=("boat",), imgsz=a.imgsz,
                        device=a.device)

    # warm-up (model load + first CUDA kernels out of the timing)
    d = det()
    d(frames[0])

    # ---- detection only -------------------------------------------------
    t0 = time.time()
    n_det = 0
    for f in frames:
        n_det += len(d(f))
    el = time.time() - t0
    rows.append(("detection only", "-", f"{len(frames)/el:.1f}",
                 f"{n_det/len(frames):.1f} boats/frame"))

    # ---- LK pipeline at N = 1 / 5 / 10 ----------------------------------
    for n in (1, 5, 10):
        pipe = Pipeline(det(), Segmenter(device=a.device),
                        PipelineConfig(redetect_every=n))
        ids = set()
        t0 = time.time()
        for f in frames:
            r = pipe.step(f)
            ids.update(t.track_id for t in r.tracks)
        el = time.time() - t0
        rows.append((f"detect+SAM+LK", str(n), f"{len(frames)/el:.1f}",
                     f"{len(ids)} ids"))

    # ---- neural pipeline -------------------------------------------------
    for seg_len in (24, 48):
        pipe = NeuralPipeline(det(), Segmenter(device=a.device),
                              NeuralConfig(redetect_every=seg_len),
                              device=a.device)
        ids = set()
        t0 = time.time()
        for s in range(0, len(frames) - 1, seg_len):
            seg = pipe.run_segment(frames[s:s + seg_len], s)
            for trks in seg.tracks_per_frame:
                ids.update(t.track_id for t in trks)
        el = time.time() - t0
        rows.append((f"detect+SAM+CoTracker3", str(seg_len),
                     f"{len(frames)/el:.1f}", f"{len(ids)} ids"))

    print("| config | re-detect every N | FPS | notes |")
    print("|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
