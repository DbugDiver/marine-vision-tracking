#!/usr/bin/env python3
"""Compose the staged demo video: cover -> YOLO -> SAM -> bootstrap -> tracking.

Each pipeline stage gets its own segment of the clip, labelled, with the
visualization that explains it: raw detections for YOLO, masks for SAM, the
seeded query points for bootstrapping, then full neural tracking with trails.
Both panes are zoomed to the action region (union of detections, padded).

    python scripts/make_demo.py --input data/samples/sample_open.mp4 \
        --output media/demo.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marine_tracking.detect import Detector                      # noqa: E402
from marine_tracking.neural_pipeline import (NeuralConfig,       # noqa: E402
                                             NeuralPipeline,
                                             _sample_mask_points)
from marine_tracking.segment import Segmenter                    # noqa: E402
from marine_tracking.visualize import draw_tracks                # noqa: E402

PANE_W, PANE_H = 960, 540
BAND = 46                         # label band height, px


def banner(img, text, sub=""):
    import cv2
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], BAND), (18, 18, 18), -1)
    cv2.putText(out, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (out.shape[1] - 12 - 9 * len(sub), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)
    return out


def crop_zoom(img, box):
    import cv2
    x1, y1, x2, y2 = box
    return cv2.resize(img[y1:y2, x1:x2], (PANE_W, PANE_H))


def action_box(frames, detector, W, H):
    """Union of detections over a few probe frames, padded, 16:9."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for f in frames[:: max(1, len(frames) // 6)]:
        for d in detector(f):
            xs1.append(d.box[0]); ys1.append(d.box[1])
            xs2.append(d.box[2]); ys2.append(d.box[3])
    if not xs1:
        return 0, 0, W, H
    x1, y1 = max(min(xs1) - 60, 0), max(min(ys1) - 60, 0)
    x2, y2 = min(max(xs2) + 60, W), min(max(ys2) + 60, H)
    # expand to 16:9 around the centre
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    if w / h > 16 / 9:
        h = w * 9 / 16
    else:
        w = h * 16 / 9
    x1, x2 = int(max(cx - w / 2, 0)), int(min(cx + w / 2, W))
    y1, y2 = int(max(cy - h / 2, 0)), int(min(cy + h / 2, H))
    return x1, y1, x2, y2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cover-seconds", type=float, default=1.2)
    ap.add_argument("--stage-seconds", type=float, default=4.0)
    ap.add_argument("--skip-seconds", type=float, default=6.0,
                    help="skip the source's own intro/title card")
    a = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(a.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.skip_seconds * fps))
    frames = []
    need = int(fps * (3 * a.stage_seconds + 14))
    while len(frames) < need:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    H, W = frames[0].shape[:2]

    det = Detector(conf=a.conf, classes=("boat",), device=a.device)
    seg = Segmenter(device=a.device)
    zoom = action_box(frames, det, W, H)
    print("zoom box:", zoom)

    vw = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (2 * PANE_W, PANE_H))

    def emit(raw, vis, label, sub=""):
        pair = np.hstack([crop_zoom(raw, zoom), crop_zoom(vis, zoom)])
        vw.write(banner(pair, label, sub))

    n_stage = int(fps * a.stage_seconds)
    i = 0

    # ---- cover: short, just a card ------------------------------------
    cover = np.full((PANE_H, 2 * PANE_W, 3), 16, np.uint8)
    for line, y, sc in [("MARINE VISION TRACKING", 240, 1.6),
                        ("YOLO detection  >  SAM segmentation  >  "
                         "neural point tracking", 300, 0.8)]:
        cv2.putText(cover, line, (2 * PANE_W // 2 - int(9 * sc * len(line) / 2),
                    y), cv2.FONT_HERSHEY_SIMPLEX, sc, (240, 240, 240), 2,
                    cv2.LINE_AA)
    for _ in range(int(fps * a.cover_seconds)):
        vw.write(cover)

    # ---- stage 1: YOLO ------------------------------------------------
    for k in range(n_stage):
        f = frames[i + k]
        vis = f.copy()
        for d in det(f):
            x1, y1, x2, y2 = [int(v) for v in d.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (66, 135, 245), 2)
            cv2.putText(vis, f"boat {d.score:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 135, 245), 2,
                        cv2.LINE_AA)
        emit(f, vis, "STAGE 1 / RUNNING YOLO",
             "per-frame detection, conf shown")
    i += n_stage

    # ---- stage 2: SAM -------------------------------------------------
    masks, last_det_frame = [], -99
    for k in range(n_stage):
        f = frames[i + k]
        if k - last_det_frame >= 5:
            boxes = [d.box for d in det(f)]
            masks = seg(f, boxes) if boxes else []
            last_det_frame = k
        vis = f.copy()
        if masks:
            overlay = vis.copy()
            pal = [(66, 135, 245), (52, 199, 89), (255, 149, 0),
                   (175, 82, 222), (255, 59, 48), (90, 200, 250)]
            for m_i, m in enumerate(masks):
                overlay[m] = (overlay[m] * 0.4 +
                              np.array(pal[m_i % 6]) * 0.6).astype(np.uint8)
            vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
        emit(f, vis, "STAGE 2 / RUNNING SAM",
             "one mask per detection")
    i += n_stage

    # ---- stage 3: bootstrap points (brief) -----------------------------
    f = frames[i]
    boxes = [d.box for d in det(f)]
    masks = seg(f, boxes) if boxes else []
    vis = f.copy()
    for m in masks:
        for px, py in _sample_mask_points(m, 24):
            cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255), -1)
    for _ in range(int(fps * 1.5)):
        emit(f, vis, "STAGE 3 / BOOTSTRAPPING TRACK POINTS",
             "query points seeded inside each mask")

    # ---- stage 4: neural tracking -------------------------------------
    pipe = NeuralPipeline(det, seg, NeuralConfig(), device=a.device)
    rest = frames[i:]
    for s in range(0, len(rest) - 1, 24):
        chunk = rest[s:s + 24]
        segr = pipe.run_segment(chunk, i + s)
        for f_i, (fr, trks) in enumerate(zip(segr.frames,
                                             segr.tracks_per_frame)):
            vis = draw_tracks(fr, trks, None, boxes=False)
            emit(fr, vis, "STAGE 4 / NEURAL POINT TRACKING",
                 "CoTracker3: per-point identity, trails")
        print(f"  tracking {s + len(chunk)}/{len(rest)}")

    vw.release()
    print("wrote", a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
