"""Glue: detect -> segment -> track per frame, with a re-detection cadence.

Main idea: YOLO + SAM every frame is slow; LK flow is
cheap. So run the heavy models every `redetect_every` frames and coast on
point tracking in between, shifting each track's box by its points' median
motion. The FPS-vs-accuracy trade-off for N = 1 / 5 / 10 is measured
in RESULTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Detector
from .segment import Segmenter
from .track import (PointTracker, Track, associate, expire_tracks,
                    seed_points)


@dataclass
class PipelineConfig:
    redetect_every: int = 5
    iou_thr: float = 0.3
    dist_thr: float = 80.0
    max_age: int = 15          # frames a track may coast unmatched
    max_points: int = 60
    min_points: int = 5        # fewer than this -> track asks for re-detect


@dataclass
class FrameResult:
    index: int
    tracks: list[Track] = field(default_factory=list)
    detections: int = 0
    redetected: bool = False
    masks: list[np.ndarray] = field(default_factory=list)


class Pipeline:
    def __init__(self, detector: Detector, segmenter: Segmenter,
                 cfg: PipelineConfig | None = None):
        self.detector = detector
        self.segmenter = segmenter
        self.cfg = cfg or PipelineConfig()
        self.tracks: list[Track] = []
        self.point_tracker = PointTracker()
        self._next_id = 1
        self._prev_gray: np.ndarray | None = None
        self._frame_i = -1

    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _starved(self) -> bool:
        """A live track with too few points forces an early re-detect."""
        return any(t.points is not None and len(t.points) < self.cfg.min_points
                   for t in self.tracks)

    def step(self, frame_bgr: np.ndarray) -> FrameResult:
        import cv2
        self._frame_i += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        res = FrameResult(index=self._frame_i)

        # 1. coast every live track on optical flow
        if self._prev_gray is not None:
            for t in self.tracks:
                if t.points is None or len(t.points) == 0:
                    t.misses += 1
                    continue
                pts, shift = self.point_tracker.step(self._prev_gray, gray,
                                                     t.points)
                t.points = pts
                if len(pts) > 0:
                    t.box = t.box + np.array([shift[0], shift[1],
                                              shift[0], shift[1]],
                                             np.float32)
                else:
                    t.misses += 1

        # 2. heavy models on cadence, on starvation, and on the first frame
        due = (self._frame_i % max(1, self.cfg.redetect_every) == 0
               or self._starved() or self._prev_gray is None)
        if due:
            dets = self.detector(frame_bgr)
            res.detections = len(dets)
            res.redetected = True
            boxes = [d.box for d in dets]
            matches, unmatched_t, unmatched_d = associate(
                self.tracks, boxes, self.cfg.iou_thr, self.cfg.dist_thr,
                self.cfg.max_age)
            masks = self.segmenter(frame_bgr, boxes) if boxes else []
            res.masks = masks

            for t_i, d_i in matches:
                t = self.tracks[t_i]
                t.box = dets[d_i].box.copy()
                t.cls_name = dets[d_i].cls_name
                t.misses = 0
                t.points = seed_points(masks[d_i], self.cfg.max_points,
                                       gray=gray)
            for t_i in unmatched_t:
                self.tracks[t_i].misses += 1
            for d_i in unmatched_d:
                t = Track(track_id=self._new_id(), box=dets[d_i].box.copy(),
                          cls_name=dets[d_i].cls_name)
                t.points = seed_points(masks[d_i], self.cfg.max_points,
                                       gray=gray)
                self.tracks.append(t)

        # 3. bookkeeping
        self.tracks = expire_tracks(self.tracks, self.cfg.max_age)
        for t in self.tracks:
            t.age += 1
            t.record()
        res.tracks = list(self.tracks)
        self._prev_gray = gray
        return res
