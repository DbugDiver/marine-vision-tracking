"""Neural point-tracking pipeline: YOLO -> SAM -> CoTracker3 (TAPIR-family).

Where the base pipeline tracks points with Lucas-Kanade frame pairs, this one
uses a learned dense point tracker (CoTracker3, public Meta weights, loaded
via torch.hub). TAPIR-family trackers see a temporal window, not a frame
pair, so they survive glare flicker and brief occlusion that kills LK - the
exact failure modes of water video.

Integration shape: the online model consumes frames in strides of
`model.step`; queries (points to track) can only be declared at init. So the
pipeline re-initialises the tracker at every re-detection boundary: YOLO+SAM
give fresh masks, points are re-seeded inside the masks (plus survivors from
the previous segment), and the tracker runs the next segment. Object
identity is carried by WHICH QUERY BELONGS TO WHICH TRACK, not by box IoU -
the per-point id map is the association.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Detector
from .segment import Segmenter
from .track import Track, associate, expire_tracks


def _sample_mask_points(mask: np.ndarray, k: int = 24) -> np.ndarray:
    """Uniformly sample up to k (x, y) points inside a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    idx = np.linspace(0, len(xs) - 1, min(k, len(xs))).astype(int)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


@dataclass
class NeuralConfig:
    redetect_every: int = 48      # frames per segment (tracker re-init)
    points_per_obj: int = 24
    iou_thr: float = 0.2
    dist_thr: float = 120.0
    max_age: int = 3              # segments, not frames
    vis_thr: float = 0.5          # CoTracker visibility to count a point
    carry_points: bool = True     # unmatched-but-alive tracks keep their own
                                  # points as queries across a re-init: a boat
                                  # YOLO misses at a segment boundary keeps its
                                  # identity instead of minting a new id
    box_pct: float = 8.0          # percentile box from points (min/max lets a
                                  # single stray point balloon the box)


@dataclass
class SegmentResult:
    start: int
    frames: list[np.ndarray] = field(default_factory=list)      # BGR
    tracks_per_frame: list[list[Track]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)       # at seg start


class NeuralPipeline:
    def __init__(self, detector: Detector, segmenter: Segmenter,
                 cfg: NeuralConfig | None = None, device: str = "cuda"):
        self.detector = detector
        self.segmenter = segmenter
        self.cfg = cfg or NeuralConfig()
        self.device = device
        self._model = None
        self.tracks: list[Track] = []
        self._next_id = 1

    def _tracker(self):
        if self._model is None:
            import torch
            self._model = torch.hub.load("facebookresearch/co-tracker",
                                         "cotracker3_offline",
                                         trust_repo=True).to(self.device)
        return self._model

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # ------------------------------------------------------------------

    def _detect_and_assign(self, frame_bgr) -> tuple[list, list[np.ndarray]]:
        """YOLO+SAM on one frame; update self.tracks via association."""
        dets = self.detector(frame_bgr)
        boxes = [d.box for d in dets]
        masks = self.segmenter(frame_bgr, boxes) if boxes else []
        matches, unmatched_t, unmatched_d = associate(
            self.tracks, boxes, self.cfg.iou_thr, self.cfg.dist_thr)
        # Build the id -> detection map BEFORE expiry: `matches` indexes the
        # CURRENT tracks list, and expire_tracks() renumbers it. The first
        # version of this method mapped after expiry and crashed on the first
        # clip where a track actually expired.
        det_mask_of_track: dict[int, int] = {}
        for t_i, d_i in matches:
            t = self.tracks[t_i]
            t.box = dets[d_i].box.copy()
            t.cls_name = dets[d_i].cls_name
            t.misses = 0
            det_mask_of_track[t.track_id] = d_i
        for t_i in unmatched_t:
            self.tracks[t_i].misses += 1
        for d_i in unmatched_d:
            t = Track(track_id=self._new_id(), box=dets[d_i].box.copy(),
                      cls_name=dets[d_i].cls_name)
            self.tracks.append(t)
            det_mask_of_track[t.track_id] = d_i
        self.tracks = expire_tracks(self.tracks, self.cfg.max_age)
        live = {t.track_id for t in self.tracks}
        det_mask_of_track = {k: v for k, v in det_mask_of_track.items()
                             if k in live}
        return det_mask_of_track, masks

    def run_segment(self, frames_bgr: list[np.ndarray],
                    start_index: int) -> SegmentResult:
        """Detect on the first frame, then neural-track through the segment."""
        import torch
        res = SegmentResult(start=start_index, frames=frames_bgr)
        first = frames_bgr[0]
        det_mask_of_track, masks = self._detect_and_assign(first)
        res.masks = masks

        # queries: per live track, points inside its mask on frame 0 - and
        # for tracks the detector MISSED this boundary, their surviving points
        # from the previous segment, so identity outlives a detection gap
        owners: list[int] = []
        pts: list[np.ndarray] = []
        for t in self.tracks:
            d_i = det_mask_of_track.get(t.track_id)
            if d_i is not None and d_i < len(masks):
                p = _sample_mask_points(masks[d_i], self.cfg.points_per_obj)
            elif (self.cfg.carry_points and t.points is not None
                  and len(t.points) >= 3):
                p = t.points.astype(np.float32)
            else:
                continue
            owners += [t.track_id] * len(p)
            pts.append(p)
        if not pts:
            res.tracks_per_frame = [list(self.tracks) for _ in frames_bgr]
            return res
        q = np.concatenate(pts, 0)                       # (N, 2)
        queries = torch.tensor(
            np.concatenate([np.zeros((len(q), 1), np.float32), q], 1),
            device=self.device)[None]                    # (1, N, 3) t,x,y

        video = torch.tensor(
            np.stack([f[..., ::-1].copy() for f in frames_bgr]),
            device=self.device).permute(0, 3, 1, 2)[None].float()
        with torch.no_grad():
            tr, vis = self._tracker()(video, queries=queries)
        tr = tr[0].cpu().numpy()                         # (T, N, 2)
        vis = vis[0].cpu().numpy() > self.cfg.vis_thr    # (T, N)

        owner_arr = np.array(owners)
        by_id = {t.track_id: t for t in self.tracks}
        for f_i in range(len(frames_bgr)):
            for tid in np.unique(owner_arr):
                sel = (owner_arr == tid) & vis[f_i]
                t = by_id.get(int(tid))
                if t is None:
                    continue
                if sel.sum() >= 3:
                    p = tr[f_i][sel]
                    t.points = p.astype(np.float32)
                    lo = np.percentile(p, self.cfg.box_pct, axis=0)
                    hi = np.percentile(p, 100 - self.cfg.box_pct, axis=0)
                    t.box = np.array([lo[0], lo[1], hi[0], hi[1]], np.float32)
                t.record()
            res.tracks_per_frame.append(
                [Track(t.track_id, t.box.copy(), t.cls_name,
                       None if t.points is None else t.points.copy(),
                       t.misses, t.age, list(t.trajectory))
                 for t in self.tracks])
        return res
