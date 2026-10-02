"""Point tracking and ID association.

Two separate pieces:

1. PointTracker - Lucas-Kanade optical flow on points seeded inside each
   object's mask, with a median-motion gate: a point whose displacement
   disagrees with the object's median displacement is dropped (on water the
   usual offender is a point that latched onto a wave crest or glare, which
   moves with the water, not the boat).

2. associate() - pure-function matching of new detections to existing tracks
   by IoU with a centroid-distance fallback, plus a max-age rule. Pure and
   numpy-only so it unit-tests on synthetic data with no models loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------- geometry

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


# -------------------------------------------------------------------- track

@dataclass
class Track:
    track_id: int
    box: np.ndarray                    # xyxy, float32
    cls_name: str = "object"
    points: np.ndarray | None = None   # (N, 2) float32, current LK points
    misses: int = 0                    # consecutive frames without a match
    age: int = 0                       # frames since creation
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0,
                float(self.box[1] + self.box[3]) / 2.0)

    def record(self):
        self.trajectory.append(self.centroid)


# -------------------------------------------------------------- association

def associate(tracks: list[Track], det_boxes: list[np.ndarray],
              iou_thr: float = 0.3, dist_thr: float = 80.0,
              max_age: int = 15) -> tuple[list[tuple[int, int]],
                                          list[int], list[int]]:
    """Match detections to tracks. -> (matches, unmatched_tracks,
    unmatched_dets), as index pairs/lists.

    Greedy best-IoU first (strongest signal), then a centroid-distance pass
    for the leftovers (rescues re-detections whose box shifted too much for
    IoU, common after a few seconds of coasting). Callers expire a track when
    `misses > max_age`; expire_tracks() below implements that rule.
    """
    matches: list[tuple[int, int]] = []
    free_t = set(range(len(tracks)))
    free_d = set(range(len(det_boxes)))

    pairs = sorted(((iou_xyxy(tracks[t].box, det_boxes[d]), t, d)
                    for t in free_t for d in free_d), reverse=True)
    for score, t, d in pairs:
        if score < iou_thr:
            break
        if t in free_t and d in free_d:
            matches.append((t, d))
            free_t.discard(t)
            free_d.discard(d)

    # centroid fallback for what IoU could not claim
    def cdist(t: int, d: int) -> float:
        tc = tracks[t].centroid
        db = det_boxes[d]
        dc = ((db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0)
        return float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))

    pairs2 = sorted((cdist(t, d), t, d) for t in free_t for d in free_d)
    for dist, t, d in pairs2:
        if dist > dist_thr:
            break
        if t in free_t and d in free_d:
            matches.append((t, d))
            free_t.discard(t)
            free_d.discard(d)

    return matches, sorted(free_t), sorted(free_d)


def expire_tracks(tracks: list[Track], max_age: int) -> list[Track]:
    """Drop tracks that have gone unmatched for more than max_age frames."""
    return [t for t in tracks if t.misses <= max_age]


# ----------------------------------------------------------- point tracking

def seed_points(mask: np.ndarray, max_points: int = 60,
                quality: float = 0.01, min_dist: int = 7,
                gray: np.ndarray | None = None) -> np.ndarray:
    """Corner points inside the mask. -> (N, 2) float32, possibly empty."""
    import cv2
    m8 = mask.astype(np.uint8) * 255
    if gray is None:
        gray = m8  # degenerate but keeps the signature usable in tests
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_points,
                                  qualityLevel=quality, minDistance=min_dist,
                                  mask=m8)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def median_motion_gate(prev_pts: np.ndarray, new_pts: np.ndarray,
                       status: np.ndarray, tol: float = 4.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Keep points whose displacement agrees with the median displacement.

    tol is in pixels of disagreement with the median motion vector. Returns
    (kept_new_pts, kept_mask_over_input). Pure numpy -> unit-testable.
    """
    ok = status.reshape(-1).astype(bool)
    if ok.sum() == 0:
        return np.zeros((0, 2), np.float32), ok
    disp = new_pts - prev_pts
    med = np.median(disp[ok], axis=0)
    err = np.linalg.norm(disp - med, axis=1)
    keep = ok & (err <= tol)
    return new_pts[keep].astype(np.float32), keep


class PointTracker:
    """LK flow between consecutive frames for one object's point set."""

    def __init__(self, win: int = 21, levels: int = 3, tol_px: float = 4.0):
        self.win = win
        self.levels = levels
        self.tol_px = tol_px

    def step(self, prev_gray: np.ndarray, gray: np.ndarray,
             pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (new_pts, median_shift[2]). Empty input -> empty output."""
        import cv2
        if len(pts) == 0:
            return pts, np.zeros(2, np.float32)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts.reshape(-1, 1, 2), None,
            winSize=(self.win, self.win), maxLevel=self.levels)
        new_pts = new_pts.reshape(-1, 2)
        kept, keep_mask = median_motion_gate(pts, new_pts, status, self.tol_px)
        if keep_mask.sum() == 0:
            return np.zeros((0, 2), np.float32), np.zeros(2, np.float32)
        shift = np.median(new_pts[keep_mask] - pts[keep_mask], axis=0)
        return kept, shift.astype(np.float32)
