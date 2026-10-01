"""Unit tests for the association logic and the median-motion gate.

Synthetic data only - no models, no video, no cv2. These test the logic the
pipeline's correctness actually hangs on: who gets matched to whom, when an
ID dies, and which points get thrown out.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marine_tracking.track import (Track, associate, expire_tracks,  # noqa: E402
                                   iou_xyxy, median_motion_gate)


def box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], np.float32)


def track(tid, b, misses=0):
    return Track(track_id=tid, box=b, misses=misses)


# ------------------------------------------------------------------- iou

def test_iou_identical():
    assert iou_xyxy(box(0, 0, 10, 10), box(0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert iou_xyxy(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    v = iou_xyxy(box(0, 0, 10, 10), box(5, 0, 15, 10))
    assert abs(v - (50 / 150)) < 1e-6


# ------------------------------------------------------------- association

def test_match_by_iou():
    tracks = [track(1, box(0, 0, 10, 10)), track(2, box(100, 100, 120, 120))]
    dets = [box(101, 99, 121, 119), box(1, 1, 11, 11)]
    matches, un_t, un_d = associate(tracks, dets)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert un_t == [] and un_d == []


def test_no_double_assignment():
    # two tracks near one detection: only the better IoU gets it
    tracks = [track(1, box(0, 0, 10, 10)), track(2, box(2, 2, 12, 12))]
    dets = [box(1, 1, 11, 11)]
    matches, un_t, un_d = associate(tracks, dets)
    assert len(matches) == 1
    assert len(un_t) == 1 and un_d == []


def test_centroid_fallback_rescues_drifted_track():
    # zero IoU but centroids 30 px apart -> matched by the fallback
    tracks = [track(1, box(0, 0, 10, 10))]
    dets = [box(30, 0, 40, 10)]
    matches, un_t, un_d = associate(tracks, dets, iou_thr=0.3, dist_thr=80.0)
    assert matches == [(0, 0)]


def test_centroid_fallback_respects_distance_cap():
    tracks = [track(1, box(0, 0, 10, 10))]
    dets = [box(500, 500, 510, 510)]
    matches, un_t, un_d = associate(tracks, dets, dist_thr=80.0)
    assert matches == [] and un_t == [0] and un_d == [0]


def test_unmatched_detection_reported_for_new_track():
    tracks = [track(1, box(0, 0, 10, 10))]
    dets = [box(1, 1, 11, 11), box(300, 300, 320, 330)]
    matches, un_t, un_d = associate(tracks, dets)
    assert matches == [(0, 0)] and un_d == [1]


def test_id_stability_across_small_motion():
    # the same object drifting a few px per "frame" keeps its track slot
    t = track(7, box(50, 50, 90, 90))
    for i in range(1, 6):
        d = box(50 + 3 * i, 50 + 2 * i, 90 + 3 * i, 90 + 2 * i)
        matches, _, _ = associate([t], [d])
        assert matches == [(0, 0)]
        t.box = d


# ------------------------------------------------------------------ expiry

def test_expiry_respects_max_age():
    ts = [track(1, box(0, 0, 10, 10), misses=3),
          track(2, box(0, 0, 10, 10), misses=16)]
    kept = expire_tracks(ts, max_age=15)
    assert [t.track_id for t in kept] == [1]


def test_expiry_boundary_inclusive():
    ts = [track(1, box(0, 0, 10, 10), misses=15)]
    assert len(expire_tracks(ts, max_age=15)) == 1


# -------------------------------------------------------- median-motion gate

def test_gate_keeps_consistent_motion():
    prev = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
    new = prev + np.array([5, 2], np.float32)          # everyone moves (5,2)
    status = np.ones((4, 1), np.uint8)
    kept, mask = median_motion_gate(prev, new, status, tol=4.0)
    assert mask.sum() == 4 and len(kept) == 4


def test_gate_drops_outlier():
    prev = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
    new = prev + np.array([5, 2], np.float32)
    new[3] = prev[3] + np.array([25, -10], np.float32)  # wave-rider
    status = np.ones((4, 1), np.uint8)
    kept, mask = median_motion_gate(prev, new, status, tol=4.0)
    assert mask.sum() == 3 and not mask[3]


def test_gate_respects_lk_status():
    prev = np.array([[0, 0], [10, 0]], np.float32)
    new = prev + 1
    status = np.array([[1], [0]], np.uint8)             # LK lost point 1
    kept, mask = median_motion_gate(prev, new, status, tol=4.0)
    assert mask.tolist() == [True, False]


def test_gate_all_lost():
    prev = np.zeros((3, 2), np.float32)
    new = prev
    status = np.zeros((3, 1), np.uint8)
    kept, mask = median_motion_gate(prev, new, status)
    assert len(kept) == 0
