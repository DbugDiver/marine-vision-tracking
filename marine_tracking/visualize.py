"""Drawing: boxes, mask contours, IDs, trajectories, side-by-side composer."""
from __future__ import annotations

import numpy as np

# one fixed palette, id -> colour, stable across frames
_PALETTE = [(66, 135, 245), (52, 199, 89), (255, 149, 0), (175, 82, 222),
            (255, 59, 48), (90, 200, 250), (255, 204, 0), (88, 86, 214)]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(frame_bgr: np.ndarray, tracks, masks=None,
                trail: int = 50) -> np.ndarray:
    import cv2
    out = frame_bgr.copy()
    if masks:
        overlay = out.copy()
        for i, m in enumerate(masks):
            overlay[m] = (overlay[m] * 0.5 +
                          np.array(_PALETTE[i % len(_PALETTE)]) * 0.5
                          ).astype(np.uint8)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for t in tracks:
        c = _color(t.track_id)
        x1, y1, x2, y2 = [int(round(v)) for v in t.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"#{t.track_id} {t.cls_name}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        if t.points is not None:
            for px, py in t.points:
                cv2.circle(out, (int(px), int(py)), 2, c, -1)
        tr = t.trajectory[-trail:]
        for a, b in zip(tr, tr[1:]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     c, 2)
    return out


def side_by_side(raw_bgr: np.ndarray, tracked_bgr: np.ndarray,
                 max_width: int = 1280) -> np.ndarray:
    """raw | tracked, scaled to a shareable width (the demo GIF format)."""
    import cv2
    pair = np.hstack([raw_bgr, tracked_bgr])
    if pair.shape[1] > max_width:
        s = max_width / pair.shape[1]
        pair = cv2.resize(pair, (max_width, int(pair.shape[0] * s)))
    return pair
