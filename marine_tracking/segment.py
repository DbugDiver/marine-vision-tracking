"""SAM segmentation wrapper: frame + boxes -> one boolean mask per box.

Why segment at all: on water, a detection box is mostly NOT the object - it
is waves, wake and glare around a hull. Points seeded in the whole box latch
onto water texture and drift; points seeded inside the mask stay on the
object. The mask is the point tracker's permission slip.

Uses Ultralytics' SAM interface so detection and segmentation share one
dependency. Default weights are the small SAM2 checkpoint; mobile_sam is a
faster drop-in for CPU runs. Both are public weights.
"""
from __future__ import annotations

import numpy as np


class Segmenter:
    def __init__(self, weights: str = "sam2_b.pt", device: str | None = None):
        self.weights = weights
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM
            self._model = SAM(self.weights)
        return self._model

    def __call__(self, frame_bgr: np.ndarray,
                 boxes_xyxy: list[np.ndarray]) -> list[np.ndarray]:
        """-> list of bool masks (H, W), aligned with boxes_xyxy.

        A failed mask (empty result) falls back to the filled box, so the
        pipeline degrades to box-seeded tracking rather than dropping the
        object - degrade gracefully, never disappear.
        """
        h, w = frame_bgr.shape[:2]
        if not boxes_xyxy:
            return []
        model = self._load()
        res = model.predict(frame_bgr, bboxes=[b.tolist() for b in boxes_xyxy],
                            device=self.device, verbose=False)[0]
        masks: list[np.ndarray] = []
        got = (res.masks.data.cpu().numpy().astype(bool)
               if res.masks is not None else np.zeros((0, h, w), bool))
        for i, box in enumerate(boxes_xyxy):
            if i < len(got) and got[i].any():
                m = got[i]
                if m.shape != (h, w):
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                masks.append(m)
            else:
                m = np.zeros((h, w), bool)
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                m[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = True
                masks.append(m)
        return masks
