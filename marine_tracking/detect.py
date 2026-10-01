"""YOLO detection wrapper: frame -> boxes, classes, scores.

Keeps the Ultralytics dependency behind one class so the rest of the pipeline
only sees plain numpy. Default weights are the public COCO-pretrained nano
model; on water the relevant COCO class is 'boat' (id 8), but the filter is
configurable because buoys and paddleboards show up as other classes or not
at all, and a fine-tune may come later.

Thresholds here are DELIBERATELY unset in code and chosen per run: the right
confidence floor depends on the footage (glare, target size, camera motion),
so it is a CLI argument, and the measured trade-off goes in RESULTS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One detection in pixel coordinates (x1, y1, x2, y2)."""
    box: np.ndarray          # float32 [4], xyxy
    score: float
    cls_id: int
    cls_name: str

    @property
    def centroid(self) -> tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0,
                float(self.box[1] + self.box[3]) / 2.0)


class Detector:
    """Thin wrapper over an Ultralytics YOLO model.

    Loading is lazy so unit tests and --help never pay the torch import cost.
    """

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.25,
                 classes: tuple[str, ...] | None = ("boat",),
                 imgsz: int = 640, device: str | None = None):
        self.weights = weights
        self.conf = conf
        self.classes = classes
        self.imgsz = imgsz
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        model = self._load()
        res = model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                            device=self.device, verbose=False)[0]
        names = res.names
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls_id = int(b.cls.item())
            name = names.get(cls_id, str(cls_id))
            if self.classes is not None and name not in self.classes:
                continue
            out.append(Detection(
                box=b.xyxy[0].cpu().numpy().astype(np.float32),
                score=float(b.conf.item()),
                cls_id=cls_id,
                cls_name=name,
            ))
        return out
