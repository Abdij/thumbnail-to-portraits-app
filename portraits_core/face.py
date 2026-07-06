"""Optional face-aware re-centering, using OpenCV's bundled Haar cascade."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_CASCADE_PATH = Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
    return _cascade


def detect_faces(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes for detected faces in the full image."""
    if not _CASCADE_PATH.exists():
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24)
    )
    return [tuple(int(v) for v in f) for f in faces]


def recenter_on_face(
    box: Tuple[int, int, int, int],
    faces: List[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    head_room_frac: float = 0.35,
) -> Tuple[int, int, int, int]:
    """
    If a face falls inside `box`, shift the crop so the face sits in the
    upper third with head-room above it, instead of centered on the whole
    detected garment silhouette. Returns `box` unchanged if no face matches.
    """
    x1, y1, x2, y2 = box
    image_w, image_h = image_size
    candidates = [
        f for f in faces
        if f[0] >= x1 - 5 and f[1] >= y1 - 5 and f[0] + f[2] <= x2 + 5 and f[1] + f[3] <= y2 + 5
    ]
    if not candidates:
        return box

    fx, fy, fw, fh = max(candidates, key=lambda f: f[2] * f[3])
    box_h = y2 - y1
    box_w = x2 - x1
    face_cy = fy + fh / 2
    face_cx = fx + fw / 2

    desired_top = face_cy - box_h * head_room_frac
    new_y1 = int(round(max(0, min(desired_top, image_h - box_h))))
    new_y2 = new_y1 + box_h

    desired_left = face_cx - box_w / 2
    new_x1 = int(round(max(0, min(desired_left, image_w - box_w))))
    new_x2 = new_x1 + box_w

    return new_x1, new_y1, new_x2, new_y2
