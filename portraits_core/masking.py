"""Background estimation and foreground mask extraction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def estimate_background_bgr(bgr: np.ndarray) -> np.ndarray:
    """Estimate the light background color from the image border."""
    h, w = bgr.shape[:2]
    bw = max(5, w // 50)
    bh = max(5, h // 50)
    border = np.concatenate(
        [
            bgr[:bh, :, :].reshape(-1, 3),
            bgr[-bh:, :, :].reshape(-1, 3),
            bgr[:, :bw, :].reshape(-1, 3),
            bgr[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border, axis=0).astype(np.uint8)


def make_clean_mask(
    bgr: np.ndarray,
    lab_threshold: float = 30.0,
    tiny_height: int = 35,
    tiny_area: int = 800,
) -> np.ndarray:
    """
    Build a foreground mask while suppressing tiny text/noise components.

    LAB color distance is more stable than RGB distance for light-background
    fashion thumbnails. tiny_height/tiny_area scale with the image via the
    caller so this generalizes past one fixed source resolution.
    """
    bg = estimate_background_bgr(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    bg_lab = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2LAB).astype(np.int16)[0, 0]
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    raw = (dist > lab_threshold).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    clean = np.zeros_like(raw)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if h < tiny_height and area < tiny_area:
            continue
        clean[labels == i] = 255
    return clean


def scaled_noise_thresholds(image_h: int, image_w: int) -> tuple[int, int]:
    """
    Derive tiny-component noise thresholds from image size instead of the
    fixed 35px / 800px^2 that only suited one reference resolution.
    """
    tiny_height = max(12, int(round(image_h * 0.018)))
    tiny_area = max(200, int(round(image_h * image_w * 0.00035)))
    return tiny_height, tiny_area
