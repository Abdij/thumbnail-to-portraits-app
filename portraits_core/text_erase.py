"""Optional inpainting of small heading text (e.g. WOMEN/KIDS labels) above rows."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .masking import estimate_background_bgr


def erase_heading_text_bgr(
    bgr: np.ndarray,
    row_cuts: Sequence[int],
    band_frac: float = 0.16,
) -> np.ndarray:
    """
    Inpaint small dark heading text near the top of each row.

    Thresholds are derived from the image itself (background luminance, row
    height, image width) rather than fixed pixel constants, so this is not
    tied to one heading font size or one source resolution. It is still an
    opt-in step: callers should only apply this when they expect a heading
    band, since any dark small blob (a button, a belt buckle) can otherwise
    be mistaken for text.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg_bgr = estimate_background_bgr(bgr)
    bg_gray = float(cv2.cvtColor(np.uint8([[bg_bgr]]), cv2.COLOR_BGR2GRAY)[0, 0])
    # Heading text is dark relative to the light background; scale the cutoff
    # off the actual background brightness instead of a fixed value like 165.
    gray_threshold = max(60, int(round(bg_gray * 0.72)))

    erase = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(row_cuts) - 1):
        y1, y2 = row_cuts[i], row_cuts[i + 1]
        row_h = y2 - y1
        band_h = int(max(row_h * 0.05, min(row_h * band_frac, row_h * 0.3)))
        band_bottom = min(y2, y1 + band_h)
        region = gray[y1:band_bottom, :]
        dark = (region < gray_threshold).astype(np.uint8) * 255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)

        # Text glyphs/underlines are short and narrow relative to the band and
        # image width; garment edges that happen to be dark are much larger.
        max_h = max(10, int(round(band_h * 0.85)))
        max_area = max(150, int(round(band_h * w * 0.01)))
        max_w = max(20, int(round(w * 0.22)))

        for label_id in range(1, num):
            x, y, bw, bh, area = stats[label_id]
            if bh <= max_h and area <= max_area and bw <= max_w:
                erase[y1:band_bottom, :][labels == label_id] = 255

    if not erase.any():
        return bgr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erase = cv2.dilate(erase, kernel, iterations=1)
    return cv2.inpaint(bgr, erase, 3, cv2.INPAINT_TELEA)
