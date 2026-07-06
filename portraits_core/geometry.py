"""Row/column layout detection: turning a mask into a grid of crop boxes."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .masking import make_clean_mask, scaled_noise_thresholds

DEFAULT_ROW_CUTS = "0,0.2734,0.5156,0.7471,1"

#: Passing this instead of an explicit row_cuts string/sequence tells
#: extract_portraits() to detect row bands automatically instead of
#: requiring the caller to already know the layout.
AUTO = "auto"


def parse_grid_spec(spec: str) -> Tuple[int, int]:
    """Parse a "rows x cols" grid spec, e.g. "4x5", "4X5", or "4,5"."""
    cleaned = spec.strip().lower().replace(",", "x")
    parts = [p.strip() for p in cleaned.split("x") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"grid spec must be 'rows x cols', got {spec!r}")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"grid rows/cols must be >= 1, got {spec!r}")
    return rows, cols


def parse_row_cuts(row_cuts: str | Sequence[float], image_height: int) -> List[int]:
    """
    Convert row cut points into pixel coordinates.

    Accepted formats:
      - ratios:      "0,0.2734,0.5156,0.7471,1"
      - percentages: "0,27.34,51.56,74.71,100"
      - pixels:      "0,280,528,765,1024"
    """
    if isinstance(row_cuts, str):
        values = [float(x.strip()) for x in row_cuts.split(",") if x.strip()]
    else:
        values = [float(x) for x in row_cuts]

    if len(values) < 2:
        raise ValueError("row_cuts must contain at least a start and end value")

    if all(0.0 <= v <= 1.0 for v in values):
        cuts = [int(round(v * image_height)) for v in values]
    elif all(0.0 <= v <= 100.0 for v in values) and values[-1] == 100.0:
        cuts = [int(round((v / 100.0) * image_height)) for v in values]
    else:
        cuts = [int(round(v)) for v in values]

    cuts[0] = 0
    cuts[-1] = image_height
    cuts = sorted(set(max(0, min(image_height, c)) for c in cuts))
    if cuts[0] != 0:
        cuts.insert(0, 0)
    if cuts[-1] != image_height:
        cuts.append(image_height)
    return cuts


def auto_row_cuts(
    mask: np.ndarray,
    min_gap_frac: float = 0.012,
    min_row_fill_frac: float = 0.01,
    min_band_frac: float = 0.06,
) -> List[int]:
    """
    Find horizontal band boundaries from the foreground mask itself, instead
    of requiring the caller to supply row-cut percentages by hand.

    This mirrors column_segments()'s approach (projection + gap finding) but
    rotated onto rows: a run of consecutive rows with almost no foreground,
    at least ``min_gap_frac`` of the image tall, is treated as the gutter
    between two bands. Runs shorter than that are assumed to be gaps between
    limbs/garments within a single band, not a band boundary.
    """
    h, w = mask.shape[:2]
    row_fill = (mask > 0).sum(axis=1).astype(float) / max(1, w)

    smooth = max(1, int(round(h * 0.004)))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        row_fill = np.convolve(row_fill, np.ones(smooth) / smooth, mode="same")

    empty = row_fill < min_row_fill_frac
    min_gap = max(2, int(round(h * min_gap_frac)))

    gaps: List[Tuple[int, int]] = []
    start = None
    for y, is_empty in enumerate(empty):
        if is_empty and start is None:
            start = y
        if (not is_empty or y == h - 1) and start is not None:
            end = y if not is_empty else y + 1
            # A gap touching either edge is outer margin, not a boundary
            # between two bands, so it must not produce a cut.
            if end - start >= min_gap and start > 0 and end < h:
                gaps.append((start, end))
            start = None

    cuts = [0]
    for gap_start, gap_end in gaps:
        mid = (gap_start + gap_end) // 2
        if 0 < mid < h:
            cuts.append(mid)
    cuts.append(h)

    min_band = max(1, int(round(h * min_band_frac)))
    merged = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] < min_band:
            continue
        merged.append(c)
    if merged[-1] != h:
        if h - merged[-1] < min_band and len(merged) > 1:
            merged[-1] = h
        else:
            merged.append(h)

    return merged


def column_segments(
    mask: np.ndarray,
    row_bounds: Tuple[int, int],
    min_col_frac: float = 0.03,
    smooth_frac: float = 0.008,
    min_width_frac: float = 0.013,
    gap_merge_frac: float = 0.003,
) -> List[Tuple[int, int]]:
    """Find x-ranges inside a row that likely correspond to individual outfits."""
    image_h, image_w = mask.shape[:2]
    y1, y2 = row_bounds
    sub = (mask[y1:y2, :] > 0).astype(np.uint8)
    proj = sub.sum(axis=0).astype(float)

    smooth = max(3, int(round(image_w * smooth_frac)))
    if smooth % 2 == 0:
        smooth += 1
    if smooth > 1:
        proj = np.convolve(proj, np.ones(smooth) / smooth, mode="same")

    min_col_sum = max(3, int(round((y2 - y1) * min_col_frac)))
    active = proj > min_col_sum

    min_width = max(8, int(round(image_w * min_width_frac)))
    segments: List[Tuple[int, int]] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        at_end = i == len(active) - 1
        if (not is_active or at_end) and start is not None:
            end = i if not is_active else i + 1
            if end - start >= min_width:
                segments.append((start, end))
            start = None

    gap = max(1, int(round(image_w * gap_merge_frac)))
    merged: List[Tuple[int, int]] = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_boxes(
    bgr: np.ndarray,
    row_cuts: Sequence[int],
    lab_threshold: float = 30.0,
    padding_frac: float = 0.05,
    min_col_frac: float = 0.03,
    smooth_frac: float = 0.008,
    min_width_frac: float = 0.013,
    gap_merge_frac: float = 0.003,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Return crop boxes as (row, item, x1, y1, x2, y2)."""
    image_h, image_w = bgr.shape[:2]
    tiny_height, tiny_area = scaled_noise_thresholds(image_h, image_w)
    mask = make_clean_mask(bgr, lab_threshold=lab_threshold, tiny_height=tiny_height, tiny_area=tiny_area)
    boxes: List[Tuple[int, int, int, int, int, int]] = []

    for row_index in range(len(row_cuts) - 1):
        y1, y2 = row_cuts[row_index], row_cuts[row_index + 1]
        segments = column_segments(
            mask,
            (y1, y2),
            min_col_frac=min_col_frac,
            smooth_frac=smooth_frac,
            min_width_frac=min_width_frac,
            gap_merge_frac=gap_merge_frac,
        )

        for item_index, (seg_x1, seg_x2) in enumerate(segments, start=1):
            sub = mask[y1:y2, seg_x1:seg_x2] > 0
            ys, xs = np.where(sub)
            if len(xs) == 0 or len(ys) == 0:
                box_x1, box_y1, box_x2, box_y2 = seg_x1, y1, seg_x2, y2
            else:
                # Use the segment's x range because white clothing often blends
                # into the background and would otherwise be cropped too tightly.
                box_x1 = min(seg_x1 + int(xs.min()), seg_x1)
                box_x2 = max(seg_x1 + int(xs.max()) + 1, seg_x2)
                box_y1 = y1 + int(ys.min())
                box_y2 = y1 + int(ys.max()) + 1

            bw = box_x2 - box_x1
            bh = box_y2 - box_y1
            pad_x = int(round(bw * padding_frac))
            pad_y = int(round(bh * padding_frac))
            box_x1 = max(0, box_x1 - pad_x)
            box_x2 = min(image_w, box_x2 + pad_x)
            box_y1 = max(0, box_y1 - pad_y)
            box_y2 = min(image_h, box_y2 + pad_y)
            boxes.append((row_index + 1, item_index, box_x1, box_y1, box_x2, box_y2))

    return boxes


def detect_grid_boxes(
    bgr: np.ndarray,
    n_rows: int,
    n_cols: int,
    lab_threshold: float = 25.0,
    padding_frac: float = 0.05,
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Slice the image into a fixed n_rows x n_cols grid of equal cells, then
    tightly crop each cell's own foreground.

    This is for contact-sheet-style thumbnails with a known, regular layout
    (e.g. an evenly spaced batch export), as opposed to detect_boxes()'s
    content-aware row/column search. Each cell gets its own local background
    estimate (via make_clean_mask on just that cell), so it also handles
    alternating/checkerboard cell backgrounds that a single global background
    estimate cannot: no single color describes every cell in that case.
    """
    image_h, image_w = bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int, int, int]] = []

    for row_index in range(n_rows):
        y1 = int(round(row_index * image_h / n_rows))
        y2 = int(round((row_index + 1) * image_h / n_rows))
        for col_index in range(n_cols):
            x1 = int(round(col_index * image_w / n_cols))
            x2 = int(round((col_index + 1) * image_w / n_cols))
            cell_bgr = bgr[y1:y2, x1:x2]
            cell_h, cell_w = cell_bgr.shape[:2]

            tiny_height, tiny_area = scaled_noise_thresholds(cell_h, cell_w)
            mask = make_clean_mask(cell_bgr, lab_threshold=lab_threshold, tiny_height=tiny_height, tiny_area=tiny_area)
            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                box_x1, box_y1, box_x2, box_y2 = x1, y1, x2, y2
            else:
                bw = int(xs.max()) - int(xs.min()) + 1
                bh = int(ys.max()) - int(ys.min()) + 1
                pad_x = int(round(bw * padding_frac))
                pad_y = int(round(bh * padding_frac))
                box_x1 = x1 + max(0, int(xs.min()) - pad_x)
                box_x2 = x1 + min(cell_w, int(xs.max()) + 1 + pad_x)
                box_y1 = y1 + max(0, int(ys.min()) - pad_y)
                box_y2 = y1 + min(cell_h, int(ys.max()) + 1 + pad_y)

            boxes.append((row_index + 1, col_index + 1, box_x1, box_y1, box_x2, box_y2))

    return boxes
