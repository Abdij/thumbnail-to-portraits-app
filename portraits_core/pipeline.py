"""End-to-end: source collage in, list of Portrait crops out."""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

from .face import detect_faces, recenter_on_face
from .geometry import AUTO, auto_row_cuts, detect_boxes, detect_grid_boxes, parse_row_cuts
from .masking import bgr_to_pil, estimate_background_bgr, make_clean_mask, pil_to_bgr, scaled_noise_thresholds
from .text_erase import erase_heading_text_bgr
from .upscale import get_upscaler
from .watermark import add_watermark

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class Portrait:
    index: int
    row: int
    item: int
    box: Tuple[int, int, int, int]
    image: Image.Image

    @property
    def filename(self) -> str:
        return f"portrait_{self.index:02d}_row{self.row}_item{self.item}.png"


def pad_to_aspect(
    img: Image.Image,
    target_ratio: float = 3 / 4,
    fill_rgb: Tuple[int, int, int] = (236, 227, 222),
) -> Image.Image:
    """Pad an image to a target width/height ratio without cropping more."""
    w, h = img.size
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 0.001:
        return img

    if current_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        pad = max(0, new_h - h)
        top = pad // 2
        bottom = pad - top
        return ImageOps.expand(img, border=(0, top, 0, bottom), fill=fill_rgb)

    new_w = int(round(h * target_ratio))
    pad = max(0, new_w - w)
    left = pad // 2
    right = pad - left
    return ImageOps.expand(img, border=(left, 0, right, 0), fill=fill_rgb)


def resolve_row_cuts(bgr, row_cuts, lab_threshold: float) -> List[int]:
    h, w = bgr.shape[:2]
    if isinstance(row_cuts, str) and row_cuts.strip().lower() == AUTO:
        tiny_height, tiny_area = scaled_noise_thresholds(h, w)
        mask = make_clean_mask(bgr, lab_threshold=lab_threshold, tiny_height=tiny_height, tiny_area=tiny_area)
        cuts = auto_row_cuts(mask)
        return cuts if len(cuts) >= 2 else [0, h]
    return parse_row_cuts(row_cuts, h)


def extract_portraits(
    source: Image.Image,
    row_cuts: str | Sequence[float] = AUTO,
    lab_threshold: float = 30.0,
    padding_frac: float = 0.05,
    target_ratio: float = 3 / 4,
    output_width: int = 768,
    erase_headings: bool = True,
    face_aware: bool = False,
    watermark: bool = False,
    upscaler_name: str = "lanczos",
    min_col_frac: float = 0.03,
    smooth_frac: float = 0.008,
    min_width_frac: float = 0.013,
    gap_merge_frac: float = 0.003,
    grid: Tuple[int, int] | None = None,
) -> Tuple[List[Portrait], List[int], List[Tuple[int, int, int, int, int, int]]]:
    """Detect, crop, pad, and optionally resize/enhance individual portraits.

    row_cuts="auto" (the default) detects horizontal bands from the image
    itself; pass explicit ratios/percentages/pixels to override for a
    layout the auto-detector gets wrong.

    grid=(n_rows, n_cols) bypasses row/column detection entirely and instead
    slices the image into a fixed, known grid, cropping each cell to its own
    foreground. Use this for contact-sheet-style batches (e.g. an evenly
    spaced NxM export) where the layout is known upfront and garments of
    differing length make content-aware row detection unreliable.
    """
    source = source.convert("RGB")
    bgr = pil_to_bgr(source)
    h, w = bgr.shape[:2]

    if grid is not None:
        n_rows, n_cols = grid
        cuts = [int(round(i * h / n_rows)) for i in range(n_rows + 1)]
        boxes = detect_grid_boxes(bgr, n_rows, n_cols, lab_threshold=lab_threshold, padding_frac=padding_frac)
    else:
        cuts = resolve_row_cuts(bgr, row_cuts, lab_threshold)
        boxes = detect_boxes(
            bgr,
            cuts,
            lab_threshold=lab_threshold,
            padding_frac=padding_frac,
            min_col_frac=min_col_frac,
            smooth_frac=smooth_frac,
            min_width_frac=min_width_frac,
            gap_merge_frac=gap_merge_frac,
        )

    output_bgr = erase_heading_text_bgr(bgr, cuts) if erase_headings else bgr.copy()
    output_pil = bgr_to_pil(output_bgr)
    bg_bgr = estimate_background_bgr(bgr)
    fill_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))

    faces = detect_faces(bgr) if face_aware else []
    upscaler = get_upscaler(upscaler_name)

    portraits: List[Portrait] = []
    for index, (row, item, x1, y1, x2, y2) in enumerate(boxes, start=1):
        box = (x1, y1, x2, y2)
        if face_aware and faces:
            box = recenter_on_face(box, faces, (w, h))
        crop = output_pil.crop(box)
        crop = pad_to_aspect(crop, target_ratio=target_ratio, fill_rgb=fill_rgb)
        if output_width and output_width > 0:
            crop = upscaler.upscale(crop, output_width)
        if watermark:
            crop = add_watermark(crop)
        portraits.append(Portrait(index=index, row=row, item=item, box=box, image=crop))

    return portraits, cuts, boxes


def draw_overlay(source: Image.Image, row_cuts: Sequence[int], boxes: Sequence[Tuple[int, int, int, int, int, int]]) -> Image.Image:
    """Draw row lines and crop boxes for preview/debugging."""
    preview = source.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    for y in row_cuts[1:-1]:
        draw.line((0, y, width, y), fill=(0, 160, 255), width=max(1, width // 512))
    for index, (row, item, x1, y1, x2, y2) in enumerate(boxes, start=1):
        line_w = max(1, width // 768)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_w)
        draw.text((x1 + 3, y1 + 3), str(index), fill=(255, 0, 0))
    return preview


def portraits_to_zip_bytes(portraits: Sequence[Portrait]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for portrait in portraits:
            img_buffer = io.BytesIO()
            portrait.image.save(img_buffer, format="PNG")
            zf.writestr(portrait.filename, img_buffer.getvalue())
    return buffer.getvalue()


def save_portraits(portraits: Sequence[Portrait], out_dir: str | os.PathLike[str]) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for portrait in portraits:
        portrait.image.save(path / portrait.filename)
