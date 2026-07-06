from __future__ import annotations

import numpy as np

from portraits_core.pipeline import draw_overlay, extract_portraits, portraits_to_zip_bytes
from portraits_core.watermark import add_watermark


def test_extract_portraits_auto_detects_grid(collage_3x4):
    portraits, cuts, boxes = extract_portraits(collage_3x4, row_cuts="auto", output_width=0)
    assert len(portraits) == 12
    assert len(cuts) - 1 == 3


def test_extract_portraits_manual_row_cuts_still_supported(collage_3x4):
    portraits, cuts, boxes = extract_portraits(
        collage_3x4, row_cuts="0,0.33,0.66,1", output_width=0
    )
    assert len(portraits) == 12


def test_extract_portraits_resizes_to_output_width(collage_3x4):
    portraits, _, _ = extract_portraits(collage_3x4, row_cuts="auto", output_width=200)
    for p in portraits:
        assert p.image.size[0] == 200


def test_extract_portraits_watermark_changes_pixels(collage_3x4):
    plain, _, _ = extract_portraits(collage_3x4, row_cuts="auto", output_width=300, watermark=False)
    marked, _, _ = extract_portraits(collage_3x4, row_cuts="auto", output_width=300, watermark=True)
    a = np.array(plain[0].image.convert("RGB"))
    b = np.array(marked[0].image.convert("RGB"))
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_extract_portraits_face_aware_does_not_crash_without_faces(collage_3x4):
    portraits, _, _ = extract_portraits(collage_3x4, row_cuts="auto", output_width=0, face_aware=True)
    assert len(portraits) == 12


def test_zip_bytes_contains_one_entry_per_portrait(collage_3x4):
    portraits, _, _ = extract_portraits(collage_3x4, row_cuts="auto", output_width=0)
    import zipfile
    import io

    data = portraits_to_zip_bytes(portraits)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert len(zf.namelist()) == len(portraits)


def test_draw_overlay_returns_same_size_image(collage_3x4):
    portraits, cuts, boxes = extract_portraits(collage_3x4, row_cuts="auto", output_width=0)
    overlay = draw_overlay(collage_3x4, cuts, boxes)
    assert overlay.size == collage_3x4.size


def test_add_watermark_preserves_size():
    from PIL import Image

    img = Image.new("RGB", (300, 400), (240, 238, 232))
    out = add_watermark(img)
    assert out.size == img.size
