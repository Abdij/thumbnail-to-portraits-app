from __future__ import annotations

import pytest
from PIL import Image, ImageDraw


def make_collage(
    rows: int = 3,
    cols: int = 4,
    cell_w: int = 120,
    cell_h: int = 160,
    gap_row: int = 50,
    gap_col: int = 30,
    margin: int = 40,
    bg=(240, 238, 232),
    fg=(70, 65, 90),
) -> Image.Image:
    """A synthetic light-background collage with `rows` x `cols` solid cells."""
    width = margin * 2 + cols * cell_w + (cols - 1) * gap_col
    height = margin * 2 + rows * cell_h + (rows - 1) * gap_row
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for r in range(rows):
        y1 = margin + r * (cell_h + gap_row)
        y2 = y1 + cell_h
        for c in range(cols):
            x1 = margin + c * (cell_w + gap_col)
            x2 = x1 + cell_w
            draw.rectangle((x1, y1, x2, y2), fill=fg)
    return img


@pytest.fixture
def collage_3x4():
    return make_collage(rows=3, cols=4)


@pytest.fixture
def collage_1x1():
    return make_collage(rows=1, cols=1)
