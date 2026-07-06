"""Regression tests against real fashion-thumbnail exports, not synthetic fixtures.

These catch the two distinct layout styles the pipeline has to handle:
grid-mode contact sheets (known, regular rows x cols) and auto-detected
collage boards (uneven row heights, single background).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from portraits_core import extract_portraits

IMAGE_DIR = Path(__file__).parent.parent / "test_images"

GRID_CASES = [
    ("men.png", (3, 6), 18),
    ("desins.png", (4, 5), 20),
    ("traditional_clothes3.png", (4, 7), 28),
]


@pytest.mark.parametrize("filename, grid, expected_count", GRID_CASES)
def test_grid_mode_matches_known_layout(filename, grid, expected_count):
    path = IMAGE_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} not available")

    source = Image.open(path).convert("RGB")
    portraits, cuts, boxes = extract_portraits(source, grid=grid, output_width=0)

    assert len(portraits) == expected_count
    assert len(cuts) - 1 == grid[0]


def test_auto_mode_finds_the_tuned_thumbnail_close_to_expected():
    path = IMAGE_DIR / "all_designs.png"
    if not path.exists():
        pytest.skip(f"{path} not available")

    source = Image.open(path).convert("RGB")
    portraits, cuts, boxes = extract_portraits(source, row_cuts="auto", output_width=0)

    assert len(cuts) - 1 == 4
    assert 40 <= len(portraits) <= 43
