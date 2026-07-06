from __future__ import annotations

from portraits_core.geometry import auto_row_cuts, parse_row_cuts
from portraits_core.masking import make_clean_mask, pil_to_bgr, scaled_noise_thresholds


def test_parse_row_cuts_ratios():
    assert parse_row_cuts("0,0.5,1", 200) == [0, 100, 200]


def test_parse_row_cuts_percentages():
    assert parse_row_cuts("0,50,100", 200) == [0, 100, 200]


def test_parse_row_cuts_pixels():
    assert parse_row_cuts("0,80,200", 200) == [0, 80, 200]


def test_parse_row_cuts_clamps_and_dedupes():
    cuts = parse_row_cuts("0,0.5,0.5,1", 100)
    assert cuts[0] == 0 and cuts[-1] == 100
    assert len(cuts) == len(set(cuts))


def test_auto_row_cuts_finds_each_band(collage_3x4):
    bgr = pil_to_bgr(collage_3x4)
    h, w = bgr.shape[:2]
    tiny_h, tiny_a = scaled_noise_thresholds(h, w)
    mask = make_clean_mask(bgr, tiny_height=tiny_h, tiny_area=tiny_a)
    cuts = auto_row_cuts(mask)
    assert len(cuts) - 1 == 3
    assert cuts[0] == 0
    assert cuts[-1] == h


def test_auto_row_cuts_single_band(collage_1x1):
    bgr = pil_to_bgr(collage_1x1)
    h, w = bgr.shape[:2]
    tiny_h, tiny_a = scaled_noise_thresholds(h, w)
    mask = make_clean_mask(bgr, tiny_height=tiny_h, tiny_area=tiny_a)
    cuts = auto_row_cuts(mask)
    assert len(cuts) == 2
    assert cuts == [0, h]
