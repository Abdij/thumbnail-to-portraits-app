"""Core, UI-independent image pipeline for splitting collages into portraits.

Nothing in this package imports Streamlit, FastAPI, or any web framework —
it is plain PIL/OpenCV/numpy so it can be called from a CLI, a Streamlit app,
or an HTTP API without change.
"""

from .pipeline import (
    Portrait,
    draw_overlay,
    extract_portraits,
    portraits_to_zip_bytes,
    save_portraits,
)
from .geometry import DEFAULT_ROW_CUTS, auto_row_cuts, parse_grid_spec, parse_row_cuts
from .upscale import LanczosUpscaler, RealESRGANUpscaler, Upscaler, get_upscaler
from .watermark import add_watermark

__all__ = [
    "Portrait",
    "draw_overlay",
    "extract_portraits",
    "portraits_to_zip_bytes",
    "save_portraits",
    "DEFAULT_ROW_CUTS",
    "auto_row_cuts",
    "parse_row_cuts",
    "parse_grid_spec",
    "Upscaler",
    "LanczosUpscaler",
    "RealESRGANUpscaler",
    "get_upscaler",
    "add_watermark",
]
