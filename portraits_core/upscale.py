"""Pluggable upscaling: a free Lanczos default plus a hook for a real model.

Splitting only re-frames pixels that already exist; it cannot add detail the
source thumbnail never had. A learned super-resolution model is a genuine
"enhance" step, which is why it is a distinct paid feature rather than the
default path — it needs extra, fairly heavy dependencies (torch + model
weights) that a free/local install should not be forced to carry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLE_LANCZOS = Image.LANCZOS


class Upscaler(ABC):
    name: str = "base"

    @abstractmethod
    def upscale(self, image: Image.Image, target_width: int) -> Image.Image:
        """Return image resized so its width is target_width."""


class LanczosUpscaler(Upscaler):
    """Default, dependency-free resize. This is a resize, not an enhancement."""

    name = "lanczos"

    def upscale(self, image: Image.Image, target_width: int) -> Image.Image:
        w, h = image.size
        if target_width <= 0 or w == target_width:
            return image
        target_height = int(round(h * (target_width / w)))
        return image.resize((target_width, target_height), RESAMPLE_LANCZOS)


class RealESRGANUpscaler(Upscaler):
    """
    Learned super-resolution, gated behind the 'pro' tier.

    Requires the optional `realesrgan` + `torch` packages and downloaded model
    weights, which are intentionally not bundled with the base install. See
    README "AI upscale (optional)" for setup. Falls back with a clear error
    rather than silently degrading to Lanczos, so callers can tell the paid
    feature is misconfigured instead of just getting an ordinary resize.
    """

    name = "realesrgan"

    def __init__(self, model_path: str | None = None, scale: int = 2):
        try:
            import torch  # noqa: F401
            from realesrgan import RealESRGANer  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "AI upscale requires the optional 'torch' and 'realesrgan' "
                "packages plus downloaded model weights. Install them with "
                "`pip install -r backend/requirements-ai.txt` and set "
                "REALESRGAN_MODEL_PATH, or use the default Lanczos upscaler."
            ) from exc
        self._scale = scale
        self._model_path = model_path
        self._impl = None  # lazily constructed on first use

    def upscale(self, image: Image.Image, target_width: int) -> Image.Image:
        raise RuntimeError(
            "RealESRGANUpscaler is a wired integration point, not a bundled "
            "model: provide REALESRGAN_MODEL_PATH and implement the forward "
            "pass here, or select the 'lanczos' upscaler."
        )


def get_upscaler(name: str = "lanczos") -> Upscaler:
    if name == "lanczos":
        return LanczosUpscaler()
    if name == "realesrgan":
        return RealESRGANUpscaler()
    raise ValueError(f"Unknown upscaler: {name!r}")
