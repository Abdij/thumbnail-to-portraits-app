"""Free-tier watermark, applied only when a caller opts a job into the free tier."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def add_watermark(image: Image.Image, text: str = "Made with ThumbSplit • remove on Pro") -> Image.Image:
    """Stamp a small translucent label along the bottom edge of the image."""
    image = image.convert("RGBA")
    w, h = image.size
    font_size = max(11, int(round(h * 0.028)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(6, font_size // 2)
    bar_h = text_h + pad * 2

    draw.rectangle((0, h - bar_h, w, h), fill=(0, 0, 0, 110))
    draw.text(((w - text_w) / 2, h - bar_h + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 235))

    return Image.alpha_composite(image, overlay).convert("RGB")
