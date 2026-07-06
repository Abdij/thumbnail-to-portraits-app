#!/usr/bin/env python3
"""
Split a fashion thumbnail/collage into individual portrait crop images.

CLI wrapper around the portraits_core pipeline (see portraits_core/pipeline.py
for the actual detection/crop/upscale logic).
"""

from __future__ import annotations

import argparse

from PIL import Image

from portraits_core import (
    DEFAULT_ROW_CUTS,
    draw_overlay,
    extract_portraits,
    parse_grid_spec,
    portraits_to_zip_bytes,
    save_portraits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a fashion collage into individual portrait crops.")
    parser.add_argument("image", help="Input collage image, e.g. all_designs.png")
    parser.add_argument("--out", default="portraits", help="Output directory for PNG crops")
    parser.add_argument("--row-cuts", default=DEFAULT_ROW_CUTS, help="Row cuts as ratios, percentages, or pixels, or 'auto'")
    parser.add_argument(
        "--grid",
        default="",
        help="Known grid layout as 'rows x cols' (e.g. '4x5') for contact-sheet batches. "
        "Bypasses --row-cuts and slices evenly, cropping each cell to its own foreground.",
    )
    parser.add_argument("--threshold", type=float, default=30.0, help="LAB foreground threshold; try 25-40")
    parser.add_argument("--padding", type=float, default=0.05, help="Padding around each detected outfit, as a fraction")
    parser.add_argument("--aspect", type=float, default=0.75, help="Output width/height ratio. 0.75 means 3:4 portrait")
    parser.add_argument("--output-width", type=int, default=768, help="Output crop width in pixels. Use 0 to keep natural size")
    parser.add_argument("--no-erase-headings", action="store_true", help="Do not remove row heading text from crops")
    parser.add_argument("--face-aware", action="store_true", help="Recenter crops on detected faces when present")
    parser.add_argument("--watermark", action="store_true", help="Add a watermark to each crop")
    parser.add_argument("--upscaler", default="lanczos", help="Upscaler to use: lanczos or realesrgan")
    parser.add_argument("--preview", default="crop_preview.png", help="Preview image showing crop boxes")
    parser.add_argument("--zip", default="", help="Optional ZIP filename for all output crops")
    args = parser.parse_args()

    grid = parse_grid_spec(args.grid) if args.grid.strip() else None

    source = Image.open(args.image).convert("RGB")
    portraits, cuts, boxes = extract_portraits(
        source,
        row_cuts=args.row_cuts,
        lab_threshold=args.threshold,
        padding_frac=args.padding,
        target_ratio=args.aspect,
        output_width=args.output_width,
        erase_headings=not args.no_erase_headings,
        face_aware=args.face_aware,
        watermark=args.watermark,
        upscaler_name=args.upscaler,
        grid=grid,
    )

    save_portraits(portraits, args.out)
    overlay = draw_overlay(source, cuts, boxes)
    overlay.save(args.preview)

    if args.zip:
        with open(args.zip, "wb") as f:
            f.write(portraits_to_zip_bytes(portraits))

    print(f"Saved {len(portraits)} portraits to: {args.out}")
    print(f"Saved crop preview to: {args.preview}")
    if args.zip:
        print(f"Saved ZIP to: {args.zip}")


if __name__ == "__main__":
    main()
