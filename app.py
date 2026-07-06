import io

import streamlit as st
from PIL import Image

from portraits_core import (
    DEFAULT_ROW_CUTS,
    draw_overlay,
    extract_portraits,
    parse_grid_spec,
    portraits_to_zip_bytes,
)

st.set_page_config(page_title="Thumbnail to Portraits", layout="wide")
st.title("Thumbnail to Individual Portraits")
st.write(
    "Upload a collage/thumbnail, detect each outfit, and export one portrait PNG per outfit. "
    "The defaults are tuned for the attached fashion-design thumbnail."
)

uploaded = st.file_uploader("Upload collage image", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("Detection settings")
    st.caption("For the attached image, the defaults should find about 43 portraits.")
    threshold = st.slider("Foreground threshold", 10.0, 60.0, 30.0, 1.0)
    padding = st.slider("Padding around each outfit", 0.0, 0.20, 0.05, 0.01)
    output_width = st.slider("Output portrait width", 256, 1536, 768, 64)
    erase_headings = st.checkbox("Erase row headings from crops", True)

    st.header("Grid layout")
    st.caption(
        "For contact-sheet-style batches with a known, regular layout (e.g. a 4x5 export), "
        "enter rows x cols to slice evenly instead of auto-detecting bands. Leave blank for "
        "collage-style boards with uneven row heights, like the attached thumbnail."
    )
    grid_spec = st.text_input("Grid layout, rows x cols (optional)", value="", placeholder="e.g. 4x5")

    st.header("Row cuts")
    st.caption("Ignored when a grid layout is set above. Adjust these if your collage uses different horizontal bands.")
    cut1 = st.slider("Cut 1, percent of image height", 10.0, 45.0, 27.34, 0.1)
    cut2_min = min(90.0, cut1 + 5.0)
    cut2 = st.slider("Cut 2, percent of image height", cut2_min, 80.0, max(51.56, cut2_min), 0.1)
    cut3_min = min(95.0, cut2 + 5.0)
    cut3 = st.slider("Cut 3, percent of image height", cut3_min, 95.0, max(74.71, cut3_min), 0.1)
    row_cuts = f"0,{cut1},{cut2},{cut3},100"

if uploaded is None:
    st.info("Upload the attached thumbnail or another similar collage to start.")
    st.stop()

source = Image.open(uploaded).convert("RGB")
st.subheader("Input")
st.image(source, use_container_width=True)

if st.button("Split into portraits", type="primary"):
    grid = None
    if grid_spec.strip():
        try:
            grid = parse_grid_spec(grid_spec)
        except ValueError as exc:
            st.error(f"Invalid grid layout: {exc}")
            st.stop()

    portraits, cuts, boxes = extract_portraits(
        source,
        row_cuts=row_cuts,
        lab_threshold=threshold,
        padding_frac=padding,
        target_ratio=3 / 4,
        output_width=output_width,
        erase_headings=erase_headings,
        grid=grid,
    )

    st.success(f"Detected {len(portraits)} portraits.")

    st.subheader("Crop-box preview")
    overlay = draw_overlay(source, cuts, boxes)
    st.image(overlay, use_container_width=True)

    zip_bytes = portraits_to_zip_bytes(portraits)
    st.download_button(
        "Download all portraits as ZIP",
        data=zip_bytes,
        file_name="individual_portraits.zip",
        mime="application/zip",
    )

    st.subheader("Portrait preview")
    cols = st.columns(6)
    for i, portrait in enumerate(portraits):
        with cols[i % len(cols)]:
            st.image(portrait.image, caption=portrait.filename, use_container_width=True)
