# Thumbnail to Individual Portraits

This small local Python app splits a fashion collage/thumbnail into separate portrait PNGs. It is tuned for the attached design thumbnail, which has four horizontal bands and a light background.

## Quick start

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

Upload the collage image, then click **Split into portraits**. The app will show detected crop boxes and provide a ZIP download.

## Command-line use

```bash
pip install -r requirements.txt
python split_portraits.py all_designs.png --out portraits --zip individual_portraits.zip
```

The default row cuts are ratios tuned for the attached image:

```text
0,0.2734,0.5156,0.7471,1
```

For a different collage layout, pass your own row cuts as ratios, percentages, or pixels,
or let the detector find the bands itself with `--row-cuts auto`:

```bash
python split_portraits.py my_collage.png --row-cuts auto --out portraits
python split_portraits.py my_collage.png --row-cuts 0,280,528,765,1024 --out portraits
python split_portraits.py my_collage.png --row-cuts 0,27.34,51.56,74.71,100 --out portraits
```

`split_portraits.py` is a thin CLI wrapper; the actual detection/crop/upscale
pipeline lives in `portraits_core/`, which is also what `app.py` uses.

## Notes

- This separates and upscales the thumbnail crops; it cannot recover detail that is not present in the original image.
- If crops are merged, increase the foreground threshold slightly or lower the gap merge value inside `split_portraits.py`.
- If crops are missed, lower the foreground threshold or adjust the row cuts.
