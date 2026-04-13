# FITS Labeler System

Reusable Tkinter app for labeling 4-band FITS samples with binary decisions saved to CSV.

## What it supports now

- 4 panel display (VIS, Y, J, H by default)
- Per-band scaling controls:
  - low percentile
  - high percentile
  - asinh `a`
- Per-slider keyboard toggle buttons (`Keys On`/`Keys Off`) for fine arrow-key adjustments
- Two label buttons (True/False by default)
- CSV append after every click (no deferred save)
- Optional resume behavior: already-labeled sample IDs are skipped when `--skip-handled` is used
- In no-skip mode (default), labeling an existing `sample_id` updates its CSV row instead of creating a duplicate
- Pluggable loading methods:
  - four separate FITS files per sample
  - one FITS file with one HDU per band

## Install

```bash
pip install -r requirements.txt
```

## Run (separate files mode)

```bash
python run_labeler.py \
  --images-root /path/to/fits/folder \
  --csv-path /path/to/output/labels.csv \
  --loader separate

# add this flag to skip IDs already present in the CSV
# --skip-handled
```

Default expected suffixes for separate mode:

- `_VIS.fits`
- `_NIR_Y.fits`
- `_NIR_J.fits`
- `_NIR_H.fits`

Override if needed:

```bash
python run_labeler.py \
  --images-root /path/to/fits/folder \
  --loader separate \
  --vis-suffix _band1.fits \
  --y-suffix _band2.fits \
  --j-suffix _band3.fits \
  --h-suffix _band4.fits
```

## Run (single-file multi-HDU mode)

```bash
python run_labeler.py \
  --images-root /path/to/fits/folder \
  --loader multi-hdu \
  --single-pattern "*.fits" \
  --hdu-map "VIS:1,Y:2,J:3,H:4"
```

## Keyboard shortcuts

- With no slider toggle active:
  - Left arrow = False button
  - Right arrow = True button
- With one slider toggle active (`Keys On`):
  - Left/Down = decrease active slider
  - Right/Up = increase active slider
  - Only one slider can be active at a time

## Specialize later

Add new loader classes in `labeler/loaders.py` by implementing `BaseFitsLoader`:

- `discover() -> list[str]`
- `load(sample_id: str) -> ImageSample`

This keeps UI and CSV logic unchanged while allowing custom input formats.
