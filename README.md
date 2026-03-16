# HDRI Calibration Pipeline

A photometrically accurate HDRI reconstruction and calibration tool for VFX / IBL workflows.

Takes a clipped or LDR equirectangular environment map, finds the sun lobe, reconstructs missing energy, applies white balance from a ColorChecker chart (or grey sphere), and outputs a calibrated EXR suitable for image-based lighting in production renderers.

---

## Files

| File | Purpose |
|---|---|
| `hdri_cal.py` | Core pipeline — CLI and callable API |
| `colorchecker_erp.py` | ColorChecker detection in equirectangular images |
| `hdri_cal_gui.py` | PySide6 GUI frontend |

---

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. OpenEXR backend

`imageio` needs an OpenEXR backend to read and write `.exr` files. The easiest path is to let OpenCV handle it (already installed via `opencv-python`), which works automatically on most platforms. If you hit EXR read errors:

**Windows**
```bash
pip install imageio[openexr]
```

**macOS**
```bash
brew install openexr
pip install openexr
```

**Linux**
```bash
sudo apt install libopenexr-dev
pip install openexr
```

### 3. YOLOv8 model (optional, auto-downloads)

The first time the pipeline searches for a ColorChecker it will download the YOLOv8 segmentation model (~88 MB) from Hugging Face into:
```
~/.colour-science/colour-checker-detection/
```
This only happens once. Requires an internet connection on first run. Set `--wb-source sphere` to skip chart search entirely if offline.

---

## Quick Start

```bash
# Auto mode — if a chart is found, use it; otherwise leave WB/exposure alone
python hdri_cal.py shot.exr --out shot_cal.exr

# Treat a linear-sRGB HDR/EXR correctly and convert it to ACEScg on load
python hdri_cal.py shot_srgb.exr --colorspace srgb --out shot_cal.exr

# Validate only (no EXR written)
python hdri_cal.py shot.exr --validate-only

# GUI
python hdri_cal_gui.py
```

---

## Current Pipeline

The pipeline now works in this order:

```
load -> input-primaries conversion -> ACEScg working space
  -> orientation validate
  -> optional ColorChecker search / rectified pass-2 read
  -> WB source solve
  -> exposure source solve
  -> hot lobe extraction
  -> HDRI centering
  -> sun solve
  -> optional final balance trim
  -> save EXR + previews + report.json
```

Important behavior:
- Internal processing is always in `ACEScg`.
- `--colorspace srgb` means: interpret HDR/EXR input as linear sRGB, then convert to linear ACEScg before any processing.
- LDR inputs such as `.jpg`, `.jpeg`, `.png`, and `.webp` are treated as sRGB-encoded images, decoded to linear sRGB, then converted to ACEScg.
- In `auto` mode, if no chart is found, WB and exposure both fall back to `none`.

---

## CLI Reference

### Basic usage

```bash
python hdri_cal.py INPUT [options]
```

Accepted inputs:
- `.exr`
- `.hdr`
- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

### Core flags

| Flag | Default | Description |
|---|---|---|
| `--out PATH` | `corrected.exr` | Output calibrated EXR |
| `--debug-dir DIR` | `debug_hdri` | Debug previews and `report.json` |
| `--res` | `full` | `full`, `half`, `quarter` |
| `--colorspace` | auto | Force input primaries: `acescg` or `srgb` |
| `--validate-only` | off | Validate only, no EXR output |
| `--center-hdri` | on | Center azimuth on the dominant hot lobe |
| `--no-center-hdri` | off | Disable centering |

### White balance and exposure

| Flag | Default | Description |
|---|---|---|
| `--wb-source` | `auto` | `auto`, `chart`, `none` |
| `--exposure-source` | `auto` | `auto`, `chart`, `none` |
| `--albedo` | `0.18` | Target grey reflectance |

Auto behavior:
- `WB`: chart if found, else none
- `Exposure`: chart if found, else none

### Sun solve and final trim

| Flag | Default | Description |
|---|---|---|
| `--sphere-solve` | `energy_conservation` | `energy_conservation`, `sun_facing_card`, `sun_facing_vertical`, `none` |
| `--final-balance-target` | `none` | `none` or `auto` |
| `--sun-threshold` | `0.1` | Lobe boundary fraction below peak |
| `--lobe-neutralise` | `1.0` | White the lobe before gain solve |
| `--sun-gain-ceiling` | `2000` | Hard cap on per-channel sun gain |
| `--sun-gain-rolloff` | `500` | Soft rolloff start |

`--sphere-solve` meanings:
- `energy_conservation`: upward-facing grey card target
- `sun_facing_card`: grey card whose normal points at the sun
- `sun_facing_vertical`: vertical grey card rotated toward the sun azimuth
- `none`: skip sun solve

`--final-balance-target auto`:
- measures the imaginary target card implied by `--sphere-solve`
- computes one final global RGB multiplier
- trims the output so that target is neutral at `albedo`

### ColorChecker options

| Flag | Default | Description |
|---|---|---|
| `--colorchecker PATH` | — | Separate ColorChecker plate |
| `--colorchecker-in-hdri` | off | Search chart inside the HDRI |
| `--cc-read-backend` | `colour` | `auto`, `colour`, `contour` |
| `--cc-compare-backends` | off | Save comparison overlays |

Notes:
- The current default read path is `colour`.
- The detector uses a coarse tile sweep plus centered refinement.
- Final swatch measurement comes from the rectified pass-2 checker read.

---

## Common Workflows

### Recommended auto mode
```bash
python hdri_cal.py shot.exr --out shot_cal.exr
```

### Sun solve only, no chart WB/exposure
```bash
python hdri_cal.py shot.exr --out shot_cal.exr   --wb-source none   --exposure-source none   --center-hdri   --sphere-solve energy_conservation
```

### Target a card facing the sun
```bash
python hdri_cal.py shot.exr --out shot_cal.exr   --wb-source none   --exposure-source none   --center-hdri   --sphere-solve sun_facing_card
```

### Target a vertical card facing the sun azimuth
```bash
python hdri_cal.py shot.exr --out shot_cal.exr   --wb-source none   --exposure-source none   --center-hdri   --sphere-solve sun_facing_vertical
```

### Add the final post-balance trim
```bash
python hdri_cal.py shot.exr --out shot_cal.exr   --wb-source none   --exposure-source none   --center-hdri   --sphere-solve sun_facing_vertical   --final-balance-target auto
```

### Validate only
```bash
python hdri_cal.py shot.exr --validate-only
```

---

## GUI

The GUI now has two top-level modes:
- `auto`
- `advanced`

### Auto mode
- If a chart is found, use chart WB/exposure.
- If no chart is found, leave WB/exposure unchanged.
- You can still set:
  - `Input primaries`
  - `Intensity`
  - `Temperature`
  - `Tint`
  - `Centre HDRI on sun`

### Advanced mode
Advanced reveals explicit controls for:
- `WB source`
- `Exposure source`
- `Sun solve`
- `Final balance`
- `Albedo`
- `Sun threshold`
- `Lobe neutralise`
- `Gain ceiling`
- `Gain rolloff`

### Source preview
The GUI provides an immediate `Source` preview on file drop/select.
It is re-rendered when you change, per queued file:
- `Input primaries`
- `Intensity`
- `Temperature`
- `Tint`

Each queued HDRI stores its own settings, so one file can be `acescg` and another `srgb` with different photographic adjustments.

### Photographic controls
- `Intensity`: global preview/manual input multiplier
- `Temperature`: photographic WB temperature
- `Tint`: `+1.00 = magenta`, `-1.00 = green`
- `Reset Photographic`: restore `1.0 / 6500 K / 0.0`

The controls now support both:
- slider scrubbing
- direct typed numeric entry

### Preview tabs
The GUI preview tabs now include:
- `Source`
- `WB`
- `Exposed`
- `Lobe`
- `Chart Tile`
- `Rectified`
- `Swatches`
- `Solved`
- `Balanced`
- `Final`

### Queue actions
You can now:
- remove a selected file
- requeue a selected file
- clear all files
- open the output folder

---
## Troubleshooting

**ColorChecker not found**
- Ensure the chart is visible and reasonably well-lit in the HDRI
- Try `--wb-source sphere` to skip chart search
- Check `debug_dir/colorchecker/` for detection debug images

**WB disagreement warning**
- Chart may be in shadow — use `--wb-source chart --exposure-source sphere`
- Chart may be misidentified — check `cc_swatch_comparison.jpg` in debug dir

**Gain ceiling hit warnings**
- Input is severely clipped — raise `--sun-gain-ceiling 5000`
- Very low remaining lobe energy — some clipping is unrecoverable

**Blue/wrong-coloured sun disc**
- Ensure `--lobe-neutralise` is at `1.0` (default)
- Check if base dome E_upper is already coloured (coloured sky is expected, sun will compensate)

**EXR read errors**
- Install OpenEXR system library (see Installation above)
- Try renaming to `.hdr` and running — imageio has a different backend path for Radiance HDR

---

## Notes on colorspace

- `.exr` / `.hdr` files are assumed **ACEScg (AP1 linear)** unless `--colorspace srgb` is set
- `.jpg` / `.png` files are assumed **sRGB** and linearised automatically
- CC24 reference patch values are spectral reflectances — neutral patches are achromatic in both colorspaces, so WB derivation is colorspace-agnostic
- All internal math runs in linear light

---

## License

MIT
