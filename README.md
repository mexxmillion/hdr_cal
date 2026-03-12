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
# Auto mode — finds chart if present, falls back to sphere WB
python hdri_cal.py shot.exr --out shot_cal.exr

# Check a file without processing (fast triage)
python hdri_cal.py shot.exr --validate-only

# GUI
python hdri_cal_gui.py
```

---

## Algorithm

The pipeline runs these stages in order:

```
load → colorspace detect → resolution scale → orientation validate
  → ColorChecker search (gnomonic tile sweep)
  → white balance (chart / sphere / dome / kelvin / manual)
  → WB sanity check (irradiance cross-check)
  → exposure solve (chart patch 22 / sphere render / metering)
  → hot lobe extraction (sun disc mask)
  → HDRI centering (azimuth shift → sun at φ=0)
  → base dome irradiance integrate (sky with sun zeroed)
  → sun lobe neutralise (desaturate to white)
  → per-channel gain solve (card-up neutral method)
  → apply gains → save EXR
  → final sphere render (validation)
  → report.json
```

### White Balance

White balance is derived from one of several sources and applied as per-channel RGB multipliers before any energy solve.

**Chart (default when found):** The pipeline sweeps the equirectangular image as six gnomonic (rectilinear) cube-face tiles. Each tile is passed to `colour-checker-detection`:

1. Segmentation detects the quad and returns 24 swatch colours
2. If YOLOv8 is available, it runs on a tight crop of the detected quad — whichever result has lower neutral-ramp chroma error wins
3. Patch 22 (18% grey neutral) is Reinhard-inverted from tonemapped space back to linear
4. WB scale = `reference_linear / measured_linear` per channel, G-normalised
5. A sanity check compares chart WB against sphere render WB — large disagreement (>0.15) raises a warning

**Sphere render:** Renders a Lambertian grey sphere into the HDRI, takes the mean RGB of illuminated pixels, and solves for the scale that neutralises it.

**Dome grey-world:** Cosine-weighted mean of the upper hemisphere, assumed achromatic.

**Kelvin / Manual:** Direct colour temperature conversion or explicit R,G,B multipliers.

### Exposure

Exposure sets the absolute photometric scale — the relationship between pixel values and real-world irradiance.

**Chart (patch 22):** After white balance, patch 22 should read `albedo × E_on_chart / π`. For a chart inside the HDRI facing upward, `E_on_chart = π` (full hemisphere irradiance), giving `exposure_scale = albedo / patch22_luma`. This is the most accurate method.

**Sphere render:** Renders a grey sphere, compares mean luma to `albedo`, solves the scale directly.

**Metering:** Measures median or irradiance-weighted luma of a region (lower dome, full scene, upper hemisphere) and scales to `meter_target`.

### Sun Lobe Reconstruction (Card-Up Neutral Solve)

Clipped HDRIs have their sun disc saturated at the sensor ceiling. The pipeline reconstructs physically plausible sun energy using the **card-up neutral method**:

**Goal:** after reconstruction, a grey card (albedo = 0.18) facing directly upward should read exactly `[0.18, 0.18, 0.18]` — neutral on all channels.

**Method:**

1. Zero the sun lobe to get the base dome (sky only)
2. Compute upper-hemisphere irradiance from the base dome per channel:
   `E_base_c = Σ base[c] · cos+(y) · dΩ`
3. What the base dome delivers to a card facing up:
   `card_base_c = albedo × E_base_c / π`
   (may be blue on a blue sky, warm on sunset — whatever the sky colour is)
4. The sun lobe must supply the deficit per channel:
   `E_needed_c = π - E_base_c`
5. Neutralise the lobe to `[L, L, L]` white (removes sky contamination from clipped pixels)
6. Per-channel gains on the neutralised lobe:
   `gain_c = E_needed_c / E_lobe_neutral_c`

**Result:**
- Sky stays its natural colour (untouched)
- Sun disc starts white, then gets warm gains (high R, low B on blue sky scenes) to fill what the sky is missing → sun appears physically warm/orange
- Upward card reads `[0.18, 0.18, 0.18]` exactly on all channels

Gains are soft-rolled above `--sun-gain-rolloff` and hard-capped at `--sun-gain-ceiling` to prevent runaway values on severely clipped inputs.

### Validation

After the gain solve the pipeline renders a Lambertian grey sphere into the final corrected HDRI and computes predicted vs analytical irradiance values. These are written to `report.json` and printed in the log.

---

## CLI Reference

### Basic usage

```bash
python hdri_cal.py INPUT.exr [options]
```

### Essential flags

| Flag | Default | Description |
|---|---|---|
| `--out PATH` | `input_cal.exr` | Output EXR path |
| `--debug-dir DIR` | `debug_hdri` | Folder for preview images and report.json |
| `--res` | `full` | Processing resolution: `full`, `half`, `quarter` |
| `--validate-only` | off | Fast triage pass — no EXR written |
| `--no-center-hdri` | off | Skip azimuth shift (keep sun where it is) |

### White balance

| Flag | Default | Description |
|---|---|---|
| `--wb-source` | `auto` | `auto` `chart` `sphere` `dome` `kelvin` `manual` `none` |
| `--kelvin VALUE` | — | Colour temperature, used when `--wb-source kelvin` |
| `--rgb-scale R,G,B` | — | Manual multipliers, used when `--wb-source manual` |
| `--dome-wb MODE` | `upper_dome` | Sub-mode for `--wb-source dome`: `upper_dome` `full_dome` `hot_exclude` |

**WB source fallback chain (auto):**
```
chart (found, conf ≥ 0.2) → sphere → none
```

### Exposure

| Flag | Default | Description |
|---|---|---|
| `--exposure-source` | `auto` | `auto` `chart` `sphere` `metering` `none` |
| `--metering-mode` | `bottom_dome` | Used when `--exposure-source metering`: `bottom_dome` `whole_scene` `upper_hemi_irradiance` `swatch` |
| `--meter-target` | `0.18` | Target value for metering (0.18 = 18% grey) |
| `--albedo` | `0.18` | Reflectance of the reference grey card / sphere |

**Exposure source fallback chain (auto):**
```
chart (found, conf ≥ 0.2) → sphere
```

### Sun lobe

| Flag | Default | Description |
|---|---|---|
| `--sun-threshold` | `0.1` | Fraction below peak defining the lobe boundary. Lower = tighter disc |
| `--lobe-neutralise` | `1.0` | Desaturate sun lobe before boost: `1.0` = fully white, `0.0` = keep colour |
| `--sun-gain-ceiling` | `2000` | Hard cap on per-channel gain |
| `--sun-gain-rolloff` | `500` | Gains above this value soft-roll (sqrt curve) to the ceiling |
| `--sun-upper-only` | off | Restrict lobe search to upper hemisphere only |

### Advanced

| Flag | Default | Description |
|---|---|---|
| `--colorspace` | auto | Force `acescg` or `srgb` (auto-detected from extension) |
| `--sphere-res` | `96` | Validation sphere render resolution |
| `--ref-sphere PATH` | — | Grey ball plate photograph for physical exposure calibration |

---

## Common workflows

### Auto (recommended default)
```bash
python hdri_cal.py shot.exr --out shot_cal.exr
```
Finds ColorChecker if present, derives WB and exposure from patch 22, reconstructs sun energy using card-up neutral solve.

### Chart in shadow — use sphere for exposure only
```bash
python hdri_cal.py shot.exr --wb-source chart --exposure-source sphere
```
Chart WB is still valid (it's a ratio). Sphere exposure is more reliable when the chart isn't in direct sun.

### No chart in shot
```bash
python hdri_cal.py shot.exr --wb-source sphere --exposure-source sphere
```

### Keep original azimuth (don't centre on sun)
```bash
python hdri_cal.py shot.exr --no-center-hdri
```

### Batch validate before processing
```bash
# PowerShell
Get-ChildItem .\plates\*.exr | ForEach-Object {
    python hdri_cal.py $_.FullName --validate-only --debug-dir .\validate_tmp
}

# bash
for f in plates/*.exr; do
    python hdri_cal.py "$f" --validate-only --debug-dir ./validate_tmp
done
```

### Severely clipped input — raise gain ceiling
```bash
python hdri_cal.py shot.exr --sun-gain-ceiling 5000 --sun-gain-rolloff 1000
```

### Half resolution for faster turnaround
```bash
python hdri_cal.py shot.exr --res half
```

---

## GUI

```bash
python hdri_cal_gui.py
```

The GUI wraps the same pipeline with a drag-and-drop queue, live preview tabs, and a simple/advanced settings toggle.

**Simple mode** exposes the three most common controls:
- **WB source** — how to derive white balance
- **Exposure source** — how to set photometric scale
- **Centre HDRI on sun** — whether to shift azimuth

**Advanced mode** (click the toggle) reveals:
- WB detail — kelvin value, manual RGB, dome sub-mode
- Exposure detail — metering mode, meter target, albedo
- Sun/lobe — threshold, neutralise strength, gain ceiling/rolloff
- Misc — colorspace override, sphere resolution, reference sphere plate

**Validate button** runs a fast triage pass on all queued files without processing them. Reports clip fraction, orientation, and chroma imbalance. Useful for checking a whole shoot before committing to full calibration.

**Preview tabs:**
| Tab | Contents |
|---|---|
| WB | HDRI after white balance |
| Exposed | After exposure scale |
| Lobe | Sun disc mask |
| Chart | Detected ColorChecker swatches in ERP |
| Swatches | Patch comparison: measured vs reference |
| Final | Validation sphere render |

---

## Output files

After processing, the `--debug-dir` folder contains:

| File | Description |
|---|---|
| `01_wb_preview.png` | HDRI after white balance |
| `02_exposed_preview.png` | After exposure scaling |
| `03_hot_mask.png` | Sun lobe mask |
| `07_corrected_preview.png` | After gain solve |
| `08_verify_sphere_final.png` | Final validation sphere render |
| `base_dome_calibrated.exr` | Base sky with sun zeroed (useful for debugging) |
| `report.json` | Full JSON report with all metrics |
| `colorchecker/` | Chart detection debug images (if chart found) |

The calibrated EXR is written to `--out`.

---

## report.json structure

```json
{
  "input": "shot.exr",
  "working_resolution": [2048, 4096],
  "colorspace": "acescg",
  "orientation_energy_ratio": 3.65,
  "wb": {
    "source": "colorchecker_patch22",
    "rgb_scale": [0.77, 1.0, 1.21]
  },
  "exposure": {
    "source": "chart",
    "scale": 0.82
  },
  "sun": {
    "theta_deg": 41.2,
    "phi_deg": 0.0,
    "elevation_deg": 48.8
  },
  "gain_solve": {
    "mode": "card_up_neutral",
    "gains": [198.1, 174.8, 132.2],
    "card_base": [0.072, 0.085, 0.108],
    "card_final": [0.180, 0.180, 0.180]
  },
  "energy_validation": {
    "E_upper": 1.82,
    "pred_flat_card_up": 0.18000,
    "pred_sphere_mean": 0.04500,
    "rendered_sphere_mean": 0.04487,
    "rendered_vs_analytical_err": 0.0029
  }
}
```

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
