# hdri_cal — HDRI Calibration Pipeline for VFX / IBL Compositing

Physically calibrated latlong HDRI processing for on-set VFX.  
Takes a raw HDR panorama (EXR / HDR / LDR bracket), performs white balance, photometric exposure calibration, and sun energy reconstruction, and outputs a production-ready EXR suitable for IBL and Nuke/Maya/Houdini compositing.

---

## What it does

```
Input EXR/HDR
     │
     ├─ ColorChecker detection (cubemap tile sweep)
     │    └─ WB scale from patch 22 (Neutral 5, ~18% grey)
     │    └─ Absolute exposure from flat-patch irradiance equation
     │
     ├─ White balance  (chart > rgb_scale > kelvin > swatch > sphere > dome)
     │
     ├─ Exposure calibration
     │    ├─ Chart mode:  exposure_scale = albedo / patch22_luma
     │    └─ No-chart:   dome metering → 18% grey target
     │
     ├─ Sun lobe extraction  (smoothstep mask, solid-angle centroid)
     │
     ├─ HDRI centering  (azimuth shift so sun sits at φ=0)
     │
     ├─ Sun energy reconstruction
     │    └─ Per-channel gain solve: restores clipped lobe energy
     │       so a Lambertian sphere renders neutral grey at target
     │
     └─ Output calibrated EXR + debug folder + JSON report
```

---

## Installation

```bash
# Recommended: conda environment
conda create -n hdri_cal python=3.11
conda activate hdri_cal
conda install -c conda-forge numpy opencv scipy imageio openexr

# Then pip for the remaining packages
pip install pyexr colour-science colour-checker-detection
```

Or with pip only:
```bash
pip install -r requirements.txt
```

---

## Quick Start

**Simplest — no chart, sphere WB:**
```bash
python hdri_cal.py input.exr --out calibrated.exr --sphere-wb
```

**With a ColorChecker inside the HDRI (on the floor, tripod, etc.):**
```bash
python hdri_cal.py input.exr --out calibrated.exr --colorchecker-in-hdri
```

**With a separate chart plate image:**
```bash
python hdri_cal.py input.exr --out calibrated.exr --colorchecker chart_plate.jpg
```

**With a physical grey ball reference photo:**
```bash
python hdri_cal.py input.exr --out calibrated.exr \
    --sphere-wb \
    --ref-sphere grayball.jpg \
    --ref-sphere-albedo 0.18
```

Debug images and a JSON report are always written to `debug_hdri/` (override with `--debug-dir`).

---

## Full CLI Reference

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(required)* | Input panorama: `.exr`, `.hdr`, `.jpg`, `.png` |
| `--out` | `calibrated.exr` | Output EXR path |
| `--res` | `full` | Processing resolution: `full`, `half`, `quarter` |
| `--debug-dir` | `debug_hdri` | Folder for debug images and JSON report |

### White Balance

Priority order: ColorChecker > `--rgb-scale` > `--kelvin` > `--wb-swatch` > `--sphere-wb` > `--dome-wb` > passthrough.

| Flag | Description |
|------|-------------|
| `--colorchecker-in-hdri` | Search HDRI for ColorChecker (cubemap sweep). Sets WB and absolute exposure. |
| `--colorchecker PATH` | Load a separate chart plate image for WB/colour correction. |
| `--rgb-scale R,G,B` | Manual WB multipliers, e.g. `1.05,1.0,0.92` |
| `--kelvin K` | Colour temperature WB, e.g. `5600` |
| `--wb-swatch X,Y` | Sample a neutral pixel at this ERP coordinate for WB |
| `--sphere-wb` | Render a Lambertian sphere, neutralise its mean colour |
| `--dome-wb MODE` | Average pixels in `upper_dome`, `full_dome`, or `hot_exclude` region |

### Exposure

Only used when **no chart** is present. If a ColorChecker is detected, exposure is derived directly from patch 22.

| Flag | Default | Description |
|------|---------|-------------|
| `--metering-mode` | `bottom_dome` | `bottom_dome`, `whole_scene`, `swatch` |
| `--meter-stat` | `median` | `mean` or `median` |
| `--meter-target` | `0.18` | Target luminance (linear). 0.18 = 18% grey |
| `--swatch X,Y` | — | Pixel to sample for swatch metering mode |
| `--swatch-size N` | `5` | Sample window size in pixels |

### Sun Lobe

| Flag | Default | Description |
|------|---------|-------------|
| `--sun-threshold` | `0.1` | Width of smoothstep mask around peak luma. 0.1 = narrow tight mask around sun disc. |
| `--sun-upper-only` | off | Restrict hot lobe to upper hemisphere |
| `--sun-blur-px N` | `0` | Gaussian blur on lobe mask (odd px, 0 = off) |

### HDRI Centering

| Flag | Default | Description |
|------|---------|-------------|
| `--center-hdri` / `--no-center-hdri` | on | Shift azimuth so dominant light sits at φ=0 (centre column). Useful for match-moving workflows. |
| `--center-elevation` | off | Also shift vertically to put sun on equator row. Changes sun elevation — use with care. |

### Sphere Solve

| Flag | Default | Description |
|------|---------|-------------|
| `--sphere-solve` | `energy_conservation` | Gain solve mode. `energy_conservation` (default): per-channel solve ensuring final sphere renders neutral grey at target. Legacy modes: `direct_highlight`, `iterative_peak_ratio`, `none`. |
| `--lobe-neutralise` | `1.0` | Desaturate hot lobe before gain boost. `1.0` = fully white sun. `0.0` = preserve colour cast. |
| `--albedo` | `0.18` | Grey ball / chart patch 22 albedo |
| `--sphere-res` | `96` | Lambertian sphere render resolution in pixels |

### Reference Grey Ball

Optional physical exposure calibration from an on-set grey ball photograph.

| Flag | Description |
|------|-------------|
| `--ref-sphere PATH` | Path to grey ball photo (any format) |
| `--ref-sphere-cx/cy/r` | Ball centre and radius in pixels (auto-detected if omitted) |
| `--ref-sphere-albedo` | Albedo of the reference ball (default 0.18) |

---

## Pipeline Math

### ColorChecker Exposure Calibration

When a ColorChecker is present, the pipeline derives absolute photometric exposure without any scene metering heuristics.

**Patch 22 (Neutral 5) is a flat Lambertian surface with known albedo `a = 0.1882`.**

A flat Lambertian surface under irradiance E reads:
```
pixel = a × E / π
```

After WB correction (G-normalised, so colour only, no brightness change), patch 22 reads `meas_luma` in raw HDRI units. We rescale so it reads exactly `a`:

```
exposure_scale = a / meas_luma
```

After this, a Lambertian sphere rendered under the calibrated HDRI should read:
```
sphere_mean = a × E / 4 = a × π / 4  ≈  0.1414  (for a = 0.18)
```

This is the `sphere_target` for the sun gain solve.

### Sun Energy Reconstruction

The hot lobe (clipped sun disc) is masked out. A per-channel gain is computed so that when the lobe pixels are inflated by that gain, a rendered Lambertian sphere hits `sphere_target`:

```
gain_c = (target - base_mean_c) / lobe_mean_c
corrected = base + lobe × gain
```

---

## Debug Outputs

All written to `--debug-dir` (default `debug_hdri/`):

| File | Contents |
|------|----------|
| `01_wb_preview.png` | HDRI after white balance (tonemapped) |
| `01_wb_sphere_check.png` | Rendered sphere — should be neutral grey |
| `02_exposed_preview.png` | After exposure scaling |
| `03_hot_mask.png` | Sun lobe mask (white = lobe pixels) |
| `03b_hot_mask_after_centre.png` | Mask after HDRI centering |
| `03c_centred_preview.png` | Centred HDRI preview |
| `04a_verify_sphere_pre.png` | Sphere before sun solve |
| `04_grayball_full.png` | Full sphere render |
| `05_grayball_base.png` | Base (non-lobe) contribution only |
| `06_grayball_lobe.png` | Lobe contribution only |
| `07_corrected_preview.png` | Final HDRI preview |
| `08_grayball_after_solve.png` | Sphere after gain solve |
| `09_verify_sphere_final.png` | Final verification — should read target |
| `base_dome_calibrated.exr` | WB + exposure applied, sun zeroed. Sample chart patch 22 here — should read ~0.1882 on all channels. |
| `report.json` | Full pipeline log in JSON |
| `colorchecker/` | CC detection debug images (if `--colorchecker-in-hdri`) |
| `colorchecker/sweep_cube-*.jpg` | All 12 cubemap tiles the library searched |
| `colorchecker/cc_erp_swatches.jpg` | ERP with detected swatch positions overlaid |
| `colorchecker/cc_swatch_comparison.jpg` | 3-row strip: measured / reference / post-WB |

---

## Files

| File | Description |
|------|-------------|
| `hdri_cal.py` | Main pipeline — load, WB, expose, solve, save |
| `colorchecker_erp.py` | ColorChecker detection inside ERP panoramas via cubemap tile sweep |
| `diagnose_swatches.py` | Diagnostic: run detection on any EXR and print per-patch values |
| `requirements.txt` | Python dependencies |

---

## Verification Workflow

After running, check these in order:

1. **`base_dome_calibrated.exr`** — open in Nuke/Houdini, sample the chart patch 22 pixel. Should read `R≈G≈B≈0.188`. If it reads too bright, exposure_scale is wrong. If it has a colour cast, WB is wrong.

2. **`cc_swatch_comparison.jpg`** — Row **W** (post-WB) should be visually neutral on the bottom-row grey patches. Patch 22 has a red border.

3. **`09_verify_sphere_final.png`** — Rendered Lambertian sphere should be a neutral grey disc with no visible colour cast.

4. **`report.json`** — Check `"colorchecker"` block for confidence score (>0.8 = good detection), and `"sphere_verify_final"` for luma vs target.

---

## Known Limitations

- ColorChecker detection requires the chart to be at least ~3–5% of the panorama width and not in extreme highlight/shadow.
- The cubemap sweep uses 90° FOV tiles. Charts at exactly the face boundary of the first pass will be caught by the 45°-rotated second pass.
- Sun energy reconstruction assumes the lobe mask correctly isolates the hot disc. Overcast HDRIs with no clear lobe will produce a near-zero lobe mean and the gain solve will be unstable — use `--sphere-solve none` in that case.
- LDR input (JPG/PNG) is treated as sRGB and linearised automatically. Ensure your input is not double-gamma-encoded.
