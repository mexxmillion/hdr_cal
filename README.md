# HDRI Calibration Pipeline

A photometrically accurate HDRI reconstruction and calibration tool for VFX / IBL workflows.

Takes a clipped or LDR equirectangular environment map, applies white balance and exposure from a ColorChecker chart you place by hand, gain-solves the sun lobe, and writes a calibrated EXR ready for image-based lighting.

The workflow is intentionally narrow:
1. **Place the chart** — draw a search rectangle on the latlong preview, click 4 corners.
2. **Process** — WB and exposure from patch 22, sun-lobe gain solve, save EXR.

That's it. No batch queue, no auto-detect black box, no second-pass rebalance steps.

---

## Files

| File | Purpose |
|---|---|
| `hdri_cal_gui.py` | PySide6 GUI — primary entry point |
| `hdri_cal.py` | Pipeline core — CLI + callable API |
| `colorchecker_erp.py` | ColorChecker sampling on equirectangular images |
| `cc_debug.py` | Stand-alone CC detection debugger (drop a JPG/PNG to test) |

---

## Installation

```bash
pip install -r requirements.txt
```

OpenEXR backend (needed for `.exr` read/write) — usually picked up automatically from `opencv-python`. If EXR fails:

```bash
# Windows
pip install imageio[openexr]

# macOS
brew install openexr && pip install openexr

# Linux
sudo apt install libopenexr-dev && pip install openexr
```

---

## GUI usage (recommended)

```bash
python hdri_cal_gui.py
```

### Workflow

1. **Drag and drop** an EXR / HDR / PNG / WebP onto the window (or click **📂 Open**).
2. On the **Source** tab, **drag a rectangle** over the chart. The rect is just a hint for the corner-picker — it controls the FOV/yaw/pitch of the rectilinear crop you'll work on.
3. Click **Place Chart Corners**. A second window opens showing a rectilinear projection of your rect. Click 4 corners in order **TL → TR → BR → BL**. As soon as 4 are placed, the CC24 reference-swatch overlay renders in your quad so you can visually verify the placement. **Drag any corner** to fine-tune.
4. Click **▶ Process**. The pipeline runs the chart WB + exposure, sun-lobe gain solve, and writes the EXR.

### Viewer controls

- **Viewer EV slider** (top of preview): −10 … +6 EV scrub. Display only — doesn't change the saved data. Reads the `.f16.npy` linear companion the pipeline writes next to each preview PNG.
- **Pixel probe** (bottom of preview): hover any tab to see `x / y / R / G / B / luma` in scene-linear (ACEScg) plus the resolved display swatch at the current viewer EV.

### Preview tabs

| Tab | Source |
|---|---|
| **Source** | The loaded HDRI (where you draw the search rect) |
| **WB** | After white balance applied |
| **Exposed** | After exposure scale applied |
| **Lobe** | The hot-lobe mask |
| **Chart Tile** | Rectilinear crop of the search rect |
| **Rectified** | Chart warped to a canonical CC24 grid |
| **Swatches** | Measured swatches vs reference |
| **Solved** | Post sun-lobe gain |
| **Final** | The actual output EXR |
| **Sphere** | Rendered grey sphere — only when **Validation** is enabled |

### Settings

**Output**
- `Suffix` / `Output dir` / `Debug dir` / `Resolution` — file locations and processing res.

**Calibration**
- `Mode`: `auto` (the supported workflow) or `advanced` (exposes every internal knob).
- `Input primaries`: `auto` / `acescg` / `srgb`. `auto` interprets `.exr`/`.hdr` as ACEScg and LDR images as sRGB.
- `Exposure`: ±8 EV camera-style stops. Drives the input intensity multiplier (`2^EV`).
- `Temperature` / `Tint`: photographic WB.
- `Reset Photographic`: snap back to `0 EV / 6500 K / 0.0`.
- `Centre HDRI on sun`: rotate the latlong so the sun sits at the centre column.

**Chart Detection**
- Status of the search rect + 4-corner placement, with `✕` Clear icons.
- `Place Chart Corners` opens the picker.

**Advanced (only visible in Advanced mode)**
- `WB source` / `Exposure source`: `auto` / `chart` / `sphere` / `meter` / `none`. Auto mode forces `chart` when the user has placed corners, otherwise falls back to pixel-average meter.
- `Sun solve`: `auto` / `energy_conservation` / `sun_facing_card` / `sun_facing_vertical` / `none`.
- `Albedo`: target grey reflectance (default 0.18).
- `Sun threshold` / `Lobe neutralise` / `Gain ceiling` / `Gain rolloff`.
- **`Run energy & calibration validation`** (off by default) — opt-in diagnostic that renders a grey sphere and prints the irradiance map / E=π check / grey-card predictions. Adds ~1–2 s.

### Actions

Bottom of the settings column:
- **📂 Open** / **✕ Clear** — load or unload a file.
- **▶ Process** — full pipeline → output EXR.
- **⚡ Validate** — quick metrics, no EXR written.
- **■ Abort** — kill a running job.
- **📁 Output Folder** — open the output directory.

---

## CLI usage

```bash
# Process with a chart (you've placed corners in the GUI and saved a config),
# or pass --colorchecker-in-hdri to use the manual flow at the CLI level
python hdri_cal.py shot.exr --out shot_cal.exr

# Treat a linear-sRGB HDR as sRGB primaries
python hdri_cal.py shot_srgb.exr --colorspace srgb --out shot_cal.exr

# Validate only — no EXR written, just a report.json
python hdri_cal.py shot.exr --validate-only
```

### Inputs

`.exr`, `.hdr`, `.jpg`, `.jpeg`, `.png`, `.webp`. EXR/HDR default to ACEScg; LDR formats default to sRGB and are linearised+converted on load.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--out PATH` | `corrected.exr` | Output calibrated EXR |
| `--debug-dir DIR` | `debug_hdri` | Debug previews + `report.json` |
| `--res` | `full` | `full` / `half` / `quarter` |
| `--colorspace` | auto | `acescg` or `srgb` — forces input-primaries interpretation |
| `--validate-only` | off | No EXR written, just report.json |
| `--center-hdri` / `--no-center-hdri` | on | Rotate sun to phi=0 |
| `--wb-source` | `auto` | `auto` / `chart` / `sphere` / `meter` / `none` |
| `--exposure-source` | `auto` | same set as `--wb-source` |
| `--albedo` | `0.18` | Target grey reflectance |
| `--sphere-solve` | `auto` | `auto` (= energy_conservation) / `sun_facing_card` / `sun_facing_vertical` / `none` |
| `--sun-threshold` | `0.1` | Lobe boundary fraction below peak |
| `--lobe-neutralise` | `1.0` | Desaturate the lobe before gain (1.0 = fully achromatic) |
| `--sun-gain-ceiling` | `2000` | Hard cap on per-channel sun gain |
| `--sun-gain-rolloff` | `500` | Soft rolloff start before the ceiling |

When `--wb-source chart` is requested and a chart was found, the chart is used **unconditionally** — no confidence gating.

---

## Pipeline

```
load → input-primaries conversion → ACEScg working space
  → orientation validate
  → chart sampling (manual corners on the latlong)
  → WB scale from patch 22  (R = G = B at patch 22)
  → exposure scale = albedo / patch22_luma
  → hot-lobe mask + sun gain solve
  → save EXR + previews + .f16.npy companions + report.json
```

What's **not** in the pipeline (by design):
- No WB blending. The chart is ground truth.

---

## Algorithm

This section describes what each step actually does so you can reason about the output.

### 1. Chart sampling (manual placement)

The user clicks 4 corners (TL → TR → BR → BL) on a rectilinear crop of the latlong. We:

1. Build a gnomonic projection of the search rect: `yaw = (cu - 0.5) · 360°`, `pitch = (cv - 0.5) · 180°`, `fov = max(span_u_deg, span_v_deg)`. The output tile aspect matches the rect aspect so the chart isn't squished.
2. Map the 4 corner UVs to tile pixel coordinates by nearest-neighbour lookup through the projection's `map_uv`.
3. Warp the linear HDR tile through a perspective transform from the user's quad into a canonical CC24 rectangle. Rectified target size and per-swatch sampling window scale to the source quad's pixel span.
4. Sample 24 swatches inside an inner sub-rect of each grid cell. Try all 4 rotations of the quad and pick the one whose neutral-ramp luma best correlates with the CC24 reference.

The output is **24 RGB samples in linear ACEScg**. Patch index 21 (`Neutral 5`) drives WB and exposure; the other patches are stored for later debug/diagnostics but don't drive the calibration.

> The pre-WB chroma of the measured patch 22 is logged but never used as a gating metric. A warm-lit chart will read non-neutral before WB — that's *the* condition WB exists to correct.

### 2. White balance

Goal: make `Neutral 5` read `R = G = B`.

```
measured  = cc_measured[21]                        # (3,) ACEScg linear
reference = CC24_LINEAR_SRGB[21]                   # ≈ (0.188, 0.188, 0.188)
scale_raw = reference / measured                   # per-channel
wb_scale  = scale_raw / scale_raw[1]               # G-normalise → green stays at 1.0
wb_img    = img * wb_scale                         # apply globally
```

G-normalising preserves luminance — only chromaticity shifts. After this step `cc_measured[21] · wb_scale` has `R = G = B` exactly.

### 3. Exposure

Goal: make `Neutral 5` read `albedo` (default `0.18`) on all channels.

```
meas_luma_post = luminance(cc_measured[21] · wb_scale)
exposure_scale = albedo / meas_luma_post
exposed        = wb_img · exposure_scale
```

Why this is `albedo / measured_luma` and not something fancier:

For a chart inside the HDRI, the chart pixels **are** scene pixels — they carry the same exposure relationship as everything else. The calibration definition is "make the chart read its own albedo." Substitute `E_on_chart = π` into the general predicted-luma formula `albedo · E / π` and you get exactly `albedo`; the formula collapses to `exposure_scale = albedo / measured`. This is independent of the chart's orientation, because we're only normalising its already-measured radiance, not predicting it from external irradiance.

After exposure: `cc_measured[21]` in the corrected HDRI = `(albedo, albedo, albedo)` exactly.

### 4. Hot-lobe extraction

The sun is detected by thresholding luminance:

```
peak_lum  = percentile(lum, 99.99)
threshold = peak_lum · sun_threshold       # default 0.1 → top 10% of peak
mask      = lum > threshold                # boolean per pixel
```

Connected components are merged into a single lobe centred on the brightest cluster. The lobe's direction and solid angle drop into `hot["center_dir"]`, `hot["mask"]`, etc.

`--center-hdri` (default on) rotates the latlong so the lobe centre lands on `phi = 0` (centre column).

### 5. Sun lobe gain solve

Goal: each colour channel's total upward irradiance equals π (the value a perfectly-lit Lambertian grey card would integrate to).

Decompose the corrected HDRI:

```
base = exposed · (1 - lobe_mask)      # everything except the sun
lobe = exposed · lobe_mask            # the sun
```

Then compute per-channel cosine-weighted irradiance integrals over the upper hemisphere:

```
E_base[c] = Σ base[..., c] · max(cos(θ), 0) · dΩ
E_lobe[c] = Σ lobe[..., c] · max(cos(θ), 0) · dΩ
```

Solve for per-channel gain:

```
gain[c] = (π - E_base[c]) / E_lobe[c]      # what makes E_base + gain·E_lobe = π
gain[c] = clamp(gain[c], 0, ceiling) with smooth rolloff above ceiling - rolloff
```

`apply_sun_gain_per_channel`:

1. **Neutralise the lobe** (`--lobe-neutralise`, default 1.0): inside the lobe mask, replace each pixel's RGB by its luminance broadcast across all channels. Sun discs read as ~5800 K blackbodies; after WB they should be near-neutral anyway, and any residual per-channel imbalance in a clipped sun is sensor artefact, not real spectral content.
2. **Apply gain** to lobe pixels only: `lobe_gained = lobe_neutralised · gain[None, None, :]`.
3. **Recombine**: `corrected = base + lobe_gained`.

Critically, `gain` is applied **only inside `lobe_mask`**. The chart on the ground (`lobe_mask = 0` there) is untouched. Patch 22 stays at `(albedo, albedo, albedo)` from step 3.

### 6. What the chart guarantees

After all five steps the output EXR has these invariants when a chart was placed:

| Pixel | RGB |
|---|---|
| Patch 22 (Neutral 5) | exactly `(albedo, albedo, albedo)` |
| Patch 19 (white) | albedo · CC24_LINEAR_SRGB[18] / CC24_LINEAR_SRGB[21] in each channel — preserved by WB+exposure ratio |
| Sun-disc lobe pixels | neutral and gain-solved so the integrated upper hemisphere is `(π, π, π)` |
| Everything else | preserved chromatic content, exposure rescaled by `wb_scale · exposure_scale` |

Equivalent statement: **a grey card placed on the ground in this HDRI renders to exactly `albedo` luminance in all three channels.** That's the calibration definition.

### 7. Resolution helpers

Three knobs that influence the lobe gain solve:

- `--sun-threshold` (default `0.1`): fraction of peak luma below which pixels are NOT considered part of the lobe. Higher = tighter sun, lower = wider sun.
- `--sun-gain-ceiling` (default `2000`): maximum per-channel gain. Severely clipped suns hit this; raise if energy validation reports < 90% recovery.
- `--sun-gain-rolloff` (default `500`): the gain transitions smoothly into the ceiling instead of snapping. `gain = ceiling - (ceiling - raw) · exp(-(raw - rolloff)/rolloff)` once `raw > rolloff`.

### 8. Optional validation suite

Off by default. When the **Run energy & calibration validation** checkbox is on (or `validate_energy = True` in the config), after the main pipeline:

- Render an actual Lambertian grey ball from the corrected HDRI at `sphere_res` resolution.
- Integrate the 6-direction irradiance map (`+y`, `-y`, `±x`, `±z`).
- Compute `E_upper`, `E_lower`, `E_sun`, `E_upper_chroma_norm`, etc.
- Report per-direction predicted card values, sphere-mean RGB, rendered-vs-analytical deviation, and chroma imbalance.
- Adds ~1–2 s and writes `meta["energy_validation"]` to `report.json` + `08_verify_sphere_final.png`.

Nothing in this suite modifies the saved EXR — it's purely a diagnostic for verifying the math closed.

### 9. What's deliberately absent

| Step | Why we don't do it |
|---|---|
| WB sphere blend | Sphere WB is contaminated by the very cast the chart removes. Mixing them makes WB worse on coloured-light scenes. |
| Final post-balance rescale | Would un-do the chart neutrality we just set. |
| Chart-neutral guarantee snap | Patch 22 is already exact by construction (steps 2+3). A "guarantee" step that re-samples is just a bug surface — and was empirically tinting the output when the re-sample window landed on swatch borders on the warped ERP. |
| Patch-confidence gate | Pre-WB chroma is meaningless as a confidence metric. If a chart was found, use it. |
| Pose-aware exposure for in-HDRI charts | The chart pixels carry their own exposure relationship to the scene; pose correction is only meaningful for separately-exposed reference plates. |

---

## Debug outputs

Per-run, in `<debug-dir>/`:

```
01_wb_preview.png             + .f16.npy + .f16.json
02_exposed_preview.png        + .f16.npy + .f16.json
03_hot_mask.png
07_corrected_preview.png      + .f16.npy + .f16.json
08_final_hdri_preview.png     + .f16.npy + .f16.json
08_verify_sphere_final.png    (only if Validation is enabled)
colorchecker/
    manual_tile_detected.jpg
    manual_rectified.png
    cc_swatch_comparison.jpg
report.json
```

`.f16.npy` is float16 RGB scene-linear (working colorspace). `.f16.json` is a sidecar with the colorspace tag and the 99.5-percentile anchor the PNG was normalised to. The GUI uses both for the viewer-EV slider and the pixel probe.

The GUI's **Detect** path (currently hidden — manual placement is the supported workflow) additionally writes `cc_debug_out/detect_<stem>/` containing `rect_tile_*.{jpg,exr}` and `cc_detected_tile.jpg` for triage with `cc_debug.py`.

---

## Troubleshooting

**Chart corners drift / probe says non-neutral after Process**
- Make sure you've placed exactly 4 corners and they sit inside the swatches, not on the black surround.
- Drag corners to fine-tune. The reference-swatch overlay updates live; aim for visual coincidence.

**Output looks blue / has a cast**
- Check that **WB source** wasn't forced off in Advanced mode. In auto mode with corners placed, the chart is used unconditionally.

**GUI freezes briefly when I change settings**
- Only colorspace / Exposure / Temperature / Tint rebuild the Source preview (because they change what gets displayed). Everything else now just persists the config — no disk reload.

**EXR read errors**
- Install the OpenEXR system library (see Installation).

**Manual placement on EXRs with the chart in the lower half showed sky**
- Fixed in `_rect_uv_to_tile_params` — pitch sign was inverted. Update to the latest commit.

---

## Notes on colorspace

- `.exr` / `.hdr` assumed **ACEScg (AP1 linear)** unless `--colorspace srgb`.
- `.jpg` / `.png` / `.webp` assumed **sRGB**, linearised on load, then converted to ACEScg.
- CC24 neutral patches are achromatic (R=G=B), so chart WB derivation is colorspace-agnostic.
- All internal math runs in linear light.

---

## License

MIT
