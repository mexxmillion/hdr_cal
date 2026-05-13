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
- No final post-WB rebalance. WB+exposure already lock patch 22 to albedo.
- No chart-vs-sphere disagreement warnings.
- No automatic colour-cast warnings.
- No "chart-neutral guarantee" snap. The math is correct by construction.

What patch 22 reads in the final EXR: **exactly `albedo / albedo / albedo`**.

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
