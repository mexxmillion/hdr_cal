# A Direct, Chart-Anchored HDRI Calibration Pipeline for VFX Image-Based Lighting

## Abstract

We describe a small, opinionated tool for calibrating equirectangular high-dynamic-range environment maps so they can be used as physically correct light sources in image-based lighting (IBL) workflows. The pipeline takes a single HDRI plus four user-placed ColorChecker (CC24) corner clicks and produces a calibrated output EXR in which (i) the Neutral 5 patch reads exactly the target albedo in all three colour channels, and (ii) the integrated upper-hemisphere irradiance per channel equals π (the closed-form value required for a Lambertian grey card at the origin to render to its own albedo). The pipeline deliberately omits several plausible "polish" steps — WB cross-checks, sphere blending, post-balance rescales, confidence gating — because each one is provably either redundant or actively destructive of the chart-derived calibration. The result is a workflow that is short to describe, short to implement, and easy to audit when output ever looks wrong.

## 1. Background and motivation

A typical VFX shot composites synthetic geometry into a live-action plate using image-based lighting from a captured panorama. For the synthetic surfaces to integrate believably, the captured HDRI must be photometrically calibrated: its dynamic range must match the scene's true radiance ratios, its white point must match the scene's dominant illuminant, and its total integrated upward irradiance must place a virtual grey card at the correct exposure.

Three failure modes dominate uncalibrated HDRIs in practice:

1. **Clipped sun.** Even bracketed HDR captures saturate the camera sensor inside the sun disc, leading to a flat, low-energy lobe and incorrect cast-shadow energy on CG objects.
2. **Wrong white balance.** The capture's nominal colour temperature is rarely the true illuminant; renderers compose images in linear light, so any residual cast multiplies through every lighting calculation.
3. **Wrong absolute exposure.** Cameras meter relatively; lookdev artists need an absolute anchor (typically the 18 % grey card) to align CG and live-action exposure.

The standard workflow for addressing these is to (a) place a CC24 ColorChecker in the scene during the capture, (b) sample its neutral patches off-line to derive WB and exposure, and (c) apply a hot-lobe gain solve to recover clipped sun energy. The art is in doing this without contaminating the calibration with the very errors it is trying to remove.

## 2. Method

The pipeline runs five steps after loading and converting the input to a working ACEScg linear-light representation.

### 2.1 Chart sampling

We do not perform automatic chart detection in the deployed workflow. Auto-detectors that work on rectified studio images fail predictably on charts seen at oblique angles, in shadow, on grass, or in scenes where the dominant chroma fights the chart's own swatches. Instead, the user draws a search rectangle on the latlong, then clicks four chart corners (TL → TR → BR → BL) on a rectilinear projection of that rectangle. The rectilinear FOV is set to the rectangle's exact angular span (no margin, no clamp), so the chart fills the picker canvas at the same aspect it occupies in the captured scene. A live CC24 reference-swatch overlay registered through the perspective transform of the user's quad lets the user drag corners until the reference colours visually coincide with the captured swatches.

Given the four corners, we warp the linear HDR through a perspective transform into a canonical 6 × 4 swatch grid, sample 24 patches in an inner sub-rect of each cell, and try all four rotations of the quad — selecting the rotation whose neutral-ramp luma best correlates with the CC24 reference. The output is 24 RGB samples in linear ACEScg.

### 2.2 White balance

Let `m` be the measured patch 22 RGB and `r` the reference. We compute a per-channel scale, then green-normalise to preserve luminance:

```
s = r / m
wb = s / s_G
img'  = img · wb
```

After this step `m · wb` has R = G = B by construction.

### 2.3 Exposure

The exposure scale forces the white-balanced patch 22 to read the user's target albedo (0.18 by default):

```
e   = albedo / luminance(m · wb)
img'' = img' · e
```

This formulation is the standard camera grey-card meter rearranged. A more general form would be `e = (albedo · E_chart / π) / luminance(m · wb)` where `E_chart` is the incident irradiance on the chart, but for a chart embedded in the captured environment the calibration definition fixes `E_chart = π` and the formula collapses to the simple case. Substituting any other irradiance (computed, e.g., from a pose-aware upward integration of the pre-exposure image) destroys self-consistency: the chart pixels stop reading their own albedo.

### 2.4 Hot-lobe mask

The sun is localised by thresholding luminance against a fraction of the peak:

```
peak = percentile(luminance(img''), 99.99)
mask = luminance(img'') > peak · sun_threshold
```

Connected components are merged to a single lobe whose centre direction and solid angle are recorded. Optionally the latlong is rotated so the lobe lands on `φ = 0`.

### 2.5 Per-channel sun gain solve

The energy-conservation target is `E_upper[c] = π` per channel, where `E_upper[c]` is the cosine-weighted irradiance from the upper hemisphere on a virtual grey card oriented upward. Decomposing the corrected HDRI into base + lobe contributions:

```
E_base[c] = Σ base[..., c] · max(cos θ, 0) · dΩ
E_lobe[c] = Σ lobe[..., c] · max(cos θ, 0) · dΩ
```

the closed-form per-channel gain is:

```
g[c] = (π − E_base[c]) / E_lobe[c]
```

with a smooth rolloff applied above a configurable ceiling to prevent unbounded boosts on severely clipped suns. The lobe pixels are first neutralised (RGB replaced by broadcast luminance) to eliminate sensor-clipping chromatic artefacts, then multiplied by `g`, then recombined with the unchanged base. **The mask is binary at recombination time, so the chart on the ground — which is never inside `mask` — is preserved bit-for-bit through the gain step.** The calibration invariant established in §2.2–§2.3 is therefore unaffected by §2.5.

## 3. What the pipeline does not do

A surprising amount of design effort went into *removing* polish steps that look helpful but degrade the calibration:

- **WB cross-check / chart-vs-sphere blending.** Sphere-derived WB is contaminated by the same illuminant cast the chart is removing. Blending the two on coloured-light scenes (golden hour, overcast blue, etc.) makes WB systematically worse than chart-only.
- **Pre-WB chart confidence.** A common reflex is to score chart detections by chroma deviation from the neutral reference and demote low-confidence detections. But a chart's pre-WB chroma is exactly the metric WB is correcting — using it as a gate downgrades correctly-identified warm-lit charts to "low confidence" and silently falls back to pixel-average WB, producing a heavily cast output.
- **Post-pipeline rebalance.** A "final balance" step that re-makes the upward irradiance or sun direction neutral undoes the chart neutrality just established in §2.2–§2.3.
- **Chart-neutrality guarantee snap.** Sampling the corrected EXR at the chart's UV centres and rescaling so patch 22 reads exactly albedo seems like a safety net. In practice §2.2–§2.3 already establish that exactly; a guarantee step that re-samples is purely a new bug surface — and one that, in our testing, picked up swatch borders on the warped ERP and tinted the entire HDRI.
- **Pose-aware exposure for in-HDRI charts.** Substituting a pose-corrected `E_chart` into the exposure formula breaks the self-consistency described in §2.3. Pose correction is meaningful only for separately-exposed reference plates.

These exclusions are the central design contribution. The point is not that they are subtly buggy; it is that each one *correctly* implements an algorithm whose premise is wrong for an in-HDRI chart.

## 4. Implementation notes

The tool is a single-file GUI written in PySide6 driving a NumPy/OpenCV pipeline. The supported workflow is one HDRI at a time, drag-and-drop, with two action buttons (Process, Validate). All intermediate stages write a display-mapped PNG plus a float16 raw-scene-linear `.npy` companion and a JSON sidecar carrying the working-colorspace tag and a 99.5-percentile display anchor. A viewer-EV slider and a pixel probe read these companions to display true scene-linear RGB at any exposure stop the user dials in, including −10 EV inspections of clipped highlights that the PNG-only display cannot show.

A small per-stage validation suite (off by default) renders a Lambertian grey ball from the corrected HDRI, integrates the six cardinal irradiance directions, and reports rendered-vs-analytical deviation; this is a diagnostic, not a calibration step, and never modifies the saved EXR.

## 5. Discussion

The calibration produced by this pipeline is meaningful in exactly one sense: a Lambertian grey card placed at the world origin and oriented upward, lit by the output HDRI, renders to its own albedo on every colour channel. Every step in §2 follows mechanically from that definition. Steps that violate the definition are absent. Steps that preserve the definition (the sun gain solve) operate on disjoint pixel sets from the calibration anchor (the chart).

The tool is small (~3 kLOC of pipeline, ~2 kLOC of GUI), opinionated, and intentionally easy to audit. When a user reports a cast in the output, the absence of optional rebalance / blend / guarantee steps means there are exactly three places to look — chart sampling, WB scale derivation, exposure scale derivation — and each is one screen of code.

## 6. Availability

Source: <https://github.com/mexxmillion/hdr_cal>. License: MIT.
