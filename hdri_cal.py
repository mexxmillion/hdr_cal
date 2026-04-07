#!/usr/bin/env python3
"""
hdri_cal.py — HDRI calibration pipeline for VFX / IBL compositing

Pipeline:
  1. Load any latlong (LDR jpg/png, .hdr, .exr)
  2. Optional white balance (Kelvin / RGB scale / swatch)
  3. Metering (bottom_dome / whole_scene / swatch)
  4. Exposure solve
  5. Hot-lobe extraction (smoothstep mask, solid-angle-correct centroid)
  6. HDRI centering — shift azimuth so dominant light sits at φ=0 (centre column)
     Works correctly for extreme luminance ranges (sun >> sky) via log-domain
     weighted centroid on the hot mask, not raw pixel values.
  7. Sun gain solve (vectorised Lambertian sphere render, split base+lobe)
  8. Optional: reference grey/chrome ball image for physical exposure calibration
  9. Optional: ColorChecker detection & 3×3 matrix colour correction
 10. Save calibrated EXR + debug outputs + JSON report

Dependencies (all pip-installable):
  pip install numpy opencv-python imageio scipy
  pip install openexr                             # EXR I/O (or pyexr as fallback)
  pip install colour-science                     # colour correction
  pip install colour-checker-detection           # checker segmentation
"""

import os
import sys
import json
import math
import argparse
import warnings
import traceback

import cv2
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

# Point colour-checker-detection at bundled model so it never downloads
_BUNDLED_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if os.path.isdir(_BUNDLED_MODELS):
    os.environ.setdefault(
        "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY", _BUNDLED_MODELS
    )

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    import pyexr
    HAVE_PYEXR = True
except Exception:
    HAVE_PYEXR = False

try:
    import OpenEXR
    HAVE_OPENEXR = True
except Exception:
    HAVE_OPENEXR = False

try:
    import Imath
    HAVE_IMATH = True
except Exception:
    HAVE_IMATH = False

try:
    import colour
    import colour_checker_detection as ccd
    HAVE_COLOUR = True
except Exception:
    HAVE_COLOUR = False


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def log(msg):
    print(f"[hdri-cal] {msg}", flush=True)

def warn(msg):
    print(f"[hdri-cal] WARNING: {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Colour / IO helpers
# ══════════════════════════════════════════════════════════════════════════════

def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)

def _to_display_srgb_linear(img_linear, source_colorspace="acescg"):
    """Convert a linear working image into linear sRGB for display."""
    img_linear = np.clip(np.asarray(img_linear, dtype=np.float32), 0.0, None)
    cs = (source_colorspace or "acescg").lower().replace("-", "").replace("_", "")
    if cs in ("acescg", "aces", "ap1") and acescg_to_srgb_linear is not None:
        return np.clip(acescg_to_srgb_linear(img_linear), 0.0, None)
    return img_linear

def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _read_exr_with_openexr_file(path):
    """Read EXR via the modern OpenEXR.File API (v3.3+)."""
    with OpenEXR.File(path) as infile:
        channels = infile.channels()
        if "RGB" in channels:
            return np.asarray(channels["RGB"].pixels)
        if "RGBA" in channels:
            return np.asarray(channels["RGBA"].pixels)[..., :3]
        if all(name in channels for name in ("R", "G", "B")):
            return np.stack([
                np.asarray(channels["R"].pixels),
                np.asarray(channels["G"].pixels),
                np.asarray(channels["B"].pixels),
            ], axis=-1)
        if "Y" in channels:
            return np.asarray(channels["Y"].pixels)
        raise RuntimeError(f"Unsupported EXR channel layout in '{path}': {sorted(channels.keys())}")


def _read_exr_with_openexr_legacy(path):
    """Read EXR via the legacy OpenEXR.InputFile API."""
    if not (HAVE_OPENEXR and HAVE_IMATH):
        raise RuntimeError("Legacy OpenEXR EXR fallback requires both OpenEXR and Imath")

    exr = OpenEXR.InputFile(path)
    try:
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

        channel_map = header.get("channels", {})
        names = set(channel_map.keys())
        if all(name in names for name in ("R", "G", "B")):
            r = np.frombuffer(exr.channel("R", pixel_type), np.float32).reshape(height, width)
            g = np.frombuffer(exr.channel("G", pixel_type), np.float32).reshape(height, width)
            b = np.frombuffer(exr.channel("B", pixel_type), np.float32).reshape(height, width)
            return np.stack([r, g, b], axis=-1)
        if "Y" in names:
            return np.frombuffer(exr.channel("Y", pixel_type), np.float32).reshape(height, width)
        raise RuntimeError(f"Unsupported EXR channel layout in '{path}': {sorted(names)}")
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()


def _read_exr(path):
    """Read EXR using whichever backend is available in the environment."""
    if HAVE_OPENEXR and hasattr(OpenEXR, "File"):
        return _read_exr_with_openexr_file(path)

    if HAVE_OPENEXR and hasattr(OpenEXR, "InputFile"):
        return _read_exr_with_openexr_legacy(path)

    if HAVE_PYEXR:
        return pyexr.read(path)

    raise RuntimeError("EXR requires OpenEXR or pyexr Python bindings")


def _write_exr_with_openexr_file(path, img):
    """Write EXR via the modern OpenEXR.File API (v3.3+)."""
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"RGB": np.asarray(img, dtype=np.float32)}
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(path)


def _write_exr_with_openexr_legacy(path, img):
    """Write EXR via the legacy OpenEXR.OutputFile API."""
    if not (HAVE_OPENEXR and HAVE_IMATH):
        raise RuntimeError("Legacy OpenEXR EXR fallback requires both OpenEXR and Imath")

    img = np.asarray(img, dtype=np.float32)
    height, width = img.shape[:2]
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pixel_type),
        "G": Imath.Channel(pixel_type),
        "B": Imath.Channel(pixel_type),
    }
    if hasattr(OpenEXR, "ZIP_COMPRESSION"):
        header["compression"] = OpenEXR.ZIP_COMPRESSION

    exr = OpenEXR.OutputFile(path, header)
    try:
        exr.writePixels({
            "R": np.ascontiguousarray(img[..., 0]).astype(np.float32).tobytes(),
            "G": np.ascontiguousarray(img[..., 1]).astype(np.float32).tobytes(),
            "B": np.ascontiguousarray(img[..., 2]).astype(np.float32).tobytes(),
        })
    finally:
        close = getattr(exr, "close", None)
        if callable(close):
            close()


def _write_exr(path, img):
    """Write EXR using whichever backend is available in the environment."""
    if HAVE_OPENEXR and hasattr(OpenEXR, "File"):
        _write_exr_with_openexr_file(path, img)
        return

    if HAVE_OPENEXR and hasattr(OpenEXR, "OutputFile"):
        _write_exr_with_openexr_legacy(path, img)
        return

    if HAVE_PYEXR:
        pyexr.write(path, img)
        return

    raise RuntimeError("EXR output requires OpenEXR or pyexr Python bindings")

def load_image_any(path, target_colorspace="acescg", input_colorspace=None):
    """
    Load any image to float32 linear RGB in the requested colorspace.

    `input_colorspace` describes the source primaries for already-linear HDR data.
    The pipeline now processes internally in ACEScg, so linear sRGB HDR inputs are
    converted up front when requested.
    """
    ext = os.path.splitext(path)[1].lower()
    log(f"Loading: {path}")
    if ext == ".exr":
        img = _read_exr(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        src_cs = (input_colorspace or target_colorspace or "acescg").lower().replace("-", "").replace("_", "")
        dst_cs = (target_colorspace or src_cs).lower().replace("-", "").replace("_", "")
        if src_cs in ("srgb", "rec709", "linear") and dst_cs in ("acescg", "aces", "ap1") and srgb_linear_to_acescg is not None:
            img = srgb_linear_to_acescg(img)
            img = np.clip(img, 0.0, None)
            log("Loaded EXR — interpreted as linear sRGB and converted to ACEScg")
        else:
            log(f"Loaded EXR — interpreted as {input_colorspace or target_colorspace or 'acescg'}")
        return img, True
    if ext == ".hdr":
        img = imageio.imread(path).astype(np.float32)
        if img.shape[-1] > 3:
            img = img[..., :3]
        src_cs = (input_colorspace or target_colorspace or "acescg").lower().replace("-", "").replace("_", "")
        dst_cs = (target_colorspace or src_cs).lower().replace("-", "").replace("_", "")
        if src_cs in ("srgb", "rec709", "linear") and dst_cs in ("acescg", "aces", "ap1") and srgb_linear_to_acescg is not None:
            img = srgb_linear_to_acescg(img)
            img = np.clip(img, 0.0, None)
            log("Loaded .hdr — interpreted as linear sRGB and converted to ACEScg")
        else:
            log(f"Loaded .hdr — interpreted as {input_colorspace or target_colorspace or 'acescg'}")
        return img, True
    # LDR path — linearise from sRGB then optionally convert to ACEScg
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    img = srgb_to_linear(img)
    cs = target_colorspace.lower().replace("-", "").replace("_", "")
    if cs in ("acescg", "aces", "ap1") and srgb_linear_to_acescg is not None:
        img = srgb_linear_to_acescg(img)
        img = np.clip(img, 0.0, None)
        log(f"Loaded LDR → linearised sRGB → ACEScg")
    else:
        log(f"Loaded LDR → linearised sRGB")
    return img.astype(np.float32), False

def save_exr(path, img):
    img = np.asarray(img, dtype=np.float32)
    if not path.lower().endswith(".exr"):
        path += ".exr"
    _write_exr(path, img)
    log(f"Saved EXR: {path}")

def save_png_preview(path, img_linear, percentile=99.5, source_colorspace="acescg"):
    img_linear = np.clip(img_linear, 0.0, None)
    lum = luminance(img_linear)
    valid = lum[np.isfinite(lum)]
    denom = max(1e-6, np.percentile(valid, percentile)) if valid.size else 1.0
    display_linear = _to_display_srgb_linear(np.clip(img_linear / denom, 0.0, None), source_colorspace)
    view = linear_to_srgb(np.clip(display_linear, 0.0, 1.0))
    imageio.imwrite(path, np.clip(view * 255 + 0.5, 0, 255).astype(np.uint8))
    log(f"Saved preview: {path}")

def save_mask_preview(path, mask):
    m = np.clip(mask, 0.0, 1.0)
    imageio.imwrite(path, np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8))
    log(f"Saved mask: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

def parse_rgb_scale(s):
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("RGB multiplier must be like: 1.0,1.0,1.0")
    return vals

def parse_xy(s):
    vals = [int(x.strip()) for x in s.split(",")]
    if len(vals) != 2:
        raise ValueError("Coordinates must be like: x,y")
    return vals


# ══════════════════════════════════════════════════════════════════════════════
# White balance
# ══════════════════════════════════════════════════════════════════════════════

def kelvin_to_rgb_scale(kelvin):
    t = kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(max(t, 1e-6)) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0
    rgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 1e-4, None)
    scale = 1.0 / rgb
    scale /= np.mean(scale)
    return scale.astype(np.float32)

def _scale_to_kelvin_approx(scale):
    """
    Very rough WB scale → equivalent Kelvin via the Planckian locus.
    Good enough for logging; not for precision use.
    R/B ratio correlates with colour temperature.
    """
    s = scale / (scale[1] + 1e-8)
    rb = s[0] / (s[2] + 1e-8)
    kelvin = float(np.clip(5600.0 * (rb ** -0.8), 1500.0, 20000.0))
    return kelvin

def estimate_wb_from_dome(img, mode="upper_dome", lum_lo_pct=2.0, lum_hi_pct=97.0):
    """
    Grey-world white balance estimation from the environment map dome.

    Assumes the solid-angle-weighted average colour of the chosen region
    should be achromatic (R=G=B).  Returns a (3,) scale vector, normalised
    so luminance is unchanged (mean(scale)=1).

    This is a HEURISTIC — it works well when:
      - The scene has no dominant single-colour tint (e.g. golden hour will
        be pushed toward cooler tones because the algorithm thinks orange IS
        the neutral)
      - The dome samples a reasonably balanced mix of light colours

    It fails gracefully — if the estimate is implausible (< 0.5 or > 2.0 on
    any channel relative to green) it clamps and warns rather than blowing up.

    Args:
        img        : (H, W, 3) linear float, the ERP environment map
        mode       : region to average over —
                       "full_dome"   — entire 360° sphere (most inclusive)
                       "upper_dome"  — upper hemisphere only (y > 0, sky+sun)
                                       avoids ground colour biasing the estimate
                       "hot_exclude" — upper dome minus the top 3% luminance
                                       pixels (strips the sun disc whose extreme
                                       brightness would dominate even after solid-
                                       angle weighting)
        lum_lo_pct : exclude pixels below this luminance percentile (noise floor)
        lum_hi_pct : exclude pixels above this luminance percentile
                       — for hot_exclude mode this is overridden to 97

    Returns:
        scale      : (3,) float32 RGB multipliers, luminance-neutral
        info       : dict with diagnostics
    """
    h, w = img.shape[:2]

    # Solid-angle weight per pixel — mandatory for ERP.
    # Without this, pole pixels (tiny solid angle) would be over-represented.
    theta = (np.arange(h) + 0.5) / h * np.pi   # (H,)
    sin_theta = np.sin(theta)                    # solid-angle row weight
    dOmega = sin_theta[:, None] * (np.pi / h) * (2 * np.pi / w)  # (H, W)

    lum = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]

    # ── Region mask ───────────────────────────────────────────────────────
    # Build y-coordinate map for upper hemisphere test (y-up ERP convention)
    cos_theta = np.cos(theta)   # y component of unit direction
    y_map = np.broadcast_to(cos_theta[:, None], (h, w))

    if mode == "full_dome":
        region = np.ones((h, w), dtype=bool)
    elif mode == "upper_dome":
        region = y_map > 0.0
    elif mode == "hot_exclude":
        region = y_map > 0.0
        lum_hi_pct = 97.0   # always strip top 3% in this mode
    else:
        raise ValueError(f"Unknown dome WB mode: {mode!r}. "
                         f"Choose full_dome | upper_dome | hot_exclude")

    # ── Luminance range filter ────────────────────────────────────────────
    # Exclude noise floor and saturated/clipped pixels.
    # We compute percentiles only within the region to avoid cross-contamination.
    region_lum = lum[region]
    if region_lum.size < 100:
        warn("Dome WB: fewer than 100 valid pixels in region — falling back to no WB.")
        return np.ones(3, dtype=np.float32), {"mode": mode, "applied": False,
                                               "reason": "insufficient pixels"}

    lo = float(np.percentile(region_lum, lum_lo_pct))
    hi = float(np.percentile(region_lum, lum_hi_pct))

    valid = region & (lum >= lo) & (lum <= hi)

    if valid.sum() < 50:
        warn("Dome WB: too few pixels after luminance filtering.")
        return np.ones(3, dtype=np.float32), {"mode": mode, "applied": False,
                                               "reason": "too few after lum filter"}

    # ── Solid-angle-weighted mean RGB ─────────────────────────────────────
    w_map = dOmega * valid.astype(np.float32)
    w_sum = float(w_map.sum()) + 1e-8

    mean_rgb = np.array([
        float(np.sum(img[..., c] * w_map)) / w_sum
        for c in range(3)
    ], dtype=np.float32)

    # ── Solve for neutral scale ───────────────────────────────────────────
    # We want mean_rgb * scale = [L, L, L] where L = luminance(mean_rgb).
    # That means scale_c = L / mean_rgb_c.
    # Normalise so luma is preserved (mean(scale) = 1).
    L = float(0.2126*mean_rgb[0] + 0.7152*mean_rgb[1] + 0.0722*mean_rgb[2])
    if L < 1e-8:
        warn("Dome WB: mean luminance is near zero.")
        return np.ones(3, dtype=np.float32), {"mode": mode, "applied": False,
                                               "reason": "zero luminance"}

    scale_raw = L / np.clip(mean_rgb, 1e-8, None)

    # ── Plausibility clamp ────────────────────────────────────────────────
    # If any channel needs more than a 2-stop correction (4×) relative to
    # the green channel, something is wrong — clamp and warn.
    # This prevents runaway correction on scenes with strong intentional tints.
    scale_rel = scale_raw / (scale_raw[1] + 1e-8)   # relative to green
    MAX_CORRECTION = 3.0   # stops: 2^1.5 ≈ 3×
    if np.any(scale_rel > MAX_CORRECTION) or np.any(scale_rel < 1.0 / MAX_CORRECTION):
        warn(f"Dome WB: estimated correction is large "
             f"(R={scale_rel[0]:.2f}× G=1.00 B={scale_rel[2]:.2f}× relative). "
             f"Clamping to ±{MAX_CORRECTION}×. "
             f"Scene may have a strong intentional colour tint — "
             f"consider using --kelvin or --rgb-scale instead.")
        scale_rel = np.clip(scale_rel, 1.0/MAX_CORRECTION, MAX_CORRECTION)
        scale_raw = scale_rel * scale_raw[1]

    # Final luminance-neutral normalisation
    scale_luma = float(0.2126*scale_raw[0] + 0.7152*scale_raw[1] + 0.0722*scale_raw[2])
    scale      = scale_raw / max(scale_luma, 1e-8)

    # Estimate equivalent Kelvin (rough — for logging/report only)
    kelvin_est = _scale_to_kelvin_approx(scale)

    log(f"Dome WB ({mode}): mean_rgb={mean_rgb.tolist()}")
    log(f"  scale: R={scale[0]:.4f} G={scale[1]:.4f} B={scale[2]:.4f}")
    log(f"  ~{kelvin_est:.0f}K equivalent  "
        f"valid pixels: {int(valid.sum())}  solid-angle weighted")

    info = {
        "mode":             mode,
        "applied":          True,
        "mean_rgb_dome":    mean_rgb.tolist(),
        "scale":            scale.tolist(),
        "kelvin_approx":    kelvin_est,
        "valid_pixels":     int(valid.sum()),
        "lum_range":        [float(lo), float(hi)],
    }
    return scale.astype(np.float32), info


def estimate_wb_and_exposure_from_pixel_average(img, albedo=0.18,
                                                integration_mode="full_sphere"):
    """
    Naive camera-style metering via pixel averaging.

    No sphere rendering, no cosine weighting, no solid-angle correction —
    just average the RGB values of pixels in the chosen region, like a
    camera's built-in meter would see it.

    WB:       neutralise the average so R=G=B
    Exposure: scale so the average luminance = albedo (18% grey)

    integration_mode controls which pixels to include:
      full_sphere : all pixels
      upper_dome  : upper hemisphere only (sky)
      sun_facing  : hemisphere facing the brightest pixel
    """
    h, w = img.shape[:2]
    dirs, _ = latlong_dirs(h, w)

    # Build region mask
    if integration_mode == "upper_dome":
        region = dirs[..., 1] > 0.0
    elif integration_mode == "sun_facing":
        sun_dir = _find_peak_direction(img)
        cos_sun = (dirs * sun_dir[None, None, :]).sum(axis=-1)
        region = cos_sun > 0.0
        log(f"Meter: sun_facing — peak at [{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}]")
    else:
        region = np.ones((h, w), dtype=bool)

    pixels = img[region]
    if pixels.shape[0] < 100:
        warn("Pixel-average meter: too few pixels in region.")
        return (np.ones(3, dtype=np.float32), 1.0,
                {"mode": "pixel_average", "applied": False, "reason": "too few pixels"})

    mean_rgb = pixels.mean(axis=0).astype(np.float32)
    L = float(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2])

    if L < 1e-8:
        warn("Pixel-average meter: mean luminance near zero.")
        return (np.ones(3, dtype=np.float32), 1.0,
                {"mode": "pixel_average", "applied": False, "reason": "zero luminance"})

    # WB scale: neutralise to R=G=B
    wb_scale_raw = L / np.clip(mean_rgb, 1e-8, None)
    wb_scale_luma = float(0.2126 * wb_scale_raw[0] + 0.7152 * wb_scale_raw[1]
                          + 0.0722 * wb_scale_raw[2])
    wb_scale = (wb_scale_raw / max(wb_scale_luma, 1e-8)).astype(np.float32)

    # Exposure scale: bring average luminance to albedo
    exposure_scale = float(albedo / L)

    log(f"Pixel-average meter ({integration_mode}):")
    log(f"  mean RGB       : R={mean_rgb[0]:.5f}  G={mean_rgb[1]:.5f}  B={mean_rgb[2]:.5f}")
    log(f"  mean luminance : {L:.5f}")
    log(f"  WB scale       : R={wb_scale[0]:.4f}  G={wb_scale[1]:.4f}  B={wb_scale[2]:.4f}")
    log(f"  exposure scale : {exposure_scale:.5f}  (target={albedo})")
    log(f"  pixels used    : {pixels.shape[0]}")

    info = {
        "mode":              "pixel_average",
        "integration_mode":  integration_mode,
        "applied":           True,
        "mean_rgb":          mean_rgb.tolist(),
        "mean_luminance":    L,
        "wb_scale":          wb_scale.tolist(),
        "exposure_scale":    exposure_scale,
        "pixel_count":       int(pixels.shape[0]),
    }
    return wb_scale, float(exposure_scale), info


def estimate_wb_from_sphere_render(img, albedo=0.18, sphere_res=48,
                                   integration_mode="full_sphere"):
    """
    White balance estimation via rendered Lambertian grey sphere mean.

    The idea: render a grey ball lit by the HDR.  Its mean RGB encodes the
    integrated colour of all light reaching a diffuse surface — the same
    quantity that drives colour casts in your CG renders.  If the mean is
    not neutral (R=G=B) there is a colour cast in the lighting that needs
    correction.

    This is better than dome pixel averaging because:
      - It accounts for cosine-weighting (NdotL) — light from directly above
        contributes more than from low angles, which is physically correct
      - It accounts for solid-angle weighting automatically via the hemisphere
        integration — no ERP distortion issues
      - It integrates over the full sphere so ground-bounce colour is included
      - The sun disc's extreme luminance is naturally handled — it contributes
        proportionally to its solid angle × NdotL, not disproportionately

    The assumption: the mean of an 18% grey diffuse sphere should be achromatic.
    This holds when the scene has a mix of light directions. It will overcorrect
    strongly coloured intentional lighting (stage lights, neon signs etc).

    integration_mode: "full_sphere" | "upper_dome" | "sun_facing"

    Returns scale, info  (same interface as estimate_wb_from_dome)
    """
    # Render at low res — this is a metering operation, not a quality render
    env = _mask_env_by_integration_mode(img, integration_mode)
    env = _env_for_sphere(env, max_w=256)
    sphere, mask = render_gray_ball_vectorized(
        env, albedo=albedo, res=sphere_res, sphere_env_max_w=256)

    valid_pixels = sphere[mask]
    if valid_pixels.shape[0] < 10:
        warn("Sphere WB: too few valid sphere pixels.")
        return np.ones(3, dtype=np.float32), {"applied": False,
                                               "reason": "too few sphere pixels"}

    mean_rgb = valid_pixels.mean(axis=0)   # (3,) — mean colour of lit sphere

    L = float(0.2126*mean_rgb[0] + 0.7152*mean_rgb[1] + 0.0722*mean_rgb[2])
    if L < 1e-8:
        warn("Sphere WB: sphere mean luminance is near zero.")
        return np.ones(3, dtype=np.float32), {"applied": False,
                                               "reason": "zero sphere luminance"}

    # scale_c = L / mean_rgb_c  makes each channel equal L (neutral grey)
    scale_raw = L / np.clip(mean_rgb, 1e-8, None)

    # Plausibility clamp — same logic as dome WB
    scale_rel    = scale_raw / (scale_raw[1] + 1e-8)
    MAX_CORR     = 3.0
    if np.any(scale_rel > MAX_CORR) or np.any(scale_rel < 1.0 / MAX_CORR):
        warn(f"Sphere WB: large correction detected "
             f"(R={scale_rel[0]:.2f}× G=1.00 B={scale_rel[2]:.2f}× vs green). "
             f"Clamping to ±{MAX_CORR}×. "
             f"This may be intentional scene lighting — verify the result.")
        scale_rel = np.clip(scale_rel, 1.0/MAX_CORR, MAX_CORR)
        scale_raw = scale_rel * scale_raw[1]

    # Luminance-neutral normalisation: divide by luma of the scale vector
    # so applying it preserves scene luminance.
    # WRONG: scale / mean(scale)  — arithmetic mean is not luminance
    # RIGHT: scale / luma(scale)  — luma(scale * pixel) = luma(pixel) after
    scale_luma = float(0.2126*scale_raw[0] + 0.7152*scale_raw[1] + 0.0722*scale_raw[2])
    scale      = scale_raw / max(scale_luma, 1e-8)
    kelvin_est = _scale_to_kelvin_approx(scale)

    log(f"Sphere WB: sphere mean RGB={mean_rgb.tolist()}")
    log(f"  scale: R={scale[0]:.4f} G={scale[1]:.4f} B={scale[2]:.4f}")
    log(f"  ~{kelvin_est:.0f}K equivalent")

    info = {
        "mode":              "sphere_render",
        "integration_mode":  integration_mode,
        "applied":           True,
        "sphere_mean_rgb":   mean_rgb.tolist(),
        "sphere_res":        sphere_res,
        "scale":             scale.tolist(),
        "kelvin_approx":     kelvin_est,
    }
    return scale.astype(np.float32), info


def apply_white_balance(img, kelvin=None, rgb_scale=None, swatch_xy=None,
                        dome_wb_mode=None, sphere_wb=False,
                        integration_mode="full_sphere"):
    """
    Apply white balance to a linear ERP image.

    Priority order (first non-None/False wins):
      1. rgb_scale      — explicit manual multiplier
      2. kelvin         — colour temperature
      3. swatch_xy      — sample a neutral pixel
      4. sphere_wb      — render a grey ball, neutralise its mean RGB (recommended)
      5. dome_wb_mode   — grey-world from dome pixels (fallback heuristic)
      6. nothing        — passthrough

    sphere_wb: bool. Renders a quick low-res Lambertian sphere and uses its
               mean RGB as the WB reference. Better than dome averaging because
               it correctly weights by NdotL and solid angle.
    dome_wb_mode: "full_dome" | "upper_dome" | "hot_exclude" | None
    integration_mode: "full_sphere" | "upper_dome" | "sun_facing" — region used
                      for sphere WB integration.
    """
    out = img.copy()
    applied = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dome_info = {}

    if rgb_scale is not None:
        applied = np.array(rgb_scale, dtype=np.float32)
        log(f"WB RGB scale: {applied.tolist()}")

    elif kelvin is not None:
        applied = kelvin_to_rgb_scale(kelvin)
        log(f"WB Kelvin {kelvin}K → scale {applied.tolist()}")

    elif swatch_xy is not None:
        x, y = swatch_xy
        h, w = img.shape[:2]
        patch = img[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
        mean_rgb = np.mean(patch.reshape(-1, 3), axis=0)
        applied = 1.0 / np.clip(mean_rgb, 1e-6, None)
        applied /= np.mean(applied)
        log(f"WB swatch ({x},{y}) sample={mean_rgb.tolist()} → scale {applied.tolist()}")

    elif sphere_wb:
        applied, dome_info = estimate_wb_from_sphere_render(
            img, integration_mode=integration_mode)
        if not dome_info.get("applied", False):
            log("Sphere WB estimation failed — no WB applied.")
            applied = np.ones(3, dtype=np.float32)

    elif dome_wb_mode is not None:
        applied, dome_info = estimate_wb_from_dome(img, mode=dome_wb_mode)
        if not dome_info.get("applied", False):
            log("Dome WB estimation failed — no WB applied.")
            applied = np.ones(3, dtype=np.float32)

    else:
        log("No WB applied")

    out *= applied[None, None, :]
    return out.astype(np.float32), applied, dome_info


# ══════════════════════════════════════════════════════════════════════════════
# LatLong geometry
# ══════════════════════════════════════════════════════════════════════════════

def latlong_dirs(h, w):
    """
    Returns:
      dirs   (H, W, 3)  — unit direction vectors, y-up convention
      dOmega (H, W)     — solid angle per pixel [sr]
    """
    ys = (np.arange(h) + 0.5) / h          # v ∈ (0,1)
    xs = (np.arange(w) + 0.5) / w          # u ∈ (0,1)
    theta = ys[:, None] * np.pi            # polar  [0, π]
    phi   = (xs[None, :] * 2.0 - 1.0) * np.pi  # azimuth [-π, π]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    x = sin_t * cos_p
    y = np.broadcast_to(cos_t, (h, w)).copy()
    z = sin_t * sin_p
    dirs   = np.stack([x, y, z], axis=-1).astype(np.float32)
    dOmega = (sin_t * (np.pi / h) * (2.0 * np.pi / w)).astype(np.float32)
    dOmega = np.broadcast_to(dOmega, (h, w)).copy()
    return dirs, dOmega

def direction_to_uv(d):
    d = np.asarray(d, dtype=np.float64)
    d = d / max(1e-8, np.linalg.norm(d))
    x, y, z = d
    theta = np.arccos(np.clip(y, -1.0, 1.0))
    phi   = np.arctan2(z, x)   # azimuth: consistent with latlong_dirs (x=sin_t*cos_p, z=sin_t*sin_p)
    u = phi / (2.0 * np.pi) + 0.5
    v = theta / np.pi
    return float(u), float(v), float(theta), float(phi)


# ══════════════════════════════════════════════════════════════════════════════
# Metering
# ══════════════════════════════════════════════════════════════════════════════

def robust_stat(values, stat="median"):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return float(np.mean(values) if stat == "mean" else np.median(values))

def meter_image(env_linear, mode="bottom_dome", stat="median",
                swatch_xy=None, swatch_size=5):
    h, w = env_linear.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    lum = luminance(env_linear)
    out = {"mode": mode, "stat": stat}

    if mode == "whole_scene":
        vals = lum[np.isfinite(lum)]
        out["meter_value"] = robust_stat(vals, stat)
        out["pixel_count"] = int(vals.size)

    elif mode == "bottom_dome":
        mask = dirs[..., 1] < 0.0
        vals = lum[mask]
        out["meter_value"] = robust_stat(vals, stat)
        out["pixel_count"] = int(mask.sum())
        out["weighted_mean"] = float(
            np.sum(lum * mask * dOmega) / (np.sum(mask * dOmega) + 1e-8))

    elif mode == "upper_hemi_irradiance":
        cos_theta = np.clip(dirs[..., 1], 0.0, None)
        E_upper   = float(np.sum(lum * cos_theta * dOmega))
        out["meter_value"] = E_upper
        out["pixel_count"] = int((cos_theta > 0).sum())
        out["E_upper"]     = E_upper
        log(f"Upper-hemi irradiance E = {E_upper:.6f}")

    elif mode == "swatch":
        if swatch_xy is None:
            raise ValueError("--metering-mode swatch requires --swatch x,y")
        x, y = int(np.clip(swatch_xy[0], 0, w-1)), int(np.clip(swatch_xy[1], 0, h-1))
        r = max(1, swatch_size // 2)
        patch = env_linear[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
        patch_lum = luminance(patch).reshape(-1)
        out["meter_value"] = robust_stat(patch_lum, stat)
        out["swatch_xy"] = [x, y]
        out["swatch_size"] = swatch_size
        out["pixel_count"] = int(patch_lum.size)
        out["swatch_mean_rgb"] = np.mean(patch.reshape(-1,3), axis=0).tolist()
    else:
        raise ValueError(f"Unknown metering mode: {mode}")

    return out

def solve_exposure_scale(meter_info, target=0.18):
    current = max(1e-6, meter_info["meter_value"])
    scale = target / current
    log(f"Meter mode={meter_info['mode']} val={current:.6f} → exposure_scale={scale:.6f}")
    return float(scale)


# ══════════════════════════════════════════════════════════════════════════════
# Hot lobe / sun mask
# ══════════════════════════════════════════════════════════════════════════════

def smoothstep01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def extract_hot_lobe_key(env_linear, threshold=0.1, upper_only=True, blur_px=0):
    """
    Builds a soft mask around the hottest region of the environment.

    threshold ∈ (0,1]: controls how wide the lobe is.
      low  = (1 - threshold) * hottest
      high = hottest
    Example: threshold=0.1, hottest=80000 → low=72000, high=80000.
    The smoothstep ramp spans [low, high].

    NOTE: We do NOT compute the centroid from raw luminance.  The sun disc at
    80 000 nit occupies ~4 px; sky at 2 000 nit occupies thousands of px.
    A raw-lum centroid would still point roughly right, but the solid-angle
    weighting by dOmega fully corrects for ERP distortion at the poles.
    The final centre direction uses weighted-sum of direction vectors over the
    mask, which is correct regardless of dynamic range.
    """
    threshold = float(np.clip(threshold, 1e-6, 1.0))
    h, w = env_linear.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    lum = luminance(env_linear)

    valid = np.ones((h, w), dtype=bool)
    if upper_only:
        valid &= dirs[..., 1] > 0.0

    valid_lum = lum[valid]
    hottest   = float(valid_lum.max()) if valid_lum.size else float(lum.max())
    low  = (1.0 - threshold) * hottest
    high = hottest

    if high <= low + 1e-8:
        mask = np.zeros_like(lum, dtype=np.float32)
        py, px = np.unravel_index(np.argmax(lum), lum.shape)
        mask[py, px] = 1.0
    else:
        mask = smoothstep01((lum - low) / (high - low)).astype(np.float32)

    if upper_only:
        mask *= valid.astype(np.float32)

    if blur_px > 0:
        k = int(blur_px) | 1   # ensure odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # ── Solid-angle-weighted direction centroid ───────────────────────────
    # Weight by mask × dOmega.  This is correct for ERP: pixels near the poles
    # have smaller dOmega and are naturally down-weighted.  We do NOT weight by
    # luminance here because we already used luminance to build the mask shape;
    # weighting twice would pull the centroid toward the highest pixel rather
    # than the geometric centre of the lobe.
    w_sum = float(np.sum(mask * dOmega)) + 1e-8
    center = np.sum(dirs * (mask * dOmega)[..., None], axis=(0, 1)) / w_sum
    norm = np.linalg.norm(center)
    if norm > 1e-8:
        center /= norm
    else:
        center = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    u, v, th, ph = direction_to_uv(center)
    log(f"Hot lobe: threshold={threshold:.4f} lum=[{low:.3g},{high:.3g}]")
    log(f"  centroid UV=({u:.4f},{v:.4f}) θ={math.degrees(th):.1f}° φ={math.degrees(ph):.1f}°")
    log(f"  soft pixels: {int(np.sum(mask > 0.01))}  strong pixels: {int(np.sum(mask > 0.5))}")

    return {
        "mask":               mask.astype(np.float32),
        "center_dir":         center.astype(np.float32),
        "center_uv":          (u, v),
        "theta_deg":          math.degrees(th),
        "phi_deg":            math.degrees(ph),
        "hottest":            hottest,
        "low":                float(low),
        "high":               float(high),
        "mask_pixel_count_soft":   int(np.sum(mask > 0.01)),
        "mask_pixel_count_strong": int(np.sum(mask > 0.5)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HDRI centering
# ══════════════════════════════════════════════════════════════════════════════

def center_hdri_on_sun(env_linear, hot, center_elevation=False):
    """
    Shift the HDRI azimuthally so the dominant light source sits at the centre
    column (φ = 0, u = 0.5).

    Optionally also shift vertically to place it at the equator row (v = 0.5),
    which is useful for turntable-style setups but disabled by default because
    it changes the sun elevation and breaks the physical ground orientation.

    The shift is a pure pixel roll — no resampling artefacts because ERP rows
    are independent.  Vertical shift uses bilinear resampling (cv2.remap) to
    preserve quality.

    Returns:
      shifted_env  (H, W, 3) float32
      shift_info   dict with pixel offsets and angular deltas
    """
    h, w = env_linear.shape[:2]
    center_dir = hot["center_dir"]

    # Current azimuth of the sun in the HDRI (φ, mapped to pixel column)
    _, _, theta_sun, phi_sun = direction_to_uv(center_dir)
    # phi_sun ∈ [-π, π].  Column index at which the sun currently sits:
    col_sun = (phi_sun / (2.0 * np.pi) + 0.5) * w   # fractional
    # We want it at col_center = w/2 (u=0.5, φ=0)
    col_center = w / 2.0
    # Pixel shift needed (positive = shift right, i.e. roll the image left)
    col_shift = int(round(col_center - col_sun)) % w

    shifted = np.roll(env_linear, col_shift, axis=1)
    log(f"HDRI centre: azimuth shift {col_shift:+d} px  "
        f"(φ {math.degrees(phi_sun):.1f}° → 0°)")

    row_shift = 0
    if center_elevation:
        row_sun = (theta_sun / np.pi) * h
        row_center = h / 2.0
        row_shift = int(round(row_center - row_sun))
        if row_shift != 0:
            # Vertical shift via remap (not np.roll — would wrap pole to pole)
            map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            map_y = np.clip(
                np.tile(np.arange(h, dtype=np.float32)[:,None], (1, w)) - row_shift,
                0, h - 1
            )
            shifted = cv2.remap(shifted, map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            log(f"HDRI centre: elevation shift {row_shift:+d} px  "
                f"(θ {math.degrees(theta_sun):.1f}° → 90°)")

    shift_info = {
        "col_shift_px":      col_shift,
        "row_shift_px":      row_shift,
        "original_phi_deg":  math.degrees(phi_sun),
        "original_theta_deg":math.degrees(theta_sun),
        "center_elevation":  center_elevation,
    }
    return shifted.astype(np.float32), shift_info


# ══════════════════════════════════════════════════════════════════════════════
# Lambertian sphere render — vectorised
# ══════════════════════════════════════════════════════════════════════════════

def sphere_normals(res=160):
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    x = (xx + 0.5) / res * 2.0 - 1.0
    y = 1.0 - (yy + 0.5) / res * 2.0
    r2 = x * x + y * y
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(np.clip(1.0 - r2[mask], 0.0, 1.0))
    n = np.stack([x, y, z], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    return n.astype(np.float32), mask

def _find_peak_direction(env_linear):
    """Find the direction of the brightest pixel in the environment map."""
    h, w = env_linear.shape[:2]
    lum = luminance(env_linear)
    idx = np.argmax(lum)
    row, col = divmod(idx, w)
    theta = (row + 0.5) / h * np.pi
    phi = ((col + 0.5) / w * 2.0 - 1.0) * np.pi
    d = np.array([np.sin(theta) * np.cos(phi),
                  np.cos(theta),
                  np.sin(theta) * np.sin(phi)], dtype=np.float32)
    return d / (np.linalg.norm(d) + 1e-8)


def _mask_env_by_integration_mode(env_linear, mode="full_sphere"):
    """
    Mask environment map pixels based on integration mode.

    Returns a copy of env_linear with masked pixels zeroed out.
      full_sphere : no masking — all pixels contribute
      upper_dome  : zero out below-horizon pixels (y < 0)
      sun_facing  : find peak luminance direction, zero out the opposite hemisphere
    """
    if mode == "full_sphere":
        return env_linear
    h, w = env_linear.shape[:2]
    dirs, _ = latlong_dirs(h, w)
    out = env_linear.copy()
    if mode == "upper_dome":
        mask = dirs[..., 1] < 0.0  # y < 0 = below horizon
        out[mask] = 0.0
        log(f"Integration mode: upper_dome — zeroed {int(mask.sum())} below-horizon pixels")
    elif mode == "sun_facing":
        sun_dir = _find_peak_direction(env_linear)
        cos_sun = (dirs * sun_dir[None, None, :]).sum(axis=-1)
        mask = cos_sun < 0.0  # opposite hemisphere from sun
        out[mask] = 0.0
        log(f"Integration mode: sun_facing — sun at "
            f"[{sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f}], "
            f"zeroed {int(mask.sum())} pixels in opposite hemisphere")
    return out


def _env_for_sphere(env_linear, max_w=512):
    """
    Downsample env map for sphere rendering if needed.

    Sphere integration is a low-frequency operation — the irradiance
    at any normal direction is the weighted integral over the whole
    hemisphere. High-frequency detail (sharp sun disc edges, fine clouds)
    averages out. 512×256 is sufficient and keeps RAM bounded.

    At 4096×2048 full res: chunk matmul is (512, 8M) = 16 GB — OOM.
    At 512×256            : chunk matmul is (512, 131K) = 256 MB — fine.
    """
    h, w = env_linear.shape[:2]
    if w <= max_w:
        return env_linear
    th = max(8, h * max_w // w)
    log(f"Sphere render: downsampling env {w}×{h} → {max_w}×{th} "
        f"(integration is low-frequency, full-res not needed)")
    return cv2.resize(env_linear, (max_w, th),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def render_gray_ball_vectorized(env_linear, albedo=0.18, res=96, chunk=512,
                                 sphere_env_max_w=512):
    """
    Fully vectorised cosine-weighted Lambertian integration.
    Chunked over sphere pixels to stay within RAM.

    sphere_env_max_w: cap the env map width used for integration.
    Sphere irradiance is low-frequency — 512 px wide is indistinguishable
    from 4096 px wide for this purpose, and avoids OOM on large inputs.
    """
    env = _env_for_sphere(env_linear, max_w=sphere_env_max_w)
    h, w = env.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    normals, mask = sphere_normals(res)

    env_dirs_f  = dirs.reshape(-1, 3).astype(np.float32)
    env_rgb_f   = env.reshape(-1, 3).astype(np.float32)
    env_omega_f = dOmega.reshape(-1).astype(np.float32)
    norms_f     = normals.reshape(-1, 3).astype(np.float32)

    n_sph  = norms_f.shape[0]
    n_env  = env_dirs_f.shape[0]
    out_f  = np.zeros((n_sph, 3), dtype=np.float32)

    # Auto-tune chunk to stay under ~512 MB peak allocation
    # peak = chunk × n_env × 4 bytes
    max_bytes   = 512 * 1024 * 1024
    safe_chunk  = max(1, int(max_bytes / (n_env * 4)))
    chunk       = min(chunk, safe_chunk)

    log(f"Rendering sphere {res}×{res} (env {w}×{h}, chunk={chunk}) ...")
    for i in range(0, n_sph, chunk):
        cn  = norms_f[i:i+chunk]
        ndl = np.clip(cn @ env_dirs_f.T, 0.0, None)
        wts = ndl * env_omega_f[None, :]
        out_f[i:i+chunk] = wts @ env_rgb_f

    out = out_f.reshape(res, res, 3) * (albedo / np.pi)
    out[~mask] = 0.0
    return out, mask


def apply_gain_ceiling(raw_gain, ceiling, rolloff_start):
    """
    Apply a soft ceiling to sun gain.

    Below rolloff_start  : gain passes through unchanged (linear region).
    Above rolloff_start  : gain is compressed smoothly toward ceiling using
                           a sqrt-based curve that preserves directionality
                           while preventing hard clip artifacts.

    rolloff_start = ceiling → hard ceiling (no rolloff).

    Formula above rolloff_start:
        t = (raw - rolloff_start) / (ceiling - rolloff_start + 1e-8)
        t_clamped = min(t, 1.0)
        compressed = rolloff_start + (ceiling - rolloff_start) * sqrt(t_clamped)

    sqrt gives a gentler curve than linear — fast rise near rolloff_start
    tapering off toward ceiling, which matches how real HDR reconstruction
    benefits diminish as you push further beyond the clip point.
    """
    if raw_gain <= rolloff_start:
        return float(min(raw_gain, ceiling))
    if rolloff_start >= ceiling:
        return float(min(raw_gain, ceiling))   # hard ceiling, no rolloff
    t = (raw_gain - rolloff_start) / (ceiling - rolloff_start + 1e-8)
    t = min(t, 1.0)
    return float(rolloff_start + (ceiling - rolloff_start) * math.sqrt(t))

def neutralise_lobe(env, lobe_mask, strength=1.0):
    """
    Desaturate the hot lobe pixels toward achromatic before gain boosting.

    Physical reasoning: the sun disc is a ~5800K blackbody. After white
    balance the sky and fill lights are already neutral. The lobe should
    also be close to neutral — any remaining per-channel imbalance is
    clipping artefact or sensor response error, not real spectral content.

    We preserve the luma of each lobe pixel exactly and lerp its colour
    toward neutral by `strength` (0=no change, 1=fully achromatic).

    strength=1.0  — fully neutral (white sun, safest for IBL)
    strength=0.8  — allows a tiny hint of warmth through
    strength=0.0  — no colour neutralisation (original behaviour)

    This is applied IN-PLACE only on lobe pixels. Non-lobe pixels untouched.
    """
    if strength <= 0.0:
        return env.copy()

    out = env.copy()
    lobe_px = lobe_mask > 0.5   # hard mask for pixel selection

    rgb    = out[lobe_px]                            # (N, 3)
    luma   = (0.2126*rgb[:,0] + 0.7152*rgb[:,1]
              + 0.0722*rgb[:,2])[:, None]            # (N, 1)
    neutral = np.broadcast_to(luma, rgb.shape)       # (N, 3)  all channels = luma
    out[lobe_px] = rgb + strength * (neutral - rgb)  # lerp toward neutral
    return out.astype(np.float32)


def apply_sun_gain_per_channel(env, lobe_mask, gains,
                               lobe_neutralise_strength=1.0):
    """
    Neutralise lobe colour then apply per-channel gains.

    Order matters:
      1. Neutralise lobe pixels (desaturate toward luma, preserving energy)
      2. Apply per-channel gains to the neutralised lobe
      3. Recombine with base

    gains: (3,) float — one multiplier per channel, from solve_sun_gain_per_channel.
    lobe_neutralise_strength: 0–1, how strongly to desaturate the lobe before boost.
    """
    # Step 1: neutralise lobe colour
    env_neutral_lobe = neutralise_lobe(env, lobe_mask,
                                       strength=lobe_neutralise_strength)

    base = env_neutral_lobe * (1.0 - lobe_mask[..., None])
    lobe = env_neutral_lobe *        lobe_mask[..., None]

    # Step 2: apply gains — after neutralisation, gains should be very close
    # to scalar (R≈G≈B). Any residual per-channel difference closes the
    # energy balance from the sphere render.
    lobe_gained = lobe * gains[None, None, :]
    return np.clip(base + lobe_gained, 0.0, None).astype(np.float32)


def validate_hdri_orientation(env_linear):
    dirs, dOmega = latlong_dirs(*env_linear.shape[:2])
    lum = luminance(env_linear)
    upper = float(np.sum(lum * (dirs[..., 1] > 0) * dOmega))
    lower = float(np.sum(lum * (dirs[..., 1] < 0) * dOmega))
    ratio = upper / (lower + 1e-8)
    if ratio < 1.2:
        warn(f"Upper/lower energy ratio = {ratio:.2f} — HDRI may be tilted or "
             f"upside-down. Consider --sun-upper-only=False or pre-rotate the HDRI.")
    else:
        log(f"HDRI orientation OK: upper/lower energy ratio = {ratio:.2f}")
    return ratio


# ══════════════════════════════════════════════════════════════════════════════
# Reference sphere calibration
# ══════════════════════════════════════════════════════════════════════════════

def detect_sphere_auto(plate_bgr, min_r_frac=0.02, max_r_frac=0.25):
    """
    Auto-detect a diffuse or chrome sphere via Hough circles.
    Scores candidates by interior-smoothness (diffuse) vs highlight (chrome).
    Returns (cx, cy, r).
    """
    h, w = plate_bgr.shape[:2]
    gray    = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(w * min_r_frac * 2),
        param1=60, param2=28,
        minRadius=int(w * min_r_frac),
        maxRadius=int(w * max_r_frac),
    )
    if circles is None:
        raise RuntimeError("No sphere detected in reference image.")
    circles = np.round(circles[0]).astype(int)

    # Score: interior smoothness (low gradient std → diffuse grey ball)
    grad = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0)**2 + \
           cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1)**2
    ys, xs = np.mgrid[0:h, 0:w]
    best_score, best = -1e9, circles[0]
    for cx, cy, r in circles:
        inner = (xs-cx)**2 + (ys-cy)**2 <= (r*0.85)**2
        if inner.sum() < 30:
            continue
        boundary = ((xs-cx)**2+(ys-cy)**2 <= (r*1.3)**2) & \
                   ((xs-cx)**2+(ys-cy)**2 > (r*0.9)**2)
        score = (grad[boundary].mean() / (grad[inner].mean() + 1e-4)) * r
        if score > best_score:
            best_score, best = score, (cx, cy, r)
    return int(best[0]), int(best[1]), int(best[2])

def estimate_light_dir_from_shading(plate_linear, cx, cy, r):
    """
    Photometric stereo solve: fit L·N = I over all ball pixels.
    Returns L in camera space (unit vector).
    """
    h, w = plate_linear.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - cx).astype(np.float32) / r
    dy = (cy - ys).astype(np.float32) / r
    dist2 = dx**2 + dy**2
    valid  = dist2 <= 0.90**2
    dz     = np.sqrt(np.maximum(0.0, 1.0 - dist2))

    lum = luminance(plate_linear)
    I   = lum[valid]
    Nx, Ny, Nz = dx[valid], dy[valid], dz[valid]

    ndotv_w  = Nz**2
    p5, p95  = np.percentile(I, 5), np.percentile(I, 95)
    expo_w   = np.where(I > p95, 0.1, np.where(I < p5, 0.1, 1.0))
    W        = ndotv_w * expo_w

    A = np.stack([Nx, Ny, Nz, np.ones_like(Nx)], axis=1)
    Aw, Iw = A * W[:,None], I * W
    result, _, _, _ = np.linalg.lstsq(Aw, Iw, rcond=None)
    aL = result[:3]
    L  = aL / (np.linalg.norm(aL) + 1e-8)
    intensity = float(np.linalg.norm(aL))
    log(f"Light dir (cam): {L}  intensity={intensity:.4f}")
    return L.astype(np.float32), intensity

def calibrate_exposure_from_sphere(env_linear, plate_linear,
                                   cx, cy, r, albedo=0.18,
                                   sphere_res=64, n_normals=48):
    """
    Solve global exposure multiplier k by matching rendered irradiance
    to measured irradiance across all ball pixels (WLS).

    E_pred(N_i) × k  =  E_meas(N_i)  for all ball pixels i
    Closed-form WLS:  k = Σ(w·E_meas·E_pred) / Σ(w·E_pred²)
    """
    h_env, w_env = env_linear.shape[:2]
    dirs_env, dOmega_env = latlong_dirs(h_env, w_env)
    env_dirs_f  = dirs_env.reshape(-1,3).astype(np.float32)
    env_rgb_f   = env_linear.reshape(-1,3).astype(np.float32)
    env_omega_f = dOmega_env.reshape(-1).astype(np.float32)

    def irradiance_at(normal):
        ndl = np.clip(env_dirs_f @ normal, 0.0, None)
        wts = ndl * env_omega_f
        return (wts @ env_rgb_f)  # (3,)

    # Build per-pixel normals on the sphere
    h_p, w_p = plate_linear.shape[:2]
    ys, xs   = np.mgrid[0:h_p, 0:w_p]
    dx = (xs - cx).astype(np.float32) / r
    dy = (cy - ys).astype(np.float32) / r
    dist2  = dx**2 + dy**2
    valid  = dist2 <= 0.88**2
    dz_arr = np.sqrt(np.maximum(0.0, 1.0 - dist2))

    vy, vx = np.where(valid)
    # Stratified subsample
    rng = np.random.default_rng(42)
    lum_ball = luminance(plate_linear)[valid]
    # Prefer terminator: Gaussian weight centered on median luminance
    med_lum  = np.median(lum_ball)
    rng_lum  = np.percentile(lum_ball, 90) - np.percentile(lum_ball, 10)
    term_w   = np.exp(-0.5 * ((lum_ball - med_lum) / (rng_lum * 0.4 + 1e-6))**2)
    ndotv_w  = dz_arr[valid]**2
    combined = term_w * ndotv_w
    combined /= combined.sum() + 1e-8
    n_use    = min(n_normals, len(vy))
    idx      = rng.choice(len(vy), size=n_use, replace=False, p=combined)

    E_meas_list, E_pred_list, W_list = [], [], []
    for i in idx:
        py, px = int(vy[i]), int(vx[i])
        N = np.array([dx[py,px], dy[py,px], dz_arr[py,px]], dtype=np.float32)
        N /= max(np.linalg.norm(N), 1e-8)
        # Measured irradiance from plate pixel
        pix    = plate_linear[py, px]
        E_meas = pix * (np.pi / albedo)
        # Predicted irradiance from HDR (k=1)
        E_pred = irradiance_at(N)
        w_i    = float(ndotv_w[i] * term_w[i])
        E_meas_list.append(E_meas)
        E_pred_list.append(E_pred)
        W_list.append(w_i)

    E_meas_arr = np.array(E_meas_list)   # (N,3)
    E_pred_arr = np.array(E_pred_list)   # (N,3)
    W_arr      = np.array(W_list)

    # Luminance-weighted scalar k
    lum_meas = 0.2126*E_meas_arr[:,0] + 0.7152*E_meas_arr[:,1] + 0.0722*E_meas_arr[:,2]
    lum_pred = 0.2126*E_pred_arr[:,0] + 0.7152*E_pred_arr[:,1] + 0.0722*E_pred_arr[:,2]
    k = float(np.sum(W_arr * lum_meas * lum_pred) / (np.sum(W_arr * lum_pred**2) + 1e-8))

    # Per-channel for colour temperature check
    k_ch = np.array([
        float(np.sum(W_arr*E_meas_arr[:,c]*E_pred_arr[:,c]) /
              (np.sum(W_arr*E_pred_arr[:,c]**2)+1e-8))
        for c in range(3)
    ])

    residuals  = k * lum_pred - lum_meas
    rmse       = float(np.sqrt(np.mean(W_arr * residuals**2) / (W_arr.mean()+1e-8)))
    rel_err    = rmse / (lum_meas.mean() + 1e-8)

    log(f"Sphere calibration: k={k:.4f} ({np.log2(k+1e-8):+.2f} stops)")
    log(f"  per-channel k: R={k_ch[0]:.3f} G={k_ch[1]:.3f} B={k_ch[2]:.3f}")
    log(f"  fit RMSE={rmse:.4f} ({rel_err*100:.1f}%)")

    return float(k), k_ch, {"rmse": rmse, "rel_err_pct": rel_err*100,
                             "n_samples": n_use}


# ══════════════════════════════════════════════════════════════════════════════
# ColorChecker detection & colour correction
# ══════════════════════════════════════════════════════════════════════════════

# ── ColorChecker detection delegated to colorchecker_erp.py ──────────────────
# The old approach (feed ERP directly to colour-checker-detection) was wrong:
# the library uses contour/rectangle detection which requires straight edges.
# In ERP, straight lines become sinusoidal curves → detection fails silently.
#
# The correct approach (in colorchecker_erp.py):
#   1. Gnomonic (rectilinear) projection sweep — tiles the full panorama
#   2. Detect on each tile (straight lines stay straight in rectilinear)
#   3. Back-project swatch pixel centres through gnomonic inverse → ERP (u,v)
#   4. Sample full-res linear HDR at those ERP coords for accurate colours
#   5. solvePnP pose estimate → checker face normal in world space

try:
    from colorchecker_erp import (
        find_colorchecker_in_erp,
        solve_color_matrix_from_swatches,
        apply_color_matrix,
        CC24_LINEAR_SRGB,
    )
    HAVE_CC_ERP = True
    # Newer symbols — present only in updated colorchecker_erp.py
    try:
        from colorchecker_erp import get_cc24_reference
    except ImportError:
        def get_cc24_reference(colorspace="acescg"):
            return CC24_LINEAR_SRGB
    try:
        from colorchecker_erp import srgb_linear_to_acescg, acescg_to_srgb_linear
    except ImportError:
        srgb_linear_to_acescg = None
        acescg_to_srgb_linear = None
except Exception as _cc_import_err:
    HAVE_CC_ERP = False
    CC24_LINEAR_SRGB = None
    srgb_linear_to_acescg = None
    acescg_to_srgb_linear = None
    import traceback as _tb
    warn(f"colorchecker_erp import failed: {_cc_import_err}")
    warn("Full traceback:")
    _tb.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════════════════

def apply_res_scale(img, res="full"):
    """
    Optionally downsample the working image.

    res:
      "full"    — no change, process at native resolution (default)
      "half"    — downsample to 1/2 in each dimension (4× fewer pixels)
      "quarter" — downsample to 1/4 in each dimension (16× fewer pixels)

    Uses INTER_AREA which is the correct filter for downsampling — it
    averages over the source pixels rather than sampling, so hot pixels
    (extreme sun values) don't alias into black gaps.
    """
    h, w = img.shape[:2]
    scale = {"full": 1, "half": 2, "quarter": 4}.get(res, 1)
    if scale == 1:
        log(f"Processing at full resolution: {w}×{h}")
        return img
    tw = max(8, w // scale)
    th = max(8, h // scale)
    log(f"Downsampling {w}×{h} → {tw}×{th} ({res})")
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="HDRI calibration pipeline for VFX/IBL compositing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────
    ap.add_argument("input",
                    help="Input latlong (.jpg/.jpeg/.png/.webp/.hdr/.exr)")
    ap.add_argument("--out", default="corrected.exr",
                    help="Output calibrated EXR")
    ap.add_argument("--debug-dir", default="debug_hdri",
                    help="Directory for debug images and report")
    ap.add_argument("--res", default="full",
                    choices=["full", "half", "quarter"],
                    help="Processing resolution. Default: full (no downsampling). "
                         "half/quarter reduce both dimensions by 2× or 4×. "
                         "Use half or quarter to speed up sphere renders and "
                         "hot-lobe detection on very large inputs (>4K wide). "
                         "The output EXR is always at the chosen resolution — "
                         "save at full then re-run at half for iteration.")
    ap.add_argument("--colorspace", default=None,
                    choices=["acescg", "srgb"],
                    help="Input primaries of the source image. "
                         "Default: auto — 'acescg' for .exr/.hdr, 'srgb' for .jpg/.jpeg/.png/.webp/.tif/.tiff. "
                         "The pipeline processes internally in ACEScg; choose 'srgb' for linear-sRGB EXR/HDR inputs so they are converted on load.")

    # ── White balance — primary source selectors ─────────────────────────
    # Two independent axes: --wb-source and --exposure-source.
    # Each can be set explicitly. 'auto' applies smart fallback logic.
    ap.add_argument("--wb-source", default="auto",
                    choices=["auto", "chart", "sphere", "meter", "none"],
                    help="White balance colour source. "
                         "'auto' (default): chart if found+confident, else meter. "
                         "'chart': use ColorChecker patch 22. "
                         "'sphere': neutralise rendered grey sphere to achromatic. "
                         "'meter': pixel-average neutralisation (like a camera meter). "
                         "'none': no WB correction, passthrough.")
    ap.add_argument("--exposure-source", default="auto",
                    choices=["auto", "chart", "sphere", "meter", "none"],
                    help="Exposure magnitude source. "
                         "'auto' (default): chart if found+confident, else meter. "
                         "'chart': use ColorChecker patch 22 luma as the absolute reference. "
                         "'sphere': render grey sphere and normalise to 18%% grey. "
                         "'meter': pixel-average exposure to 18%% grey (like a camera meter). "
                         "'none': no exposure correction.")
    ap.add_argument("--integration-mode", default="full_sphere",
                    choices=["full_sphere", "upper_dome", "sun_facing"],
                    help="Region of the HDRI used for sphere WB and exposure integration. "
                         "'full_sphere': integrate entire environment (default). "
                         "'upper_dome': sky hemisphere only, excludes ground bounce. "
                         "'sun_facing': hemisphere facing the dominant light direction.")
    # ── Hot lobe ──────────────────────────────────────────────────────────
    ap.add_argument("--sun-threshold", type=float, default=0.1,
                    help="Hot lobe width: 0..1 fraction below peak. "
                         "0.1 → low=0.9×peak, high=peak")
    ap.add_argument("--sun-gain-ceiling", type=float, default=2000.0,
                    help="Maximum per-channel gain applied to the sun lobe during "
                         "irradiance reconstruction. Default 2000×. "
                         "A soft rolloff is applied above 500× so gains taper "
                         "smoothly toward the ceiling rather than hard-clipping. "
                         "Increase for better sun reconstruction on heavily clipped "
                         "inputs (at the cost of more energy). "
                         "Decrease (e.g. --sun-gain-ceiling 200) to limit "
                         "reconstruction aggressiveness on clean or slightly "
                         "clipped inputs. Set to 1.0 to disable gain solve.")
    ap.add_argument("--sun-gain-rolloff", type=float, default=500.0,
                    help="Gain level above which soft rolloff begins. Default 500×. "
                         "Below this, gain is applied linearly. Above this, gains "
                         "are compressed toward --sun-gain-ceiling with a smooth "
                         "curve. Set equal to --sun-gain-ceiling to disable rolloff "
                         "(hard ceiling).")

    # ── HDRI centering ────────────────────────────────────────────────────
    ap.add_argument("--center-hdri", action="store_true", default=True,
                    help="Shift HDRI azimuth so dominant light sits at φ=0 "
                         "(centre column). ON by default.")
    ap.add_argument("--no-center-hdri", dest="center_hdri",
                    action="store_false",
                    help="Disable HDRI azimuth centering")

    # ── Sphere solve ──────────────────────────────────────────────────────
    ap.add_argument("--albedo", type=float, default=0.18,
                    help="Grey ball albedo")
    ap.add_argument("--sphere-solve", default="energy_conservation",
                    choices=["energy_conservation", "sun_facing_card",
                             "sun_facing_vertical", "none"],
                    help="Sun gain solve mode. Default: energy_conservation. "
                         "energy_conservation targets an upward-facing grey card, "
                         "sun_facing_card targets a grey card whose normal points at the sun, "
                         "sun_facing_vertical targets a vertical card rotated toward the sun azimuth. "
                         "none disables the sun solve.")
    ap.add_argument("--final-balance-target", default="none",
                    choices=["none", "auto"],
                    help="Optional final post-balance target. "
                         "'none' keeps the solved output unchanged. "
                         "'auto' measures the imaginary card implied by the selected sun solve target, "
                         "then applies one final RGB trim so it is neutral at albedo.")
    ap.add_argument("--lobe-neutralise", type=float, default=1.0,
                    metavar="0.0-1.0",
                    help="How strongly to desaturate hot lobe pixels before gain "
                         "boosting. 1.0 = fully achromatic (white sun, default). "
                         "0.8 = allow a hint of warmth. 0.0 = no neutralisation. "
                         "Physical basis: after white balance the sun should be "
                         "near-neutral; remaining colour is clipping artefact.")

    # ── ColorChecker ──────────────────────────────────────────────────────
    ap.add_argument("--colorchecker", type=str, default=None,
                    help="Path to a separate flat plate image containing a "
                         "ColorChecker Classic 24. Used for WB and colour "
                         "correction. Requires colour-checker-detection.")
    ap.add_argument("--colorchecker-in-hdri", action="store_true", default=False,
                    help="Search for a ColorChecker inside the HDRI latlong "
                         "itself using an overlapped cubemap sweep plus centred refinement passes. "
                         "More robust than manual --colorchecker for on-set HDRI captures where the chart is on the floor.")
    ap.add_argument("--cc-read-backend", default="colour",
                    choices=["auto", "colour", "contour"],
                    help="Final swatch-read backend after the chart is located and recentered. "
                         "'colour' reruns colour-checker-detection on the rectified checker image and is the default, "
                         "'contour' uses a contour-fit quad, and 'auto' compares both and keeps the better-agreeing read.")
    ap.add_argument("--cc-compare-backends", action="store_true", default=False,
                    help="Save backend comparison overlays for the ColorChecker read stage.")

    # ── Validate-only mode ────────────────────────────────────────────────
    ap.add_argument("--validate-only", action="store_true", default=False,
                    help="Dry-run: load, check orientation, energy, clamping and "
                         "write a JSON report — no EXR output. Fast batch checking.")

    args = ap.parse_args()
    _legacy_defaults = {
        "exposure_scale": None,
        "sphere_target": "irradiance",
        "kelvin": None,
        "rgb_scale": None,
        "wb_swatch": None,
        "dome_wb": None,
        "sphere_wb": False,
        "chart_facing": "auto",
        "metering_mode": "upper_hemi_irradiance",
        "meter_stat": "median",
        "meter_target": 1.0,
        "swatch": None,
        "swatch_size": 5,
        "sun_upper_only": False,
        "sun_blur_px": 0,
        "center_elevation": False,
        "sphere_res": 96,
        "direct_highlight_target": 0.32,
        "target_peak_ratio": 2.5,
        "ref_sphere": None,
        "ref_sphere_cx": None,
        "ref_sphere_cy": None,
        "ref_sphere_r": None,
        "ref_sphere_albedo": 0.18,
    }
    for _k, _v in _legacy_defaults.items():
        if not hasattr(args, _k):
            setattr(args, _k, _v)
    _run_pipeline(args)


def _run_validate_only(args, img, meta):
    """
    Fast validate-only pass.  No WB, no gain solve, no EXR written.
    Produces a JSON report with energy, clamping, and orientation checks.
    """
    h, w = img.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    lum = luminance(img)

    # ── Energy integrals ──────────────────────────────────────────────────
    cos_up   = np.clip(dirs[..., 1],  0.0, None)
    cos_down = np.clip(-dirs[..., 1], 0.0, None)
    E_upper  = float(np.sum(lum * cos_up   * dOmega))
    E_lower  = float(np.sum(lum * cos_down * dOmega))
    E_full   = float(np.sum(lum * dOmega))

    E_upper_rgb = np.array([float(np.sum(img[..., c] * cos_up * dOmega))
                             for c in range(3)])
    E_norm      = E_upper_rgb / (np.mean(E_upper_rgb) + 1e-8)
    chroma_imbal = float(np.max(np.abs(E_norm - 1.0)))

    # ── Clamping check ─────────────────────────────────────────────────────
    lum_max    = float(lum.max())
    lum_p999   = float(np.percentile(lum, 99.9))
    lum_p99    = float(np.percentile(lum, 99.0))
    clip_frac  = float(np.mean(lum >= lum_max * 0.99))

    # Clamping likelihood: compare p999/p99 gradient vs p99/p90 gradient.
    # A genuine sun disc has a steep gradient at the very top.
    # A clipped HDRI has many pixels at exactly the same max value.
    lum_p90    = float(np.percentile(lum, 90.0))
    headroom   = lum_max / (lum_p999 + 1e-6)   # >1 = headroom, 1 = flat top

    # ── Hot lobe detection ────────────────────────────────────────────────
    hot = extract_hot_lobe_key(img, threshold=0.1, upper_only=False)

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings = []
    if meta.get("orientation_energy_ratio", 2.0) < 1.2:
        warnings.append(
            f"Low upper/lower energy ratio {meta['orientation_energy_ratio']:.2f} "
            f"— HDRI may be upside-down or tilted")
    if clip_frac > 0.001:
        warnings.append(
            f"Possible clamping: {clip_frac:.3%} of pixels at/near max luma "
            f"({lum_max:.2f}).  Headroom ratio: {headroom:.2f}× "
            f"({'likely clipped' if headroom < 1.05 else 'some headroom'})")
    if chroma_imbal > 0.12:
        warnings.append(
            f"Upper-hemi chroma imbalance {chroma_imbal:.3f} > 0.12 "
            f"— coloured illuminant (golden hour / overcast coloured sky). "
            f"")
    if chroma_imbal > 0.05:
        warnings.append(
            f"Upper-hemi chroma imbalance {chroma_imbal:.3f} > 0.05 "
            f"— mild colour cast in upper hemisphere")

    for w_ in warnings:
        warn(w_)

    if not warnings:
        log("✓ Validation passed: orientation, energy and clamping all look healthy")

    # ── Report ─────────────────────────────────────────────────────────────
    meta["validate_only"] = True
    meta["energy"] = {
        "E_upper":              E_upper,
        "E_lower":              E_lower,
        "E_full":               E_full,
        "E_upper_rgb":          E_upper_rgb.tolist(),
        "E_upper_chroma_norm":  E_norm.tolist(),
        "chroma_imbalance":     chroma_imbal,
    }
    meta["clamping"] = {
        "lum_max":      lum_max,
        "lum_p999":     lum_p999,
        "lum_p99":      lum_p99,
        "lum_p90":      lum_p90,
        "clip_fraction": clip_frac,
        "headroom_ratio": headroom,
        "likely_clipped": headroom < 1.05 and clip_frac > 0.0005,
    }
    meta["hot_lobe"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in hot.items() if k != "mask"}
    meta["warnings"] = warnings

    report_path = os.path.join(args.debug_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"Validate-only report: {report_path}")


def _run_pipeline(args):
    """
    Run the full calibration pipeline given an argparse Namespace (or any
    SimpleNamespace with the same attributes).  This is the callable API
    used by the GUI worker — main() is just an argparse front-end.
    """
    os.makedirs(args.debug_dir, exist_ok=True)

    rgb_scale       = parse_rgb_scale(args.rgb_scale) if args.rgb_scale else None
    wb_swatch_xy    = parse_xy(args.wb_swatch)        if args.wb_swatch else None
    meter_swatch_xy = parse_xy(args.swatch)           if args.swatch    else None

    # ── Input colorspace / working space ─────────────────────────────────
    # The pipeline now processes internally in ACEScg.
    if args.colorspace:
        input_cs = args.colorspace
    else:
        ext = os.path.splitext(args.input)[1].lower()
        input_cs = "srgb" if ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff") else "acescg"
    working_cs = "acescg"
    log(f"Input colorspace: {input_cs}")
    log(f"Working colorspace: {working_cs}")

    meta = {"input": args.input, "args": vars(args), "input_colorspace": input_cs, "colorspace": working_cs}

    # ── 1. Load ───────────────────────────────────────────────────────────
    img, is_hdr = load_image_any(args.input, target_colorspace=working_cs, input_colorspace=input_cs)
    img = np.clip(img, 0.0, None).astype(np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError("Input must be RGB")
    img = apply_res_scale(img, args.res)
    meta["is_hdr_input"] = bool(is_hdr)
    meta["working_resolution"] = list(img.shape[:2])

    # ── 2. Validate orientation ───────────────────────────────────────────
    orient_ratio = validate_hdri_orientation(img)
    meta["orientation_energy_ratio"] = float(orient_ratio)

    # ── Validate-only early exit ──────────────────────────────────────────
    if getattr(args, "validate_only", False):
        _run_validate_only(args, img, meta)
        return

    # ── 3. ColorChecker detection (runs FIRST if available — drives WB) ───
    # Correct pipeline order when a chart is present:
    #   a) Find the chart in the raw linear HDR
    #   b) Use the neutral patches (19-24) to derive the WB scale directly
    #      — more accurate than sphere render because neutrals are physical
    #        references with known ground-truth RGB values
    #   c) Apply WB, do the rest of the pipeline normally
    #   d) After gain solve, apply the full 3×3 colour matrix as a residual
    #      correction for spectral cross-talk that a 3-multiplier WB can't fix
    #
    # If no chart: fall back to sphere WB / dome WB / kelvin / manual.

    cc_info        = {"applied": False}
    cc_measured    = None   # (24,3) linear RGB from chart, set if detected
    cc_det_info    = {}
    cc_matrix      = None   # 3×3, applied after gain solve
    wb_from_chart  = False

    checker_src = None
    _wb_src_early  = getattr(args, "wb_source",       "auto")
    _exp_src_early = getattr(args, "exposure_source", "auto")
    # Chart search logic:
    #   - Explicit plate file → always use it
    #   - --colorchecker-in-hdri → explicit HDRI search
    #   - --wb-source/--exposure-source=chart → implicit HDRI search
    #   - auto (default) → always search HDRI unless BOTH axes are explicitly
    #     set to non-chart sources. If the user says nothing, we look.
    #     Only skip search if user has explicitly opted out of chart for both.
    _chart_explicitly_unwanted = (
        _wb_src_early  not in ("auto", "chart") and
        _exp_src_early not in ("auto", "chart")
    )
    if args.colorchecker:
        checker_src = args.colorchecker
        log("Chart search: using plate file provided via --colorchecker")
    elif args.colorchecker_in_hdri:
        checker_src = "__hdri__"
        log("Chart search: explicit --colorchecker-in-hdri")
    elif _chart_explicitly_unwanted:
        log(f"Chart search: skipped — both --wb-source={_wb_src_early!r} and "
            f"--exposure-source={_exp_src_early!r} are non-chart sources")
    else:
        # Default: always search. Covers:
        #   - auto auto  (both default)     → search, use if found+confident
        #   - chart auto / auto chart       → search, required for chart axis
        #   - sphere chart / chart sphere   → search, required for chart axis
        checker_src = "__hdri__"
        if _wb_src_early == "auto" and _exp_src_early == "auto":
            log("Chart search: auto-searching HDRI (default behaviour — "
                "use --wb-source sphere --exposure-source sphere to skip)")
        else:
            log(f"Chart search: auto-enabled because wb-source={_wb_src_early!r} "
                f"or exposure-source={_exp_src_early!r} requires chart")

    if checker_src and HAVE_CC_ERP:
        cc_debug = os.path.join(args.debug_dir, "colorchecker")
        os.makedirs(cc_debug, exist_ok=True)

        if checker_src == "__hdri__":
            log("Searching for ColorChecker in HDRI latlong "
                "(gnomonic tile sweep) ...")
            cc_measured, cc_det_info = find_colorchecker_in_erp(
                img,   # raw linear, no WB yet
                colorspace=working_cs,
                debug_dir=cc_debug,
                read_backend=getattr(args, "cc_read_backend", "auto"),
                compare_backends=getattr(args, "cc_compare_backends", False),
            )
        else:
            log(f"Loading ColorChecker plate: {checker_src}")
            cc_img_raw, _ = load_image_any(checker_src, target_colorspace=working_cs)
            from colorchecker_erp import _linear_to_u8_for_detection
            import colour_checker_detection as _ccd_direct
            tile_u8_bgr = cv2.cvtColor(
                _linear_to_u8_for_detection(cc_img_raw), cv2.COLOR_RGB2BGR)
            try:
                results = _ccd_direct.detect_colour_checkers_segmentation(
                    tile_u8_bgr, show=False, additional_data=False)
                if results:
                    sw = results[0]   # (24,3) float BGR
                    if sw is not None and sw.shape == (24, 3):
                        sw_rgb = sw[:, ::-1].astype(np.float32)
                        cc_measured = srgb_to_linear(np.clip(sw_rgb, 0, 1))
                        cc_det_info = {"source": checker_src, "found": True}
            except Exception as e:
                warn(f"Flat plate CC detection failed: {e}")

        if cc_measured is not None:
            # WB from the 18% grey patch — CC24 "Neutral 5" (patch 22, index 21).
            # This is the patch closest to 0.18 linear reflectance, same as a
            # standard grey card. It's the 4th patch counting from white in the
            # neutral ramp: white(19), N8(20), N6.5(21), N5(22), N3.5(23), black(24).
            #
            # CC24 patch 22: Neutral 5, ~18% grey, row-major index 21.
            grey_patch_idx = 21

            # Reference in working colorspace — ACEScg for EXR, sRGB for LDR.
            # Neutral patches (R=G=B) are identical in both colorspaces.
            cc24_ref = get_cc24_reference(working_cs)
            meas = cc_measured[grey_patch_idx]       # (3,) linear in working_cs, from HDR
            ref  = cc24_ref[grey_patch_idx]          # (3,) linear in working_cs, ~[0.18, 0.18, 0.18]

            # ── Patch #22 confidence diagnostics ─────────────────────────
            # Chroma check: after luma normalisation the grey patch should be
            # achromatic (R≈G≈B). R-G and B-G deviation reveals WB error or
            # wrong patch identification.
            meas_luma = float(0.2126*meas[0] + 0.7152*meas[1] + 0.0722*meas[2])
            ref_luma  = float(0.2126*ref[0]  + 0.7152*ref[1]  + 0.0722*ref[2])
            if meas_luma > 1e-6:
                meas_norm = meas / meas_luma   # should be ~[1,1,1] for grey
                chroma_rg = float(meas_norm[0] - meas_norm[1])
                chroma_bg = float(meas_norm[2] - meas_norm[1])
            else:
                chroma_rg = chroma_bg = 0.0
            exposure_ratio    = float(meas_luma / max(ref_luma, 1e-8))
            patch22_chroma_err = float(np.sqrt(chroma_rg**2 + chroma_bg**2))
            patch22_confidence = float(np.clip(
                1.0 - patch22_chroma_err * 3.0
                    - abs(np.log2(max(exposure_ratio, 1e-4))) * 0.1,
                0.0, 1.0))

            log(f"── ColorChecker patch #22 (Neutral 5, ~18% grey) diagnostics ──")
            log(f"  measured  linear RGB : R={meas[0]:.5f}  G={meas[1]:.5f}  B={meas[2]:.5f}")
            log(f"  reference linear RGB : R={ref[0]:.5f}  G={ref[1]:.5f}  B={ref[2]:.5f}  (~0.18)")
            log(f"  measured luma        : {meas_luma:.5f}  (ref={ref_luma:.5f}, "
                f"ratio={exposure_ratio:.3f}×)")
            log(f"  chroma deviation     : R-G={chroma_rg:+.4f}  B-G={chroma_bg:+.4f}  "
                f"(|err|={patch22_chroma_err:.4f}, should be <0.02 after correct WB)")
            log(f"  patch confidence     : {patch22_confidence:.3f}"
                + (" ⚠ LOW — patch may be misidentified or chart is clipped/in shadow"
                   if patch22_confidence < 0.7 else " ✓"))

            # ── Pose logging ──────────────────────────────────────────────
            theta = cc_det_info.get("checker_normal_theta_deg")
            phi   = cc_det_info.get("checker_normal_phi_deg")
            n     = cc_det_info.get("checker_normal_world")
            if theta is not None:
                if theta < 70:
                    pose_desc = "facing upward — chart flat on the floor"
                elif theta < 110:
                    pose_desc = "roughly vertical — chart on tripod/stand (expected)"
                else:
                    pose_desc = "facing downward — chart tilted away"
                log(f"── ColorChecker pose ──")
                log(f"  normal : θ={theta:.1f}°  φ={phi:.1f}°  → {pose_desc}")
                if n:
                    log(f"  vector : [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")

            # ── Derive WB scale ───────────────────────────────────────────
            # scale_c = ref_c / meas_c  →  applied to image makes grey patch neutral.
            # Normalise to G channel (standard camera WB convention: G gain = 1.0).
            # This preserves overall scene exposure while correcting colour cast.
            scale_raw = ref / np.clip(meas, 1e-8, None)
            wb_from_chart_scale = (scale_raw / max(float(scale_raw[1]), 1e-8)).astype(np.float32)
            wb_from_chart = True
            log(f"  WB scale (G-normalised) : "
                f"R={wb_from_chart_scale[0]:.4f}  "
                f"G={wb_from_chart_scale[1]:.4f}  "
                f"B={wb_from_chart_scale[2]:.4f}")
            rgb_scale = wb_from_chart_scale

            _cc_patch_diag = {
                "patch_index":          grey_patch_idx,
                "patch_name":           "Neutral 5 (~18% grey)",
                "measured_linear_rgb":  meas.tolist(),
                "reference_linear_rgb": ref.tolist(),
                "measured_luma":        meas_luma,
                "reference_luma":       ref_luma,
                "exposure_ratio":       exposure_ratio,
                "chroma_rg":            chroma_rg,
                "chroma_bg":            chroma_bg,
                "chroma_error":         patch22_chroma_err,
                "confidence":           patch22_confidence,
            }
        else:
            warn("ColorChecker not found — falling back to other WB method.")
            _cc_patch_diag = None
    elif checker_src and not HAVE_CC_ERP:
        warn("colorchecker_erp.py not found — CC detection unavailable.")
        _cc_patch_diag = None
    else:
        _cc_patch_diag = None

    # ── 4. Source resolution ─────────────────────────────────────────────
    # --wb-source and --exposure-source are the primary controls.
    # --sphere-wb is a backward-compat alias for --wb-source sphere.
    #
    # 'auto' fallback chains:
    #   WB:       chart (found+confident) → none
    #   Exposure: chart (found+confident) → none
    #
    # Each axis is fully independent — you can mix and match freely:
    #   --wb-source chart --exposure-source sphere   (chart colour, sphere brightness)
    #   --wb-source sphere --exposure-source chart   (sphere colour, chart brightness)
    #   --wb-source sphere --exposure-source sphere  (full sphere, no chart)

    # Resolve --sphere-wb alias
    _wb_src_raw  = getattr(args, "wb_source",       "auto")
    _exp_src_raw = getattr(args, "exposure_source", "auto")
    if getattr(args, "sphere_wb", False) and _wb_src_raw == "auto":
        _wb_src_raw = "sphere"
        log("WB: --sphere-wb alias → --wb-source sphere")

    # Chart availability + confidence gate
    CHART_CONF_MIN = 0.20
    chart_confident = (wb_from_chart and
                       float(cc_det_info.get("confidence") or 0.0) >= CHART_CONF_MIN)

    # ── Resolve sources with fallback chain ──────────────────────────────
    # Rules:
    #   'auto'          → chart if found+confident, else none
    #   explicit source → use it IF the required data is available, else fallback
    #                     with a loud warning (never silently do nothing)
    #
    # Fallback chain when requested source is unavailable:
    #   chart missing/low-conf → sphere → (metering for exposure)
    #   sphere always available (just renders the env)
    #   kelvin/manual always available if args provided

    FALLBACK_WB  = "meter"     # when chart unavailable: pixel-average WB
    FALLBACK_EXP = "meter"     # when chart unavailable: pixel-average exposure

    def _resolve_wb(src_raw, chart_ok):
        if src_raw == "auto":
            if chart_ok:
                return "chart"
            log(f"WB auto: no chart found — falling back to pixel-average metering")
            return FALLBACK_WB
        if src_raw == "chart" and not chart_ok:
            warn(f"⚠ --wb-source chart requested but chart not found or low confidence "
                 f"(conf={float(cc_det_info.get('confidence') or 0.0):.3f} < {CHART_CONF_MIN}) "
                 f"— add --colorchecker-in-hdri to search for chart, "
                 f"or falling back to '{FALLBACK_WB}'")
            return FALLBACK_WB
        return src_raw

    def _resolve_exp(src_raw, chart_ok):
        if src_raw == "auto":
            if chart_ok:
                return "chart"
            log(f"Exposure auto: no chart found — falling back to pixel-average metering")
            return FALLBACK_EXP
        if src_raw == "chart" and not chart_ok:
            warn(f"⚠ --exposure-source chart requested but chart not found or low confidence "
                 f"(conf={float(cc_det_info.get('confidence') or 0.0):.3f} < {CHART_CONF_MIN}) "
                 f"— add --colorchecker-in-hdri to search for chart, "
                 f"or falling back to '{FALLBACK_EXP}'")
            return FALLBACK_EXP
        return src_raw

    wb_src  = _resolve_wb (_wb_src_raw,  chart_confident)
    exp_src = _resolve_exp(_exp_src_raw, chart_confident)

    chart_use_wb  = (wb_src  == "chart") and wb_from_chart
    chart_use_exp = (exp_src == "chart") and wb_from_chart

    # Log resolution table
    _conf_val = float(cc_det_info.get("confidence") or 0.0)
    log(f"── Source resolution ──")
    log(f"  chart found/confident : {wb_from_chart} / {chart_confident}  "
        f"(conf={_conf_val:.3f}  min={CHART_CONF_MIN})")
    log(f"  --wb-source           : {_wb_src_raw!r}  →  {wb_src!r}"
        + (f"  [FALLBACK from chart]" if _wb_src_raw == "chart" and wb_src != "chart" else "")
        + (f"  [auto resolved]"       if _wb_src_raw == "auto"  else ""))
    log(f"  --exposure-source     : {_exp_src_raw!r}  →  {exp_src!r}"
        + (f"  [FALLBACK from chart]" if _exp_src_raw == "chart" and exp_src != "chart" else "")
        + (f"  [auto resolved]"       if _exp_src_raw == "auto"  else ""))
    if not chart_use_wb and wb_from_chart and _wb_src_raw not in ("chart", "auto"):
        log(f"  note: chart found but wb-source={wb_src!r} explicitly set — chart WB suppressed by user")
    if not chart_use_exp and wb_from_chart and _exp_src_raw not in ("chart", "auto"):
        log(f"  note: chart found but exposure-source={exp_src!r} explicitly set — chart exposure suppressed by user")

    # ── Pixel-average metering (shared by WB and/or exposure) ──────────
    _meter_info_pixel = None
    _meter_wb_scale   = None
    _meter_exp_scale  = None
    _integration_mode = getattr(args, "integration_mode", "full_sphere")
    if wb_src == "meter" or exp_src == "meter":
        _meter_wb_scale, _meter_exp_scale, _meter_info_pixel = \
            estimate_wb_and_exposure_from_pixel_average(
                img, albedo=args.albedo, integration_mode=_integration_mode)

    # ── Resolve WB colour ─────────────────────────────────────────────────
    if chart_use_wb:
        wb_source  = "colorchecker_patch22"
        _wb_rgb    = rgb_scale   # already set from chart detection above
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = None
    elif wb_src == "meter":
        wb_source  = f"pixel_average_{_integration_mode}"
        _wb_rgb    = _meter_wb_scale
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = None
    elif wb_src == "sphere":
        wb_source  = "sphere_render"
        _wb_rgb    = None
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = True
        _wb_swatch = None
    elif wb_src == "kelvin":
        wb_source  = f"kelvin_{args.kelvin:.0f}K" if args.kelvin else "kelvin_5600K"
        _wb_rgb    = None
        _wb_kelvin = args.kelvin or 5600.0
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = None
    elif wb_src == "manual":
        wb_source  = "rgb_scale_manual"
        _wb_rgb    = rgb_scale
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = None
    elif wb_src == "dome":
        wb_source  = args.dome_wb or "upper_dome"
        _wb_rgb    = None
        _wb_kelvin = None
        _wb_dome   = args.dome_wb or "upper_dome"
        _wb_sphere = False
        _wb_swatch = None
    elif wb_src == "none":
        wb_source  = "none"
        _wb_rgb    = None
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = None
    else:
        # fallthrough: swatch or any unlisted
        wb_source  = "wb_swatch_pixel" if wb_swatch_xy else "none"
        _wb_rgb    = None
        _wb_kelvin = None
        _wb_dome   = None
        _wb_sphere = False
        _wb_swatch = wb_swatch_xy

    log(f"WB method: {wb_source}")

    wb_img, wb_scale, wb_dome_info = apply_white_balance(
        img,
        kelvin      = _wb_kelvin,
        rgb_scale   = _wb_rgb,
        swatch_xy   = _wb_swatch,
        dome_wb_mode= _wb_dome,
        sphere_wb   = _wb_sphere,
        integration_mode= _integration_mode)

    save_png_preview(os.path.join(args.debug_dir, "01_wb_preview.png"), wb_img)
    meta["white_balance"] = {
        "source":           wb_source,
        "wb_src_resolved":  wb_src,
        "exp_src_resolved": exp_src,
        "chart_use_wb":     chart_use_wb,
        "chart_use_exp":    chart_use_exp,
        "wb_scale":         wb_scale.tolist(),
        "dome_wb_mode":     _wb_dome,
        "sphere_wb":        _wb_sphere,
        "wb_info":          wb_dome_info,
    }

    # ── WB cross-check: always render sphere after WB and report neutrality ──
    # Run a low-res sphere render on the WB-applied image.
    # This is the single most useful sanity check:
    #   - If WB is correct, sphere mean R≈G≈B (achromatic)
    #   - R-G and B-G deviation tells you the residual colour cast in stops
    # When chart WB was used, also compute sphere WB independently and compare.
    # Large disagreement → chart patch may be misidentified or in shadow.
    log("── WB sanity check (irradiance) ──")
    # Use direct upper-hemisphere irradiance instead of sphere renders — faster, same info.
    _dirs_wb, _dOmega_wb = latlong_dirs(wb_img.shape[0], wb_img.shape[1])
    _cos_up_wb = np.clip(_dirs_wb[..., 1], 0.0, None)
    _E_wb_rgb = np.array([
        float(np.sum(wb_img[..., c] * _cos_up_wb * _dOmega_wb)) for c in range(3)])
    _E_wb_luma = float(0.2126*_E_wb_rgb[0] + 0.7152*_E_wb_rgb[1] + 0.0722*_E_wb_rgb[2])
    # Chroma deviation: how far from neutral is the upper-hemi irradiance?
    _rg_dev = float((_E_wb_rgb[0] - _E_wb_rgb[1]) / (_E_wb_luma + 1e-8))
    _bg_dev = float((_E_wb_rgb[2] - _E_wb_rgb[1]) / (_E_wb_luma + 1e-8))
    log(f"  E_upper RGB : R={_E_wb_rgb[0]:.4f}  G={_E_wb_rgb[1]:.4f}  B={_E_wb_rgb[2]:.4f}")
    log(f"  chroma (R-G)/luma={_rg_dev:+.4f}  (B-G)/luma={_bg_dev:+.4f}  (ideal=0.0)")
    _chroma_total = abs(_rg_dev) + abs(_bg_dev)

    if wb_from_chart:
        # Cross-check: derive WB scale implied by raw upper-hemi irradiance
        _E_raw_rgb = np.array([
            float(np.sum(img[..., c] * _cos_up_wb * _dOmega_wb)) for c in range(3)])
        _E_raw_luma = float(0.2126*_E_raw_rgb[0] + 0.7152*_E_raw_rgb[1] + 0.0722*_E_raw_rgb[2])
        _sphere_implied_scale = _E_raw_luma / np.clip(_E_raw_rgb, 1e-8, None)
        _sphere_implied_luma  = float(0.2126*_sphere_implied_scale[0]
                                      + 0.7152*_sphere_implied_scale[1]
                                      + 0.0722*_sphere_implied_scale[2])
        _sphere_implied_neutral = (_sphere_implied_scale / max(_sphere_implied_luma, 1e-8)
                                   ).astype(np.float32)
        # Compare chart scale vs sphere scale
        _scale_diff = wb_scale - _sphere_implied_neutral
        _diff_mag   = float(np.max(np.abs(_scale_diff)))
        log(f"  chart  WB scale : R={wb_scale[0]:.4f}  G={wb_scale[1]:.4f}  B={wb_scale[2]:.4f}")
        log(f"  sphere WB scale : R={_sphere_implied_neutral[0]:.4f}  "
            f"G={_sphere_implied_neutral[1]:.4f}  B={_sphere_implied_neutral[2]:.4f}")
        log(f"  max channel diff: {_diff_mag:.4f}")
        if _diff_mag > 0.15:
            warn(f"⚠⚠ LARGE disagreement between chart WB and sphere WB "
                 f"(max_diff={_diff_mag:.3f} > 0.15). Chart patch may be "
                 f"misidentified, in shadow, or clipped. "
                 f"Consider --sphere-wb as fallback.")
        elif _diff_mag > 0.08:
            warn(f"⚠ Moderate disagreement between chart WB and sphere WB "
                 f"(max_diff={_diff_mag:.3f} > 0.08). Verify chart patch in debug image.")
        else:
            log(f"  chart/sphere agreement: OK (diff={_diff_mag:.4f} < 0.08) ✓")
        meta["white_balance"]["sphere_cross_check"] = {
            "sphere_implied_scale":   _sphere_implied_neutral.tolist(),
            "chart_scale":            wb_scale.tolist(),
            "max_channel_diff":       _diff_mag,
            "status": "LARGE_DISAGREEMENT" if _diff_mag > 0.15
                      else "MODERATE_DISAGREEMENT" if _diff_mag > 0.08
                      else "OK",
        }

    # ── Adaptive WB blend (chart + sphere) ───────────────────────────────
    # When chart is found, we now blend chart WB scale and sphere WB scale
    # based on their disagreement:
    #
    #   blend_t = smoothstep(LOW, HIGH, disagreement)
    #   LOW  = 0.05 → below this, chart and sphere agree: trust chart fully
    #   HIGH = 0.25 → above this, chart is unreliable: trust sphere fully
    #
    # Physical justification:
    #   chart scale = correct IFF chart is in same light as scene (high sun, no shadow)
    #   sphere scale = correct IFF scene has identifiable neutral illuminant
    #   disagreement = proxy for "chart was in different light than scene"
    #
    # CRITICAL: blend only affects WB colour (hue/chromaticity).
    #           Exposure magnitude stays from patch 22 luma — that's illuminant-independent.
    #
    # The blend is applied as a secondary correction on top of the already-WB'd image:
    #   correction_c = blended_scale_c / chart_scale_c
    #   wb_img_final = wb_img * correction_c

    wb_blend_t    = 0.0
    wb_blend_info = {"applied": False, "blend_t": 0.0, "method": "chart_only"}

    # Blend is only valid when auto-mode is uncertain about the chart.
    # If the user explicitly said --wb-source chart, OR auto resolved to chart
    # at high confidence (>= 0.5), trust the chart unconditionally.
    # Rationale: on a red/coloured sky the sphere WB is contaminated by the
    # very cast the chart is trying to remove — blending toward sphere makes
    # the WB worse, not better. The blend exists only to catch misidentified
    # or shadowed charts, not legitimately coloured scenes.
    _chart_explicit  = (_wb_src_raw == "chart")
    _chart_confident_hi = wb_from_chart and (_conf_val >= 0.5)
    _blend_allowed   = (not _chart_explicit) and (not _chart_confident_hi)

    if not _blend_allowed and wb_from_chart and "_sphere_implied_neutral" in locals():
        _reason = "user set --wb-source chart" if _chart_explicit else f"chart confidence {_conf_val:.3f} >= 0.5"
        log(f"WB blend: skipped — {_reason}. Chart is ground truth.")
        wb_blend_info = {"applied": False, "blend_t": 0.0,
                         "method": "chart_only_explicit", "reason": _reason}

    if _blend_allowed and wb_from_chart and "_sphere_implied_neutral" in locals():
        LOW, HIGH = 0.05, 0.25
        wb_blend_t = float(smoothstep01(((_diff_mag - LOW) / (HIGH - LOW))))

        if wb_blend_t < 0.01:
            log(f"WB blend: chart/sphere agree (diff={_diff_mag:.4f}) — chart only (t={wb_blend_t:.3f})")
            wb_blend_info = {"applied": False, "blend_t": wb_blend_t,
                             "method": "chart_only", "diff": _diff_mag}
        else:
            # G-normalise both scales before blending so they're comparable
            chart_g_norm  = wb_scale / max(float(wb_scale[1]), 1e-8)
            sphere_g_norm = _sphere_implied_neutral / max(float(_sphere_implied_neutral[1]), 1e-8)

            blended_g_norm = (chart_g_norm  * (1.0 - wb_blend_t)
                            + sphere_g_norm *        wb_blend_t).astype(np.float32)

            # Secondary correction: what multiplier turns chart-WB'd image into blended-WB'd
            # wb_img is already chart_scale applied: wb_img = img * chart_scale
            # We want:  img * blended  =  wb_img * (blended / chart_scale)
            correction = blended_g_norm / np.clip(chart_g_norm, 1e-8, None)
            wb_img     = (wb_img * correction[None, None, :]).astype(np.float32)

            # Update wb_scale to reflect what was actually applied
            wb_scale = (chart_g_norm * correction).astype(np.float32)

            method = ("sphere_dominant" if wb_blend_t > 0.75 else
                      "blended_sphere_heavy" if wb_blend_t > 0.5 else
                      "blended_chart_heavy" if wb_blend_t > 0.25 else
                      "blended_slight")

            log(f"WB blend: disagreement={_diff_mag:.4f}  t={wb_blend_t:.3f}  → {method}")
            log(f"  chart  G-norm : R={chart_g_norm[0]:.4f}  G={chart_g_norm[1]:.4f}  B={chart_g_norm[2]:.4f}")
            log(f"  sphere G-norm : R={sphere_g_norm[0]:.4f}  G={sphere_g_norm[1]:.4f}  B={sphere_g_norm[2]:.4f}")
            log(f"  blended       : R={blended_g_norm[0]:.4f}  G={blended_g_norm[1]:.4f}  B={blended_g_norm[2]:.4f}")
            log(f"  correction    : R={correction[0]:.4f}  G={correction[1]:.4f}  B={correction[2]:.4f}")

            wb_blend_info = {
                "applied":          True,
                "blend_t":          wb_blend_t,
                "method":           method,
                "diff":             _diff_mag,
                "chart_g_norm":     chart_g_norm.tolist(),
                "sphere_g_norm":    sphere_g_norm.tolist(),
                "blended_g_norm":   blended_g_norm.tolist(),
                "correction":       correction.tolist(),
            }

            # Re-render sphere check with blended WB to confirm improvement
            _env_blended_check = _env_for_sphere(wb_img, max_w=256)
            _sp_blended, _sp_blended_mask = render_gray_ball_vectorized(
                _env_blended_check, albedo=args.albedo, res=48, chunk=512)
            _sp_bl_rgb  = np.mean(_sp_blended[_sp_blended_mask], axis=0)
            _rg_bl = float(_sp_bl_rgb[0] - _sp_bl_rgb[1])
            _bg_bl = float(_sp_bl_rgb[2] - _sp_bl_rgb[1])
            log(f"  post-blend sphere: R={_sp_bl_rgb[0]:.4f} G={_sp_bl_rgb[1]:.4f} B={_sp_bl_rgb[2]:.4f}  "
                f"R-G={_rg_bl:+.4f} B-G={_bg_bl:+.4f}  "
                f"(was R-G={_rg_dev:+.4f} B-G={_bg_dev:+.4f})")
            wb_blend_info["post_blend_sphere_rgb"]    = _sp_bl_rgb.tolist()
            wb_blend_info["post_blend_chroma_rg"]     = _rg_bl
            wb_blend_info["post_blend_chroma_bg"]     = _bg_bl
            wb_blend_info["pre_blend_chroma_rg"]      = _rg_dev
            wb_blend_info["pre_blend_chroma_bg"]      = _bg_dev

            # Update for downstream chroma checks
            _sp_rgb       = _sp_bl_rgb
            _rg_dev       = _rg_bl
            _bg_dev       = _bg_bl
            _chroma_total = abs(_rg_bl) + abs(_bg_bl)

            save_png_preview(os.path.join(args.debug_dir, "01_wb_blended_preview.png"), wb_img)

    meta["white_balance"]["wb_blend"] = wb_blend_info

    if _chroma_total > 0.10:
        warn(f"⚠⚠ Sphere render shows strong colour cast after WB "
             f"(R-G={_rg_dev:+.4f} B-G={_bg_dev:+.4f}). "
             f"WB may be wrong — check 01_wb_sphere_check.png")
    elif _chroma_total > 0.04:
        warn(f"⚠ Sphere render shows minor colour cast after WB "
             f"(R-G={_rg_dev:+.4f} B-G={_bg_dev:+.4f}).")

    # Save sphere debug image for this WB check
    save_png_preview(os.path.join(args.debug_dir, "01_wb_preview.png"), wb_img)
    meta["white_balance"]["sphere_check"] = {
        "mean_rgb":   _E_wb_rgb.tolist(),
        "chroma_rg":  _rg_dev,
        "chroma_bg":  _bg_dev,
    }

    # cc_measured holds the raw linear swatches from the chart (if found).
    # wb_from_chart_scale holds the derived WB multiplier (already applied via rgb_scale).

    # ── 4. Metering + exposure solve ──────────────────────────────────────
    # If WB was derived from the ColorChecker, the grey patch (patch 22, ~18%
    # reflectance) gives absolute exposure — but ONLY if we account for which
    # direction the chart was actually facing.
    #
    # Physical model:
    #   patch22_luma = albedo × E_incident_on_chart_face / π
    #
    # where E_incident depends on chart orientation:
    #   chart flat on floor  →  E_incident = E_upper  (cosine-weighted sky hemisphere)
    #   chart on tripod      →  E_incident = E_toward(chart_normal)
    #   pose unknown         →  assume E_incident = π  (chart was perfectly metered)
    #
    # The classic formula (exposure = albedo / patch_luma) is only correct when
    # E_incident = π, i.e. the chart was held facing the key light like a perfect
    # incident meter. A chart flat on the floor under a low sun will read lower
    # than albedo even after correct WB, because E_upper < π.
    if wb_from_chart and chart_use_exp:
        exposure_scale_from = wb_from_chart_scale  # original chart WB, not blended
        meas_post_wb   = cc_measured[21] * exposure_scale_from
        meas_luma_post = float(0.2126*meas_post_wb[0]
                               + 0.7152*meas_post_wb[1]
                               + 0.0722*meas_post_wb[2])

        # ── Determine E_incident on the chart face ─────────────────────────
        #
        # CRITICAL: Pose correction (--chart-facing up/sun) is only valid when
        # the chart was photographed as a SEPARATE REFERENCE PLATE in known
        # correct exposure (--colorchecker plate.jpg).
        #
        # When the chart is detected INSIDE the HDRI being calibrated
        # (--colorchecker-in-hdri or auto-search), the HDRI itself is
        # uncalibrated — E_upper computed from it is meaningless as an
        # incident irradiance reference. Using it would be circular:
        #   "scale HDRI based on E_upper of the uncalibrated HDRI"
        #
        # In that case, patch 22 luma IS the absolute reference directly:
        #   exposure_scale = albedo / patch22_luma  (classic formula)
        # This is always correct for an in-HDRI chart regardless of orientation,
        # because the chart pixels ARE the HDRI pixels — they carry the same
        # exposure relationship.
        #
        # Pose correction IS meaningful for external plates because E_upper
        # is computed from a correctly exposed scene (or at least independently).

        chart_is_in_hdri = (checker_src == "__hdri__")
        chart_normal_world = cc_det_info.get("checker_normal_world")
        chart_normal_theta = cc_det_info.get("checker_normal_theta_deg")
        chart_confidence   = float(cc_det_info.get("confidence") or 0.0)
        chart_facing_override = getattr(args, "chart_facing", "auto")

        # Pre-compute E_upper from the WB image (only used for external plate)
        _dirs_exp, _dOmega_exp = latlong_dirs(wb_img.shape[0], wb_img.shape[1])
        _lum_wb_exp = luminance(wb_img)
        _cos_up_exp = np.clip(_dirs_exp[..., 1], 0.0, None)
        E_upper_for_exp = float(np.sum(_lum_wb_exp * _cos_up_exp * _dOmega_exp))

        if chart_is_in_hdri:
            # Chart is inside the HDRI — classic formula: exposure = albedo / patch22_luma.
            # Assumes chart is cleanly lit. Use --colorchecker plate.jpg + --chart-facing
            # for pose-corrected exposure when chart is in shadow or at a known angle.
            pose_mode  = "hdri_internal_classic"
            E_on_chart = math.pi   # E=π → predicted = albedo → scale = albedo / measured
            pose_note  = "Chart inside HDRI — classic formula (albedo / patch22_luma). Assumed clean."
            log(f"  chart inside HDRI — using classic exposure formula (assumed clean illumination)")

        def _E_toward_exp(direction):
            D   = np.asarray(direction, dtype=np.float32)
            D  /= np.linalg.norm(D) + 1e-8
            cos = np.clip((_dirs_exp * D[None, None, :]).sum(axis=-1), 0.0, None)
            return float(np.sum(_lum_wb_exp * cos * _dOmega_exp))

        POSE_FLOOR_MAX  = 50    # theta < 50°  → chart flat on floor
        POSE_TRIPOD_MAX = 130   # 50–130°      → vertical / tripod
        POSE_CONF_MIN   = 0.25  # below this, don't trust the detected normal

        # Pose correction only runs for external reference plates.
        # chart_is_in_hdri already set pose_mode/E_on_chart/pose_note above.
        if not chart_is_in_hdri:
          if chart_facing_override == "up":
            pose_mode  = "floor_forced"
            E_on_chart = E_upper_for_exp
            pose_note  = "--chart-facing up: E_upper used as incident"

          elif chart_facing_override == "sun":
            _sun_dir_for_exp = hot["center_dir"] if "hot" in dir() else np.array([0,1,0])
            pose_mode  = "sun_forced"
            E_on_chart = _E_toward_exp(_sun_dir_for_exp)
            pose_note  = "--chart-facing sun: E toward sun direction used"

          elif (chart_normal_world is not None
                and chart_normal_theta is not None
                and chart_confidence >= POSE_CONF_MIN):
            n = np.asarray(chart_normal_world, dtype=np.float32)
            if chart_normal_theta < POSE_FLOOR_MAX:
                pose_mode  = "floor_detected"
                E_on_chart = E_upper_for_exp
                pose_note  = (f"chart flat on floor detected (θ={chart_normal_theta:.1f}°) "
                              f"— E_upper={E_upper_for_exp:.4f} used as incident")
            elif chart_normal_theta < POSE_TRIPOD_MAX:
                pose_mode  = "tripod_detected"
                E_on_chart = _E_toward_exp(n)
                pose_note  = (f"chart vertical/tripod (θ={chart_normal_theta:.1f}°) "
                              f"— E toward chart normal used")
            else:
                pose_mode  = "fallback_bad_pose"
                E_on_chart = math.pi
                pose_note  = (f"chart normal facing down (θ={chart_normal_theta:.1f}°) "
                              f"— detection unreliable, assuming E=π")
          else:
            pose_mode  = "fallback_no_pose"
            E_on_chart = math.pi
            pose_note  = (f"pose confidence {chart_confidence:.3f} < {POSE_CONF_MIN} "
                          f"or no pose data — assuming chart was metered toward key light (E=π)")

        # ── Compute exposure scale ─────────────────────────────────────────
        # predicted_patch22_luma: what patch 22 SHOULD measure if scene is
        # correctly exposed — i.e. albedo × E_on_chart / π.
        # exposure_scale = predicted / measured  →  rescales HDRI so that a
        # Lambertian grey card at the chart's position reads exactly albedo.
        if meas_luma_post > 1e-6 and E_on_chart > 1e-6:
            predicted_patch_luma = args.albedo * E_on_chart / math.pi
            exposure_scale = predicted_patch_luma / meas_luma_post
        else:
            log("WARNING: Patch 22 post-WB luma or E_on_chart near zero — defaulting exposure to 1.0.")
            exposure_scale = 1.0
            predicted_patch_luma = args.albedo

        classic_scale = args.albedo / meas_luma_post if meas_luma_post > 1e-6 else 1.0

        meter_info = {
            "mode":                   "colorchecker_patch22_pose_aware",
            "pose_mode":              pose_mode,
            "chart_facing_override":  chart_facing_override,
            "E_on_chart":             E_on_chart,
            "E_upper":                E_upper_for_exp,
            "predicted_patch22_luma": predicted_patch_luma,
            "measured_patch22_luma":  meas_luma_post,
            "classic_exposure_scale": classic_scale,
            "target":                 args.albedo,
            "exposure_scale":         exposure_scale,
            "note":                   pose_note,
        }
        exposed = wb_img * exposure_scale

        log(f"Exposure from chart patch 22 (pose-aware):")
        log(f"  pose mode      : {pose_mode}")
        log(f"  {pose_note}")
        log(f"  E_upper        = {E_upper_for_exp:.5f}  (sky hemisphere irradiance in wb_img)")
        log(f"  E_on_chart     = {E_on_chart:.5f}  (irradiance on chart face)")
        log(f"  E_on_chart/π   = {E_on_chart/math.pi:.4f}  "
            f"({'chart well-metered toward key light' if abs(E_on_chart/math.pi - 1.0) < 0.15 else 'chart NOT at E=π — pose correction applied'})")
        log(f"  measured luma  = {meas_luma_post:.5f}  (patch 22 post-WB)")
        log(f"  predicted luma = {predicted_patch_luma:.5f}  (albedo × E_on_chart / π)")
        log(f"  exposure_scale = {exposure_scale:.5f}  (predicted / measured)")
        log(f"  classic_scale  = {classic_scale:.5f}  (albedo / measured, for reference — "
            f"{'same' if abs(classic_scale - exposure_scale) < 0.01 else 'DIFFERS — pose correction active'})")
        if wb_blend_info.get("applied"):
            log(f"  note: WB blend t={wb_blend_t:.3f} — colour corrected by sphere blend, "
                f"exposure anchored to chart luma")
    elif exp_src == "sphere":
        # ── Exposure from sphere / irradiance target ──────────────────────
        # Two sub-modes via --sphere-target:
        #
        #   sphere     : rendered grey sphere mean = albedo × π/4
        #                Physically: what a Lambertian sphere reads when E_upper = π.
        #                Your renderer will show the sphere as the right grey.
        #
        #   irradiance : E_upper = π (cosine-weighted upper hemisphere irradiance)
        #                Physically: a flat grey card facing up reads exactly albedo.
        #                Same as the incident light meter assumption.
        #
        # These are equivalent only when E_upper drives the sphere evenly (uniform
        # hemisphere). They differ for low-sun, indoor, or mixed lighting scenes.
        _sphere_tgt_mode = getattr(args, "sphere_target", "sphere")
        log(f"Exposure from sphere render (--exposure-source sphere  --sphere-target {_sphere_tgt_mode}):")

        if _sphere_tgt_mode == "irradiance":
            # ── Irradiance normalisation: scale so E_upper = π ────────────
            _dirs_ei, _dOmega_ei = latlong_dirs(wb_img.shape[0], wb_img.shape[1])
            _cos_up_ei = np.clip(_dirs_ei[..., 1], 0.0, None)
            _lum_ei    = luminance(wb_img)
            E_upper_ei = float(np.sum(_lum_ei * _cos_up_ei * _dOmega_ei))
            if E_upper_ei > 1e-6:
                exposure_scale = math.pi / E_upper_ei
            else:
                log("WARNING: E_upper near zero — defaulting exposure to 1.0.")
                exposure_scale = 1.0
            E_upper_after = E_upper_ei * exposure_scale
            pred_card_up  = args.albedo * E_upper_after / math.pi   # should = albedo
            meter_info = {
                "mode":              "irradiance_normalise",
                "sphere_target":     _sphere_tgt_mode,
                "E_upper_before":    E_upper_ei,
                "E_upper_target":    math.pi,
                "E_upper_after":     E_upper_after,
                "pred_card_facing_up": pred_card_up,
                "exposure_scale":    exposure_scale,
                "note":              "E_upper scaled to π — flat card facing sky reads albedo",
            }
            log(f"  E_upper (before) = {E_upper_ei:.5f}")
            log(f"  E_target         = π = {math.pi:.5f}")
            log(f"  exposure_scale   = {exposure_scale:.5f}  (π / E_upper)")
            log(f"  E_upper (after)  = {E_upper_after:.5f}  (should be π)")
            log(f"  pred card up     = {pred_card_up:.5f}  (should be albedo={args.albedo:.4f})")

        else:
            # ── Sphere render: scale so sphere mean = albedo × π/4 ────────
            # albedo × π/4 ≈ 0.1414 is what a sphere reads when E_upper=π.
            # We render at low-res (integration is diffuse, high-res not needed).
            _env_exp_sphere = _env_for_sphere(wb_img, max_w=256)
            _sp_exp, _sp_exp_mask = render_gray_ball_vectorized(
                _env_exp_sphere, albedo=args.albedo, res=48, chunk=512)
            _sp_exp_lum  = luminance(_sp_exp)
            _valid        = _sp_exp_mask & (_sp_exp_lum > 1e-6)
            _sp_exp_mean  = float(np.mean(_sp_exp_lum[_valid])) if _valid.any() else 0.0
            sphere_exp_target = args.albedo * math.pi / 4.0
            if _sp_exp_mean > 1e-6:
                exposure_scale = sphere_exp_target / _sp_exp_mean
            else:
                log("WARNING: Sphere render mean near zero — defaulting exposure to 1.0.")
                exposure_scale = 1.0
            meter_info = {
                "mode":             "sphere_render",
                "sphere_target":    _sphere_tgt_mode,
                "sphere_mean_luma": _sp_exp_mean,
                "sphere_exp_target": sphere_exp_target,
                "exposure_scale":   exposure_scale,
                "note":             "Rendered grey sphere mean scaled to albedo × π/4",
            }
            log(f"  sphere mean luma = {_sp_exp_mean:.5f}")
            log(f"  sphere target    = {sphere_exp_target:.5f}  (albedo × π/4 = {args.albedo:.4f} × π/4)")
            log(f"  exposure_scale   = {exposure_scale:.5f}  (target / mean)")

        exposed = wb_img * exposure_scale

    elif exp_src == "meter":
        # ── Pixel-average exposure ────────────────────────────────────────
        # The WB was already applied, so re-meter the WB'd image for exposure.
        # If WB also used meter, the colour is already neutral — just need brightness.
        _wb_lum = luminance(wb_img)
        _int_mode = getattr(args, "integration_mode", "full_sphere")
        _h, _w = wb_img.shape[:2]
        _dirs_m, _ = latlong_dirs(_h, _w)
        if _int_mode == "upper_dome":
            _region_m = _dirs_m[..., 1] > 0.0
        elif _int_mode == "sun_facing":
            _sun_d = _find_peak_direction(wb_img)
            _region_m = (_dirs_m * _sun_d[None, None, :]).sum(axis=-1) > 0.0
        else:
            _region_m = np.ones((_h, _w), dtype=bool)
        _region_lum = _wb_lum[_region_m]
        if _region_lum.size > 0:
            _avg_lum = float(_region_lum.mean())
            exposure_scale = float(args.albedo / max(_avg_lum, 1e-8))
        else:
            exposure_scale = 1.0
            _avg_lum = 0.0
        log(f"Exposure from pixel-average meter ({_int_mode}):")
        log(f"  mean luminance (post-WB) = {_avg_lum:.5f}")
        log(f"  target                   = {args.albedo:.5f}")
        log(f"  exposure_scale           = {exposure_scale:.5f}")
        meter_info = {
            "mode":              "pixel_average",
            "integration_mode":  _int_mode,
            "mean_luminance":    _avg_lum,
            "target":            args.albedo,
            "exposure_scale":    exposure_scale,
        }
        exposed = wb_img * exposure_scale

    elif exp_src == "manual":
        # ── Manual exposure override ──────────────────────────────────────
        manual_exp = getattr(args, "exposure_scale", None) or 1.0
        log(f"Exposure: manual override --exposure-scale {manual_exp:.5f}")
        exposure_scale = float(manual_exp)
        meter_info = {
            "mode":           "manual",
            "exposure_scale": exposure_scale,
            "note":           f"Manual override: --exposure-scale {manual_exp}",
        }
        exposed = wb_img * exposure_scale

    elif exp_src == "none":
        log("Exposure: no correction (--exposure-source none)")
        exposure_scale = 1.0
        meter_info = {"mode": "none", "exposure_scale": 1.0}
        exposed = wb_img.copy()

    else:
        # ── Metering fallback (dome / irradiance modes) ───────────────────
        log(f"Exposure from metering (--exposure-source {exp_src!r} → metering mode {args.metering_mode!r}):")
        meter_info = meter_image(wb_img, mode=args.metering_mode,
                                 stat=args.meter_stat,
                                 swatch_xy=meter_swatch_xy,
                                 swatch_size=args.swatch_size)
        exposure_scale = solve_exposure_scale(meter_info, args.meter_target)
        exposed = wb_img * exposure_scale
    save_png_preview(os.path.join(args.debug_dir, "02_exposed_preview.png"), exposed)
    meta["metering"]       = meter_info
    meta["exposure_scale"] = float(exposure_scale)

    # ── 5. Hot lobe extraction ────────────────────────────────────────────
    hot = extract_hot_lobe_key(exposed,
                               threshold=args.sun_threshold,
                               upper_only=args.sun_upper_only,
                               blur_px=args.sun_blur_px)
    save_mask_preview(os.path.join(args.debug_dir, "03_hot_mask.png"), hot["mask"])
    meta["hot_lobe"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in hot.items() if k != "mask"}

    # ── 6. HDRI centering ─────────────────────────────────────────────────
    shift_info = {"applied": False}
    if args.center_hdri:
        exposed, shift_info = center_hdri_on_sun(
            exposed, hot, center_elevation=args.center_elevation)
        shift_info["applied"] = True
        # Re-extract hot lobe on centred image for correct debug masks
        hot = extract_hot_lobe_key(exposed,
                                   threshold=args.sun_threshold,
                                   upper_only=args.sun_upper_only,
                                   blur_px=args.sun_blur_px)
        save_mask_preview(
            os.path.join(args.debug_dir, "03b_hot_mask_after_centre.png"),
            hot["mask"])
        save_png_preview(
            os.path.join(args.debug_dir, "03c_centred_preview.png"), exposed)
        log(f"HDRI centred: col_shift={shift_info['col_shift_px']}px  "
            f"row_shift={shift_info['row_shift_px']}px")
    else:
        log("HDRI centering disabled (--no-center-hdri)")
    meta["hdri_centering"] = shift_info

    # ── 7. Split render + sun gain solve ──────────────────────────────────
    # Determine sphere solve target — what mean value should a Lambertian
    # sphere render to after the hot-lobe energy is reconstructed.

    sun_elevation_deg = 90.0 - hot["theta_deg"]
    log(f"Sun elevation: {sun_elevation_deg:.1f}°")

    # ── Save calibrated base dome EXR (WB applied, sun lobe zeroed) ─────
    # Lets you verify chart patch reads ~0.18 and colours are correct
    # before the sun energy is reconstructed.
    base_dome = exposed * (1.0 - hot["mask"][..., None])
    base_dome_path = os.path.join(args.debug_dir, "base_dome_calibrated.exr")
    save_exr(base_dome_path, base_dome)
    log(f"Saved calibrated base dome (sun zeroed): {base_dome_path}")

    # Pre-solve sphere renders removed — solve uses direct irradiance integration,
    # no sphere render needed. Single final render after solve for validation.
    pre_verify_rgb = np.zeros(3, dtype=np.float32)  # placeholder for report
    pre_metrics    = {}

    _gain_ceiling = float(getattr(args, "sun_gain_ceiling", 2000.0))
    _gain_rolloff = float(getattr(args, "sun_gain_rolloff", 500.0))
    _gain_rolloff = min(_gain_rolloff, _gain_ceiling)  # rolloff can't exceed ceiling
    log(f"Sun gain ceiling: {_gain_ceiling:.0f}×  rolloff start: {_gain_rolloff:.0f}×"
        + ("  (hard ceiling — rolloff disabled)" if _gain_rolloff >= _gain_ceiling else
           f"  (soft rolloff above {_gain_rolloff:.0f}×, sqrt curve to {_gain_ceiling:.0f}×)"))

    # ── Sun disc gain solve ───────────────────────────────────────────────
    # Solve the hot lobe against an explicit target card orientation.

    _dirs_lobe, _dOmega_lobe = latlong_dirs(exposed.shape[0], exposed.shape[1])

    # Base dome only (lobe zeroed)
    _base_env = exposed * (1.0 - hot["mask"][..., None])

    # Neutralised lobe (sky contamination removed, sun is white [L,L,L])
    _lobe_env_neutral = neutralise_lobe(exposed, hot["mask"], strength=args.lobe_neutralise)
    _lobe_env_n = _lobe_env_neutral * hot["mask"][..., None]

    def _irradiance_rgb_toward(_env, _direction):
        D = np.asarray(_direction, dtype=np.float32)
        D /= np.linalg.norm(D) + 1e-8
        _cos = np.clip((_dirs_lobe * D[None, None, :]).sum(axis=-1), 0.0, None)
        return np.array([
            float(np.sum(_env[..., c] * _cos * _dOmega_lobe))
            for c in range(3)
        ], dtype=np.float32)

    _solve_mode = getattr(args, "sphere_solve", "energy_conservation")
    _sun_dir = np.asarray(hot["center_dir"], dtype=np.float32)
    _sun_dir /= np.linalg.norm(_sun_dir) + 1e-8
    _sun_vertical_dir = np.array([_sun_dir[0], 0.0, _sun_dir[2]], dtype=np.float32)
    if float(np.linalg.norm(_sun_vertical_dir)) < 1e-6:
        _sun_vertical_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        _sun_vertical_dir /= np.linalg.norm(_sun_vertical_dir) + 1e-8

    _solve_target_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _solve_target_key = "card_up"
    _solve_target_summary = "grey card facing sky = albedo"
    _solve_target_metric = "E_upper / pi"

    if _solve_mode == "sun_facing_card":
        _solve_target_dir = _sun_dir
        _solve_target_key = "card_sun"
        _solve_target_summary = "grey card facing sun = albedo"
        _solve_target_metric = "E_sun / pi"
    elif _solve_mode == "sun_facing_vertical":
        _solve_target_dir = _sun_vertical_dir
        _solve_target_key = "card_vertical_sun"
        _solve_target_summary = "vertical grey card facing sun azimuth = albedo"
        _solve_target_metric = "E_vertical_sun / pi"

    if _solve_mode == "none":
        _gains_solve = np.ones(3, dtype=np.float32)
        _E_base_rgb = _irradiance_rgb_toward(_base_env, _solve_target_dir)
        _E_lobe_rgb = _irradiance_rgb_toward(_lobe_env_n, _solve_target_dir)
        _card_base_rgb = args.albedo * _E_base_rgb / math.pi
        _card_final_rgb = _card_base_rgb.copy()
        log("Sun lobe solve disabled (--sphere-solve none)")
        solution = {
            "gain": 1.0,
            "gains_per_channel": _gains_solve,
            "metrics": {},
            "gain_diag": {
                "method": "disabled",
                "E_base_rgb": _E_base_rgb.tolist(),
                "E_lobe_rgb": _E_lobe_rgb.tolist(),
                "E_target": math.pi,
                "gains": _gains_solve.tolist(),
                "card_base": _card_base_rgb.tolist(),
                "card_final": _card_final_rgb.tolist(),
                "mode": _solve_target_key,
                "target_summary": _solve_target_summary,
                "target_metric": _solve_target_metric,
            },
        }
        corrected = exposed.copy()
    else:
        _E_base_rgb = _irradiance_rgb_toward(_base_env, _solve_target_dir)
        _E_lobe_rgb = _irradiance_rgb_toward(_lobe_env_n, _solve_target_dir)
        _card_base_rgb = args.albedo * _E_base_rgb / math.pi
        log(f"Sun lobe solve — base dome {_solve_target_key}: "
            f"R={_card_base_rgb[0]:.4f}  G={_card_base_rgb[1]:.4f}  B={_card_base_rgb[2]:.4f}  "
            f"(target={args.albedo:.4f} each)")

        _gains_solve = np.ones(3, dtype=np.float32)
        for _c, _name in enumerate(('R', 'G', 'B')):
            _E_needed = math.pi - _E_base_rgb[_c]
            _E_lobe_c = _E_lobe_rgb[_c]
            if _E_lobe_c < 1e-10:
                log(f"  {_name}: lobe irradiance near zero — gain=1.0")
                _gains_solve[_c] = 1.0
            elif _E_needed <= 0:
                log(f"  {_name}: base already ≥ π (E_base={_E_base_rgb[_c]:.4f}) — gain=1.0")
                _gains_solve[_c] = 1.0
            else:
                _g_raw = _E_needed / _E_lobe_c
                _g = apply_gain_ceiling(_g_raw, _gain_ceiling, _gain_rolloff)
                _gains_solve[_c] = float(_g)
                _card_final_c = args.albedo * (_E_base_rgb[_c] + _g * _E_lobe_c) / math.pi
                if _g_raw > _gain_ceiling:
                    log(f"  {_name}: raw={_g_raw:.1f}× → ceiling={_g:.1f}×  card={_card_final_c:.4f}")
                else:
                    log(f"  {_name}: gain={_g:.2f}×  card={_card_final_c:.4f}")

        _card_final_rgb = args.albedo * (_E_base_rgb + _gains_solve * _E_lobe_rgb) / math.pi
        log(f"Sun lobe solve — final {_solve_target_key} predicted: "
            f"R={_card_final_rgb[0]:.4f}  G={_card_final_rgb[1]:.4f}  B={_card_final_rgb[2]:.4f}  "
            f"(should be {args.albedo:.4f} each)")

        solution = {
            "gain": float(_gains_solve.mean()),
            "gains_per_channel": _gains_solve,
            "metrics": {},
            "gain_diag": {
                "method": "irradiance",
                "E_base_rgb": _E_base_rgb.tolist(),
                "E_lobe_rgb": _E_lobe_rgb.tolist(),
                "E_target": math.pi,
                "gains": _gains_solve.tolist(),
                "card_base": _card_base_rgb.tolist(),
                "card_final": _card_final_rgb.tolist(),
                "mode": _solve_target_key,
                "target_summary": _solve_target_summary,
                "target_metric": _solve_target_metric,
            },
        }

        corrected = apply_sun_gain_per_channel(exposed, hot["mask"], _gains_solve,
                                               lobe_neutralise_strength=args.lobe_neutralise)

    # Apply per-channel gains (energy_conservation mode) or scalar (legacy modes)
    gains_pc = solution["gains_per_channel"]
    save_png_preview(os.path.join(args.debug_dir, "07_corrected_preview.png"), corrected)

    _final_balance_target = getattr(args, "final_balance_target", "none")
    final_balance_info = {
        "applied": False,
        "target": _final_balance_target,
        "target_mode": _solve_target_key,
        "target_summary": _solve_target_summary,
    }
    if _final_balance_target != "none":
        _lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        _fb_E_rgb = _irradiance_rgb_toward(corrected, _solve_target_dir)
        _fb_card_before = args.albedo * _fb_E_rgb / math.pi
        _fb_luma_before = float(np.dot(_fb_card_before, _lum_w))

        _neutral_level = max(_fb_luma_before, 1e-6)
        _fb_wb_scale = (_neutral_level / np.clip(_fb_card_before, 1e-6, None)).astype(np.float32)

        _fb_card_after_wb = _fb_card_before * _fb_wb_scale
        _fb_luma_after_wb = float(np.dot(_fb_card_after_wb, _lum_w))

        _fb_exposure_scale = float(args.albedo / max(_fb_luma_after_wb, 1e-6))

        _fb_total_scale = (_fb_wb_scale * _fb_exposure_scale).astype(np.float32)
        corrected = corrected * _fb_total_scale[None, None, :]
        _fb_card_after = _fb_card_before * _fb_total_scale
        _fb_luma_after = float(np.dot(_fb_card_after, _lum_w))

        log(f"Final post-balance — target {_solve_target_key}: "
            f"card before R={_fb_card_before[0]:.4f} G={_fb_card_before[1]:.4f} B={_fb_card_before[2]:.4f}  "
            f"(luma={_fb_luma_before:.4f})")
        log(f"  wb scale      : R={_fb_wb_scale[0]:.4f}  G={_fb_wb_scale[1]:.4f}  B={_fb_wb_scale[2]:.4f}")
        log(f"  exposure scale: {_fb_exposure_scale:.4f}")
        log(f"  total scale   : R={_fb_total_scale[0]:.4f}  G={_fb_total_scale[1]:.4f}  B={_fb_total_scale[2]:.4f}")
        log(f"  card after    : R={_fb_card_after[0]:.4f}  G={_fb_card_after[1]:.4f}  B={_fb_card_after[2]:.4f}  "
            f"(luma={_fb_luma_after:.4f}, target={args.albedo:.4f})")

        save_png_preview(os.path.join(args.debug_dir, "07b_final_balanced_preview.png"), corrected)
        final_balance_info.update({
            "applied": True,
            "card_before_rgb": _fb_card_before.tolist(),
            "card_before_luma": _fb_luma_before,
            "wb_scale": _fb_wb_scale.tolist(),
            "exposure_scale": _fb_exposure_scale,
            "total_scale": _fb_total_scale.tolist(),
            "card_after_rgb": _fb_card_after.tolist(),
            "card_after_luma": _fb_luma_after,
        })

    # ── Validation render: final HDR → grey sphere ─────────────────────────
    # This is the ground truth check. If the pipeline is correct, the sphere
    # mean RGB must equal meter_target on every channel independently.
    log("── Final validation sphere render ──")
    final_sphere, final_sphere_mask = render_gray_ball_vectorized(
        corrected, albedo=args.albedo, res=args.sphere_res)
    save_png_preview(os.path.join(args.debug_dir, "08_verify_sphere_final.png"), final_sphere)
    # Compute validation stats from the single render (no second render needed)
    _fsph_valid = final_sphere[final_sphere_mask]
    final_verify_rgb = _fsph_valid.mean(axis=0) if len(_fsph_valid) else np.zeros(3)
    _fv_luma = float(0.2126*final_verify_rgb[0] + 0.7152*final_verify_rgb[1] + 0.0722*final_verify_rgb[2])
    _fv_rg = float((final_verify_rgb[0]-final_verify_rgb[1]) / (_fv_luma+1e-8))
    _fv_bg = float((final_verify_rgb[2]-final_verify_rgb[1]) / (_fv_luma+1e-8))
    log(f"Sphere verify [final]: mean RGB={final_verify_rgb.tolist()}")
    log(f"  luma={_fv_luma:.4f}  R-G={_fv_rg:+.4f}  B-G={_fv_bg:+.4f} (should be ~0)")
    if abs(_fv_rg) > 0.05 or abs(_fv_bg) > 0.05:
        warn(f"Sphere [final] colour cast: R-G={_fv_rg:+.3f} B-G={_fv_bg:+.3f}")

    meta["sphere"] = {
        "albedo":                args.albedo,
        "sphere_res":            args.sphere_res,
        "sphere_solve":          args.sphere_solve,
        "gray_ball_pre_solve":   pre_metrics,
        "sun_gain_mean":         float(solution["gain"]),
        "sun_gains_per_channel": gains_pc.tolist(),
        "gray_ball_post_solve":  solution["metrics"],
        "peak_ratio":            float(solution.get("ratio", 0.0)),
        "verify_pre_wb_mean_rgb":  pre_verify_rgb.tolist(),
        "verify_final_mean_rgb":   final_verify_rgb.tolist(),
        "verify_final_neutral":    bool(
            abs(float(final_verify_rgb[0]) - float(final_verify_rgb[1])) < 0.01 and
            abs(float(final_verify_rgb[2]) - float(final_verify_rgb[1])) < 0.01
        ),
    }
    meta["final_balance"] = final_balance_info

    # ── 8. Reference sphere calibration ───────────────────────────────────
    sphere_cal_info = {"applied": False}
    if args.ref_sphere:
        log(f"Loading reference sphere image: {args.ref_sphere}")
        ref_img_linear, _ = load_image_any(args.ref_sphere, target_colorspace=working_cs)
        ref_bgr = cv2.cvtColor(
            np.clip(linear_to_srgb(np.clip(_to_display_srgb_linear(ref_img_linear, working_cs), 0.0, 1.0)) * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)

        # Auto-detect or use manual coords
        if all(v is not None for v in [args.ref_sphere_cx,
                                        args.ref_sphere_cy,
                                        args.ref_sphere_r]):
            cx, cy, r = args.ref_sphere_cx, args.ref_sphere_cy, args.ref_sphere_r
            log(f"Using manual sphere coords: cx={cx} cy={cy} r={r}")
        else:
            log("Auto-detecting sphere in reference image...")
            cx, cy, r = detect_sphere_auto(ref_bgr)
            log(f"Detected sphere: cx={cx} cy={cy} r={r}")

        # Estimate light direction from shading
        L_cam, intensity = estimate_light_dir_from_shading(ref_img_linear, cx, cy, r)

        # Solve k via WLS irradiance matching
        k_global, k_per_ch, cal_diag = calibrate_exposure_from_sphere(
            corrected, ref_img_linear,
            cx, cy, r,
            albedo=args.ref_sphere_albedo,
        )

        # Apply: scale the entire HDR
        corrected = corrected * k_global

        # Per-channel correction if spread > 5%
        k_spread = float(np.std(k_per_ch) / (np.mean(k_per_ch) + 1e-8))
        if k_spread > 0.05:
            lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            k_neutral = k_per_ch / np.dot(k_per_ch, lum_w)
            corrected = corrected * k_neutral[None, None, :]
            log(f"Per-channel colour correction applied (spread={k_spread:.2%})")
        else:
            log(f"Per-channel spread {k_spread:.2%} < 5%% — skipping colour trim")

        save_png_preview(os.path.join(args.debug_dir, "09_ref_sphere_calibrated.png"),
                         corrected)

        sphere_cal_info = {
            "applied":              True,
            "ref_sphere_path":      args.ref_sphere,
            "ball_cx":              cx, "ball_cy": cy, "ball_r": r,
            "light_dir_cam":        L_cam.tolist(),
            "k_global":             k_global,
            "k_per_channel":        k_per_ch.tolist(),
            "k_spread":             k_spread,
            "per_channel_applied":  k_spread > 0.05,
            "calibration":          cal_diag,
        }
        log(f"Reference sphere calibration: k={k_global:.4f} "
            f"({np.log2(k_global+1e-8):+.2f} stops)")

    meta["ref_sphere_calibration"] = sphere_cal_info

    # ── 9. ColorChecker result ────────────────────────────────────────────
    if cc_measured is not None:
        cc_info = {
            "applied":               True,
            "source":                checker_src,
            "wb_from_neutral5_grey": True,
            "wb_scale":              wb_from_chart_scale.tolist() if wb_from_chart else None,
            "patch_diagnostics":     _cc_patch_diag,
            "pose": {
                "checker_normal_theta_deg": cc_det_info.get("checker_normal_theta_deg"),
                "checker_normal_phi_deg":   cc_det_info.get("checker_normal_phi_deg"),
                "checker_normal_world":     cc_det_info.get("checker_normal_world"),
                "refinement_pass":          cc_det_info.get("refinement_pass"),
            },
            "detection_confidence": cc_det_info.get("confidence"),
            "tiles_searched":       cc_det_info.get("tiles_searched"),
            "detection":            cc_det_info,
            "note": "WB derived from CC24 patch #22 (Neutral 5, ~18% grey). No 3×3 matrix.",
        }
    else:
        cc_info = {"applied": False,
                   "reason": "no chart detected" if checker_src else "not requested"}

    meta["colorchecker"] = cc_info

    # ── 10. Save final EXR ────────────────────────────────────────────────
    save_exr(args.out, corrected)

    # ── 10b. Energy validation ────────────────────────────────────────────
    # Analytical integration of the final EXR — no rendering loop, purely
    # vectorised. Confirms the gain solve didn't add phantom energy and
    # that a Lambertian shading call will return physically correct values.
    #
    # Definitions (y-up, ERP, θ=0 at zenith):
    #   dΩ  = sin(θ) dθ dφ  (solid angle element)
    #   E_upper = ∫_upper  L(ω) cos(θ) dΩ   irradiance on upward flat surface
    #   E_lower = ∫_lower  L(ω) |cos(θ)| dΩ irradiance on downward flat surface
    #   E_full  = ∫_all    L(ω) dΩ           total radiant flux (no cosine)
    #
    # Expected Lambertian results:
    #   flat card facing up     = albedo × E_upper / π
    #   Lambertian sphere mean  = albedo × (E_upper + E_lower) / (4π) × π
    #                           = albedo × (E_upper + E_lower) / 4
    #   (averaged over full sphere surface, each point sees its own hemisphere)

    # ══════════════════════════════════════════════════════════════════════
    # 10b. Full energy & calibration validation suite
    # ══════════════════════════════════════════════════════════════════════
    #
    # PHYSICAL MODEL (y-up ERP, viewer at origin):
    #   dΩ      = solid angle element (sin-weighted in ERP)
    #   E_dir   = Σ L(ω) · max(dot(ω, dir), 0) · dΩ   — irradiance on a flat
    #             Lambertian receiver facing `dir`  (incident light meter reading)
    #   L_card  = albedo × E_dir / π                   — reflected radiance of
    #             a grey card facing that direction
    #   sphere  = albedo × (E_upper + E_lower) / 4      — Lambertian sphere mean
    #             (integrates over full sphere surface)
    #
    # E_target = π means: "a grey card facing the light reads its own albedo."
    #   → albedo × π / π = albedo   ✓
    #   → grey sphere = albedo × π / 4 ≈ 0.1414  (sphere ≠ flat card)

    log("── Energy & calibration validation ──────────────────────────────────")
    a = args.albedo

    _dirs_v, _dOmega_v = latlong_dirs(corrected.shape[0], corrected.shape[1])
    _lum_v = luminance(corrected)

    # ── Helper: irradiance toward an arbitrary unit direction ─────────────
    def _E_dir(direction, img=corrected):
        """Cosine-weighted irradiance on a flat Lambertian receiver facing `direction`."""
        D   = np.asarray(direction, dtype=np.float32)
        D  /= np.linalg.norm(D) + 1e-8
        cos = np.clip((_dirs_v * D[None, None, :]).sum(axis=-1), 0.0, None)
        lum = luminance(img)
        return float(np.sum(lum * cos * _dOmega_v))

    def _E_dir_rgb(direction, img=corrected):
        """Per-channel version."""
        D   = np.asarray(direction, dtype=np.float32)
        D  /= np.linalg.norm(D) + 1e-8
        cos = np.clip((_dirs_v * D[None, None, :]).sum(axis=-1), 0.0, None)
        return np.array([float(np.sum(img[..., c] * cos * _dOmega_v))
                         for c in range(3)])

    # ── 6-direction cardinal irradiance map ───────────────────────────────
    # Each value = what an incident light meter reads pointing that direction.
    # After correct calibration E_sun_facing should be ≈ π.
    sun_dir = hot["center_dir"].astype(np.float32)
    sun_dir_neg = -sun_dir

    cardinal = {
        "↑  sky     ": ( 0,  1,  0),
        "↓  ground  ": ( 0, -1,  0),
        "→  east    ": ( 1,  0,  0),
        "←  west    ": (-1,  0,  0),
        "↗  north   ": ( 0,  0,  1),
        "↙  south   ": ( 0,  0, -1),
        "☀  sun     ": sun_dir.tolist(),
        "☀  anti-sun": sun_dir_neg.tolist(),
    }

    E_upper = _E_dir(( 0,  1,  0))
    E_lower = _E_dir(( 0, -1,  0))
    E_sun   = _E_dir(sun_dir)
    sun_vertical_dir = np.array([sun_dir[0], 0.0, sun_dir[2]], dtype=np.float32)
    if float(np.linalg.norm(sun_vertical_dir)) < 1e-6:
        sun_vertical_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        sun_vertical_dir /= np.linalg.norm(sun_vertical_dir) + 1e-8
    E_vertical_sun = _E_dir(sun_vertical_dir)
    E_full  = float(np.sum(_lum_v * _dOmega_v))

    E_upper_rgb = _E_dir_rgb(( 0,  1,  0))
    E_sun_rgb   = _E_dir_rgb(sun_dir)
    E_vertical_sun_rgb = _E_dir_rgb(sun_vertical_dir)
    E_upper_norm = E_upper_rgb / (np.mean(E_upper_rgb) + 1e-8)
    E_sun_norm   = E_sun_rgb   / (np.mean(E_sun_rgb)   + 1e-8)
    E_vertical_sun_norm = E_vertical_sun_rgb / (np.mean(E_vertical_sun_rgb) + 1e-8)
    E_lower_rgb  = _E_dir_rgb(( 0, -1,  0))

    log(f"")
    log(f"  Sun direction : θ={hot['theta_deg']:.1f}°  elevation={90-hot['theta_deg']:.1f}°  "
        f"φ={hot['phi_deg']:.1f}°")

    log(f"")
    log(f"  ── Incident irradiance (light meter readings) ──────────────────")
    log(f"  {'Direction':<18}  {'E (luma)':<10}  {'R':<8}  {'G':<8}  {'B':<8}  "
        f"{'R/G':<6}  {'B/G':<6}  {'Card (luma)':<12}  status")
    log(f"  {'─'*18}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  "
        f"{'─'*6}  {'─'*6}  {'─'*12}  {'─'*12}")

    for label, direction in cardinal.items():
        E_rgb = _E_dir_rgb(direction)
        E_lum = float(np.mean(E_rgb))          # approx luma of irradiance
        card  = a * E_lum / math.pi            # what a grey card reads
        rg    = E_rgb[0] / (E_rgb[1] + 1e-8)
        bg    = E_rgb[2] / (E_rgb[1] + 1e-8)
        # Status: is this channel balanced and is E ≈ π?
        chroma_ok = abs(rg - 1.0) < 0.10 and abs(bg - 1.0) < 0.10
        e_ok      = abs(E_lum - math.pi) / math.pi < 0.15
        status = ("✓" if (chroma_ok and e_ok) else
                  "⚠ cast" if not chroma_ok else
                  "⚠ level")
        log(f"  {label:<18}  {E_lum:<10.4f}  {E_rgb[0]:<8.4f}  {E_rgb[1]:<8.4f}  "
            f"{E_rgb[2]:<8.4f}  {rg:<6.3f}  {bg:<6.3f}  {card:<12.5f}  {status}")

    # ── Key E=π check (the main calibration invariant) ────────────────────
    log(f"")
    log(f"  ── E = π calibration check ─────────────────────────────────────")
    log(f"  Physical target  : π = {math.pi:.5f}")
    log(f"  E_upper (sky ↑)  : {E_upper:.5f}  {'✓' if abs(E_upper-math.pi)/math.pi < 0.15 else '⚠'}")
    log(f"  E_sun   (☀ dir)  : {E_sun:.5f}  {'✓' if abs(E_sun-math.pi)/math.pi < 0.15 else '⚠'}")
    log(f"  E_sun/π          : {E_sun/math.pi:.4f}  (1.00 = perfectly calibrated toward sun)")
    log(f"  E_upper/π        : {E_upper/math.pi:.4f}  (1.00 = perfectly calibrated upward)")
    log(f"")

    # ── Grey card / grey sphere prediction table ──────────────────────────
    pred_card_up     = a * E_upper / math.pi
    pred_card_dn     = a * E_lower / math.pi
    pred_card_sun    = a * E_sun   / math.pi
    pred_card_rgb_up = a * E_upper_rgb / math.pi
    pred_card_rgb_sun= a * E_sun_rgb   / math.pi
    pred_sphere_full = a * (E_upper + E_lower) / 4.0
    pred_sphere_up   = a * E_upper / 4.0
    pred_sphere_sun  = a * E_sun   / 4.0    # sphere placed at sun, upper-hemi only

    log(f"  ── Grey card predictions (albedo={a:.4f}) ─────────────────────")
    log(f"  {'Surface':<28}  {'luma':<8}  {'R':<8}  {'G':<8}  {'B':<8}  note")
    log(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")

    def _card_row(label, E_lum_val, E_rgb_val, note=""):
        card_l = a * E_lum_val / math.pi
        card_r = a * E_rgb_val[0] / math.pi
        card_g = a * E_rgb_val[1] / math.pi
        card_b = a * E_rgb_val[2] / math.pi
        log(f"  {label:<28}  {card_l:<8.5f}  {card_r:<8.5f}  {card_g:<8.5f}  {card_b:<8.5f}  {note}")

    _card_row("flat card facing ↑ (sky)",
              E_upper, E_upper_rgb,
              f"{'≈ albedo ✓' if abs(pred_card_up - a) < a*0.15 else '⚠ not metered'}")
    _card_row("flat card facing ↓ (ground)",
              E_lower, E_lower_rgb, "fill light / bounce")
    _card_row("flat card facing ☀ (sun dir)",
              E_sun, E_sun_rgb,
              f"{'≈ albedo ✓' if abs(pred_card_sun - a) < a*0.15 else '⚠ not at target'}")

    log(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    log(f"  {'grey sphere (full env)':<28}  {pred_sphere_full:<8.5f}  "
        f"{'':8}  {'':8}  {'':8}  albedo × (E_up+E_dn) / 4")
    log(f"  {'grey sphere (upper hemi)':<28}  {pred_sphere_up:<8.5f}  "
        f"{'':8}  {'':8}  {'':8}  albedo × E_upper / 4")
    log(f"  {'grey sphere (sun facing)':<28}  {pred_sphere_sun:<8.5f}  "
        f"{'':8}  {'':8}  {'':8}  albedo × E_sun / 4")
    log(f"")

    # ── Rendered sphere (actual ray integration, ground truth) ────────────
    log(f"  ── Rendered sphere (ray integration) ────────────────────────────")
    _rend_sphere, _rend_mask = render_gray_ball_vectorized(corrected, albedo=a, res=args.sphere_res)
    _rend_rgb   = np.array([float(np.mean(_rend_sphere[..., c][_rend_mask])) for c in range(3)])
    rendered_sphere_mean = float(np.mean(_rend_rgb))

    log(f"  rendered sphere RGB : R={_rend_rgb[0]:.5f}  G={_rend_rgb[1]:.5f}  "
        f"B={_rend_rgb[2]:.5f}  mean={rendered_sphere_mean:.5f}")
    log(f"  analytical (full)   : {pred_sphere_full:.5f}")
    energy_err = abs(rendered_sphere_mean - pred_sphere_full) / (pred_sphere_full + 1e-8)
    log(f"  deviation           : {energy_err:.2%}  "
        f"{'✓ within 10%' if energy_err < 0.10 else '⚠ > 10% — check gain ceiling or lobe neutralisation'}")
    log(f"")

    # ── WB quality ────────────────────────────────────────────────────────
    log(f"  ── White balance quality ────────────────────────────────────────")
    log(f"  E_upper chroma (R/G B/G) : "
        f"R={E_upper_norm[0]:.4f}  G={E_upper_norm[1]:.4f}  B={E_upper_norm[2]:.4f}  "
        f"(ideal 1:1:1)")
    log(f"  E_sun   chroma (R/G B/G) : "
        f"R={E_sun_norm[0]:.4f}  G={E_sun_norm[1]:.4f}  B={E_sun_norm[2]:.4f}")
    chroma_upper = float(np.max(np.abs(E_upper_norm - 1.0)))
    chroma_sun   = float(np.max(np.abs(E_sun_norm   - 1.0)))
    log(f"  max chroma imbalance     : upper={chroma_upper:.4f}  sun={chroma_sun:.4f}  "
        f"(< 0.05 good, < 0.10 acceptable)")

    if wb_from_chart:
        log(f"  WB source: chart patch 22  (photometric ground truth)")
    else:
        log(f"  WB source: {meta.get('white_balance', {}).get('method', 'unknown')}")
    log(f"")

    # ── Sun disc diagnostics ──────────────────────────────────────────────
    log(f"  ── Sun disc diagnostics ─────────────────────────────────────────")
    lobe_solid_angle = float(np.sum(hot["mask"] * _dOmega_v))
    sun_solid_angle_deg2 = math.degrees(math.sqrt(lobe_solid_angle)) ** 2
    # Real sun solid angle ≈ 6.8e-5 sr = 0.00022 sr
    REAL_SUN_SR = 6.8e-5
    log(f"  lobe solid angle  : {lobe_solid_angle:.6f} sr  "
        f"(real sun ≈ {REAL_SUN_SR:.2e} sr, "
        f"lobe is {lobe_solid_angle/REAL_SUN_SR:.0f}× larger — "
        f"{'reasonable halo' if lobe_solid_angle < REAL_SUN_SR * 500 else 'very large — threshold may be too low'})")
    log(f"  peak luma         : {hot['hottest']:.2f}  "
        f"(at threshold {args.sun_threshold:.3f}: "
        f"low={hot['low']:.2f} high={hot['high']:.2f})")
    log(f"  lobe pixel count  : soft={hot['mask_pixel_count_soft']}  "
        f"strong={hot['mask_pixel_count_strong']}")

    # Estimated physical sun peak (what it should be if calibrated)
    # A disc of solid angle Ω_sun with E=π contributes: L_sun = π / Ω_sun
    estimated_physical_peak = math.pi / (REAL_SUN_SR + 1e-10)
    log(f"  estimated physical sun peak radiance : {estimated_physical_peak:.0f} "
        f"(E=π / Ω_sun)")
    log(f"  measured lobe peak in input          : {hot['hottest'] / (args.albedo / 0.18):.2f}  "
        f"(exposure-adjusted)")
    compression = estimated_physical_peak / (hot['hottest'] / (args.albedo / 0.18) + 1.0)
    log(f"  estimated compression ratio          : {compression:.0f}× "
        f"({'severe clipping' if compression > 100 else 'moderate clipping' if compression > 10 else 'mild'})")
    log(f"")

    # ── Per-channel gain solve summary ────────────────────────────────────
    if "gain_diag" in solution:
        gd = solution["gain_diag"]
        log(f"  ── Sun gain solve summary ───────────────────────────────────────")
        log(f"  method    : {gd.get('method', '?')}")
        if gd.get("method") == "irradiance":
            e_base = np.array(gd.get("E_base_rgb", gd.get("E_base", [0.0, 0.0, 0.0])), dtype=np.float32)
            e_lobe = np.array(gd.get("E_lobe_rgb", gd.get("E_lobe", [0.0, 0.0, 0.0])), dtype=np.float32)
            log(f"  target    : {gd.get('target_summary', gd.get('mode', '?'))}")
            log(f"  E_base    : R={e_base[0]:.4f}  G={e_base[1]:.4f}  B={e_base[2]:.4f}")
            log(f"  E_lobe×1  : R={e_lobe[0]:.6f}  G={e_lobe[1]:.6f}  B={e_lobe[2]:.6f}")
            log(f"  E_target  : {gd['E_target']:.5f}  ({gd.get('mode', '?')})")
            E_b = e_base
            E_l = e_lobe
            E_t = gd['E_target']
            E_final_per_ch = E_b + E_l * solution['gains_per_channel']
            log(f"  E_final   : R={E_final_per_ch[0]:.4f}  G={E_final_per_ch[1]:.4f}  "
                f"B={E_final_per_ch[2]:.4f}  (should be ≈ {E_t:.4f} each)")
            log(f"  gains     : R={solution['gains_per_channel'][0]:.2f}×  "
                f"G={solution['gains_per_channel'][1]:.2f}×  "
                f"B={solution['gains_per_channel'][2]:.2f}×")
        log(f"")

    # ── Final summary table ───────────────────────────────────────────────
    log(f"  ╔══════════════════════════════════════════════════════════════╗")
    log(f"  ║  CALIBRATION SUMMARY                                        ║")
    log(f"  ╠══════════════════════════════════════════════════════════════╣")
    _solve_summary_text = solution.get("gain_diag", {}).get("target_summary", "grey card facing sky = albedo")
    log(f"  ║  Target: {_solve_summary_text} ({a:.4f})            ║")
    log(f"  ║                                                              ║")
    log(f"  ║  Predicted values (use to validate on-set references):      ║")
    log(f"  ║    Grey card facing ↑ sky   : {pred_card_up:>8.5f}                  ║")
    log(f"  ║    Grey card facing ☀ sun   : {pred_card_sun:>8.5f}                  ║")
    log(f"  ║    Grey card facing ↓ ground: {pred_card_dn:>8.5f}                  ║")
    log(f"  ║    Grey sphere (full env)   : {pred_sphere_full:>8.5f}                  ║")
    log(f"  ║    Grey sphere (upper only) : {pred_sphere_up:>8.5f}                  ║")
    log(f"  ║    Rendered sphere (actual) : {rendered_sphere_mean:>8.5f}                  ║")
    log(f"  ║                                                              ║")
    _solve_mode = solution.get("gain_diag", {}).get("mode", "card_up")
    if _solve_mode == "card_sun":
        e_pi_check = E_sun / math.pi
        _solve_metric_label = "E_sun / pi"
        chroma_ok_final = chroma_sun < 0.10
    elif _solve_mode == "card_vertical_sun":
        e_pi_check = E_vertical_sun / math.pi
        _solve_metric_label = "E_vertical_sun / pi"
        chroma_ok_final = float(np.max(np.abs(E_vertical_sun_norm - 1.0))) < 0.10
    else:
        e_pi_check = E_upper / math.pi
        _solve_metric_label = "E_upper / pi"
        chroma_ok_final = chroma_upper < 0.10
    solve_ok = abs(e_pi_check - 1.0) < 0.15
    log(f"  ║  {_solve_metric_label} = {e_pi_check:.4f}  {'✓ calibrated' if solve_ok else '⚠ off — check solve'}            ║")
    log(f"  ║  Sun chroma = {chroma_sun:.4f}  {'✓ neutral' if chroma_ok_final else '⚠ coloured — expected for low sun'}        ║")
    log(f"  ║  Rendered vs analytical = {energy_err:.2%}   {'✓' if energy_err < 0.10 else '⚠'}                    ║")
    log(f"  ╚══════════════════════════════════════════════════════════════╝")

    # Sanity check warnings
    if energy_err > 0.10:
        warn(f"⚠ Rendered sphere ({rendered_sphere_mean:.4f}) deviates "
             f"{energy_err:.1%} from analytical ({pred_sphere_full:.4f}). "
             f"Check lobe neutralisation or gain ceiling hit.")
    if chroma_upper > 0.05:
        warn(f"⚠ Upper-hemi chroma imbalance {chroma_upper:.3f} — WB may be off, "
             f"")
    if chroma_sun > 0.10:
        warn(f"⚠ Sun-dir chroma imbalance {chroma_sun:.3f} after solve — "
             f"sky colour cast is large relative to sun energy. "
             f"Rendered output may have residual colour tint.")

    meta["energy_validation"] = {
        "E_upper":                  E_upper,
        "E_lower":                  E_lower,
        "E_sun":                    E_sun,
        "E_full":                   E_full,
        "E_upper_rgb":              E_upper_rgb.tolist(),
        "E_lower_rgb":              E_lower_rgb.tolist(),
        "E_sun_rgb":                E_sun_rgb.tolist(),
        "E_upper_chroma_norm":      E_upper_norm.tolist(),
        "E_sun_chroma_norm":        E_sun_norm.tolist(),
        "E_upper_over_pi":          E_upper / math.pi,
        "E_sun_over_pi":            E_sun   / math.pi,
        "pred_card_facing_up":      pred_card_up,
        "pred_card_facing_down":    pred_card_dn,
        "pred_card_facing_sun":     pred_card_sun,
        "pred_card_rgb_facing_up":  pred_card_rgb_up.tolist(),
        "pred_card_rgb_facing_sun": pred_card_rgb_sun.tolist(),
        "pred_sphere_full":         pred_sphere_full,
        "pred_sphere_upper_only":   pred_sphere_up,
        "pred_sphere_sun_facing":   pred_sphere_sun,
        "rendered_sphere_mean":     rendered_sphere_mean,
        "rendered_sphere_rgb":      _rend_rgb.tolist(),
        "rendered_vs_analytical_err": energy_err,
        "sun_lobe_solid_angle_sr":  lobe_solid_angle,
        "sun_compression_estimate": compression,
        "chroma_imbalance_upper":   chroma_upper,
        "chroma_imbalance_sun":     chroma_sun,
        "solve_target_mode":        "card_up_neutral",
        "albedo":                   a,
    }

    # ── 11. JSON report ───────────────────────────────────────────────────
    meta["notes"] = [
        "Pipeline: load → WB → meter → expose → hot-lobe → [centre] → split-render → sun-gain → [sphere-cal] → [CC-correct] → save.",
        "HDRI centering uses solid-angle-weighted centroid of the hot mask — correct for any luminance range.",
        "Sun gain solve uses vectorised Lambertian sphere render split into base+lobe for fast iteration.",
        "Reference sphere calibration solves k via WLS irradiance matching across all ball pixels.",
        "ColorChecker correction solves a 3×3 linear RGB matrix against CC24 D65 reference values.",
        "This is a practical IBL calibration heuristic, not a full radiometric recovery.",
    ]

    report_path = os.path.join(args.debug_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"Report: {report_path}")
    log(f"Done. Output: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
