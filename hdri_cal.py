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
  pip install pyexr                              # EXR I/O
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

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    import pyexr
    HAVE_PYEXR = True
except Exception:
    HAVE_PYEXR = False

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

def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

def load_image_any(path):
    ext = os.path.splitext(path)[1].lower()
    log(f"Loading: {path}")
    if ext == ".exr":
        if not HAVE_PYEXR:
            raise RuntimeError("EXR requires pyexr: pip install pyexr")
        img = pyexr.read(path).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
        log("Loaded EXR (linear)")
        return img, True
    if ext == ".hdr":
        img = imageio.imread(path).astype(np.float32)
        if img.shape[-1] > 3:
            img = img[..., :3]
        log("Loaded .hdr (linear Radiance)")
        return img, True
    # LDR path
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
    log("Loaded LDR → linearised sRGB")
    return img.astype(np.float32), False

def save_exr(path, img):
    if not HAVE_PYEXR:
        raise RuntimeError("EXR output requires pyexr: pip install pyexr")
    img = np.asarray(img, dtype=np.float32)
    if not path.lower().endswith(".exr"):
        path += ".exr"
    pyexr.write(path, img)
    log(f"Saved EXR: {path}")

def save_png_preview(path, img_linear, percentile=99.5):
    img_linear = np.clip(img_linear, 0.0, None)
    lum = luminance(img_linear)
    valid = lum[np.isfinite(lum)]
    denom = max(1e-6, np.percentile(valid, percentile)) if valid.size else 1.0
    view = linear_to_srgb(np.clip(img_linear / denom, 0.0, 1.0))
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


def estimate_wb_from_sphere_render(img, albedo=0.18, sphere_res=48):
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

    Returns scale, info  (same interface as estimate_wb_from_dome)
    """
    # Render at low res — this is a metering operation, not a quality render
    env = _env_for_sphere(img, max_w=256)
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
        "applied":           True,
        "sphere_mean_rgb":   mean_rgb.tolist(),
        "sphere_res":        sphere_res,
        "scale":             scale.tolist(),
        "kelvin_approx":     kelvin_est,
    }
    return scale.astype(np.float32), info


def apply_white_balance(img, kelvin=None, rgb_scale=None, swatch_xy=None,
                        dome_wb_mode=None, sphere_wb=False):
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
        applied, dome_info = estimate_wb_from_sphere_render(img)
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
    phi   = np.arctan2(z, x)
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

def sphere_metrics(sphere_img, mask):
    lum  = luminance(sphere_img)
    vals = lum[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95":    float(np.percentile(vals, 95)),
        "max":    float(np.max(vals)),
    }

def gray_ball_from_split(env_linear, lobe_mask, albedo=0.18, res=96,
                         lobe_neutralise_strength=1.0):
    """
    Render full, base, and lobe sphere contributions separately.

    The lobe env is neutralised before rendering so that the gain solve
    works on the same colour content that will actually be boosted —
    avoiding a situation where the solve targets a red lobe but the
    application uses a neutralised (white) lobe.
    """
    sphere_full, mask = render_gray_ball_vectorized(env_linear, albedo, res)
    base_env = env_linear * (1.0 - lobe_mask[..., None])
    lobe_env = env_linear *        lobe_mask[..., None]

    # Neutralise the lobe before rendering for the gain solve
    lobe_env_neutral = neutralise_lobe(lobe_env, lobe_mask,
                                       strength=lobe_neutralise_strength)

    sphere_base, _ = render_gray_ball_vectorized(base_env, albedo, res)
    sphere_lobe, _ = render_gray_ball_vectorized(lobe_env_neutral, albedo, res)
    return sphere_full, sphere_base, sphere_lobe, mask


# ══════════════════════════════════════════════════════════════════════════════
# Sun gain solve — per-channel energy conservation
# ══════════════════════════════════════════════════════════════════════════════
#
# The fundamental invariant:
#   A correctly calibrated HDR, when used to light an 18% grey Lambertian
#   sphere, should produce a sphere whose mean RGB equals albedo × 0.18 on
#   every channel independently.
#
# The pipeline:
#   1. Render sphere from WB'd base (no hot lobe) → sphere_base_mean (3,)
#   2. Render sphere from lobe at gain=1            → sphere_lobe_mean (3,)
#   3. Per-channel gain = (target_c - base_mean_c) / lobe_mean_c
#      where target_c = albedo × meter_target  (default 0.18 × 1.0 = 0.18)
#      This is the gain that, when applied to the lobe, exactly restores
#      the missing energy on every channel independently.
#   4. Apply per-channel gain to lobe pixels in the HDR
#   5. Render final sphere → should be [0.18, 0.18, 0.18] mean — log this
#
# Scalar gain (old approach) was wrong because:
#   - It solved on luminance only → R/G/B gain was equal
#   - But sun colour ≠ sky colour, so a scalar boost shifts the highlight
#     colour balance, introducing a colour cast in the final renders

MAX_PHYSICAL_SUN_GAIN = 200.0

def solve_sun_gain_per_channel(sphere_base, sphere_lobe, mask,
                               target_mean=0.18, albedo=0.18):
    """
    Solve per-channel gain to restore clipped/compressed sun energy.

    For each channel c:
        gain_c = (target_c - base_mean_c) / lobe_mean_c

    target_c = albedo × target_mean / albedo = target_mean
    (albedo cancels since both sphere renders used the same albedo)

    We use the mean over ALL sphere pixels (not just highlights) because
    the goal is total energy conservation, not highlight matching.
    Highlight matching was the old wrong approach — it set the peak value
    to some target, ignoring whether the total integrated energy was correct.

    Returns per-channel gains (3,) and diagnostics.
    """
    valid_base = sphere_base[mask]   # (N, 3)
    valid_lobe = sphere_lobe[mask]   # (N, 3)

    base_mean = valid_base.mean(axis=0)   # (3,)
    lobe_mean = valid_lobe.mean(axis=0)   # (3,)

    log(f"Energy solve: base_mean={base_mean.tolist()}")
    log(f"Energy solve: lobe_mean={lobe_mean.tolist()}")
    log(f"Energy solve: target_mean={target_mean:.4f} per channel")

    gains = np.ones(3, dtype=np.float32)
    for c, name in enumerate(('R', 'G', 'B')):
        missing = target_mean - float(base_mean[c])
        lobe_c  = float(lobe_mean[c])
        if lobe_c < 1e-8:
            log(f"  {name}: lobe contribution negligible, gain=1.0")
            gains[c] = 1.0
        elif missing <= 0:
            log(f"  {name}: base already at/above target "
                f"({base_mean[c]:.4f} >= {target_mean:.4f}), gain=1.0")
            gains[c] = 1.0
        else:
            g = missing / lobe_c
            g = float(np.clip(g, 1.0, MAX_PHYSICAL_SUN_GAIN))
            if g >= MAX_PHYSICAL_SUN_GAIN * 0.9:
                warn(f"Sun gain channel {name} hit ceiling {MAX_PHYSICAL_SUN_GAIN}× "
                     f"— check --sun-threshold or input exposure.")
            log(f"  {name}: missing={missing:.4f} lobe={lobe_c:.4f} → gain={g:.4f}")
            gains[c] = g

    if gains.max() >= MAX_PHYSICAL_SUN_GAIN * 0.9:
        warn("One or more channels hit MAX_PHYSICAL_SUN_GAIN — "
             "sun lobe may be too dim or threshold too tight.")

    log(f"Per-channel gains: R={gains[0]:.3f} G={gains[1]:.3f} B={gains[2]:.3f}")
    return gains


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


def verify_sphere_neutrality(env, albedo=0.18, target=0.18, label="final"):
    """
    Render a sphere from env and check if its mean RGB is neutral.
    Logs per-channel mean and deviation from target.
    Returns the mean RGB (3,) for the report.
    """
    sphere, mask = render_gray_ball_vectorized(env, albedo=albedo, res=96)
    valid = sphere[mask]
    if valid.shape[0] == 0:
        warn("verify_sphere_neutrality: no valid sphere pixels")
        return np.array([0.0, 0.0, 0.0])
    mean_rgb = valid.mean(axis=0)
    luma     = float(0.2126*mean_rgb[0] + 0.7152*mean_rgb[1] + 0.0722*mean_rgb[2])
    dev_r    = (mean_rgb[0] - mean_rgb[1]) / (luma + 1e-8)
    dev_b    = (mean_rgb[2] - mean_rgb[1]) / (luma + 1e-8)
    log(f"Sphere verify [{label}]: mean RGB={mean_rgb.tolist()}")
    log(f"  luma={luma:.4f}  target={target:.4f}  "
        f"R-G={dev_r:+.4f}  B-G={dev_b:+.4f} (should be ~0)")
    if abs(dev_r) > 0.05 or abs(dev_b) > 0.05:
        warn(f"Sphere [{label}] is not neutral: R-G={dev_r:+.3f} B-G={dev_b:+.3f}. "
             f"Check WB or gain solve.")
    if abs(luma - target) / (target + 1e-8) > 0.1:
        warn(f"Sphere [{label}] mean luma {luma:.4f} deviates >10% from "
             f"target {target:.4f}. Check exposure metering.")
    return mean_rgb


# Keep old scalar solve for --sphere-solve=direct_highlight / iterative_peak_ratio
# compatibility, but they are no longer the default path.
def solve_sun_gain_direct(sphere_base, sphere_lobe, mask,
                          target_highlight_mean=0.32):
    lum_ref = luminance(sphere_base + sphere_lobe)
    thr     = np.percentile(lum_ref[mask], 97.0)
    hi_mask = (lum_ref >= thr) & mask
    b = float(np.mean(luminance(sphere_base)[hi_mask]))
    s = float(np.mean(luminance(sphere_lobe)[hi_mask]))
    if s < 1e-8:
        gain = 1.0
    else:
        gain = max(1.0, (target_highlight_mean - b) / s)
    gain   = float(np.clip(gain, 1.0, MAX_PHYSICAL_SUN_GAIN))
    sphere = sphere_base + gain * sphere_lobe
    met    = sphere_metrics(sphere, mask)
    log(f"Direct solve (legacy scalar): b={b:.6f} s={s:.6f} → gain={gain:.4f}")
    return {"gain": gain, "gains_per_channel": np.array([gain,gain,gain]),
            "sphere": sphere, "metrics": met,
            "ratio": float(met["p95"] / max(1e-6, met["mean"]))}

def solve_sun_gain_iterative(sphere_base, sphere_lobe, mask,
                             target_peak_ratio=2.5, max_iters=20):
    gain = 1.0
    best = None
    for i in range(max_iters):
        sphere = sphere_base + gain * sphere_lobe
        met    = sphere_metrics(sphere, mask)
        ratio  = met["p95"] / max(1e-6, met["mean"])
        best   = {"gain": gain, "gains_per_channel": np.array([gain,gain,gain]),
                  "sphere": sphere.copy(), "metrics": met, "ratio": ratio}
        log(f"Sun gain iter {i:02d}: gain={gain:.4f} p95/mean={ratio:.4f}")
        if abs(ratio - target_peak_ratio) < 0.03:
            break
        gain = float(np.clip(gain * (1.6 if ratio < target_peak_ratio else 0.8),
                             1.0, MAX_PHYSICAL_SUN_GAIN))
    return best

def solve_sphere_gain(sphere_base, sphere_lobe, mask, mode="energy_conservation",
                      target_peak_ratio=2.5, direct_highlight_target=0.32,
                      target_mean=0.18, albedo=0.18):
    if mode == "none":
        sphere = sphere_base + sphere_lobe
        met    = sphere_metrics(sphere, mask)
        log("Sphere solve disabled (gains=1.0)")
        return {"gain": 1.0, "gains_per_channel": np.ones(3),
                "sphere": sphere, "metrics": met,
                "ratio": float(met["p95"] / max(1e-6, met["mean"]))}
    if mode == "energy_conservation":
        gains  = solve_sun_gain_per_channel(sphere_base, sphere_lobe, mask,
                                            target_mean=target_mean, albedo=albedo)
        sphere = sphere_base + sphere_lobe * gains[None, None, :]
        met    = sphere_metrics(sphere, mask)
        return {"gain": float(gains.mean()), "gains_per_channel": gains,
                "sphere": sphere, "metrics": met,
                "ratio": float(met["p95"] / max(1e-6, met["mean"]))}
    if mode == "direct_highlight":
        return solve_sun_gain_direct(sphere_base, sphere_lobe, mask,
                                     direct_highlight_target)
    if mode == "iterative_peak_ratio":
        return solve_sun_gain_iterative(sphere_base, sphere_lobe, mask,
                                        target_peak_ratio)
    raise ValueError(f"Unknown sphere solve mode: {mode}")

def apply_sun_gain(env_scaled, lobe_mask, gain):
    """Legacy scalar apply — kept for compatibility with old solve modes."""
    base = env_scaled * (1.0 - lobe_mask[..., None])
    sun  = env_scaled *        lobe_mask[..., None]
    return np.clip(base + sun * gain, 0.0, None).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# HDRI orientation check
# ══════════════════════════════════════════════════════════════════════════════

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
    _CC24_LINEAR_SRGB = CC24_LINEAR_SRGB   # keep old alias in case anything references it
    HAVE_CC_ERP = True
except ImportError:
    HAVE_CC_ERP = False
    CC24_LINEAR_SRGB = None
    warn("colorchecker_erp.py not found — ColorChecker detection unavailable. "
         "Place colorchecker_erp.py alongside hdri_cal.py.")


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
                    help="Input latlong (.jpg/.png/.hdr/.exr)")
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

    # ── White balance ─────────────────────────────────────────────────────
    ap.add_argument("--kelvin", type=float, default=None,
                    help="Colour temperature WB in Kelvin, e.g. 5600")
    ap.add_argument("--rgb-scale", type=str, default=None,
                    help="Manual RGB multiplier, e.g. 1.0,0.97,1.05")
    ap.add_argument("--wb-swatch", type=str, default=None,
                    help="Neutral swatch pixel for WB, e.g. 123,456")
    ap.add_argument("--dome-wb", type=str, default=None,
                    choices=["upper_dome", "full_dome", "hot_exclude"],
                    help="Grey-world WB estimation from dome pixels. OFF by default. "
                         "  upper_dome  — sky hemisphere only (recommended) "
                         "  full_dome   — entire 360° sphere "
                         "  hot_exclude — upper dome minus brightest 3%% "
                         "WARNING: will overcorrect intentional colour tints.")
    ap.add_argument("--sphere-wb", action="store_true", default=False,
                    help="Estimate WB by rendering a Lambertian grey sphere and "
                         "neutralising its mean RGB. OFF by default. "
                         "Better than --dome-wb because it correctly weights by "
                         "NdotL and solid angle — the same way your renderer sees "
                         "the lighting. Runs a fast low-res sphere render internally. "
                         "Takes priority over --dome-wb if both are set.")

    # ── Metering ──────────────────────────────────────────────────────────
    ap.add_argument("--metering-mode", default="bottom_dome",
                    choices=["whole_scene", "bottom_dome", "swatch"],
                    help="Exposure metering region")
    ap.add_argument("--meter-stat", default="median",
                    choices=["mean", "median"])
    ap.add_argument("--meter-target", type=float, default=0.18,
                    help="Target metered luminance (linear). 0.18 = 18%% grey")
    ap.add_argument("--swatch", type=str, default=None,
                    help="Pixel for swatch metering, e.g. 123,456")
    ap.add_argument("--swatch-size", type=int, default=5)

    # ── Hot lobe ──────────────────────────────────────────────────────────
    ap.add_argument("--sun-threshold", type=float, default=0.1,
                    help="Hot lobe width: 0..1 fraction below peak. "
                         "0.1 → low=0.9×peak, high=peak")
    ap.add_argument("--sun-upper-only", action="store_true",
                    help="Restrict hot lobe to upper hemisphere (y>0)")
    ap.add_argument("--sun-blur-px", type=int, default=0,
                    help="Gaussian blur on hot mask (odd px, 0=off)")

    # ── HDRI centering ────────────────────────────────────────────────────
    ap.add_argument("--center-hdri", action="store_true", default=True,
                    help="Shift HDRI azimuth so dominant light sits at φ=0 "
                         "(centre column). ON by default.")
    ap.add_argument("--no-center-hdri", dest="center_hdri",
                    action="store_false",
                    help="Disable HDRI azimuth centering")
    ap.add_argument("--center-elevation", action="store_true", default=False,
                    help="Also shift vertically to place sun at equator row. "
                         "Useful for turntables; changes sun elevation so "
                         "use with care.")

    # ── Sphere solve ──────────────────────────────────────────────────────
    ap.add_argument("--albedo", type=float, default=0.18,
                    help="Grey ball albedo")
    ap.add_argument("--sphere-res", type=int, default=96,
                    help="Lambertian sphere render resolution")
    ap.add_argument("--sphere-solve", default="energy_conservation",
                    choices=["energy_conservation", "direct_highlight",
                             "iterative_peak_ratio", "none"],
                    help="Sun gain solve mode. Default: energy_conservation — "
                         "per-channel solve that restores missing integrated energy "
                         "so the final sphere renders neutral grey at meter_target. "
                         "direct_highlight / iterative_peak_ratio are legacy scalar "
                         "modes kept for compatibility.")
    ap.add_argument("--lobe-neutralise", type=float, default=1.0,
                    metavar="0.0-1.0",
                    help="How strongly to desaturate hot lobe pixels before gain "
                         "boosting. 1.0 = fully achromatic (white sun, default). "
                         "0.8 = allow a hint of warmth. 0.0 = no neutralisation. "
                         "Physical basis: after white balance the sun should be "
                         "near-neutral; remaining colour is clipping artefact.")
    ap.add_argument("--direct-highlight-target", type=float, default=0.32,
                    help="Target bright-side mean for direct_highlight solve")
    ap.add_argument("--target-peak-ratio", type=float, default=2.5,
                    help="Target p95/mean ratio for iterative_peak_ratio solve")

    # ── Reference sphere image ────────────────────────────────────────────
    ap.add_argument("--ref-sphere", type=str, default=None,
                    help="Path to photograph of grey/diffuse ball shot on-set. "
                         "Enables physical photometric exposure calibration via "
                         "WLS irradiance matching across all ball pixels.")
    ap.add_argument("--ref-sphere-cx", type=int, default=None,
                    help="Ball centre X in ref-sphere image (auto-detected if omitted)")
    ap.add_argument("--ref-sphere-cy", type=int, default=None,
                    help="Ball centre Y in ref-sphere image (auto-detected if omitted)")
    ap.add_argument("--ref-sphere-r", type=int, default=None,
                    help="Ball radius in px in ref-sphere image (auto-detected if omitted)")
    ap.add_argument("--ref-sphere-albedo", type=float, default=0.18,
                    help="Albedo of the reference ball (0.18 = 18%% grey)")

    # ── ColorChecker ──────────────────────────────────────────────────────
    ap.add_argument("--colorchecker", type=str, default=None,
                    help="Image containing a ColorChecker Classic 24 for colour "
                         "correction. Can be the HDRI itself or a separate plate. "
                         "Requires: pip install colour-science colour-checker-detection")
    ap.add_argument("--colorchecker-in-hdri", action="store_true", default=False,
                    help="Look for ColorChecker inside the HDRI latlong itself "
                         "rather than a separate image.")
    ap.add_argument("--cc-fov", type=float, default=70.0,
                    help="Horizontal FOV (degrees) of each rectilinear tile "
                         "used when sweeping HDRI for ColorChecker. "
                         "Larger = wider view, more distortion at edges.")
    ap.add_argument("--cc-tile-w", type=int, default=900,
                    help="Width of each rectilinear tile in pixels")
    ap.add_argument("--cc-tile-h", type=int, default=675,
                    help="Height of each rectilinear tile in pixels")
    ap.add_argument("--cc-yaw-step", type=float, default=40.0,
                    help="Azimuth step between tiles (degrees). "
                         "Overlap = FOV - step. Smaller = denser sweep, slower.")
    ap.add_argument("--cc-pitches", type=float, nargs="+",
                    default=[-45.0, -20.0, 0.0, 20.0, 45.0],
                    help="Elevation angles to sweep (degrees, 0=horizon). "
                         "Covers ±45° around horizon — chart is always near "
                         "horizon (tripod/floor), never in the sky extremes.")

    args = ap.parse_args()
    os.makedirs(args.debug_dir, exist_ok=True)

    rgb_scale       = parse_rgb_scale(args.rgb_scale) if args.rgb_scale else None
    wb_swatch_xy    = parse_xy(args.wb_swatch)        if args.wb_swatch else None
    meter_swatch_xy = parse_xy(args.swatch)           if args.swatch    else None

    meta = {"input": args.input, "args": vars(args)}

    # ── 1. Load ───────────────────────────────────────────────────────────
    img, is_hdr = load_image_any(args.input)
    img = np.clip(img, 0.0, None).astype(np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError("Input must be RGB")
    img = apply_res_scale(img, args.res)
    meta["is_hdr_input"] = bool(is_hdr)
    meta["working_resolution"] = list(img.shape[:2])

    # ── 2. Validate orientation ───────────────────────────────────────────
    orient_ratio = validate_hdri_orientation(img)
    meta["orientation_energy_ratio"] = float(orient_ratio)

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
    if args.colorchecker:
        checker_src = args.colorchecker
    elif args.colorchecker_in_hdri:
        checker_src = "__hdri__"

    if checker_src and HAVE_CC_ERP:
        cc_debug = os.path.join(args.debug_dir, "colorchecker")
        os.makedirs(cc_debug, exist_ok=True)

        if checker_src == "__hdri__":
            log("Searching for ColorChecker in HDRI latlong "
                "(gnomonic tile sweep) ...")
            cc_measured, cc_det_info = find_colorchecker_in_erp(
                img,   # raw linear, no WB yet
                fov_deg=args.cc_fov,
                tile_w=args.cc_tile_w,
                tile_h=args.cc_tile_h,
                yaw_step_deg=args.cc_yaw_step,
                pitch_values=tuple(args.cc_pitches),
                debug_dir=cc_debug,
            )
        else:
            log(f"Loading ColorChecker plate: {checker_src}")
            cc_img_raw, _ = load_image_any(checker_src)
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
            # CC24_LINEAR_SRGB stores display-referred sRGB values — must linearise
            # before comparing against the HDR measured values which are linear.
            grey_patch_idx = 21   # CC24 patch 22, Neutral 5, 0-based index

            def _srgb_to_lin(v):
                v = np.clip(v, 0.0, 1.0)
                return np.where(v <= 0.04045,
                                v / 12.92,
                                ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)

            meas = cc_measured[grey_patch_idx]                          # (3,) linear RGB, from HDR
            ref  = _srgb_to_lin(CC24_LINEAR_SRGB[grey_patch_idx])      # (3,) linear RGB, ~0.18

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

    # ── 4. White balance ──────────────────────────────────────────────────
    # Priority (first applies, rest ignored):
    #   1. ColorChecker white patch  — most accurate, physical reference
    #   2. --rgb-scale               — explicit manual override
    #   3. --kelvin                  — colour temperature
    #   4. --wb-swatch               — sample a neutral pixel
    #   5. --sphere-wb               — integrated sphere render (heuristic)
    #   6. --dome-wb                 — dome pixel average (heuristic)
    #   7. nothing                   — passthrough

    if wb_from_chart:
        if args.sphere_wb:
            log("WB: ColorChecker found — --sphere-wb ignored (chart takes priority)")
        if args.dome_wb:
            log(f"WB: ColorChecker found — --dome-wb={args.dome_wb} ignored (chart takes priority)")
        if args.kelvin:
            log(f"WB: ColorChecker found — --kelvin={args.kelvin} ignored (chart takes priority)")
        wb_source = "colorchecker_patch19"
    elif rgb_scale is not None:
        wb_source = "rgb_scale_manual"
    elif args.kelvin:
        wb_source = f"kelvin_{args.kelvin:.0f}K"
    elif wb_swatch_xy:
        wb_source = "wb_swatch_pixel"
    elif args.sphere_wb:
        wb_source = "sphere_render"
    elif args.dome_wb:
        wb_source = args.dome_wb
    else:
        wb_source = "none"

    log(f"WB method: {wb_source}")

    wb_img, wb_scale, wb_dome_info = apply_white_balance(
        img,
        kelvin=args.kelvin if not wb_from_chart else None,
        rgb_scale=rgb_scale,
        swatch_xy=wb_swatch_xy if not wb_from_chart else None,
        dome_wb_mode=args.dome_wb if not wb_from_chart else None,
        sphere_wb=args.sphere_wb and not wb_from_chart)

    save_png_preview(os.path.join(args.debug_dir, "01_wb_preview.png"), wb_img)
    meta["white_balance"] = {
        "source":        wb_source,
        "wb_scale":      wb_scale.tolist(),
        "dome_wb_mode":  args.dome_wb if not wb_from_chart else None,
        "sphere_wb":     args.sphere_wb and not wb_from_chart,
        "wb_info":       wb_dome_info,
    }

    # ── WB cross-check: always render sphere after WB and report neutrality ──
    # Run a low-res sphere render on the WB-applied image.
    # This is the single most useful sanity check:
    #   - If WB is correct, sphere mean R≈G≈B (achromatic)
    #   - R-G and B-G deviation tells you the residual colour cast in stops
    # When chart WB was used, also compute sphere WB independently and compare.
    # Large disagreement → chart patch may be misidentified or in shadow.
    log("── WB sanity check (sphere render) ──")
    _env_wb_check = _env_for_sphere(wb_img, max_w=256)
    _sphere_check, _sphere_mask = render_gray_ball_vectorized(
        _env_wb_check, albedo=args.albedo, res=48, chunk=512)
    _sp_rgb  = np.mean(_sphere_check[_sphere_mask], axis=0)   # mean over valid pixels only
    _sp_luma = float(0.2126*_sp_rgb[0] + 0.7152*_sp_rgb[1] + 0.0722*_sp_rgb[2])
    _rg_dev  = float(_sp_rgb[0] - _sp_rgb[1])
    _bg_dev  = float(_sp_rgb[2] - _sp_rgb[1])
    log(f"  sphere mean RGB : R={_sp_rgb[0]:.4f}  G={_sp_rgb[1]:.4f}  B={_sp_rgb[2]:.4f}")
    log(f"  chroma deviation: R-G={_rg_dev:+.4f}  B-G={_bg_dev:+.4f}  "
        f"(ideal=0.0, |total|={abs(_rg_dev)+abs(_bg_dev):.4f})")
    _chroma_total = abs(_rg_dev) + abs(_bg_dev)

    if wb_from_chart:
        # Also compute sphere WB independently for comparison
        _sphere_wb_scale, _ = estimate_wb_from_sphere_render.__wrapped__(wb_img, args.albedo, 48) \
            if hasattr(estimate_wb_from_sphere_render, '__wrapped__') \
            else (None, None)
        # Simpler: just compute the sphere mean on the raw (pre-WB) image
        _env_raw_check = _env_for_sphere(img, max_w=256)
        _sp_raw, _sp_raw_mask = render_gray_ball_vectorized(_env_raw_check, albedo=args.albedo, res=48, chunk=512)
        _sp_raw_rgb = np.mean(_sp_raw[_sp_raw_mask], axis=0)
        _sp_raw_luma = float(0.2126*_sp_raw_rgb[0] + 0.7152*_sp_raw_rgb[1] + 0.0722*_sp_raw_rgb[2])
        _sphere_implied_scale = _sp_raw_luma / np.clip(_sp_raw_rgb, 1e-8, None)
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

    if _chroma_total > 0.10:
        warn(f"⚠⚠ Sphere render shows strong colour cast after WB "
             f"(R-G={_rg_dev:+.4f} B-G={_bg_dev:+.4f}). "
             f"WB may be wrong — check 01_wb_sphere_check.png")
    elif _chroma_total > 0.04:
        warn(f"⚠ Sphere render shows minor colour cast after WB "
             f"(R-G={_rg_dev:+.4f} B-G={_bg_dev:+.4f}).")

    # Save sphere debug image for this WB check
    save_png_preview(os.path.join(args.debug_dir, "01_wb_sphere_check.png"), _sphere_check)
    meta["white_balance"]["sphere_check"] = {
        "mean_rgb":   _sp_rgb.tolist(),
        "chroma_rg":  _rg_dev,
        "chroma_bg":  _bg_dev,
    }

    # cc_measured holds the raw linear swatches from the chart (if found).
    # wb_from_chart_scale holds the derived WB multiplier (already applied via rgb_scale).

    # ── 4. Metering + exposure solve ──────────────────────────────────────
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
    # WB sphere verify: render from the exposed (WB+metered) image to check
    # that WB is correct before we do the gain solve.
    log("── Pre-solve sphere verify (should be neutral, luma≈meter_target) ──")
    pre_verify_rgb = verify_sphere_neutrality(
        exposed, albedo=args.albedo, target=args.meter_target, label="post-WB")
    save_png_preview(os.path.join(args.debug_dir, "04a_verify_sphere_pre.png"),
                     render_gray_ball_vectorized(exposed, albedo=args.albedo,
                                                 res=args.sphere_res)[0])

    sphere_full, sphere_base, sphere_lobe, sphere_mask = gray_ball_from_split(
        exposed, hot["mask"], albedo=args.albedo, res=args.sphere_res,
        lobe_neutralise_strength=args.lobe_neutralise)

    save_png_preview(os.path.join(args.debug_dir, "04_grayball_full.png"),    sphere_full)
    save_png_preview(os.path.join(args.debug_dir, "05_grayball_base.png"),    sphere_base)
    save_png_preview(os.path.join(args.debug_dir, "06_grayball_lobe.png"),    sphere_lobe)

    pre_metrics = sphere_metrics(sphere_full, sphere_mask)
    log(f"Sphere before sun solve: {pre_metrics}")

    solution = solve_sphere_gain(
        sphere_base, sphere_lobe, sphere_mask,
        mode=args.sphere_solve,
        target_peak_ratio=args.target_peak_ratio,
        direct_highlight_target=args.direct_highlight_target,
        target_mean=args.meter_target,
        albedo=args.albedo,
    )

    # Apply per-channel gains (energy_conservation mode) or scalar (legacy modes)
    gains_pc = solution["gains_per_channel"]
    corrected = apply_sun_gain_per_channel(exposed, hot["mask"], gains_pc,
                                           lobe_neutralise_strength=args.lobe_neutralise)
    save_png_preview(os.path.join(args.debug_dir, "07_corrected_preview.png"), corrected)
    save_png_preview(os.path.join(args.debug_dir, "08_grayball_after_solve.png"),
                     solution["sphere"])

    # ── Validation render: final HDR → grey sphere ─────────────────────────
    # This is the ground truth check. If the pipeline is correct, the sphere
    # mean RGB must equal meter_target on every channel independently.
    log("── Final validation sphere render ──")
    final_verify_rgb = verify_sphere_neutrality(
        corrected, albedo=args.albedo, target=args.meter_target, label="final")
    final_sphere, _ = render_gray_ball_vectorized(
        corrected, albedo=args.albedo, res=args.sphere_res)
    save_png_preview(os.path.join(args.debug_dir, "09_verify_sphere_final.png"),
                     final_sphere)

    meta["sphere"] = {
        "albedo":                args.albedo,
        "sphere_res":            args.sphere_res,
        "sphere_solve":          args.sphere_solve,
        "gray_ball_pre_solve":   pre_metrics,
        "sun_gain_mean":         float(solution["gain"]),
        "sun_gains_per_channel": gains_pc.tolist(),
        "gray_ball_post_solve":  solution["metrics"],
        "peak_ratio":            float(solution["ratio"]),
        "verify_pre_wb_mean_rgb":  pre_verify_rgb.tolist(),
        "verify_final_mean_rgb":   final_verify_rgb.tolist(),
        "verify_final_neutral":    bool(
            abs(float(final_verify_rgb[0]) - float(final_verify_rgb[1])) < 0.01 and
            abs(float(final_verify_rgb[2]) - float(final_verify_rgb[1])) < 0.01
        ),
    }

    # ── 8. Reference sphere calibration ───────────────────────────────────
    sphere_cal_info = {"applied": False}
    if args.ref_sphere:
        log(f"Loading reference sphere image: {args.ref_sphere}")
        ref_img_linear, _ = load_image_any(args.ref_sphere)
        ref_bgr = cv2.cvtColor(
            np.clip(linear_to_srgb(ref_img_linear) * 255, 0, 255).astype(np.uint8),
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
    log("Done.")


if __name__ == "__main__":
    main()
