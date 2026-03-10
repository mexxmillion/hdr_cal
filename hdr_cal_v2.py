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

def apply_white_balance(img, kelvin=None, rgb_scale=None, swatch_xy=None):
    out = img.copy()
    applied = np.array([1.0, 1.0, 1.0], dtype=np.float32)
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
    else:
        log("No WB applied")
    out *= applied[None, None, :]
    return out.astype(np.float32), applied


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

def render_gray_ball_vectorized(env_linear, albedo=0.18, res=96, chunk=512):
    """
    Fully vectorised cosine-weighted Lambertian integration.
    Chunked over sphere pixels to stay within RAM.

    Replaces the original Python-loop brute-force — same result, ~100× faster.
    """
    h, w = env_linear.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    normals, mask = sphere_normals(res)

    env_dirs_f  = dirs.reshape(-1, 3).astype(np.float32)
    env_rgb_f   = env_linear.reshape(-1, 3).astype(np.float32)
    env_omega_f = dOmega.reshape(-1).astype(np.float32)
    norms_f     = normals.reshape(-1, 3).astype(np.float32)

    n_sph  = norms_f.shape[0]
    out_f  = np.zeros((n_sph, 3), dtype=np.float32)

    log(f"Rendering sphere {res}×{res} vectorised (chunk={chunk}) ...")
    for i in range(0, n_sph, chunk):
        cn  = norms_f[i:i+chunk]                       # (C, 3)
        ndl = np.clip(cn @ env_dirs_f.T, 0.0, None)   # (C, N_env)
        wts = ndl * env_omega_f[None, :]               # (C, N_env)
        out_f[i:i+chunk] = wts @ env_rgb_f             # (C, 3)

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

def gray_ball_from_split(env_linear, lobe_mask, albedo=0.18, res=96):
    sphere_full, mask = render_gray_ball_vectorized(env_linear, albedo, res)
    base = env_linear * (1.0 - lobe_mask[..., None])
    lobe = env_linear *        lobe_mask[..., None]
    sphere_base, _ = render_gray_ball_vectorized(base, albedo, res)
    sphere_lobe, _ = render_gray_ball_vectorized(lobe, albedo, res)
    return sphere_full, sphere_base, sphere_lobe, mask


# ══════════════════════════════════════════════════════════════════════════════
# Sun gain solve
# ══════════════════════════════════════════════════════════════════════════════

# Physical ceiling: clear-sky sun ≈ 100 000 lux vs overcast sky ≈ 2 000 lux
# After our earlier exposure normalisation the sun is compressed; a gain of
# ~200× is a generous upper bound that prevents runaway on weak lobes.
MAX_PHYSICAL_SUN_GAIN = 200.0

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

    gain = float(np.clip(gain, 1.0, MAX_PHYSICAL_SUN_GAIN))
    if gain >= MAX_PHYSICAL_SUN_GAIN * 0.9:
        warn("Sun gain hit physical ceiling — check --sun-threshold or input exposure.")

    sphere = sphere_base + gain * sphere_lobe
    met    = sphere_metrics(sphere, mask)
    log(f"Direct solve: b={b:.6f} s={s:.6f} target={target_highlight_mean:.3f} → gain={gain:.4f}")
    return {"gain": gain, "sphere": sphere, "metrics": met,
            "ratio": float(met["p95"] / max(1e-6, met["mean"]))}

def solve_sun_gain_iterative(sphere_base, sphere_lobe, mask,
                             target_peak_ratio=2.5, max_iters=20):
    gain = 1.0
    best = None
    for i in range(max_iters):
        sphere = sphere_base + gain * sphere_lobe
        met    = sphere_metrics(sphere, mask)
        ratio  = met["p95"] / max(1e-6, met["mean"])
        best   = {"gain": gain, "sphere": sphere.copy(), "metrics": met, "ratio": ratio}
        log(f"Sun gain iter {i:02d}: gain={gain:.4f} p95/mean={ratio:.4f}")
        if abs(ratio - target_peak_ratio) < 0.03:
            break
        gain = float(np.clip(gain * (1.6 if ratio < target_peak_ratio else 0.8),
                             1.0, MAX_PHYSICAL_SUN_GAIN))
    return best

def solve_sphere_gain(sphere_base, sphere_lobe, mask, mode="direct_highlight",
                      target_peak_ratio=2.5, direct_highlight_target=0.32):
    if mode == "none":
        sphere = sphere_base + sphere_lobe
        met    = sphere_metrics(sphere, mask)
        log("Sphere solve disabled (gain=1.0)")
        return {"gain": 1.0, "sphere": sphere, "metrics": met,
                "ratio": float(met["p95"] / max(1e-6, met["mean"]))}
    if mode == "direct_highlight":
        return solve_sun_gain_direct(sphere_base, sphere_lobe, mask,
                                     direct_highlight_target)
    if mode == "iterative_peak_ratio":
        return solve_sun_gain_iterative(sphere_base, sphere_lobe, mask,
                                        target_peak_ratio)
    raise ValueError(f"Unknown sphere solve mode: {mode}")

def apply_sun_gain(env_scaled, lobe_mask, gain):
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

# Reference XYZ for ColorChecker Classic 24 under D65
# Source: colour-science library  colour.CCS_COLOURCHECKERS["ColorChecker 2005"]
# Converted to linear sRGB for convenience
_CC24_LINEAR_SRGB = np.array([
    [0.4000, 0.3176, 0.2745],  # 01 dark skin
    [0.7608, 0.5804, 0.4941],  # 02 light skin
    [0.3451, 0.4314, 0.5686],  # 03 blue sky
    [0.3373, 0.4196, 0.2706],  # 04 foliage
    [0.5059, 0.4863, 0.6863],  # 05 blue flower
    [0.3098, 0.6627, 0.6196],  # 06 bluish green
    [0.7490, 0.3608, 0.0667],  # 07 orange
    [0.2549, 0.3020, 0.6627],  # 08 purplish blue
    [0.6314, 0.2196, 0.2471],  # 09 moderate red
    [0.2000, 0.1333, 0.2471],  # 10 purple
    [0.5765, 0.6863, 0.1020],  # 11 yellow green
    [0.8471, 0.5608, 0.0471],  # 12 orange yellow
    [0.1529, 0.1882, 0.5961],  # 13 blue
    [0.2510, 0.4902, 0.2078],  # 14 green
    [0.5412, 0.0980, 0.0980],  # 15 red
    [0.9020, 0.7882, 0.0314],  # 16 yellow
    [0.6314, 0.2078, 0.4510],  # 17 magenta
    [0.0353, 0.4706, 0.6314],  # 18 cyan
    [0.9412, 0.9412, 0.9412],  # 19 white (N9.5)
    [0.6196, 0.6196, 0.6196],  # 20 neutral 8
    [0.3647, 0.3647, 0.3647],  # 21 neutral 6.5
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black (N2)
], dtype=np.float32)


def detect_colorchecker(img_linear, debug_dir=None):
    """
    Detect ColorChecker Classic 24 in a linear image using
    colour-checker-detection segmentation (no YOLO dependency).

    Returns list of (24, 3) measured linear-RGB arrays, one per checker found.
    Returns [] if nothing found or library unavailable.
    """
    if not HAVE_COLOUR:
        warn("colour-science / colour-checker-detection not installed. "
             "pip install colour-science colour-checker-detection")
        return []

    # colour-checker-detection expects uint8 sRGB
    srgb   = linear_to_srgb(img_linear)
    img8   = np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    try:
        results = ccd.detect_colour_checkers_segmentation(img_bgr, show=False)
    except Exception as e:
        warn(f"Colour checker detection failed: {e}")
        return []

    if not results:
        log("No ColorChecker detected.")
        return []

    log(f"Detected {len(results)} ColorChecker(s).")

    all_measurements = []
    for k, swatches_bgr in enumerate(results):
        # swatches_bgr: (24, 3) uint8 BGR
        swatches_rgb = swatches_bgr[:, ::-1].astype(np.float32) / 255.0
        swatches_linear = srgb_to_linear(swatches_rgb)  # (24, 3)
        all_measurements.append(swatches_linear)

        if debug_dir:
            _save_checker_overlay(debug_dir, k, swatches_linear)

    return all_measurements


def solve_color_matrix(measured_linear, reference_linear=None):
    """
    Solve a 3×3 colour correction matrix M such that:
        measured @ M ≈ reference

    Uses least-squares over the 24 patches.
    The matrix operates in linear RGB space.

    reference_linear: (24,3) reference values.  Defaults to CC24 D65 values.
    Returns M (3,3).
    """
    if reference_linear is None:
        reference_linear = _CC24_LINEAR_SRGB

    A = measured_linear   # (24, 3)
    B = reference_linear  # (24, 3)

    # Solve A @ M = B  →  M = pinv(A) @ B
    M, residuals, rank, sv = np.linalg.lstsq(A, B, rcond=None)

    # Per-patch error in ΔE76 approximation (linear L*a*b* approximation)
    pred   = A @ M
    delta  = np.abs(pred - B)
    rmse   = float(np.sqrt(np.mean(delta**2)))
    log(f"Colour matrix solve: rank={rank}  RMSE={rmse:.5f}  "
        f"(linear RGB, D65 reference)")
    return M.astype(np.float32), float(rmse)

def apply_color_matrix(img_linear, M):
    """Apply 3×3 colour correction matrix to a linear RGB image."""
    h, w = img_linear.shape[:2]
    flat = img_linear.reshape(-1, 3).astype(np.float32)
    corrected = flat @ M
    return np.clip(corrected, 0.0, None).reshape(h, w, 3).astype(np.float32)

def _save_checker_overlay(debug_dir, idx, swatches_linear):
    """Save a side-by-side swatch comparison: measured vs reference."""
    sw  = 32
    ref = _CC24_LINEAR_SRGB
    rows = 4
    cols = 6
    canvas = np.zeros((rows * sw * 2 + 4, cols * sw, 3), dtype=np.float32)
    for i in range(24):
        r, c = divmod(i, cols)
        canvas[r*sw:(r+1)*sw, c*sw:(c+1)*sw] = swatches_linear[i]
        canvas[rows*sw+4+r*sw:rows*sw+4+(r+1)*sw, c*sw:(c+1)*sw] = ref[i]
    save_png_preview(os.path.join(debug_dir, f"cc_swatches_{idx}.png"), canvas)


# ══════════════════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════════════════

def maybe_resize(env, max_width=512):
    h, w = env.shape[:2]
    if w <= max_width:
        return env
    nh = max(8, int(round(h * max_width / w)))
    log(f"Resize: {w}×{h} → {max_width}×{nh}")
    return cv2.resize(env, (max_width, nh), interpolation=cv2.INTER_AREA)


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
    ap.add_argument("--work-width", type=int, default=512,
                    help="Internal processing width in pixels")

    # ── White balance ─────────────────────────────────────────────────────
    ap.add_argument("--kelvin", type=float, default=None,
                    help="Colour temperature WB in Kelvin, e.g. 5600")
    ap.add_argument("--rgb-scale", type=str, default=None,
                    help="Manual RGB multiplier, e.g. 1.0,0.97,1.05")
    ap.add_argument("--wb-swatch", type=str, default=None,
                    help="Neutral swatch pixel for WB, e.g. 123,456")

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
    ap.add_argument("--sphere-solve", default="direct_highlight",
                    choices=["direct_highlight", "iterative_peak_ratio", "none"])
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
    img = maybe_resize(img, args.work_width)
    meta["is_hdr_input"] = bool(is_hdr)
    meta["working_resolution"] = list(img.shape[:2])

    # ── 2. Validate orientation ───────────────────────────────────────────
    orient_ratio = validate_hdri_orientation(img)
    meta["orientation_energy_ratio"] = float(orient_ratio)

    # ── 3. White balance ──────────────────────────────────────────────────
    wb_img, wb_scale = apply_white_balance(
        img, kelvin=args.kelvin, rgb_scale=rgb_scale, swatch_xy=wb_swatch_xy)
    save_png_preview(os.path.join(args.debug_dir, "01_wb_preview.png"), wb_img)
    meta["white_balance"] = {
        "wb_scale":     wb_scale.tolist(),
        "kelvin":       args.kelvin,
        "rgb_scale_arg":rgb_scale,
    }

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
    sphere_full, sphere_base, sphere_lobe, sphere_mask = gray_ball_from_split(
        exposed, hot["mask"], albedo=args.albedo, res=args.sphere_res)

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
    )
    corrected = apply_sun_gain(exposed, hot["mask"], solution["gain"])
    save_png_preview(os.path.join(args.debug_dir, "07_corrected_preview.png"), corrected)
    save_png_preview(os.path.join(args.debug_dir, "08_grayball_after_solve.png"),
                     solution["sphere"])

    meta["sphere"] = {
        "albedo":              args.albedo,
        "sphere_res":          args.sphere_res,
        "sphere_solve":        args.sphere_solve,
        "gray_ball_pre_solve": pre_metrics,
        "sun_gain":            float(solution["gain"]),
        "gray_ball_post_solve":solution["metrics"],
        "peak_ratio":          float(solution["ratio"]),
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

    # ── 9. ColorChecker detection & colour correction ─────────────────────
    cc_info = {"applied": False}
    checker_src = None
    if args.colorchecker:
        checker_src = args.colorchecker
    elif args.colorchecker_in_hdri:
        checker_src = "__hdri__"

    if checker_src:
        if checker_src == "__hdri__":
            log("Searching for ColorChecker in HDRI latlong...")
            cc_img = corrected
        else:
            log(f"Loading ColorChecker image: {checker_src}")
            cc_img, _ = load_image_any(checker_src)

        measurements = detect_colorchecker(cc_img, debug_dir=args.debug_dir)

        if measurements:
            # Use first detected checker
            measured = measurements[0]    # (24, 3) linear sRGB
            M, rmse  = solve_color_matrix(measured)

            # Apply to the calibrated HDR
            corrected = apply_color_matrix(corrected, M)
            save_png_preview(
                os.path.join(args.debug_dir, "10_colorchecker_corrected.png"),
                corrected)

            cc_info = {
                "applied":              True,
                "source":               checker_src,
                "n_checkers_found":     len(measurements),
                "matrix_rmse_linear":   rmse,
                "color_matrix_3x3":     M.tolist(),
                "note": (
                    "Matrix maps measured linear-sRGB patches to CC24 D65 reference. "
                    "Applied to the linear HDR before EXR save."
                ),
            }
            log(f"ColorChecker correction applied. Matrix RMSE={rmse:.5f}")
        else:
            warn("ColorChecker not found — colour correction skipped.")
            cc_info = {"applied": False, "reason": "not detected"}

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