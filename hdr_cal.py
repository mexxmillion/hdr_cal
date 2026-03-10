#!/usr/bin/env python3
import os
import json
import math
import argparse

import cv2
import numpy as np
import imageio.v2 as imageio

try:
    import pyexr
    HAVE_PYEXR = True
except Exception:
    HAVE_PYEXR = False


# ============================================================
# Logging
# ============================================================

def log(msg):
    print(f"[hdri-cal] {msg}", flush=True)


# ============================================================
# Color / IO
# ============================================================

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
    log(f"Loading image: {path}")

    if ext == ".exr":
        if HAVE_PYEXR:
            img = pyexr.read(path).astype(np.float32)
            log("Loaded EXR with pyexr")
            return img, True
        raise RuntimeError("EXR input requires pyexr. Install with: pip install pyexr")

    if ext == ".hdr":
        img = imageio.imread(path).astype(np.float32)
        log("Loaded HDR with imageio")
        return img, True

    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)

    img = srgb_to_linear(img)
    log("Loaded LDR and converted sRGB -> linear")
    return img.astype(np.float32), False

def save_exr(path, img):
    img = np.asarray(img, dtype=np.float32)
    if not path.lower().endswith(".exr"):
        path += ".exr"

    if not HAVE_PYEXR:
        raise RuntimeError("Writing EXR requires pyexr. Install with: pip install pyexr")

    pyexr.write(path, img)
    log(f"Saved EXR: {path}")

def save_png_preview(path, img_linear, percentile=99.5):
    img_linear = np.clip(img_linear, 0.0, None)
    lum = luminance(img_linear)
    valid = lum[np.isfinite(lum)]
    denom = max(1e-6, np.percentile(valid, percentile)) if valid.size else 1.0
    view = img_linear / denom
    view = linear_to_srgb(np.clip(view, 0.0, 1.0))
    img8 = np.clip(view * 255.0 + 0.5, 0, 255).astype(np.uint8)
    imageio.imwrite(path, img8)
    log(f"Saved preview: {path}")

def save_mask_preview(path, mask):
    m = np.clip(mask, 0.0, 1.0)
    img8 = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)
    imageio.imwrite(path, img8)
    log(f"Saved mask preview: {path}")


# ============================================================
# Parsing helpers
# ============================================================

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


# ============================================================
# Optional WB
# ============================================================

def kelvin_to_rgb_scale(kelvin):
    t = kelvin / 100.0

    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(max(t, 1e-6)) - 161.1195681661
        if t <= 19:
            b = 0.0
        else:
            b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0

    rgb = np.array([r, g, b], dtype=np.float32) / 255.0
    rgb = np.clip(rgb, 1e-4, None)

    scale = 1.0 / rgb
    scale /= np.mean(scale)
    return scale.astype(np.float32)

def apply_white_balance(img, kelvin=None, rgb_scale=None, swatch_xy=None):
    out = img.copy()
    applied = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    if rgb_scale is not None:
        applied = np.array(rgb_scale, dtype=np.float32)
        log(f"Applying user RGB scale: {applied.tolist()}")

    elif kelvin is not None:
        applied = kelvin_to_rgb_scale(kelvin)
        log(f"Applying Kelvin WB: {kelvin}K -> scale {applied.tolist()}")

    elif swatch_xy is not None:
        x, y = swatch_xy
        h, w = img.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        patch = img[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
        mean_rgb = np.mean(patch.reshape(-1, 3), axis=0)
        applied = 1.0 / np.clip(mean_rgb, 1e-6, None)
        applied /= np.mean(applied)
        log(f"Applying swatch WB from ({x},{y}) -> sample {mean_rgb.tolist()} -> scale {applied.tolist()}")
    else:
        log("No white balance override applied")

    out *= applied[None, None, :]
    return out.astype(np.float32), applied


# ============================================================
# LatLong geometry
# ============================================================

def latlong_dirs(h, w):
    ys = (np.arange(h) + 0.5) / h
    xs = (np.arange(w) + 0.5) / w

    theta = ys[:, None] * np.pi
    phi = (xs[None, :] * 2.0 - 1.0) * np.pi

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    x = sin_t * cos_p
    y = np.broadcast_to(cos_t, (h, w))
    z = sin_t * sin_p

    dirs = np.stack([x, y, z], axis=-1).astype(np.float32)

    dtheta = np.pi / h
    dphi = 2.0 * np.pi / w
    dOmega = (sin_t * dtheta * dphi).astype(np.float32)
    dOmega = np.broadcast_to(dOmega, (h, w)).copy()

    return dirs, dOmega

def direction_to_uv(dir3):
    d = np.asarray(dir3, dtype=np.float64)
    d /= max(1e-8, np.linalg.norm(d))
    x, y, z = d
    theta = np.arccos(np.clip(y, -1.0, 1.0))
    phi = np.arctan2(z, x)
    u = (phi / (2.0 * np.pi)) + 0.5
    v = theta / np.pi
    return float(u), float(v), float(theta), float(phi)


# ============================================================
# Metering
# ============================================================

def robust_stat(values, stat="median"):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    if stat == "mean":
        return float(np.mean(values))
    if stat == "median":
        return float(np.median(values))
    raise ValueError(f"Unsupported stat: {stat}")

def meter_image(env_linear, mode="bottom_dome", stat="median", swatch_xy=None, swatch_size=5):
    h, w = env_linear.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    lum = luminance(env_linear)

    out = {
        "mode": mode,
        "stat": stat,
    }

    if mode == "whole_scene":
        vals = lum[np.isfinite(lum)]
        meter_val = robust_stat(vals, stat=stat)
        out["pixel_count"] = int(vals.size)

    elif mode == "bottom_dome":
        mask = dirs[..., 1] < 0.0
        vals = lum[mask]
        meter_val = robust_stat(vals, stat=stat)
        out["pixel_count"] = int(mask.sum())

        weighted_mean = float(np.sum(lum * mask * dOmega) / (np.sum(mask * dOmega) + 1e-8))
        out["weighted_mean"] = weighted_mean

    elif mode == "swatch":
        if swatch_xy is None:
            raise ValueError("--metering-mode swatch requires --swatch x,y")
        x, y = swatch_xy
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        r = max(1, int(swatch_size // 2))
        patch = env_linear[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
        patch_lum = luminance(patch).reshape(-1)
        meter_val = robust_stat(patch_lum, stat=stat)
        out["swatch_xy"] = [x, y]
        out["swatch_size"] = int(swatch_size)
        out["pixel_count"] = int(patch_lum.size)
        out["swatch_mean_rgb"] = np.mean(patch.reshape(-1, 3), axis=0).tolist()

    else:
        raise ValueError(f"Unsupported metering mode: {mode}")

    out["meter_value"] = float(meter_val)
    return out

def solve_exposure_scale(meter_info, target=0.18):
    current = max(1e-6, meter_info["meter_value"])
    scale = target / current
    log(f"Metering mode={meter_info['mode']} stat={meter_info['stat']} value={current:.6f}")
    log(f"Exposure scale solved: {scale:.6f} -> target {target:.6f}")
    return float(scale)


# ============================================================
# Hot lobe / sun mask extraction
# ============================================================

def smoothstep01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def extract_hot_lobe_key(env_linear, threshold=0.1, upper_only=True, blur_px=0):
    """
    threshold in 0..1:
      low = (1-threshold) * hottest
      high = hottest
    Example:
      hottest=10.1, threshold=0.1 => low=9.09, high=10.1
    """
    threshold = float(np.clip(threshold, 1e-6, 1.0))

    h, w = env_linear.shape[:2]
    dirs, _ = latlong_dirs(h, w)
    lum = luminance(env_linear)

    valid = np.ones((h, w), dtype=bool)
    if upper_only:
        valid &= (dirs[..., 1] > 0.0)

    valid_vals = lum[valid]
    hottest = float(np.max(valid_vals)) if valid_vals.size else float(np.max(lum))
    low = (1.0 - threshold) * hottest
    high = hottest

    if high <= low + 1e-8:
        mask = np.zeros_like(lum, dtype=np.float32)
        mask[np.argmax(lum) // w, np.argmax(lum) % w] = 1.0
    else:
        t = (lum - low) / (high - low)
        mask = smoothstep01(t).astype(np.float32)

    if upper_only:
        mask *= valid.astype(np.float32)

    if blur_px > 0:
        k = int(max(1, blur_px))
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    weights = lum * mask
    wsum = float(np.sum(weights)) + 1e-8

    if wsum > 0.0:
        center = np.sum(dirs * weights[..., None], axis=(0, 1)) / wsum
        center /= max(1e-8, np.linalg.norm(center))
        u, v, th, ph = direction_to_uv(center)
    else:
        center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u, v, th, ph = direction_to_uv(center)

    log(f"Hot mask threshold={threshold:.4f}")
    log(f"Hot mask luminance range low={low:.6f}, high={high:.6f}")
    log(f"Hot mask center UV=({u:.4f},{v:.4f}) theta={math.degrees(th):.2f}deg phi={math.degrees(ph):.2f}deg")
    log(f"Hot mask active pixels (>0.01): {int(np.sum(mask > 0.01))}")
    log(f"Hot mask strong pixels (>0.5): {int(np.sum(mask > 0.5))}")

    return {
        "mask": mask.astype(np.float32),
        "center_dir": center.astype(np.float32),
        "center_uv": (u, v),
        "theta_deg": math.degrees(th),
        "phi_deg": math.degrees(ph),
        "hottest": hottest,
        "low": float(low),
        "high": float(high),
        "mask_pixel_count_soft": int(np.sum(mask > 0.01)),
        "mask_pixel_count_strong": int(np.sum(mask > 0.5)),
    }


# ============================================================
# Lambertian gray ball rendering
# ============================================================

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

def render_gray_ball_bruteforce(env_linear, albedo=0.18, res=96):
    """
    Brute-force cosine-weighted integration.
    Slow but simple and trustworthy for MVP.
    """
    h, w = env_linear.shape[:2]
    dirs, dOmega = latlong_dirs(h, w)
    normals, mask = sphere_normals(res)

    out = np.zeros((res, res, 3), dtype=np.float32)

    env_dirs = dirs.reshape(-1, 3)
    env_rgb = env_linear.reshape(-1, 3)
    env_omega = dOmega.reshape(-1)

    valid_idx = np.argwhere(mask)
    log(f"Rendering gray ball brute-force at {res}x{res} ...")

    for idx, (yy, xx) in enumerate(valid_idx):
        n = normals[yy, xx]
        ndotl = np.dot(env_dirs, n)
        ndotl = np.clip(ndotl, 0.0, None)
        irradiance = np.sum(env_rgb * (ndotl * env_omega)[:, None], axis=0)
        out[yy, xx] = (albedo / np.pi) * irradiance

        if idx % 1500 == 0:
            log(f"  sphere sample {idx}/{len(valid_idx)}")

    out[~mask] = 0.0
    return out, mask

def sphere_metrics(sphere_img, mask):
    lum = luminance(sphere_img)
    vals = lum[mask]
    vals = vals[np.isfinite(vals)]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }

def gray_ball_from_split(env_linear, lobe_mask, albedo=0.18, res=96):
    sphere_full, mask = render_gray_ball_bruteforce(env_linear, albedo=albedo, res=res)

    base = env_linear * (1.0 - lobe_mask[..., None])
    lobe = env_linear * lobe_mask[..., None]

    sphere_base, _ = render_gray_ball_bruteforce(base, albedo=albedo, res=res)
    sphere_lobe, _ = render_gray_ball_bruteforce(lobe, albedo=albedo, res=res)

    return sphere_full, sphere_base, sphere_lobe, mask


# ============================================================
# Sphere solve choices
# ============================================================

def solve_sun_gain_fast(sphere_base, sphere_lobe, mask, target_peak_ratio=2.5, max_iters=20):
    gain = 1.0
    best = None

    for i in range(max_iters):
        sphere = sphere_base + gain * sphere_lobe
        met = sphere_metrics(sphere, mask)
        ratio = met["p95"] / max(1e-6, met["mean"])

        log(f"Sun gain iter {i:02d}: gain={gain:.6f}, sphere mean={met['mean']:.6f}, p95={met['p95']:.6f}, ratio={ratio:.6f}")

        best = {
            "gain": float(gain),
            "sphere": sphere.copy(),
            "metrics": met,
            "ratio": float(ratio),
        }

        if abs(ratio - target_peak_ratio) < 0.03:
            break

        if ratio < target_peak_ratio:
            gain *= 1.6
        else:
            gain *= 0.8

    return best

def solve_sun_gain_direct(sphere_base, sphere_lobe, mask, target_highlight_mean=0.32):
    sphere_ref = sphere_base + sphere_lobe
    lum_ref = luminance(sphere_ref)

    thr = np.percentile(lum_ref[mask], 97.0)
    hi_mask = (lum_ref >= thr) & mask

    b = float(np.mean(luminance(sphere_base)[hi_mask]))
    s = float(np.mean(luminance(sphere_lobe)[hi_mask]))

    if s < 1e-8:
        gain = 1.0
    else:
        gain = max(1.0, (target_highlight_mean - b) / s)

    sphere = sphere_base + gain * sphere_lobe
    met = sphere_metrics(sphere, mask)

    log(f"Direct sun solve: base_hi={b:.6f}, lobe_hi={s:.6f}, target_hi={target_highlight_mean:.6f}, gain={gain:.6f}")

    return {
        "gain": float(gain),
        "sphere": sphere,
        "metrics": met,
        "ratio": float(met["p95"] / max(1e-6, met["mean"]))
    }

def solve_sphere_gain(sphere_base, sphere_lobe, mask, mode="direct_highlight",
                      target_peak_ratio=2.5, direct_highlight_target=0.32):
    if mode == "none":
        sphere = sphere_base + sphere_lobe
        met = sphere_metrics(sphere, mask)
        ratio = float(met["p95"] / max(1e-6, met["mean"]))
        log("Sphere solve disabled; using sun gain = 1.0")
        return {
            "gain": 1.0,
            "sphere": sphere,
            "metrics": met,
            "ratio": ratio,
        }

    if mode == "direct_highlight":
        return solve_sun_gain_direct(
            sphere_base,
            sphere_lobe,
            mask,
            target_highlight_mean=direct_highlight_target
        )

    if mode == "iterative_peak_ratio":
        return solve_sun_gain_fast(
            sphere_base,
            sphere_lobe,
            mask,
            target_peak_ratio=target_peak_ratio,
            max_iters=10
        )

    raise ValueError(f"Unsupported sphere solve mode: {mode}")

def apply_sun_gain(env_scaled, lobe_mask, gain):
    base = env_scaled * (1.0 - lobe_mask[..., None])
    sun = env_scaled * lobe_mask[..., None]
    return np.clip(base + sun * gain, 0.0, None).astype(np.float32)


# ============================================================
# Misc
# ============================================================

def maybe_resize(env, max_width=512):
    h, w = env.shape[:2]
    if w <= max_width:
        return env
    scale = max_width / float(w)
    nh = max(8, int(round(h * scale)))
    log(f"Resizing working image from {w}x{h} to {max_width}x{nh}")
    return cv2.resize(env, (max_width, nh), interpolation=cv2.INTER_AREA)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input latlong image (.jpg/.png/.hdr/.exr)")
    ap.add_argument("--out", default="corrected.exr", help="Output EXR")
    ap.add_argument("--debug-dir", default="debug_hdri", help="Debug output directory")

    # White balance
    ap.add_argument("--kelvin", type=float, default=None, help="Optional Kelvin WB")
    ap.add_argument("--rgb-scale", type=str, default=None, help="Optional RGB multiplier, e.g. 1.0,0.97,1.05")
    ap.add_argument("--wb-swatch", type=str, default=None, help="Optional WB neutral swatch pixel, e.g. 123,456")

    # Metering
    ap.add_argument("--metering-mode", type=str, default="bottom_dome",
                    choices=["whole_scene", "bottom_dome", "swatch"],
                    help="Exposure metering choice")
    ap.add_argument("--meter-stat", type=str, default="median",
                    choices=["mean", "median"],
                    help="Meter statistic")
    ap.add_argument("--meter-target", type=float, default=0.18,
                    help="Target metered luminance in linear")
    ap.add_argument("--swatch", type=str, default=None,
                    help="Metering swatch pixel for metering-mode=swatch, e.g. 123,456")
    ap.add_argument("--swatch-size", type=int, default=5,
                    help="Metering swatch size in pixels")

    # Hot lobe extraction
    ap.add_argument("--sun-threshold", type=float, default=0.1,
                    help="0..1 key width from hottest pixel. low=(1-threshold)*hottest")
    ap.add_argument("--sun-upper-only", action="store_true",
                    help="Restrict hot mask extraction to upper hemisphere")
    ap.add_argument("--sun-blur-px", type=int, default=0,
                    help="Optional gaussian blur kernel size for hot mask preview/softening")

    # Sphere / solve
    ap.add_argument("--albedo", type=float, default=0.18, help="Gray ball albedo")
    ap.add_argument("--sphere-res", type=int, default=96, help="Gray ball render resolution")
    ap.add_argument("--sphere-solve", type=str, default="direct_highlight",
                    choices=["direct_highlight", "iterative_peak_ratio", "none"],
                    help="Sphere solve algorithm choice")
    ap.add_argument("--direct-highlight-target", type=float, default=0.32,
                    help="Target bright-side mean for direct_highlight solve")
    ap.add_argument("--target-peak-ratio", type=float, default=2.5,
                    help="Target p95/mean ratio for iterative_peak_ratio solve")

    ap.add_argument("--work-width", type=int, default=512, help="Internal processing width")
    args = ap.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)

    rgb_scale = parse_rgb_scale(args.rgb_scale) if args.rgb_scale else None
    wb_swatch_xy = parse_xy(args.wb_swatch) if args.wb_swatch else None
    meter_swatch_xy = parse_xy(args.swatch) if args.swatch else None

    img, is_hdr = load_image_any(args.input)
    img = np.clip(img, 0.0, None).astype(np.float32)

    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError("Input must be RGB")

    img = maybe_resize(img, max_width=args.work_width)

    wb_img, wb_scale = apply_white_balance(
        img,
        kelvin=args.kelvin,
        rgb_scale=rgb_scale,
        swatch_xy=wb_swatch_xy
    )
    save_png_preview(os.path.join(args.debug_dir, "01_wb_preview.png"), wb_img)

    meter_info = meter_image(
        wb_img,
        mode=args.metering_mode,
        stat=args.meter_stat,
        swatch_xy=meter_swatch_xy,
        swatch_size=args.swatch_size
    )
    exposure_scale = solve_exposure_scale(meter_info, target=args.meter_target)
    exposed = wb_img * exposure_scale
    save_png_preview(os.path.join(args.debug_dir, "02_exposed_preview.png"), exposed)

    hot = extract_hot_lobe_key(
        exposed,
        threshold=args.sun_threshold,
        upper_only=args.sun_upper_only,
        blur_px=args.sun_blur_px
    )
    save_mask_preview(os.path.join(args.debug_dir, "03_hot_mask.png"), hot["mask"])

    sphere_full, sphere_base, sphere_lobe, sphere_mask = gray_ball_from_split(
        exposed,
        hot["mask"],
        albedo=args.albedo,
        res=args.sphere_res
    )

    save_png_preview(os.path.join(args.debug_dir, "04_grayball_full.png"), sphere_full)
    save_png_preview(os.path.join(args.debug_dir, "05_grayball_base.png"), sphere_base)
    save_png_preview(os.path.join(args.debug_dir, "06_grayball_lobe.png"), np.clip(sphere_lobe, 0.0, None))

    sphere_full_metrics = sphere_metrics(sphere_full, sphere_mask)
    log(
        "Gray-ball before sun solve: "
        f"mean={sphere_full_metrics['mean']:.6f}, "
        f"median={sphere_full_metrics['median']:.6f}, "
        f"p95={sphere_full_metrics['p95']:.6f}, "
        f"max={sphere_full_metrics['max']:.6f}"
    )

    sphere_solution = solve_sphere_gain(
        sphere_base,
        sphere_lobe,
        sphere_mask,
        mode=args.sphere_solve,
        target_peak_ratio=args.target_peak_ratio,
        direct_highlight_target=args.direct_highlight_target
    )

    corrected = apply_sun_gain(exposed, hot["mask"], sphere_solution["gain"])

    save_exr(args.out, corrected)
    save_png_preview(os.path.join(args.debug_dir, "07_corrected_preview.png"), corrected)
    save_png_preview(os.path.join(args.debug_dir, "08_grayball_after_solve.png"), sphere_solution["sphere"])

    meta = {
        "input": args.input,
        "is_hdr_like_input": bool(is_hdr),
        "working_resolution": list(img.shape[:2]),

        "white_balance": {
            "wb_scale": wb_scale.tolist(),
            "kelvin": args.kelvin,
            "rgb_scale_arg": rgb_scale,
            "wb_swatch": wb_swatch_xy,
        },

        "metering": meter_info,
        "meter_target": float(args.meter_target),
        "exposure_scale": float(exposure_scale),

        "hot_lobe": {
            "sun_threshold": float(args.sun_threshold),
            "sun_upper_only": bool(args.sun_upper_only),
            "sun_blur_px": int(args.sun_blur_px),
            "center_uv": [float(x) for x in hot["center_uv"]],
            "theta_deg": float(hot["theta_deg"]),
            "phi_deg": float(hot["phi_deg"]),
            "hottest": float(hot["hottest"]),
            "low": float(hot["low"]),
            "high": float(hot["high"]),
            "mask_pixel_count_soft": int(hot["mask_pixel_count_soft"]),
            "mask_pixel_count_strong": int(hot["mask_pixel_count_strong"]),
        },

        "sphere": {
            "albedo": float(args.albedo),
            "sphere_res": int(args.sphere_res),
            "sphere_solve": args.sphere_solve,
            "direct_highlight_target": float(args.direct_highlight_target),
            "target_peak_ratio": float(args.target_peak_ratio),
            "gray_ball_before_solve": sphere_full_metrics,
            "sun_gain": float(sphere_solution["gain"]),
            "gray_ball_after_solve": sphere_solution["metrics"],
            "gray_ball_after_solve_peak_ratio": float(sphere_solution["ratio"]),
        },

        "notes": [
            "Exposure is solved first from the chosen metering mode.",
            "Hot lobe is then keyed from hottest pixel using sun-threshold in 0..1.",
            "Final sun gain is solved on cached Lambertian gray-ball renders.",
            "This is a practical outdoor IBL calibration heuristic, not exact radiometric recovery."
        ]
    }

    meta_path = os.path.join(args.debug_dir, "09_report.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved report: {meta_path}")

    log("Done.")


if __name__ == "__main__":
    main()