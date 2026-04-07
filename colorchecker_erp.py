"""
colorchecker_erp.py — ColorChecker detection inside equirectangular HDR maps

The problem with detecting a ColorChecker in an ERP image:
  - The checker has straight edges and right-angle corners
  - In ERP, straight lines become sinusoidal curves
  - Contour/rectangle detectors (including colour-checker-detection) will miss it
    or produce completely wrong swatch polygons

The correct approach:
  1. Gnomonic (rectilinear) projection of overlapping tiles covering the panorama
     This is the ONLY projection that preserves straight lines — it is exactly
     what a perspective pinhole camera produces.
  2. Run colour-checker-detection on each rectilinear tile
  3. Back-project detected checker corner pixels through the gnomonic inverse
     to get ERP (u,v) coordinates
  4. Sample the original full-resolution linear HDR at those ERP coordinates
     for accurate swatch colours (not the downsampled tile)
  5. Optional: solvePnP pose estimation to recover the checker's 3D orientation
     in camera space — useful for knowing which hemisphere of the HDRI it faces

Usage (standalone):
  from colorchecker_erp import find_colorchecker_in_erp

  swatches_linear, pose_info = find_colorchecker_in_erp(
      erp_linear,          # (H, W, 3) float32 linear HDR
      debug_dir="debug",   # optional
  )
  # swatches_linear: (24, 3) float32 linear RGB, CC24 patch order
  # pose_info: dict with checker normal in spherical/cartesian coords
"""

import os
import math
import itertools
import shutil
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

# Point colour-checker-detection at bundled model so it never downloads
_BUNDLED_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if os.path.isdir(_BUNDLED_MODELS):
    os.environ.setdefault(
        "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY", _BUNDLED_MODELS
    )

try:
    import colour_checker_detection as ccd
    HAVE_CCD = True
except ImportError:
    HAVE_CCD = False

# Check if the ML inference method is available (requires ultralytics)
HAVE_CCD_INFERENCE = False
_YOLO_MODEL_PATH = None
if HAVE_CCD:
    try:
        _ = ccd.detect_colour_checkers_inference
        HAVE_CCD_INFERENCE = True
        # Resolve where the model file lives (or would be downloaded to)
        _model_repo = os.environ.get(
            "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY",
            os.path.join(os.path.expanduser("~"), ".colour-science", "colour-checker-detection"),
        )
        _candidate = os.path.join(_model_repo, "colour-checker-detection-l-seg.pt")
        _YOLO_MODEL_PATH = _candidate if os.path.isfile(_candidate) else None
    except AttributeError:
        pass

def _log_detection_backends():
    """Log which chart detection backends are available. Call once at detection time."""
    print(f"[cc-erp] Detection backends:")
    print(f"[cc-erp]   colour-checker-detection : {'yes' if HAVE_CCD else 'NO — segmentation unavailable'}")
    print(f"[cc-erp]   YOLO inference           : {'yes' if HAVE_CCD_INFERENCE else 'no (ultralytics not installed)'}")
    if HAVE_CCD_INFERENCE:
        if _YOLO_MODEL_PATH:
            print(f"[cc-erp]   YOLO model               : {_YOLO_MODEL_PATH}")
        else:
            print(f"[cc-erp]   YOLO model               : NOT FOUND — will attempt download at runtime")

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False


# ─── Colorspace conversions (via colour-science) ────────────────────────────

def srgb_linear_to_acescg(img: np.ndarray) -> np.ndarray:
    """Convert linear sRGB (D65) → ACEScg (AP1/D60) with proper chromatic adaptation."""
    if not HAVE_COLOUR:
        raise RuntimeError("colour-science is required for colorspace conversion")
    return colour.RGB_to_RGB(
        np.asarray(img, dtype=np.float32),
        "sRGB", "ACEScg",
        apply_cctf_decoding=False,
        apply_cctf_encoding=False,
    ).astype(np.float32)


def acescg_to_srgb_linear(img: np.ndarray) -> np.ndarray:
    """Convert ACEScg (AP1/D60) → linear sRGB (D65) with proper chromatic adaptation."""
    if not HAVE_COLOUR:
        raise RuntimeError("colour-science is required for colorspace conversion")
    return colour.RGB_to_RGB(
        np.asarray(img, dtype=np.float32),
        "ACEScg", "sRGB",
        apply_cctf_decoding=False,
        apply_cctf_encoding=False,
    ).astype(np.float32)


# ─── CC24 reference values ────────────────────────────────────────────────────
# Source: Macbeth ColorChecker Classic spectral data, rendered to D65.
# These are LINEAR (scene-referred) values in sRGB / Rec.709 primaries.
# Patch order: row-major, top-left = patch 1 (dark skin).
CC24_LINEAR_SRGB = np.array([
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
    [0.9412, 0.9412, 0.9412],  # 19 white  N9.5
    [0.6196, 0.6196, 0.6196],  # 20 neutral 8
    [0.3647, 0.3647, 0.3647],  # 21 neutral 6.5
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5  ← WB reference, ~18% grey
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black   N2
], dtype=np.float32)

# ACEScg (AP1) reference values.
# IMPORTANT: The CC24 sRGB values are spectral reflectances rendered under D65
# into sRGB primaries. You cannot get correct ACEScg values by applying the
# sRGB->ACEScg matrix — that matrix encodes a D65->D60 white point adaptation
# which breaks the neutral patches (makes R≠G≠B for a spectrally flat grey).
#
# Correct approach:
#   - Neutral patches (19-24): R=G=B in ANY colorspace — a flat reflector is
#     achromatic regardless of primaries. Keep identical to sRGB values.
#   - Chromatic patches: ideally re-rendered from spectral data into AP1.
#     As an approximation we use the sRGB values — the gamut difference between
#     sRGB and AP1 is modest for the CC24 patch colours, and WB is derived from
#     the neutral patches anyway.
#
# For the 3×3 colour matrix solve the chromatic error is acceptable because
# you're solving for the best-fit matrix, not comparing absolute values.
CC24_LINEAR_ACESCG = CC24_LINEAR_SRGB.copy()
# Neutrals are already correct (equal R=G=B). No transform needed.

# Alias — used internally, always points at sRGB values (correct for all colorspaces)
CC24_LINEAR = CC24_LINEAR_SRGB


def get_cc24_reference(colorspace: str = "acescg") -> np.ndarray:
    """
    Return the CC24 reference array for the requested colorspace.

    For WB purposes (patch 22, neutral grey) the result is identical in all
    colorspaces — a spectrally flat reflector has equal R=G=B regardless of
    primaries. The colorspace distinction only matters for the full 3×3 colour
    matrix solve on chromatic patches.

    colorspace: "acescg" (default for EXR) or "srgb" (for LDR inputs)
    """
    cs = colorspace.lower().replace("-", "").replace("_", "")
    if cs in ("acescg", "aces", "ap1", "srgb", "rec709", "linear"):
        return CC24_LINEAR_SRGB
    raise ValueError(f"Unknown colorspace '{colorspace}'. Use 'acescg' or 'srgb'.")


# Physical CC24 patch layout in mm (4 cols × 6 rows, 24×24 mm patches, 6 mm gap)
# Origin at top-left patch centre.  X = right, Y = down.
_PATCH_W = 24.0
_PATCH_H = 24.0
_GAP     = 6.0
_STEP_X  = _PATCH_W + _GAP
_STEP_Y  = _PATCH_H + _GAP

CC24_3D_POINTS = np.array([
    [c * _STEP_X, r * _STEP_Y, 0.0]
    for r in range(4) for c in range(6)
], dtype=np.float32)  # (24, 3)  in mm, Z=0 plane


# ─── Gnomonic (rectilinear) projection ───────────────────────────────────────

def erp_to_rectilinear(erp_img: np.ndarray,
                        yaw_deg: float, pitch_deg: float,
                        fov_deg: float = 70.0,
                        out_w: int = 800, out_h: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a rectilinear (perspective) view from an ERP panorama.

    Uses gnomonic projection: the only projection that maps great-circle arcs
    (= straight 3D lines) to straight 2D lines.  This is identical to what a
    pinhole camera produces, so rectangle/contour detectors work correctly.

    Args:
        erp_img   : (H, W, C) equirectangular image (any dtype)
        yaw_deg   : horizontal look direction, degrees. 0=forward(φ=0), +90=right
        pitch_deg : vertical look direction, degrees. 0=horizon, +90=up, -90=down
        fov_deg   : horizontal field of view of the output window
        out_w/out_h: output resolution

    Returns:
        rectilinear  : (out_h, out_w, C) projected image
        map_uv       : (out_h, out_w, 2) float32 — ERP (u,v) ∈ [0,1] for each
                       output pixel.  Use this to back-project detections.
    """
    h_erp, w_erp = erp_img.shape[:2]

    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Half-FOV → focal length in pixels
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    # Build pixel grid for output image
    # px,py are offsets from principal point (centre of output)
    py_grid, px_grid = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    px = px_grid - (out_w - 1) / 2.0
    py = py_grid - (out_h - 1) / 2.0   # y-down in image, y-up in world

    # Ray direction in camera space (z forward, x right, y up)
    ray_cam = np.stack([px / f,           # x
                        -py / f,          # y (image y-down → world y-up)
                        np.ones_like(px)  # z (forward)
                        ], axis=-1)  # (H_out, W_out, 3)

    # Rotate ray by yaw (around world-Y) then pitch (around world-X)
    # Yaw: rotates left/right in the panorama
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [ 0,             1, 0            ],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    # Pitch: tilts up/down
    Rx = np.array([[1, 0,               0             ],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]], dtype=np.float32)

    R = Ry @ Rx   # combined rotation: first apply pitch, then yaw

    # Apply rotation to each ray
    rays_flat = ray_cam.reshape(-1, 3) @ R.T   # (N, 3)
    rays_flat /= (np.linalg.norm(rays_flat, axis=-1, keepdims=True) + 1e-8)
    x, y, z = rays_flat[:,0], rays_flat[:,1], rays_flat[:,2]

    # Ray → ERP (u, v)
    phi   = np.arctan2(x, z)                        # azimuth  [-π, π]
    theta = np.arccos(np.clip(y, -1.0, 1.0))        # polar    [0, π]
    u = (phi   / (2.0 * np.pi) + 0.5)               # [0, 1]
    v = theta  / np.pi                               # [0, 1]

    map_uv = np.stack([u, v], axis=-1).reshape(out_h, out_w, 2).astype(np.float32)

    # Sample ERP image (bilinear, border-wrap on u for 360° continuity)
    map_x = (u * (w_erp - 1)).reshape(out_h, out_w).astype(np.float32)
    map_y = (v * (h_erp - 1)).reshape(out_h, out_w).astype(np.float32)
    map_x = np.mod(map_x, w_erp)   # wrap azimuth

    rectilinear = cv2.remap(erp_img, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
    return rectilinear, map_uv


def backproject_pixel_to_erp(px_out: float, py_out: float,
                              map_uv: np.ndarray) -> Tuple[float, float]:
    """
    Given a pixel (px_out, py_out) in a rectilinear tile, return its
    ERP (u, v) coordinate via bilinear interpolation of map_uv.
    """
    out_h, out_w = map_uv.shape[:2]
    px_c = float(np.clip(px_out, 0, out_w - 1))
    py_c = float(np.clip(py_out, 0, out_h - 1))

    x0, y0 = int(px_c), int(py_c)
    x1 = min(x0 + 1, out_w - 1)
    y1 = min(y0 + 1, out_h - 1)
    fx = px_c - x0
    fy = py_c - y0

    uv = (map_uv[y0, x0] * (1-fx)*(1-fy) +
          map_uv[y0, x1] * fx*(1-fy)     +
          map_uv[y1, x0] * (1-fx)*fy     +
          map_uv[y1, x1] * fx*fy)
    return float(uv[0]), float(uv[1])


def sample_erp_bilinear(erp: np.ndarray, u: float, v: float) -> np.ndarray:
    """Sample ERP image at fractional (u,v) ∈ [0,1] with azimuth wrap."""
    h, w = erp.shape[:2]
    px = (u % 1.0) * (w - 1)
    py = np.clip(v, 0.0, 1.0) * (h - 1)
    x0, y0 = int(px), int(py)
    x1 = (x0 + 1) % w
    y1 = min(y0 + 1, h - 1)
    fx = px - x0
    fy = py - y0
    return (erp[y0, x0] * (1-fx)*(1-fy) +
            erp[y0, x1] * fx*(1-fy)     +
            erp[y1, x0] * (1-fx)*fy     +
            erp[y1, x1] * fx*fy)


# ─── Tile sweep ───────────────────────────────────────────────────────────────

def _build_sweep_tiles(yaw_step: float = 40.0,
                       pitch_values: tuple = (-45.0, -20.0, 0.0, 20.0, 45.0),
                       fov_deg: float = 70.0
                       ) -> List[Tuple[float, float]]:
    """
    Generate (yaw, pitch) pairs covering the horizon band of the panorama.

    A ColorChecker on set is always near the horizon — on a tripod, on the
    floor, or held at chest height. ±45° around the horizon covers all
    realistic placements. The sky extremes (>60° elevation) are excluded.

    pitch_values: elevation angles in degrees (0=horizon, + = above, - = below).
    """
    yaws = np.arange(0, 360, yaw_step).tolist()
    return [(y, p) for y in yaws for p in pitch_values]


@dataclass
class CheckerDetection:
    """One detected ColorChecker instance."""
    swatches_linear: np.ndarray          # (24, 3) float32, ERP-sampled
    swatch_centres_uv: np.ndarray        # (24, 2) float32, ERP coords [0,1]
    swatch_centres_tile: np.ndarray      # (24, 2) float32, tile pixel coords
    quad_tile: np.ndarray                # (4, 2) float32, tile pixel coords
    quad_center_uv: np.ndarray           # (2,) float32, ERP quad centre
    tile_yaw: float
    tile_pitch: float
    tile_fov_deg: float
    checker_normal_world: np.ndarray     # (3,) unit vector, checker face direction
    checker_normal_theta_deg: float      # polar angle of checker face
    checker_normal_phi_deg: float        # azimuth of checker face
    confidence: float                    # 0–1, based on patch colour error
    raw_swatches_bgr: np.ndarray         # (24, 3) uint8, from detector
    detection_method: str = "unknown"
    stage_label: str = "coarse"
    crop_bounds: Optional[Tuple[int, int, int, int]] = None


def _linear_to_u8_for_detection(img_linear: np.ndarray) -> np.ndarray:
    """Convert linear ACEScg HDR to uint8 sRGB for detection/debug display."""
    img_linear = np.clip(np.asarray(img_linear, dtype=np.float32), 0.0, None)
    if HAVE_COLOUR:
        img_disp = np.clip(acescg_to_srgb_linear(img_linear), 0.0, None)
    else:
        img_disp = img_linear
    lum = 0.2126 * img_disp[..., 0] + 0.7152 * img_disp[..., 1] + 0.0722 * img_disp[..., 2]
    valid = lum[lum > 0]
    if valid.size:
        scale = 1.0 / max(float(np.percentile(valid, 99)), 1e-6)
    else:
        scale = 1.0
    disp = np.clip(img_disp * scale, 0.0, None)
    srgb = np.where(disp <= 0.0031308,
                    disp * 12.92,
                    1.055 * np.power(np.clip(disp, 1e-9, None), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)


def _srgb_to_linear_arr(v: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear [0,1], works on any shape."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045,
                    v / 12.92,
                    ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)




def _linear_to_srgb_display(img_linear: np.ndarray) -> np.ndarray:
    """Apply sRGB EOTF inverse (linear → display gamma). No clamping of input."""
    x = np.clip(img_linear, 0.0, None)
    return np.where(x <= 0.0031308,
                    x * 12.92,
                    1.055 * np.power(np.clip(x, 1e-9, None), 1.0 / 2.4) - 0.055
                    ).astype(np.float32)


def _compute_detection_prebalance(img_linear: np.ndarray,
                                   target_mid: float = 0.18,
                                   ) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Per-tile exposure and WB normalization for chart detection.

    All metering and scaling happens in linear space (physically correct).
    The output is converted to sRGB display space because the detection
    model was trained on normal sRGB photos.

    In sRGB display space, 0.18 linear maps to ~0.46 — comfortably in
    mid-range for the detector. The gamma curve naturally compresses
    highlights (bright ground) while expanding shadows, which is exactly
    what a camera does.

    The rgb_scale and exposure_scale are tracked so swatch values can be
    un-scaled after detection to recover the original linear values.
    """
    img = np.clip(np.asarray(img_linear, dtype=np.float32), 0.0, None)
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    valid = lum[np.isfinite(lum) & (lum > 1e-6)]
    if valid.size < 64:
        rgb_scale = np.ones(3, dtype=np.float32)
        exposure_scale = 1.0
        return _linear_to_srgb_display(img), rgb_scale, exposure_scale, {
            "rgb_scale": rgb_scale.tolist(), "exposure_scale": exposure_scale}

    # WB: neutralise colour cast using trimmed mean (10th-90th percentile)
    lo = float(np.percentile(valid, 10.0))
    hi = float(np.percentile(valid, 90.0))
    mask = np.isfinite(lum) & (lum >= lo) & (lum <= hi)
    if np.count_nonzero(mask) < 64:
        mask = np.isfinite(lum) & (lum > 1e-6)

    samples = img[mask]
    chan_mean = np.mean(samples, axis=0)
    grey = float(np.mean(chan_mean))
    rgb_scale = grey / np.clip(chan_mean, 1e-6, None)
    rgb_scale = np.clip(rgb_scale, 0.4, 2.5).astype(np.float32)
    rgb_scale /= max(float(np.mean(rgb_scale)), 1e-6)

    # Exposure: use the mean of the mid-range pixels (p10-p90).
    # This excludes the sun disc and deep shadows — the part of
    # the tile where the chart would be. Scale that to target_mid.
    mid_lum = float(np.mean(lum[mask]))
    exposure_scale = float(np.clip(target_mid / max(mid_lum, 1e-6), 0.001, 100.0))

    # Scale in linear, then convert to sRGB display space.
    # The sRGB gamma curve compresses highlights naturally (no clamping needed)
    # and the detector sees an image that looks like a normal camera photo.
    balanced_linear = np.clip(img * rgb_scale[None, None, :] * exposure_scale, 0.0, None)
    balanced_display = _linear_to_srgb_display(balanced_linear)

    info = {
        "rgb_scale": rgb_scale.tolist(),
        "exposure_scale": exposure_scale,
        "lum_mid_mean": mid_lum,
        "lum_p10": lo,
        "lum_p90": hi,
        "target_mid": target_mid,
    }
    return balanced_display, rgb_scale, exposure_scale, info


def _get_detector_target_width() -> int:
    """
    Best-effort query of the detector's working width.

    The library reformats the input image before returning the detected
    quadrilateral. We need to invert that resize correctly, otherwise the
    quad/crop/debug overlays drift. Fall back to 1440, matching the current
    library defaults for the classic checker detector.
    """
    if not HAVE_CCD:
        return 1440

    settings = getattr(ccd, "SETTINGS_DETECTION_COLORCHECKER_CLASSIC", None)
    if settings is None:
        return 1440

    for attr in ("target_width", "working_width"):
        value = getattr(settings, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return int(round(value))

    return 1440


def _detector_reformatted_size(src_w: int, src_h: int) -> Tuple[float, float]:
    """
    Return the detector's reformatted image size for the given input image.

    `colour-checker-detection` rescales the input to a fixed target width while
    preserving aspect ratio. Its reported quadrilateral coordinates are in that
    reformatted image space.
    """
    target_w = float(_get_detector_target_width())
    if src_w <= 0 or src_h <= 0:
        return target_w, target_w

    scale = target_w / float(src_w)
    target_h = max(1.0, float(src_h) * scale)
    return target_w, target_h


def _quad_detector_to_pixels(quadrilateral: np.ndarray,
                             src_w: int,
                             src_h: int) -> np.ndarray:
    """Map detector quadrilateral coordinates back into source-image pixels."""
    quad = np.array(quadrilateral, dtype=np.float32)
    det_w, det_h = _detector_reformatted_size(src_w, src_h)
    scale_x = float(src_w) / max(det_w, 1e-6)
    scale_y = float(src_h) / max(det_h, 1e-6)
    quad_px = quad.copy()
    quad_px[:, 0] *= scale_x
    quad_px[:, 1] *= scale_y
    return quad_px


def _quad_bounds_with_padding(quad_px: np.ndarray,
                              img_w: int,
                              img_h: int,
                              pad_frac: float = 0.35) -> Optional[Tuple[int, int, int, int]]:
    """Compute a padded crop around a quadrilateral in image pixel coords."""
    if quad_px.size == 0:
        return None

    qx0 = float(np.min(quad_px[:, 0]))
    qx1 = float(np.max(quad_px[:, 0]))
    qy0 = float(np.min(quad_px[:, 1]))
    qy1 = float(np.max(quad_px[:, 1]))
    qw = max(1.0, qx1 - qx0)
    qh = max(1.0, qy1 - qy0)

    pad_x = qw * pad_frac
    pad_y = qh * pad_frac

    cx0 = int(np.floor(max(0.0, qx0 - pad_x)))
    cx1 = int(np.ceil(min(float(img_w), qx1 + pad_x)))
    cy0 = int(np.floor(max(0.0, qy0 - pad_y)))
    cy1 = int(np.ceil(min(float(img_h), qy1 + pad_y)))

    if (cx1 - cx0) < 8 or (cy1 - cy0) < 8:
        return None

    return cx0, cy0, cx1, cy1


def _save_intermediate_debug(debug_dir: Optional[str],
                             basename: str,
                             image_rgb: np.ndarray,
                             quad: Optional[np.ndarray] = None,
                             crop_bounds: Optional[Tuple[int, int, int, int]] = None,
                             centres: Optional[np.ndarray] = None,
                             note: Optional[str] = None) -> None:
    """Save additive debug images without disturbing the GUI's expected files."""
    if not debug_dir:
        return

    vis = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    if crop_bounds is not None:
        x0, y0, x1, y1 = crop_bounds
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 2)

    if quad is not None and len(quad) == 4:
        cv2.polylines(vis, [np.round(quad).astype(np.int32)], True, (0, 200, 255), 2)

    if centres is not None:
        for i, (cx, cy) in enumerate(np.asarray(centres, dtype=np.float32)):
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0, 0, 0), 1)
            cv2.putText(vis, str(i + 1), (int(round(cx)) + 4, int(round(cy)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

    if note:
        cv2.putText(vis, note, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(debug_dir, basename), vis)




def _order_quad_tl_tr_br_bl(poly: np.ndarray) -> np.ndarray:
    """Return quadrilateral points ordered TL, TR, BR, BL."""
    q = np.array(poly, dtype=np.float32).reshape(4, 2)
    s = q.sum(axis=1)
    d = q[:, 0] - q[:, 1]
    tl = q[np.argmin(s)]
    br = q[np.argmax(s)]
    tr = q[np.argmin(d)]
    bl = q[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _sample_tile_bilinear(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear sample from a float RGB tile in pixel coordinates."""
    h, w = img.shape[:2]
    x = float(np.clip(x, 0.0, max(w - 1, 0)))
    y = float(np.clip(y, 0.0, max(h - 1, 0)))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    top = (1.0 - fx) * img[y0, x0] + fx * img[y0, x1]
    bot = (1.0 - fx) * img[y1, x0] + fx * img[y1, x1]
    return ((1.0 - fy) * top + fy * bot).astype(np.float32)


def _sample_swatches_from_quad(tile_linear: np.ndarray,
                               quad: np.ndarray,
                               sample_frac: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    """Sample CC24 swatches directly from a quadrilateral in tile space."""
    quad = _order_quad_tl_tr_br_bl(quad)
    tl, tr, br, bl = quad

    centres = []
    colours = []
    rows, cols = 4, 6
    for r in range(rows):
        for c in range(cols):
            s = (c + 0.5) / cols
            t = (r + 0.5) / rows
            top = tl + s * (tr - tl)
            bottom = bl + s * (br - bl)
            pt = top + t * (bottom - top)

            left = tl + t * (bl - tl)
            right = tr + t * (br - tr)
            cell_w = (right - left) / cols
            top_col = tl + s * (tr - tl)
            bottom_col = bl + s * (br - bl)
            cell_h = (bottom_col - top_col) / rows

            half_w = np.linalg.norm(cell_w) * sample_frac
            half_h = np.linalg.norm(cell_h) * sample_frac
            ux = cell_w / max(np.linalg.norm(cell_w), 1e-8)
            uy = cell_h / max(np.linalg.norm(cell_h), 1e-8)

            samples = []
            for oy in np.linspace(-half_h, half_h, 5):
                for ox in np.linspace(-half_w, half_w, 5):
                    p = pt + ux * ox + uy * oy
                    samples.append(_sample_tile_bilinear(tile_linear, float(p[0]), float(p[1])))

            colours.append(np.mean(np.asarray(samples, dtype=np.float32), axis=0))
            centres.append((float(pt[0]), float(pt[1])))

    return np.asarray(colours, dtype=np.float32), np.asarray(centres, dtype=np.float32)


def _compute_detection_confidence(swatches_linear: np.ndarray,
                                  cc24_ref: Optional[np.ndarray] = None) -> float:
    """Score a swatch read using the neutral ramp agreement."""
    if swatches_linear.shape != (24, 3):
        return 0.0
    ref = cc24_ref if cc24_ref is not None else CC24_LINEAR_SRGB
    neutrals_meas = swatches_linear[18:24]
    neutrals_ref = ref[18:24]
    lum_meas = 0.2126 * neutrals_meas[:, 0] + 0.7152 * neutrals_meas[:, 1] + 0.0722 * neutrals_meas[:, 2]
    lum_ref = 0.2126 * neutrals_ref[:, 0] + 0.7152 * neutrals_ref[:, 1] + 0.0722 * neutrals_ref[:, 2]
    scale_n = np.clip(lum_ref / (lum_meas + 1e-8), 0.0, 20.0)[:, None]
    chroma_err = float(np.mean(np.abs(neutrals_meas * scale_n - neutrals_ref)))
    return float(np.clip(1.0 - chroma_err * 5.0, 0.0, 1.0))


def _quad_agreement(quad_a: np.ndarray,
                    quad_b: np.ndarray,
                    img_w: int,
                    img_h: int) -> float:
    """Estimate how much two quads agree geometrically, 0..1."""
    qa = _order_quad_tl_tr_br_bl(np.asarray(quad_a, dtype=np.float32))
    qb = _order_quad_tl_tr_br_bl(np.asarray(quad_b, dtype=np.float32))
    diag = math.hypot(float(img_w), float(img_h))
    mean_dist = float(np.mean(np.linalg.norm(qa - qb, axis=1))) / max(diag, 1e-6)
    agreement = 1.0 - mean_dist * 6.0
    return float(np.clip(agreement, 0.0, 1.0))


def _save_backend_debug(debug_dir,
                        basename: str,
                        tile_rgb_u8: np.ndarray,
                        quad: Optional[np.ndarray],
                        centres: Optional[np.ndarray],
                        note: str):
    """Write a backend-specific overlay image for detector comparison."""
    if not debug_dir:
        return
    vis = cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2BGR)
    if quad is not None:
        cv2.polylines(vis, [np.round(np.asarray(quad)).astype(np.int32)], True, (0, 200, 255), 2)
    if centres is not None:
        for i, (cx, cy) in enumerate(np.asarray(centres, dtype=np.float32)):
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0, 0, 0), 1)
            cv2.putText(vis, str(i + 1), (int(round(cx)) + 4, int(round(cy)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
    cv2.putText(vis, note, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, basename), vis)


def _save_rectified_debug(debug_dir,
                          basename: str,
                          cc_img: np.ndarray,
                          centres_xy: np.ndarray,
                          note: str):
    """Save one rectified checker image with its own swatch centres."""
    if not debug_dir:
        return
    vis = cv2.cvtColor(np.clip(cc_img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.polylines(vis, [quad], True, (255, 128, 0), 2)
    for i, (cx, cy) in enumerate(np.asarray(centres_xy, dtype=np.float32)):
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 255, 0), -1)
        cv2.putText(vis, str(i + 1), (int(round(cx)) + 3, int(round(cy)) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 255, 255), 1)
    cv2.putText(vis, note, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, basename), vis)


def _save_debug_swatch_strip(debug_dir,
                             basename: str,
                             swatches: np.ndarray,
                             note: str):
    """Save a simple 24-patch swatch strip for backend/pass comparison."""
    if not debug_dir or swatches is None:
        return
    sw = np.asarray(swatches, dtype=np.float32)
    if sw.shape != (24, 3):
        return
    patch_w = 40
    patch_h = 40
    label_h = 18
    strip = np.zeros((patch_h + label_h, patch_w * 24, 3), dtype=np.uint8)
    sw_u8 = np.clip(sw * 255, 0, 255).astype(np.uint8)
    for i in range(24):
        x0 = i * patch_w
        strip[label_h:, x0:x0 + patch_w] = sw_u8[i][None, None, ::-1]
        cv2.putText(strip, str(i + 1), (x0 + 10, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(strip, note, (6, strip.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
    cv2.imwrite(os.path.join(debug_dir, basename), strip)


def _map_rectified_points_to_tile(points_xy: np.ndarray,
                                  H_cc2tile: Optional[np.ndarray],
                                  out_w: int,
                                  out_h: int) -> np.ndarray:
    """Map points from rectified checker image space back into tile pixels."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if H_cc2tile is None or pts.size == 0:
        return pts.copy()
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = (H_cc2tile @ pts_h.T).T
    mapped = mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-8, None)
    mapped[:, 0] = np.clip(mapped[:, 0], 0, out_w - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, out_h - 1)
    return mapped.astype(np.float32)

def _detect_in_tile(tile_linear: np.ndarray,
                    map_uv: np.ndarray,
                    erp_linear_hd: np.ndarray,
                    yaw: float, pitch: float,
                    tile_fov_deg: float,
                    debug_dir,
                    tile_idx: int,
                    stage_label: str,
                    cc24_ref: Optional[np.ndarray] = None,
                    read_backend: str = "colour",
                    compare_backends: bool = True,
                    global_rgb_scale: Optional[np.ndarray] = None,
                    global_exp_scale: float = 1.0):
    """Detect a ColorChecker in one rectilinear tile."""
    if not HAVE_CCD:
        return None

    out_h, out_w = tile_linear.shape[:2]
    tile_rgb_u8 = _linear_to_u8_for_detection(tile_linear)
    tile_raw_tm = (tile_linear / (tile_linear + 1.0)).astype(np.float32)

    # Apply global metering (same scale for every tile) then convert to
    # sRGB display space. This preserves relative brightness across tiles
    # so the detector sees them as one consistent camera exposure.
    if global_rgb_scale is not None:
        balanced_linear = np.clip(
            tile_linear * global_rgb_scale[None, None, :] * global_exp_scale,
            0.0, None)
    else:
        balanced_linear = tile_linear.copy()
    tile_detect_display = _linear_to_srgb_display(balanced_linear)
    detect_rgb_scale = global_rgb_scale if global_rgb_scale is not None else np.ones(3, dtype=np.float32)
    detect_exposure_scale = global_exp_scale
    tile_detect_tm = tile_detect_display
    tile_detect_u8 = np.clip(tile_detect_display * 255 + 0.5, 0, 255).astype(np.uint8)

    if debug_dir:
        _save_intermediate_debug(
            debug_dir,
            f"tile_{stage_label}_prebalanced.jpg",
            tile_detect_u8,
            note=(f"{stage_label}: global wb=[{detect_rgb_scale[0]:.2f},{detect_rgb_scale[1]:.2f},{detect_rgb_scale[2]:.2f}] "
                  f"exp={detect_exposure_scale:.2f}"),
        )

    seg_det = None
    chosen_det = None
    chosen_sw_tm = None
    chosen_quad_tile = None
    crop_bounds = None
    method_used = "none"
    detect_source_linear = tile_linear
    detect_scale_vec = np.ones(3, dtype=np.float32)

    # Try globally-metered version first (sRGB display space, same exposure
    # as every other tile), then raw Reinhard as fallback.
    _bracket_attempts = [
        ("global",  tile_detect_tm,  (detect_rgb_scale * detect_exposure_scale).astype(np.float32)),
        ("raw",     tile_raw_tm,     np.ones(3, dtype=np.float32)),
    ]

    for _attempt_name, _attempt_img, _attempt_scale in _bracket_attempts:
        try:
            seg_results = ccd.detect_colour_checkers_segmentation(
                _attempt_img,
                show=False,
                additional_data=True,
                apply_cctf_decoding=False,
            )
            if seg_results:
                seg_det = seg_results[0]
                chosen_det = seg_det
                chosen_sw_tm = np.array(seg_det.swatch_colours, dtype=np.float32)
                method_used = f"segmentation({_attempt_name})"
                detect_source_linear = tile_linear
                detect_scale_vec = _attempt_scale
                print(f"[cc-erp] Segmentation located chart on {_attempt_name} tile "
                      f"yaw={yaw:.0f} pitch={pitch:.0f} exp_scale={detect_exposure_scale:.3f}")
                break
        except Exception:
            pass

    if seg_det is None:
        return None

    seg_quad_tile = _quad_detector_to_pixels(seg_det.quadrilateral, out_w, out_h)
    chosen_quad_tile = seg_quad_tile.copy()
    crop_bounds = _quad_bounds_with_padding(seg_quad_tile, out_w, out_h, pad_frac=0.40)

    _save_intermediate_debug(
        debug_dir,
        f"tile_{stage_label}_locator.jpg",
        tile_rgb_u8,
        quad=seg_quad_tile,
        crop_bounds=crop_bounds,
        note=f"{stage_label}: locate yaw={yaw:.1f} pitch={pitch:.1f} fov={tile_fov_deg:.1f}",
    )

    if HAVE_CCD_INFERENCE and crop_bounds is not None:
        cx0, cy0, cx1, cy1 = crop_bounds
        crop_lin = detect_source_linear[cy0:cy1, cx0:cx1]
        crop_h, crop_w = crop_lin.shape[:2]
        zoom_w = 800
        zoom_h = max(1, int(round(crop_h * zoom_w / max(float(crop_w), 1e-8))))
        crop_up = cv2.resize(crop_lin, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
        crop_tm = (crop_up / (crop_up + 1.0)).astype(np.float32)
        crop_u8 = np.clip(crop_tm * 255, 0, 255).astype(np.uint8)

        _save_intermediate_debug(
            debug_dir,
            f"tile_{stage_label}_crop.jpg",
            crop_u8,
            note=f"{stage_label}: crop {method_used} {crop_w}x{crop_h} -> {zoom_w}x{zoom_h}",
        )
        crop_orig = tile_linear[cy0:cy1, cx0:cx1]
        _save_intermediate_debug(
            debug_dir,
            f"tile_{stage_label}_crop_original.jpg",
            _linear_to_u8_for_detection(crop_orig),
            note=f"{stage_label}: crop original {crop_w}x{crop_h}",
        )

        try:
            inf_results = ccd.detect_colour_checkers_inference(
                crop_u8,
                additional_data=True,
            )
            if inf_results:
                inf_det = inf_results[0]
                inf_sw_tm = np.array(inf_det.swatch_colours, dtype=np.float32)
                if inf_sw_tm.shape == (24, 3):
                    def _cerr(sw, ref):
                        n = sw[18:24]
                        r = ref[18:24]
                        lm = 0.2126 * n[:, 0] + 0.7152 * n[:, 1] + 0.0722 * n[:, 2]
                        lr = 0.2126 * r[:, 0] + 0.7152 * r[:, 1] + 0.0722 * r[:, 2]
                        sc = np.clip(lr / (lm + 1e-8), 0, 20)[:, None]
                        return float(np.mean(np.abs(n * sc - r)))

                    ref_arr = cc24_ref if cc24_ref is not None else CC24_LINEAR_SRGB
                    seg_sw_safe = np.clip(chosen_sw_tm, 0, 0.9999)
                    seg_sw_lin = seg_sw_safe / (1.0 - seg_sw_safe)
                    seg_err = _cerr(seg_sw_lin, ref_arr)
                    inf_sw_safe = np.clip(inf_sw_tm, 0, 0.9999)
                    inf_sw_lin = inf_sw_safe / (1.0 - inf_sw_safe)
                    inf_err = _cerr(inf_sw_lin, ref_arr)
                    print(f"[cc-erp] YOLO on crop: seg_err={seg_err:.4f}  yolo_err={inf_err:.4f}  crop={crop_w}x{crop_h} -> {zoom_w}x{zoom_h}px")

                    if inf_err < seg_err:
                        inf_quad_zoom = _quad_detector_to_pixels(inf_det.quadrilateral, zoom_w, zoom_h)
                        inf_quad_tile = inf_quad_zoom.copy()
                        inf_quad_tile[:, 0] *= crop_w / max(float(zoom_w), 1e-8)
                        inf_quad_tile[:, 1] *= crop_h / max(float(zoom_h), 1e-8)
                        inf_quad_tile[:, 0] += float(cx0)
                        inf_quad_tile[:, 1] += float(cy0)

                        chosen_det = inf_det
                        chosen_sw_tm = inf_sw_tm
                        chosen_quad_tile = inf_quad_tile.astype(np.float32)
                        method_used = "segmentation(locate)+yolo(read)"
                        print("[cc-erp] YOLO result better - using crop-local quadrilateral and swatches")
                    else:
                        print("[cc-erp] Segmentation swatches better or equal - keeping segmentation")
            else:
                print("[cc-erp] YOLO: no detection on crop - keeping segmentation swatches")
        except Exception as ye:
            print(f"[cc-erp] YOLO on crop failed ({ye}) - keeping segmentation swatches")

    cc_img = np.array(chosen_det.colour_checker, dtype=np.float32)
    H_cc, W_cc = cc_img.shape[:2]
    quad_tile = np.array(chosen_quad_tile, dtype=np.float32)

    colour_sw_tm = np.array(chosen_sw_tm, dtype=np.float32)
    if colour_sw_tm.shape != (24, 3):
        return None

    # First-pass rectified view from the tile detection.
    cc_first_img = cc_img
    cc_first_h, cc_first_w = H_cc, W_cc
    cc_first_rect = np.array([
        [0.0, 0.0],
        [cc_first_w, 0.0],
        [cc_first_w, cc_first_h],
        [0.0, cc_first_h],
    ], dtype=np.float32)
    cc_first_masks = np.array(chosen_det.swatch_masks, dtype=np.float32)
    cc_first_H_to_tile, _ = cv2.findHomography(cc_first_rect, quad_tile)
    cc_first_centres = np.column_stack([
        (cc_first_masks[:, 2] + cc_first_masks[:, 3]) * 0.5,
        (cc_first_masks[:, 0] + cc_first_masks[:, 1]) * 0.5,
    ]).astype(np.float32)
    cc_first_sw_tm = colour_sw_tm.copy()

    cc_work_img = cc_first_img
    cc_work_h, cc_work_w = cc_first_h, cc_first_w
    cc_work_masks = cc_first_masks
    cc_work_H_to_tile = cc_first_H_to_tile
    cc_work_centres = cc_first_centres
    cc_work_sw_tm = cc_first_sw_tm

    rectified_det = None
    try:
        rectified_results = ccd.detect_colour_checkers_segmentation(
            np.clip(cc_img, 0.0, 1.0).astype(np.float32),
            show=False,
            additional_data=True,
            apply_cctf_decoding=False,
        )
        if rectified_results:
            cand = rectified_results[0]
            cand_sw = np.array(cand.swatch_colours, dtype=np.float32)
            cand_masks = np.array(cand.swatch_masks, dtype=np.float32)
            cand_img = np.array(cand.colour_checker, dtype=np.float32)
            if cand_sw.shape == (24, 3) and cand_masks.shape[0] >= 24 and cand_img.ndim == 3:
                rectified_det = cand
                colour_sw_tm = cand_sw
                cc_work_img = cand_img
                cc_work_h, cc_work_w = cc_work_img.shape[:2]
                cc_work_masks = cand_masks
                cc_work_sw_tm = cand_sw.copy()
                rectified_quad_cc = _quad_detector_to_pixels(cand.quadrilateral, W_cc, H_cc)
                cc_work_rect = np.array([
                    [0.0, 0.0],
                    [cc_work_w, 0.0],
                    [cc_work_w, cc_work_h],
                    [0.0, cc_work_h],
                ], dtype=np.float32)
                H_work_to_cc, _ = cv2.findHomography(cc_work_rect, rectified_quad_cc.astype(np.float32))
                H_cc_to_tile, _ = cv2.findHomography(cc_first_rect, quad_tile)
                if H_work_to_cc is not None and H_cc_to_tile is not None:
                    cc_work_H_to_tile = H_cc_to_tile @ H_work_to_cc
                cc_work_centres = np.column_stack([
                    (cc_work_masks[:, 2] + cc_work_masks[:, 3]) * 0.5,
                    (cc_work_masks[:, 0] + cc_work_masks[:, 1]) * 0.5,
                ]).astype(np.float32)
                mapped_quad_tile = _map_rectified_points_to_tile(cc_work_rect, cc_work_H_to_tile, out_w, out_h)
                if mapped_quad_tile.shape == (4, 2):
                    quad_tile = mapped_quad_tile
                method_used = f"{method_used}+rectified(seg)"
                print("[cc-erp] Rectified re-detect succeeded - using full second-pass geometry")
    except Exception as re:
        print(f"[cc-erp] Rectified re-detect failed ({re}) - keeping first-pass geometry")

    cx_cc = cc_work_centres[:, 0]
    cy_cc = cc_work_centres[:, 1]

    colour_centres_tile = _map_rectified_points_to_tile(
        cc_work_centres,
        cc_work_H_to_tile,
        out_w,
        out_h,
    )
    if colour_centres_tile.shape[0] != 24:
        ctr = quad_tile.mean(axis=0)
        colour_centres_tile = np.asarray([(float(ctr[0]), float(ctr[1]))] * 24, dtype=np.float32)

    colour_centres_uv = np.array([
        backproject_pixel_to_erp(cx, cy, map_uv)
        for cx, cy in colour_centres_tile
    ], dtype=np.float32)
    colour_sw_safe = np.clip(colour_sw_tm, 0.0, 0.9999)
    colour_swatches_linear = colour_sw_safe / (1.0 - colour_sw_safe)
    colour_swatches_linear = colour_swatches_linear / np.clip(detect_scale_vec[None, :], 1e-6, None)
    colour_swatches_linear, colour_reorder = _reorder_swatches_to_cc24(colour_swatches_linear, cc24_ref)
    colour_centres_uv = colour_centres_uv[colour_reorder]
    colour_centres_tile = colour_centres_tile[colour_reorder]
    colour_confidence = _compute_detection_confidence(colour_swatches_linear, cc24_ref)

    contour_quad = _find_checker_polygon(cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2BGR), crop_bounds)
    contour_centres_tile = None
    contour_centres_uv = None
    contour_swatches_hdr = None
    contour_confidence = 0.0
    contour_agreement = 0.0
    if contour_quad is not None:
        contour_swatches_tile, contour_centres_tile = _sample_swatches_from_quad(tile_linear, contour_quad)
        contour_centres_tile = np.asarray(contour_centres_tile, dtype=np.float32)
        contour_centres_uv = np.array([
            backproject_pixel_to_erp(cx, cy, map_uv)
            for cx, cy in contour_centres_tile
        ], dtype=np.float32)
        contour_swatches_linear = contour_swatches_tile / np.clip(detect_scale_vec[None, :], 1e-6, None)
        contour_swatches_linear, contour_reorder = _reorder_swatches_to_cc24(contour_swatches_linear, cc24_ref)
        contour_centres_uv = contour_centres_uv[contour_reorder]
        contour_centres_tile = contour_centres_tile[contour_reorder]
        contour_confidence = _compute_detection_confidence(contour_swatches_linear, cc24_ref)
        contour_agreement = _quad_agreement(contour_quad, quad_tile, out_w, out_h)
        contour_swatches_hdr = contour_swatches_linear
        print(f"[cc-erp] Backends: colour={colour_confidence:.3f} contour={contour_confidence:.3f} agreement={contour_agreement:.3f}")
    else:
        print(f"[cc-erp] Backends: colour={colour_confidence:.3f} contour=none")

    if compare_backends:
        lbl = tile_idx if isinstance(tile_idx, str) else (tile_idx if tile_idx >= 0 else stage_label)
        _save_backend_debug(
            debug_dir,
            f"tile_{lbl}_backend_colour.jpg",
            tile_rgb_u8,
            quad_tile,
            colour_centres_tile,
            f"{stage_label} backend=colour conf={colour_confidence:.2f}",
        )
        _save_backend_debug(
            debug_dir,
            f"tile_{lbl}_backend_contour.jpg",
            tile_rgb_u8,
            contour_quad,
            contour_centres_tile,
            f"{stage_label} backend=contour conf={contour_confidence:.2f} agree={contour_agreement:.2f}",
        )

    selected_backend = "colour"
    selected_quad = quad_tile
    swatch_centres_tile = colour_centres_tile
    swatch_centres_uv = colour_centres_uv
    swatches_linear = colour_swatches_linear
    confidence = colour_confidence

    if read_backend == "contour":
        if contour_swatches_hdr is not None:
            selected_backend = "contour"
            selected_quad = contour_quad.astype(np.float32)
            swatch_centres_tile = contour_centres_tile
            swatch_centres_uv = contour_centres_uv
            swatches_linear = contour_swatches_hdr
            confidence = contour_confidence
            method_used = f"{method_used}+contour(read)"
        else:
            method_used = f"{method_used}+colour(fallback)"
    elif read_backend == "auto" and contour_swatches_hdr is not None:
        if contour_agreement >= 0.55 and contour_confidence > (colour_confidence + 0.03):
            selected_backend = "contour"
            selected_quad = contour_quad.astype(np.float32)
            swatch_centres_tile = contour_centres_tile
            swatch_centres_uv = contour_centres_uv
            swatches_linear = contour_swatches_hdr
            confidence = contour_confidence
            method_used = f"{method_used}+contour(auto)"
        else:
            method_used = f"{method_used}+colour(auto)"
    else:
        method_used = f"{method_used}+colour(read)"

    print(f"[cc-erp] Detection method: {method_used}")

    quad_sorted_tile = _order_quad_tl_tr_br_bl(selected_quad.copy())
    checker_normal_world = _estimate_checker_pose(
        [(float(x), float(y)) for x, y in quad_sorted_tile],
        out_w, out_h, yaw, pitch, fov_deg=tile_fov_deg, is_corners=True)

    n = checker_normal_world
    theta_n = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(n[0], n[2])))

    quad_center_tile = selected_quad.mean(axis=0)
    quad_center_uv = np.array(
        backproject_pixel_to_erp(float(quad_center_tile[0]), float(quad_center_tile[1]), map_uv),
        dtype=np.float32,
    )

    if debug_dir:
        lbl = tile_idx if isinstance(tile_idx, str) else (tile_idx if tile_idx >= 0 else stage_label)
        _save_rectified_debug(
            debug_dir,
            f"tile_{lbl}_rectified_pass1.jpg",
            cc_first_img,
            cc_first_centres,
            f"{stage_label} rectified pass1",
        )
        _save_debug_swatch_strip(
            debug_dir,
            f"tile_{lbl}_swatches_pass1.jpg",
            cc_first_sw_tm,
            f"{stage_label} pass1 swatches",
        )
        if rectified_det is not None:
            _save_rectified_debug(
                debug_dir,
                f"tile_{lbl}_rectified_pass2.jpg",
                cc_work_img,
                cc_work_centres,
                f"{stage_label} rectified pass2",
            )
            _save_debug_swatch_strip(
                debug_dir,
                f"tile_{lbl}_swatches_pass2.jpg",
                cc_work_sw_tm,
                f"{stage_label} pass2 swatches",
            )
        _save_rectified_debug(
            debug_dir,
            f"tile_{lbl}_rectified_checker.jpg",
            cc_work_img,
            cc_work_centres,
            f"{stage_label} rectified selected",
        )

        vis = cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2BGR)
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 1)
        cv2.polylines(vis, [np.round(selected_quad).astype(np.int32)], True, (0, 200, 255), 2)
        for i, (cx, cy) in enumerate(swatch_centres_tile):
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0, 255, 0), -1)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 7, (0, 0, 0), 1)
            cv2.putText(vis, str(i + 1), (int(round(cx)) + 4, int(round(cy)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
        cv2.putText(vis,
                    f"{stage_label} yaw={yaw:.1f} pitch={pitch:.1f} fov={tile_fov_deg:.1f} conf={confidence:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
        cv2.putText(vis, f"method={method_used}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
        fname = (f"tile_{lbl:03d}_detected.jpg" if isinstance(lbl, int)
                 else f"tile_{lbl}_detected.jpg")
        cv2.imwrite(os.path.join(debug_dir, fname), vis)

    return CheckerDetection(
        swatches_linear=swatches_linear.astype(np.float32),
        swatch_centres_uv=swatch_centres_uv.astype(np.float32),
        swatch_centres_tile=np.asarray(swatch_centres_tile, dtype=np.float32),
        quad_tile=np.asarray(selected_quad, dtype=np.float32),
        quad_center_uv=quad_center_uv,
        tile_yaw=yaw,
        tile_pitch=pitch,
        tile_fov_deg=tile_fov_deg,
        checker_normal_world=checker_normal_world,
        checker_normal_theta_deg=theta_n,
        checker_normal_phi_deg=phi_n,
        confidence=confidence,
        raw_swatches_bgr=np.clip(swatches_linear * 255, 0, 255).astype(np.uint8),
        detection_method=method_used,
        stage_label=stage_label,
        crop_bounds=crop_bounds,
    )

def _reorder_swatches_to_cc24(swatches: np.ndarray,
                              cc24_ref: Optional[np.ndarray] = None) -> tuple:
    """
    Reorder the library's 24 swatches to match CC24 row-major layout.

    cc24_ref: (24,3) reference in the working colorspace. If None, uses
              CC24_LINEAR_SRGB (backward compat). Pass get_cc24_reference(cs)
              from the caller to match the image colorspace.

    Uses luma-normalised chroma distance so illuminant doesn't affect matching.
    """
    rows, cols = 4, 6
    assert swatches.shape == (24, 3)

    if cc24_ref is None:
        cc24_ref = CC24_LINEAR_SRGB

    # Reference: luma-normalised (illuminant-independent chroma comparison)
    ref = cc24_ref.copy()                           # (24,3)
    ref_luma = (0.2126*ref[:,0] + 0.7152*ref[:,1] + 0.0722*ref[:,2])[:,None]
    ref_norm = ref / np.clip(ref_luma, 1e-4, None) # (24,3) chroma only

    sw = swatches.copy()
    sw_luma = (0.2126*sw[:,0] + 0.7152*sw[:,1] + 0.0722*sw[:,2])[:,None]
    sw_norm = sw / np.clip(sw_luma, 1e-4, None)

    # Build the 8 candidate index permutations of a 4×6 grid
    grid = np.arange(24).reshape(rows, cols)       # row-major CC24 order

    def _grid_to_idx(g):
        return g.flatten().tolist()

    candidates = []
    for flip in (False, True):
        g = grid if not flip else grid[:, ::-1]
        for rot in range(4):
            candidates.append(_grid_to_idx(np.rot90(g, rot)))

    best_idx  = candidates[0]
    best_err  = float('inf')
    for idx in candidates:
        reordered = sw_norm[idx]                   # (24,3) chroma
        err = float(np.mean(np.abs(reordered - ref_norm)))
        if err < best_err:
            best_err  = err
            best_idx  = idx

    reorder_arr = np.array(best_idx, dtype=np.int32)
    print(f"[cc-erp] Swatch reorder: best orientation err={best_err:.4f}  "
          f"idx[21]={best_idx[21]} (patch 22 in library order)")
    return swatches[reorder_arr], reorder_arr



def _find_checker_polygon(tile_u8_bgr: np.ndarray,
                          search_bounds: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """Find a plausible 4-corner checker polygon using contour detection."""
    h, w = tile_u8_bgr.shape[:2]
    if search_bounds is not None:
        x0, y0, x1, y1 = search_bounds
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, x0 + 1, w))
        y1 = int(np.clip(y1, y0 + 1, h))
        roi = tile_u8_bgr[y0:y1, x0:x1]
        offset = np.array([x0, y0], dtype=np.float32)
    else:
        roi = tile_u8_bgr
        offset = np.zeros(2, dtype=np.float32)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape[:2]
    min_area = rw * rh * 0.02
    max_area = rw * rh * 0.85
    best_poly = None
    best_score = -1.0

    for lo, hi in [(20, 60), (30, 90), (50, 150), (10, 40)]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4:
                continue
            rect = cv2.minAreaRect(approx)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area < 1.0:
                continue
            rect_score = area / rect_area
            rw2 = max(rect[1][0], rect[1][1])
            rh2 = min(rect[1][0], rect[1][1])
            aspect = rw2 / max(rh2, 1.0)
            aspect_score = 1.0 / (1.0 + abs(aspect - 1.5))
            centre = np.mean(approx.reshape(4, 2), axis=0)
            centre_score = 1.0 - min(np.linalg.norm((centre / [max(rw,1), max(rh,1)]) - 0.5), 0.7)
            score = rect_score * aspect_score * max(centre_score, 0.2)
            if score > best_score:
                best_score = score
                best_poly = approx.reshape(4, 2).astype(np.float32)

    if best_poly is None:
        return None
    return best_poly + offset



def _order_polygon_tl_tr_br_bl(poly):
    """Reorder 4 corner points to top-left, top-right, bottom-right, bottom-left."""
    return _order_quad_tl_tr_br_bl(poly)



def _grid_from_polygon(poly):
    """Compute the 24 swatch centres from a checker quadrilateral."""
    poly = _order_polygon_tl_tr_br_bl(poly)
    tl, tr, br, bl = poly

    centres = []
    rows, cols = 4, 6
    for r in range(rows):
        for c in range(cols):
            s = (c + 0.5) / cols
            t = (r + 0.5) / rows
            top = tl + s * (tr - tl)
            bottom = bl + s * (br - bl)
            pt = top + t * (bottom - top)
            centres.append((float(pt[0]), float(pt[1])))
    return centres



def _uniform_grid_fallback(out_w, out_h):
    """Last-resort: uniform 4×6 grid covering 60% of the tile."""
    margin_x = out_w * 0.2
    margin_y = out_h * 0.2
    centres = []
    for r in range(4):
        for c in range(6):
            x = margin_x + (c + 0.5) / 6 * (out_w - 2 * margin_x)
            y = margin_y + (r + 0.5) / 4 * (out_h - 2 * margin_y)
            centres.append((float(x), float(y)))
    return centres


# ─── Pose estimation ──────────────────────────────────────────────────────────
# ─── Pose estimation ──────────────────────────────────────────────────────────

def _estimate_checker_pose(pts_px: list,
                           out_w: int, out_h: int,
                           yaw_deg: float, pitch_deg: float,
                           fov_deg: float = 70.0,
                           is_corners: bool = False) -> np.ndarray:
    """
    Estimate the 3D normal of the checker plane using solvePnP.

    pts_px: either 24 swatch centres (is_corners=False) or 4 quad corners
            in TL,TR,BR,BL order (is_corners=True), in tile pixel coords.

    The checker lies in the Z=0 plane in its own coordinate system.
    solvePnP gives R such that the checker normal in camera space is R[:,2].
    We then rotate from tile-camera space → world (ERP) space via yaw/pitch.
    """
    pts_px = list(pts_px)
    n_pts = len(pts_px)

    if is_corners and n_pts == 4:
        # 4 corners of the full checker board in mm.
        # CC24 physical size: 6 cols × 4 rows of 24mm patches with 6mm gaps.
        # Border ≈ 6mm each side.
        total_w = 5 * (_PATCH_W + _GAP) + _PATCH_W + 2 * _GAP   # 6 patches + borders
        total_h = 3 * (_PATCH_H + _GAP) + _PATCH_H + 2 * _GAP   # 4 patches + borders
        # TL, TR, BR, BL
        pts_3d = np.array([
            [0.0,     0.0,     0.0],
            [total_w, 0.0,     0.0],
            [total_w, total_h, 0.0],
            [0.0,     total_h, 0.0],
        ], dtype=np.float64)
    elif n_pts >= 6:
        pts_3d = CC24_3D_POINTS[:n_pts].astype(np.float64)
    else:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # fallback: up

    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, out_w / 2.0],
                  [0, f, out_h / 2.0],
                  [0, 0, 1.0        ]], dtype=np.float64)

    pts_2d = np.array(pts_px, dtype=np.float64)

    method = cv2.SOLVEPNP_IPPE if n_pts == 4 else cv2.SOLVEPNP_ITERATIVE
    try:
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, K, np.zeros(4), flags=method)
    except Exception:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if not success:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    R_cam, _ = cv2.Rodrigues(rvec)
    normal_cam = R_cam[:, 2]   # checker +Z in camera space
    normal_cam /= np.linalg.norm(normal_cam) + 1e-8

    # Tile camera → world rotation (yaw around Y, then pitch around X)
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [ 0,             1, 0            ],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0,                 0             ],
                   [0,  math.cos(pitch),  math.sin(pitch)],
                   [0, -math.sin(pitch),  math.cos(pitch)]])

    normal_world = (Ry @ Rx) @ normal_cam
    normal_world /= np.linalg.norm(normal_world) + 1e-8
    return normal_world.astype(np.float32)


# ─── Main entry point ─────────────────────────────────────────────────────────



def find_colorchecker_in_erp(
    erp_linear: np.ndarray,
    fov_deg: float = 110.0,
    tile_w: int = 900,
    tile_h: int = 675,
    yaw_step_deg: float = 40.0,
    pitch_values: tuple = (-45.0, -20.0, 0.0, 20.0, 45.0),
    min_confidence: float = 0.05,
    colorspace: str = "acescg",
    debug_dir: Optional[str] = None,
    read_backend: str = "colour",
    compare_backends: bool = True,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Find a ColorChecker Classic 24 inside an ERP panorama.

    Detection strategy:
      1. Coarse cubemap sweep at 110° FOV for overlap between faces.
      2. Re-extract a centred 90° tile using the detected quadrilateral centre.
      3. Re-extract a tighter centred tile (about 50° FOV, adaptively derived
         from checker coverage) so swatch extraction sees a larger checker.
    """
    if not HAVE_CCD:
        return None, {"error": "colour-checker-detection not installed. "
                                "pip install colour-checker-detection"}

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    _log_detection_backends()

    best: Optional[CheckerDetection] = None
    total_tiles = 0

    cc24_ref = get_cc24_reference(colorspace)
    print(f"[cc-erp] Colorspace: {colorspace}  "
          f"(CC24 reference patch 22: "
          f"R={cc24_ref[21,0]:.4f} G={cc24_ref[21,1]:.4f} B={cc24_ref[21,2]:.4f})")

    # Global metering: compute ONE exposure + WB scale from the full ERP.
    # Every tile gets the same adjustment so relative brightness is preserved
    # (bright ground stays bright, dark sky stays dark — like a single camera exposure).
    _, _global_rgb_scale, _global_exp_scale, _global_balance_info = \
        _compute_detection_prebalance(erp_linear, target_mid=0.18)
    print(f"[cc-erp] Global metering: mid_mean_lum={_global_balance_info.get('lum_mid_mean', 0):.4f} "
          f"exp_scale={_global_exp_scale:.3f} "
          f"wb=[{_global_rgb_scale[0]:.3f},{_global_rgb_scale[1]:.3f},{_global_rgb_scale[2]:.3f}]")

    coarse_faces = [
        (0.0,   0.0),
        (90.0,  0.0),
        (180.0, 0.0),
        (270.0, 0.0),
        (0.0,  90.0),
        (0.0, -90.0),
    ]
    coarse_fov = float(np.clip(fov_deg, 90.0, 140.0))
    coarse_size = 1024

    print(f"[cc-erp] Pass cube-overlap: {len(coarse_faces)} tiles (FOV={coarse_fov:.0f}°)")
    for idx, (yaw, pitch) in enumerate(coarse_faces):
        total_tiles += 1
        tile_linear, map_uv = erp_to_rectilinear(
            erp_linear, yaw, pitch, coarse_fov, coarse_size, coarse_size)

        if debug_dir:
            tile_u8 = _linear_to_u8_for_detection(tile_linear)
            tile_bgr = cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR)
            cv2.putText(tile_bgr,
                        f"cube-overlap yaw={yaw:.0f} pitch={pitch:.0f} fov={coarse_fov:.0f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            tile_fname = f"sweep_cube_overlap_y{int(yaw):03d}_p{int(pitch):+03d}.jpg"
            cv2.imwrite(os.path.join(debug_dir, tile_fname), tile_bgr)

        det = _detect_in_tile(
            tile_linear, map_uv, erp_linear,
            yaw, pitch, coarse_fov,
            debug_dir, tile_idx=idx,
            stage_label=f"coarse_{idx:02d}",
            cc24_ref=cc24_ref,
            read_backend=read_backend,
            compare_backends=compare_backends,
            global_rgb_scale=_global_rgb_scale,
            global_exp_scale=_global_exp_scale)

        if det is None or det.confidence < min_confidence:
            continue

        if best is None or det.confidence > best.confidence:
            best = det
            print(f"[cc-erp]  [cube-overlap] tile {idx} yaw={yaw:.0f}° pitch={pitch:.0f}° confidence={det.confidence:.3f}")

    if best is None:
        print(f"[cc-erp] No ColorChecker found after coarse sweep ({total_tiles} tiles).")
        return None, {"found": False, "tiles_searched": total_tiles}

    print(f"[cc-erp] Best coarse detection: tile yaw={best.tile_yaw:.0f}° pitch={best.tile_pitch:.0f}° confidence={best.confidence:.3f}")

    def _uv_to_yaw_pitch(uv: np.ndarray) -> Tuple[float, float]:
        u_c, v_c = float(uv[0]), float(uv[1])
        refine_yaw = (u_c - 0.5) * 360.0
        refine_yaw = float((refine_yaw + 180.0) % 360.0 - 180.0)
        refine_pitch = float(np.clip((0.5 - v_c) * 180.0, -85.0, 85.0))
        return refine_yaw, refine_pitch

    def _adaptive_tight_fov(det: CheckerDetection, img_w: int, img_h: int, current_fov: float) -> float:
        quad = np.asarray(det.quad_tile, dtype=np.float32)
        if quad.shape != (4, 2):
            return 50.0
        quad_w = float(np.max(quad[:, 0]) - np.min(quad[:, 0]))
        quad_h = float(np.max(quad[:, 1]) - np.min(quad[:, 1]))
        frac = max(quad_w / max(float(img_w), 1.0), quad_h / max(float(img_h), 1.0))
        if frac <= 1e-4:
            return 50.0
        target_frac = 0.50
        fov = current_fov * frac / target_frac
        return float(np.clip(fov, 45.0, 65.0))

    locator_det = best
    refinement_stage = "coarse"
    refined = False

    wide_yaw, wide_pitch = _uv_to_yaw_pitch(locator_det.quad_center_uv)
    print(f"[cc-erp] Recentered wide pass at yaw={wide_yaw:.1f}° pitch={wide_pitch:.1f}°")
    tile_wide, map_uv_wide = erp_to_rectilinear(
        erp_linear, wide_yaw, wide_pitch, 90.0, 1024, 1024)
    det_wide = _detect_in_tile(
        tile_wide, map_uv_wide, erp_linear,
        wide_yaw, wide_pitch, 90.0,
        debug_dir, tile_idx="recenter_wide",
        stage_label="recenter_wide",
        cc24_ref=cc24_ref,
        read_backend=read_backend,
        compare_backends=compare_backends,
        global_rgb_scale=_global_rgb_scale,
        global_exp_scale=_global_exp_scale)

    if det_wide is not None:
        best = det_wide
        refined = True
        refinement_stage = "recenter_wide"
        print(f"[cc-erp] Wide centred detection selected for swatch read (conf={det_wide.confidence:.3f}; coarse locator={locator_det.confidence:.3f})")
    else:
        print(f"[cc-erp] Wide centred detection failed - keeping coarse locator result for now (conf={locator_det.confidence:.3f})")

    tight_source = det_wide if det_wide is not None else locator_det
    tight_yaw, tight_pitch = _uv_to_yaw_pitch(tight_source.quad_center_uv)
    tight_fov = _adaptive_tight_fov(tight_source, 1024, 1024, 90.0)
    print(f"[cc-erp] Recentered tight pass at yaw={tight_yaw:.1f}° pitch={tight_pitch:.1f}° fov={tight_fov:.1f}°")
    tile_tight, map_uv_tight = erp_to_rectilinear(
        erp_linear, tight_yaw, tight_pitch, tight_fov, tile_w, tile_h)
    det_tight = _detect_in_tile(
        tile_tight, map_uv_tight, erp_linear,
        tight_yaw, tight_pitch, tight_fov,
        debug_dir, tile_idx="recenter_tight",
        stage_label="recenter_tight",
        cc24_ref=cc24_ref,
        read_backend=read_backend,
        compare_backends=compare_backends,
        global_rgb_scale=_global_rgb_scale,
        global_exp_scale=_global_exp_scale)

    if det_tight is not None:
        best = det_tight
        refined = True
        refinement_stage = "recenter_tight"
        print(f"[cc-erp] Tight centred detection selected for final swatch read (conf={det_tight.confidence:.3f})")
    elif det_wide is not None:
        print(f"[cc-erp] Tight centred detection failed - keeping wide centred swatch read (conf={det_wide.confidence:.3f})")
    else:
        print(f"[cc-erp] Tight centred detection failed - falling back to coarse locator swatch read (conf={locator_det.confidence:.3f})")

    print(f"[cc-erp] Final: confidence={best.confidence:.3f}  checker θ={best.checker_normal_theta_deg:.1f}° φ={best.checker_normal_phi_deg:.1f}°")

    def _debug_label(det: CheckerDetection) -> str:
        if det.stage_label.startswith("coarse_"):
            try:
                return str(int(det.stage_label.split("_")[-1]))
            except Exception:
                return det.stage_label
        return det.stage_label

    if debug_dir:
        _save_final_debug(debug_dir, best, erp_linear, cc24_ref=cc24_ref)

        locator_label = _debug_label(locator_det)
        best_label = _debug_label(best)

        locator_candidates = [
            os.path.join(debug_dir, f"tile_{locator_label}_detected.jpg"),
            os.path.join(debug_dir, f"tile_{locator_det.stage_label}_locator.jpg"),
            os.path.join(debug_dir, f"tile_{locator_det.stage_label}_detected.jpg"),
        ]
        gui_tile = os.path.join(debug_dir, "cc_detected_tile.jpg")
        for src in locator_candidates:
            if os.path.exists(src):
                shutil.copyfile(src, gui_tile)
                break

        rectified_candidates = [
            os.path.join(debug_dir, f"tile_{best_label}_rectified_pass2.jpg"),
            os.path.join(debug_dir, f"tile_{best_label}_rectified_checker.jpg"),
            os.path.join(debug_dir, f"tile_{best_label}_rectified_pass1.jpg"),
            os.path.join(debug_dir, f"tile_{best.stage_label}_rectified_pass2.jpg"),
            os.path.join(debug_dir, f"tile_{best.stage_label}_rectified_checker.jpg"),
            os.path.join(debug_dir, f"tile_{best.stage_label}_rectified_pass1.jpg"),
        ]
        for src in rectified_candidates:
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(debug_dir, "cc_rectified_final.jpg"))
                break

    info = {
        "found": True,
        "tiles_searched": total_tiles,
        "refinement_pass": refined,
        "refinement_stage": refinement_stage,
        "best_tile_yaw_deg": best.tile_yaw,
        "best_tile_pitch_deg": best.tile_pitch,
        "best_tile_fov_deg": best.tile_fov_deg,
        "confidence": best.confidence,
        "checker_normal_world": best.checker_normal_world.tolist(),
        "checker_normal_theta_deg": best.checker_normal_theta_deg,
        "checker_normal_phi_deg": best.checker_normal_phi_deg,
        "swatch_centres_uv": best.swatch_centres_uv.tolist(),
        "quad_center_uv": best.quad_center_uv.tolist(),
        "detection_method": best.detection_method,
        "stage_label": best.stage_label,
        "read_backend": read_backend,
        "compare_backends": bool(compare_backends),
    }
    return best.swatches_linear, info


# ─── Colour correction matrix ─────────────────────────────────────────────────

def solve_color_matrix_from_swatches(
    measured_linear: np.ndarray,
    reference_linear: Optional[np.ndarray] = None,
    use_neutral_only: bool = False,
    colorspace: str = "acescg",
) -> Tuple[np.ndarray, float]:
    """
    Solve a 3×3 colour correction matrix M such that measured @ M ≈ reference.
    Operates in linear RGB. Least-squares over all 24 (or 6 neutral) patches.

    colorspace: used to select CC24 reference if reference_linear is None.
    use_neutral_only: solve on neutral patches only (more stable under coloured light).
    Returns: (M 3×3, RMSE)
    """
    if reference_linear is None:
        reference_linear = get_cc24_reference(colorspace)

    if use_neutral_only:
        A = measured_linear[18:]    # patches 19–24
        B = reference_linear[18:]
    else:
        A = measured_linear         # all 24
        B = reference_linear

    M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    pred  = measured_linear @ M
    rmse  = float(np.sqrt(np.mean((pred - reference_linear)**2)))
    return M.astype(np.float32), rmse


def apply_color_matrix(img_linear: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 3×3 colour correction matrix to a linear RGB image."""
    h, w = img_linear.shape[:2]
    corrected = img_linear.reshape(-1, 3).astype(np.float32) @ M
    return np.clip(corrected, 0.0, None).reshape(h, w, 3).astype(np.float32)


# ─── Debug helpers ────────────────────────────────────────────────────────────

def _save_tile_debug(debug_dir, tile_idx, tile_rgb, centres,
                     swatches_linear, yaw, pitch, confidence):
    """Save tile with green dots at library-derived swatch positions (RGB input)."""
    vis = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)   # RGB → BGR for cv2 drawing
    for i, (cx, cy) in enumerate(centres):
        cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (int(cx)+5, int(cy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(vis, f"yaw={yaw:.0f} pitch={pitch:.0f} conf={confidence:.2f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fname = f"tile_{tile_idx:03d}_detected.jpg" if isinstance(tile_idx, int) else f"tile_{tile_idx}_detected.jpg"
    cv2.imwrite(os.path.join(debug_dir, fname), vis)


def _save_final_debug(debug_dir, det: 'CheckerDetection', erp_linear: np.ndarray,
                      cc24_ref: Optional[np.ndarray] = None):
    """
    Save debug images:
      cc_erp_swatches.jpg      — ERP with green dots at detected swatch positions
      cc_swatch_comparison.jpg — 3 rows: measured / reference / post-WB
    """
    if cc24_ref is None:
        cc24_ref = CC24_LINEAR_SRGB
    # ── ERP overlay ───────────────────────────────────────────────────────
    erp_u8_rgb = _linear_to_u8_for_detection(erp_linear)
    erp_u8_bgr = cv2.cvtColor(erp_u8_rgb, cv2.COLOR_RGB2BGR)
    h, w = erp_u8_bgr.shape[:2]
    vis = erp_u8_bgr.copy()

    for i, (u, v) in enumerate(det.swatch_centres_uv):
        px = int(np.clip(u * w, 0, w-1))
        py = int(np.clip(v * h, 0, h-1))
        cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
        cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)   # black outline for visibility
        cv2.putText(vis, str(i+1), (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    # Draw checker face normal direction on ERP
    n = det.checker_normal_world
    phi_n   = float(np.arctan2(n[0], n[2]))
    theta_n = float(np.arccos(np.clip(n[1], -1, 1)))
    nu = int(np.clip((phi_n / (2*np.pi) + 0.5) * w, 0, w-1))
    nv = int(np.clip(theta_n / np.pi * h, 0, h-1))
    cv2.drawMarker(vis, (nu, nv), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(vis, "normal", (nu+10, nv), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    cv2.imwrite(os.path.join(debug_dir, "cc_erp_swatches.jpg"), vis)

    # ── Swatch comparison strip ───────────────────────────────────────────
    # Row 1 (M): measured HDR values from library (tonemapped for display)
    # Row 2 (R): CC24 reference values in working colorspace
    # Row 3 (W): measured × WB scale (what the swatches look like after calibration)
    sw = 48
    gap = 2
    n_rows = 3
    strip_h = sw * n_rows + gap * (n_rows - 1)
    ref_linear = cc24_ref   # (24,3) in working colorspace

    # Compute WB scale from patch 22 for the third row.
    # Use FULL per-channel scale (not G-normalised) so patch 22 W == R exactly.
    # If W[22] != R[22] the reorder is wrong or the wrong patch is being used.
    p22 = det.swatches_linear[21]
    ref22 = cc24_ref[21]
    wb_scale_full = (ref22 / np.clip(p22, 1e-8, None)).astype(np.float32)

    strip_lin = np.zeros((strip_h, sw*24, 3), dtype=np.float32)
    for i in range(24):
        meas = det.swatches_linear[i]
        row0 = 0
        row1 = sw + gap
        row2 = (sw + gap) * 2
        strip_lin[row0:row0+sw, i*sw:(i+1)*sw] = meas                      # measured
        strip_lin[row1:row1+sw, i*sw:(i+1)*sw] = ref_linear[i]             # reference
        strip_lin[row2:row2+sw, i*sw:(i+1)*sw] = meas * wb_scale_full      # post-WB: patch22 W==R by construction

    strip_u8 = _linear_to_u8_for_detection(strip_lin)
    strip_bgr = cv2.cvtColor(strip_u8, cv2.COLOR_RGB2BGR)

    # Labels
    for i in range(24):
        x = i * sw
        cv2.putText(strip_bgr, str(i+1), (x+2, sw-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
        cv2.putText(strip_bgr, "M", (x+2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200,200,200), 1)
        cv2.putText(strip_bgr, "R", (x+2, sw+gap+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200,200,200), 1)
        cv2.putText(strip_bgr, "W", (x+2, (sw+gap)*2+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200,200,200), 1)

    # Red border on patch 22
    p22_x = 21 * sw
    cv2.rectangle(strip_bgr, (p22_x, 0), (p22_x+sw-1, strip_h-1), (0,0,255), 2)
    cv2.putText(strip_bgr, "#22 WB", (p22_x, strip_h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,255), 1)

    cv2.imwrite(os.path.join(debug_dir, "cc_swatch_comparison.jpg"), strip_bgr)

    # ── Per-patch value printout ──────────────────────────────────────────
    print(f"[cc-erp] ── Swatch values (linear, Reinhard-inverted from library) ──")
    print(f"[cc-erp]   {'#':>3}  {'R':>8}  {'G':>8}  {'B':>8}  {'luma':>8}  note")
    for i in range(24):
        v = det.swatches_linear[i]
        luma = 0.2126*v[0] + 0.7152*v[1] + 0.0722*v[2]
        note = ""
        if i == 21:
            note = "  ← PATCH 22 (WB ref, should be achromatic)"
        elif i >= 18:
            note = "  ← neutral ramp"
        print(f"[cc-erp]   {i+1:>3}  {v[0]:>8.5f}  {v[1]:>8.5f}  {v[2]:>8.5f}  {luma:>8.5f}{note}")

    # ── WB multiplier from patch 22 ───────────────────────────────────────
    p22 = det.swatches_linear[21]
    ref22 = cc24_ref[21]
    scale = ref22 / np.clip(p22, 1e-8, None)
    scale_g_norm = scale / max(float(scale[1]), 1e-8)
    print(f"[cc-erp] ── Patch 22 WB derivation ──")
    print(f"[cc-erp]   measured linear  : R={p22[0]:.5f}  G={p22[1]:.5f}  B={p22[2]:.5f}")
    print(f"[cc-erp]   reference linear : R={ref22[0]:.5f}  G={ref22[1]:.5f}  B={ref22[2]:.5f}")
    print(f"[cc-erp]   scale (ref/meas) : R={scale[0]:.5f}  G={scale[1]:.5f}  B={scale[2]:.5f}")
    print(f"[cc-erp]   G-normalised WB  : R={scale_g_norm[0]:.5f}  G={scale_g_norm[1]:.5f}  B={scale_g_norm[2]:.5f}")
