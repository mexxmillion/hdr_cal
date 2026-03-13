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
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import colour_checker_detection as ccd
    HAVE_CCD = True
except ImportError:
    HAVE_CCD = False

# Check if the ML inference method is available (requires ultralytics)
HAVE_CCD_INFERENCE = False
if HAVE_CCD:
    try:
        _ = ccd.detect_colour_checkers_inference
        HAVE_CCD_INFERENCE = True
    except AttributeError:
        pass

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False


# ─── Colorspace matrices ─────────────────────────────────────────────────────
# sRGB D65 → XYZ D65 (IEC 61966-2-1)
_M_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# XYZ D65 → ACEScg AP1 (S-2014-004)
_M_XYZ_TO_ACESCG = np.array([
    [ 1.6410234, -0.3248033, -0.2364247],
    [-0.6636629,  1.6153316,  0.0167563],
    [ 0.0117219, -0.0082844,  0.9883949],
], dtype=np.float64)

# Combined: linear sRGB → ACEScg
M_SRGB_LINEAR_TO_ACESCG = (_M_XYZ_TO_ACESCG @ _M_SRGB_TO_XYZ).astype(np.float32)

# Inverse: ACEScg → linear sRGB
M_ACESCG_TO_SRGB_LINEAR = np.linalg.inv(
    M_SRGB_LINEAR_TO_ACESCG.astype(np.float64)
).astype(np.float32)


def apply_matrix_3x3(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 3×3 matrix to every pixel of a (H,W,3) or (N,3) array."""
    shape = img.shape
    return (img.reshape(-1, 3).astype(np.float32) @ M.T).reshape(shape).astype(np.float32)


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
    """Convert linear float HDR tile to uint8 sRGB for the detector."""
    # Tonemap: exposure-normalise to a sensible display range
    lum = 0.2126*img_linear[...,0] + 0.7152*img_linear[...,1] + 0.0722*img_linear[...,2]
    valid = lum[lum > 0]
    if valid.size:
        scale = 1.0 / max(float(np.percentile(valid, 99)), 1e-6)
    else:
        scale = 1.0
    srgb = np.where(img_linear*scale <= 0.0031308,
                    img_linear*scale * 12.92,
                    1.055 * np.power(np.clip(img_linear*scale, 1e-9, None), 1/2.4) - 0.055)
    return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)


def _srgb_to_linear_arr(v: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear [0,1], works on any shape."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045,
                    v / 12.92,
                    ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)




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




def _find_rectified_checker_bbox(cc_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find a tighter axis-aligned chart box inside the rectified checker image."""
    img = np.clip(cc_img * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 10:
            continue
        aspect = bw / max(float(bh), 1.0)
        rect_area = float(bw * bh)
        fill = area / max(rect_area, 1.0)
        edge_penalty = 0.0
        if x <= 2 or y <= 2 or (x + bw) >= (w - 2) or (y + bh) >= (h - 2):
            edge_penalty = 0.15
        score = fill * (1.0 / (1.0 + abs(aspect - 1.5))) - edge_penalty
        if score > best_score:
            best_score = score
            best = (x, y, x + bw, y + bh)

    return best


def _sample_swatches_from_rectified(cc_img: np.ndarray,
                                    bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Sample 24 swatches and centres from a tighter rectified checker bbox."""
    x0, y0, x1, y1 = bbox
    rect_w = max(float(x1 - x0), 1.0)
    rect_h = max(float(y1 - y0), 1.0)
    cell_w = rect_w / 6.0
    cell_h = rect_h / 4.0
    sample_w = max(3, int(round(cell_w * 0.38)))
    sample_h = max(3, int(round(cell_h * 0.38)))
    sample_w += 1 - (sample_w % 2)
    sample_h += 1 - (sample_h % 2)
    half_w = sample_w // 2
    half_h = sample_h // 2

    centres = []
    colours = []
    h, w = cc_img.shape[:2]
    for r in range(4):
        for c in range(6):
            cx = x0 + (c + 0.5) * cell_w
            cy = y0 + (r + 0.5) * cell_h
            ix0 = int(np.clip(round(cx) - half_w, 0, w - 1))
            ix1 = int(np.clip(round(cx) + half_w + 1, 1, w))
            iy0 = int(np.clip(round(cy) - half_h, 0, h - 1))
            iy1 = int(np.clip(round(cy) + half_h + 1, 1, h))
            patch = cc_img[iy0:iy1, ix0:ix1]
            colours.append(np.mean(patch.reshape(-1, 3), axis=0))
            centres.append((float(cx), float(cy)))

    return np.array(colours, dtype=np.float32), np.array(centres, dtype=np.float32)

def _detect_in_tile(tile_linear: np.ndarray,
                    map_uv: np.ndarray,
                    erp_linear_hd: np.ndarray,
                    yaw: float, pitch: float,
                    tile_fov_deg: float,
                    debug_dir,
                    tile_idx: int,
                    stage_label: str,
                    cc24_ref: Optional[np.ndarray] = None):
    """
    Detect a ColorChecker in one rectilinear tile.

    Stage 1 uses segmentation to locate the chart and recover a quadrilateral.
    Stage 2 recentres a crop around that quadrilateral and optionally runs the
    inference model on the crop, keeping whichever result yields the lower
    neutral-ramp error. Unlike the previous implementation, the selected stage's
    geometry is carried forward as well as its swatch colours.
    """
    if not HAVE_CCD:
        return None

    out_h, out_w = tile_linear.shape[:2]

    tile_tm = (tile_linear / (tile_linear + 1.0)).astype(np.float32)
    tile_rgb_u8 = np.clip(tile_tm * 255, 0, 255).astype(np.uint8)

    seg_det = None
    chosen_det = None
    chosen_sw_tm = None
    chosen_quad_tile = None
    crop_bounds = None
    method_used = "none"

    try:
        seg_results = ccd.detect_colour_checkers_segmentation(
            tile_tm,
            show=False,
            additional_data=True,
            apply_cctf_decoding=False,
        )
        if seg_results:
            seg_det = seg_results[0]
            chosen_det = seg_det
            chosen_sw_tm = np.array(seg_det.swatch_colours, dtype=np.float32)
            method_used = "segmentation"
            print(f"[cc-erp] Segmentation located chart on tile yaw={yaw:.0f} pitch={pitch:.0f}")
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
        crop_lin = tile_linear[cy0:cy1, cx0:cx1]
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
            note=f"{stage_label}: crop {crop_w}x{crop_h} -> {zoom_w}x{zoom_h}",
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
                        lm = 0.2126*n[:, 0] + 0.7152*n[:, 1] + 0.0722*n[:, 2]
                        lr = 0.2126*r[:, 0] + 0.7152*r[:, 1] + 0.0722*r[:, 2]
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
                        method_used = "segmentation(locate)+YOLO(swatches+quad)"
                        print("[cc-erp] YOLO result better - using crop-local quadrilateral and swatches")
                    else:
                        print("[cc-erp] Segmentation swatches better or equal - keeping segmentation")
            else:
                print("[cc-erp] YOLO: no detection on crop - keeping segmentation swatches")
        except Exception as ye:
            print(f"[cc-erp] YOLO on crop failed ({ye}) - keeping segmentation swatches")

    print(f"[cc-erp] Detection method: {method_used}")

    sw_tm = np.array(chosen_sw_tm, dtype=np.float32)
    if sw_tm.shape != (24, 3):
        return None

    rectified_bbox = _find_rectified_checker_bbox(cc_img)
    rectified_centres = None
    if rectified_bbox is not None:
        refined_sw_tm, rectified_centres = _sample_swatches_from_rectified(cc_img, rectified_bbox)
        if refined_sw_tm.shape == (24, 3):
            sw_tm = refined_sw_tm
            print(f"[cc-erp] Using refined rectified-grid swatch sampling inside bbox={rectified_bbox}")

    print("[cc-erp]   RAW swatch_colours from library (tonemapped, ALL 24 patches):")
    for i in range(24):
        print(f"[cc-erp]     patch {i+1:02d}: R={sw_tm[i,0]:.4f}  G={sw_tm[i,1]:.4f}  B={sw_tm[i,2]:.4f}")

    print("[cc-erp]   RAW swatch_colours from library (tonemapped, patch 19-24):")
    for i in range(18, 24):
        print(f"[cc-erp]     patch {i+1:02d}: R={sw_tm[i,0]:.4f}  G={sw_tm[i,1]:.4f}  B={sw_tm[i,2]:.4f}")

    sw_safe = np.clip(sw_tm, 0.0, 0.9999)
    swatches_linear = sw_safe / (1.0 - sw_safe)

    print("[cc-erp]   Linear (after undo Reinhard), patch 19-24:")
    for i in range(18, 24):
        print(f"[cc-erp]     patch {i+1:02d}: R={swatches_linear[i,0]:.4f}  G={swatches_linear[i,1]:.4f}  B={swatches_linear[i,2]:.4f}")

    quad_tile = np.array(chosen_quad_tile, dtype=np.float32)

    cc_img = np.array(chosen_det.colour_checker, dtype=np.float32)
    H_cc, W_cc = cc_img.shape[:2]
    masks = np.array(chosen_det.swatch_masks, dtype=np.float32)

    cx_cc = (masks[:, 2] + masks[:, 3]) * 0.5
    cy_cc = (masks[:, 0] + masks[:, 1]) * 0.5
    if rectified_centres is not None:
        cx_cc = rectified_centres[:, 0]
        cy_cc = rectified_centres[:, 1]

    def _sort_tl_tr_br_bl(q):
        c = q.mean(axis=0)
        ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
        q = q[np.argsort(ang)]
        i0 = np.argmin(q[:, 0] + q[:, 1])
        q = np.roll(q, -i0, axis=0)
        if np.cross(q[1] - q[0], q[2] - q[0]) > 0:
            q = q[[0, 3, 2, 1]]
        return q

    quad_sorted_tile = _sort_tl_tr_br_bl(quad_tile.copy())

    # Use the detector-provided quadrilateral order for swatch mapping.
    # The library already orients that quadrilateral to match the returned
    # rectified colour_checker image; re-sorting it here drifts the mask centres.
    quad_detector_tile = quad_tile.copy().astype(np.float32)

    dst_rect = np.array([
        [0.,   0.],
        [W_cc, 0.],
        [W_cc, H_cc],
        [0.,   H_cc],
    ], dtype=np.float32)

    H_cc2tile, _ = cv2.findHomography(dst_rect, quad_detector_tile)

    swatch_centres_tile = []
    if H_cc2tile is not None:
        pts_h = np.stack([cx_cc, cy_cc, np.ones(24)], axis=1).astype(np.float32)
        mapped = (H_cc2tile @ pts_h.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        for px, py in mapped:
            swatch_centres_tile.append((
                float(np.clip(px, 0, out_w - 1)),
                float(np.clip(py, 0, out_h - 1)),
            ))
    else:
        ctr = quad_tile.mean(axis=0)
        swatch_centres_tile = [(float(ctr[0]), float(ctr[1]))] * 24

    swatch_centres_uv = np.array([
        backproject_pixel_to_erp(cx, cy, map_uv)
        for cx, cy in swatch_centres_tile
    ], dtype=np.float32)

    swatches_hdr = np.array([
        sample_erp_bilinear(erp_linear_hd, float(uv[0]), float(uv[1]))
        for uv in swatch_centres_uv
    ], dtype=np.float32)

    checker_normal_world = _estimate_checker_pose(
        [(float(x), float(y)) for x, y in quad_sorted_tile],
        out_w, out_h, yaw, pitch, fov_deg=tile_fov_deg, is_corners=True)

    neutrals_meas = swatches_linear[18:24]
    ref = cc24_ref if cc24_ref is not None else CC24_LINEAR_SRGB
    neutrals_ref = ref[18:24]
    lum_meas = 0.2126*neutrals_meas[:, 0] + 0.7152*neutrals_meas[:, 1] + 0.0722*neutrals_meas[:, 2]
    lum_ref = 0.2126*neutrals_ref[:, 0] + 0.7152*neutrals_ref[:, 1] + 0.0722*neutrals_ref[:, 2]
    scale_n = np.clip(lum_ref / (lum_meas + 1e-8), 0.0, 20.0)[:, None]
    chroma_err = float(np.mean(np.abs(neutrals_meas * scale_n - neutrals_ref)))
    confidence = float(np.clip(1.0 - chroma_err * 5.0, 0.0, 1.0))

    n = checker_normal_world
    theta_n = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(n[0], n[2])))

    swatches_linear, reorder_idx = _reorder_swatches_to_cc24(swatches_linear, cc24_ref)
    swatch_centres_uv = swatch_centres_uv[reorder_idx]
    swatch_centres_tile = np.array([swatch_centres_tile[i] for i in reorder_idx], dtype=np.float32)

    quad_center_tile = quad_tile.mean(axis=0)
    quad_center_uv = np.array(
        backproject_pixel_to_erp(float(quad_center_tile[0]), float(quad_center_tile[1]), map_uv),
        dtype=np.float32,
    )

    if debug_dir:
        lbl = tile_idx if isinstance(tile_idx, str) else (tile_idx if tile_idx >= 0 else stage_label)
        cc_vis = cv2.cvtColor(np.clip(cc_img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if rectified_bbox is not None:
            x0, y0, x1, y1 = rectified_bbox
            cv2.rectangle(cc_vis, (x0, y0), (x1, y1), (255, 128, 0), 2)
        centres_vis = rectified_centres if rectified_centres is not None else np.column_stack([cx_cc, cy_cc])
        for i, (cx, cy) in enumerate(np.asarray(centres_vis, dtype=np.float32)):
            cv2.circle(cc_vis, (int(round(cx)), int(round(cy))), 4, (0, 255, 0), -1)
            cv2.putText(cc_vis, str(i + 1), (int(round(cx)) + 3, int(round(cy)) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, f"tile_{lbl}_rectified_checker.jpg"), cc_vis)

        vis = cv2.cvtColor(tile_rgb_u8, cv2.COLOR_RGB2BGR)
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 1)
        cv2.polylines(vis, [np.round(quad_tile).astype(np.int32)], True, (0, 200, 255), 2)
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
        swatches_linear=swatches_linear,
        swatch_centres_uv=swatch_centres_uv,
        swatch_centres_tile=swatch_centres_tile,
        quad_tile=quad_tile.astype(np.float32),
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



    """
    Find the ColorChecker's 4-corner polygon in the tile using contour detection.

    Works because in a rectilinear (gnomonic) projection, straight edges remain
    straight — this is precisely why we project to rectilinear before detecting.

    Tries multiple Canny thresholds to handle varying contrast/size.
    """
    gray    = cv2.cvtColor(tile_u8_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = tile_u8_bgr.shape[:2]
    min_area = w * h * 0.005   # checker can be as small as 0.5% of tile area
    max_area = w * h * 0.90

    best_poly  = None
    best_score = -1

    # Try multiple Canny threshold pairs — checker contrast varies a lot
    for lo, hi in [(20, 60), (30, 90), (50, 150), (10, 40)]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges   = cv2.Canny(blurred, lo, hi)
        edges   = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4:
                continue
            rect      = cv2.minAreaRect(approx)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area < 1:
                continue
            # Score: rectangularity × aspect ratio proximity to CC24 (6:4 = 1.5)
            rect_score = area / rect_area
            rw = max(rect[1][0], rect[1][1])
            rh = min(rect[1][0], rect[1][1])
            aspect = rw / max(rh, 1)
            aspect_score = 1.0 / (1.0 + abs(aspect - 1.5))
            score = rect_score * aspect_score
            if score > best_score:
                best_score = score
                best_poly  = approx.reshape(4, 2).astype(np.float32)

    return best_poly


def _order_polygon_tl_tr_br_bl(poly):
    """Reorder 4 corner points to top-left, top-right, bottom-right, bottom-left."""
    poly = np.array(poly, dtype=np.float32)
    centre = poly.mean(axis=0)
    angles = np.arctan2(poly[:,1] - centre[1], poly[:,0] - centre[0])
    idx = np.argsort(angles)
    ordered = poly[idx]
    # Find top-left: closest to (-inf, -inf)
    tl_idx = np.argmin(ordered[:,0] + ordered[:,1])
    ordered = np.roll(ordered, -tl_idx, axis=0)
    return ordered  # TL, TR, BR, BL  (image coords, y-down)


def _grid_from_polygon(poly):
    """
    Given the 4 corner polygon of the checker (TL, TR, BR, BL),
    compute the 24 swatch centre positions using perspective-correct
    bilinear interpolation (the checker is a planar quad).

    CC24 layout: 4 rows × 6 cols.
    Swatch centres are at (col+0.5)/6, (row+0.5)/4 in normalised quad coords.
    """
    poly = _order_polygon_tl_tr_br_bl(poly)
    tl, tr, br, bl = poly

    centres = []
    rows, cols = 4, 6
    for r in range(rows):
        for c in range(cols):
            # Normalised position within the quad
            s = (c + 0.5) / cols   # horizontal [0,1]
            t = (r + 0.5) / rows   # vertical   [0,1]
            # Bilinear interpolation over the quad
            top    = tl + s * (tr - tl)
            bottom = bl + s * (br - bl)
            pt     = top + t * (bottom - top)
            centres.append((float(pt[0]), float(pt[1])))
    return centres   # list of 24 (x,y) tuples


def _uniform_grid_fallback(out_w, out_h):
    """Last-resort: uniform 4×6 grid covering 60% of the tile."""
    margin_x = out_w * 0.2
    margin_y = out_h * 0.2
    centres = []
    for r in range(4):
        for c in range(6):
            x = margin_x + (c + 0.5) / 6 * (out_w - 2*margin_x)
            y = margin_y + (r + 0.5) / 4 * (out_h - 2*margin_y)
            centres.append((float(x), float(y)))
    return centres


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

    best: Optional[CheckerDetection] = None
    total_tiles = 0

    cc24_ref = get_cc24_reference(colorspace)
    print(f"[cc-erp] Colorspace: {colorspace}  "
          f"(CC24 reference patch 22: "
          f"R={cc24_ref[21,0]:.4f} G={cc24_ref[21,1]:.4f} B={cc24_ref[21,2]:.4f})")

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
            cc24_ref=cc24_ref)

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
        cc24_ref=cc24_ref)

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
        cc24_ref=cc24_ref)

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

    if debug_dir:
        _save_final_debug(debug_dir, best, erp_linear, cc24_ref=cc24_ref)

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
