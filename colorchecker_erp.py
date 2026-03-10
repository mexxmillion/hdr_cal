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

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False


# ─── CC24 reference (linear sRGB, D65) ───────────────────────────────────────
# Patch order: row-major, top-left = patch 1 (dark skin)
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
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black   N2
], dtype=np.float32)

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
                       pitch_values: tuple = (-50.0, -25.0, 0.0),
                       fov_deg: float = 70.0
                       ) -> List[Tuple[float, float]]:
    """
    Generate (yaw, pitch) pairs covering the lower 2/3 of the panorama.

    A ColorChecker on set is always near the horizon or below — on a tripod,
    on the floor, or held at chest height. It is never in the top third of the
    sky. Restricting to pitch ≤ 0° eliminates half the search space and avoids
    false positives on clouds / sun discs.

    pitch_values: elevation angles in degrees.
      0   = horizon
      -25 = slightly below horizon (tripod height)
      -50 = floor / low angle
    """
    yaws = np.arange(0, 360, yaw_step).tolist()
    return [(y, p) for y in yaws for p in pitch_values]


@dataclass
class CheckerDetection:
    """One detected ColorChecker instance."""
    swatches_linear: np.ndarray          # (24, 3) float32, ERP-sampled
    swatch_centres_uv: np.ndarray        # (24, 2) float32, ERP coords [0,1]
    tile_yaw: float
    tile_pitch: float
    checker_normal_world: np.ndarray     # (3,) unit vector, checker face direction
    checker_normal_theta_deg: float      # polar angle of checker face
    checker_normal_phi_deg: float        # azimuth of checker face
    confidence: float                    # 0–1, based on patch colour error
    raw_swatches_bgr: np.ndarray         # (24, 3) uint8, from detector


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


def _detect_in_tile(tile_linear: np.ndarray,
                    map_uv: np.ndarray,
                    erp_linear_hd: np.ndarray,
                    yaw: float, pitch: float,
                    debug_dir: Optional[str],
                    tile_idx: int) -> Optional[CheckerDetection]:
    """
    Run colour-checker-detection on one rectilinear tile.
    If found, back-project swatch centres to ERP and sample the full-res HDR.
    """
    if not HAVE_CCD:
        return None

    out_h, out_w = tile_linear.shape[:2]
    tile_u8_bgr = cv2.cvtColor(_linear_to_u8_for_detection(tile_linear),
                                cv2.COLOR_RGB2BGR)

    # Call WITHOUT additional_data so we always get plain NDArrayFloat results.
    # With additional_data=True the return type is DataDetectionColourChecker
    # which has no len() and whose internal structure changed across versions.
    # additional_data=False gives us (N,) tuple of (24,3) float arrays directly.
    try:
        results = ccd.detect_colour_checkers_segmentation(
            tile_u8_bgr, show=False, additional_data=False)
    except Exception:
        return None

    if not results:
        return None

    # Each element is a (24, 3) float array of swatch colours.
    # colour-checker-detection with additional_data=False returns values in
    # RGB order (colour-science convention) normalised to [0,1].
    # apply_cctf_decoding=False (default) means the values are NOT linearised —
    # they are raw uint8/255 in sRGB display encoding.
    swatches_float = results[0]   # (24, 3) float32, RGB, sRGB-encoded [0,1]

    if swatches_float is None or swatches_float.shape != (24, 3):
        return None

    # Linearise: input was uint8 sRGB (our tonemapped tile), so values are
    # sRGB-gamma encoded. Convert to linear RGB for all downstream math.
    # NO channel swap needed — library output is already RGB.
    def _srgb_to_linear(v):
        v = np.clip(v, 0.0, 1.0)
        return np.where(v <= 0.04045, v / 12.92,
                        ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)

    swatches_linear_tile = _srgb_to_linear(swatches_float)   # (24,3) linear RGB

    # ── Swatch centre estimation ─────────────────────────────────────────
    # We no longer have extra_data, so go straight to contour detection
    # on the tile to find the quad, then lay a 4×6 grid inside it.
    swatch_centres_px = _estimate_swatch_centres(
        None, None, tile_u8_bgr, out_w, out_h)

    # ── Back-project to ERP ──────────────────────────────────────────────
    swatch_centres_uv = np.array([
        backproject_pixel_to_erp(cx, cy, map_uv)
        for cx, cy in swatch_centres_px
    ], dtype=np.float32)   # (24, 2)

    # ── Sample full-resolution linear HDR at ERP coords ──────────────────
    # This is more accurate than the tile samples because it uses the
    # original full-res HDR, not the downsampled rectilinear tile.
    swatches_linear = np.array([
        sample_erp_bilinear(erp_linear_hd, float(uv[0]), float(uv[1]))
        for uv in swatch_centres_uv
    ], dtype=np.float32)   # (24, 3)

    # ── Pose estimation ──────────────────────────────────────────────────
    checker_normal_world = _estimate_checker_pose(
        swatch_centres_px, out_w, out_h, yaw, pitch)

    # ── Confidence score ─────────────────────────────────────────────────
    # Compare neutral patches (19–24) — after normalising to same luma as
    # reference, chroma error should be near zero for a real detection.
    neutrals_measured = swatches_linear[18:24]
    neutrals_ref      = CC24_LINEAR_SRGB[18:24]
    lum_meas = 0.2126*neutrals_measured[:,0] + 0.7152*neutrals_measured[:,1] + 0.0722*neutrals_measured[:,2]
    lum_ref  = 0.2126*neutrals_ref[:,0]      + 0.7152*neutrals_ref[:,1]      + 0.0722*neutrals_ref[:,2]
    scale_n  = np.clip(lum_ref / (lum_meas + 1e-8), 0, 10)
    neutrals_scaled = neutrals_measured * scale_n[:,None]
    chroma_err = float(np.mean(np.abs(neutrals_scaled - neutrals_ref)))
    confidence = float(np.clip(1.0 - chroma_err * 5.0, 0.0, 1.0))

    n = checker_normal_world
    theta_n = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    phi_n   = float(np.degrees(np.arctan2(n[0], n[2])))

    if debug_dir:
        _save_tile_debug(debug_dir, tile_idx, tile_u8_bgr, swatch_centres_px,
                         swatches_linear_tile, yaw, pitch, confidence)

    return CheckerDetection(
        swatches_linear=swatches_linear,
        swatch_centres_uv=swatch_centres_uv,
        tile_yaw=yaw,
        tile_pitch=pitch,
        checker_normal_world=checker_normal_world,
        checker_normal_theta_deg=theta_n,
        checker_normal_phi_deg=phi_n,
        confidence=confidence,
        raw_swatches_bgr=(swatches_linear_tile * 255).astype(np.uint8),  # RGB despite field name
    )


def _estimate_swatch_centres(swatches_bgr_raw, extra_data,
                              tile_u8_bgr, out_w, out_h):
    """
    Recover the 2D pixel location of each swatch centre in the tile.

    Strategy 1 (best): use extra_data polygon from detector if available.
    Strategy 2 (fallback): detect the checker bounding rectangle via contours
                            on a difference-from-neutral image, then lay a
                            4×6 grid of swatch centres inside it.
    """
    # Try to get polygon from extra_data
    if extra_data is not None:
        polygon = _extract_polygon_from_extra(extra_data, out_w, out_h)
        if polygon is not None:
            return _grid_from_polygon(polygon)

    # Fallback: find checker via contour detection on the tile itself
    polygon = _detect_checker_polygon(tile_u8_bgr)
    if polygon is not None:
        return _grid_from_polygon(polygon)

    # Last resort: uniform grid covering the full tile
    # (will give wrong colours but at least something)
    return _uniform_grid_fallback(out_w, out_h)


def _extract_polygon_from_extra(extra_data, out_w, out_h):
    """Extract the 4-corner polygon from colour-checker-detection extra data."""
    if extra_data is None:
        return None
    # extra_data format varies by library version; try common attributes
    for attr in ('quadrilateral', 'rectangle', 'contour', 'corners'):
        poly = getattr(extra_data, attr, None)
        if poly is not None:
            poly = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if len(poly) >= 4:
                return poly[:4]
    # Also try dict-style
    if isinstance(extra_data, dict):
        for key in ('quadrilateral', 'rectangle', 'corners'):
            if key in extra_data:
                poly = np.array(extra_data[key], dtype=np.float32).reshape(-1, 2)
                if len(poly) >= 4:
                    return poly[:4]
    return None


def _detect_checker_polygon(tile_u8_bgr):
    """
    Find the ColorChecker's 4-corner polygon by looking for the most
    rectangle-like large contour in the tile.

    Works because in a rectilinear projection, the checker's straight edges
    remain straight — this is precisely why gnomonic projection is required.
    """
    gray    = cv2.cvtColor(tile_u8_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges   = cv2.Canny(blurred, 30, 80)
    edges   = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = tile_u8_bgr.shape[:2]
    min_area = w * h * 0.02    # checker must be at least 2% of tile area
    max_area = w * h * 0.95

    best_poly = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        # Rectangularity score: area vs bounding box area
        rect = cv2.minAreaRect(approx)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area < 1:
            continue
        score = area / rect_area  # close to 1.0 for a clean rectangle
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

def _estimate_checker_pose(swatch_centres_px: list,
                           out_w: int, out_h: int,
                           yaw_deg: float, pitch_deg: float,
                           fov_deg: float = 70.0) -> np.ndarray:
    """
    Estimate the 3D normal of the checker plane using solvePnP.

    We know:
      - The 3D layout of the 24 patches in checker space (CC24_3D_POINTS)
      - The 2D pixel positions of those patch centres in the rectilinear tile
      - The camera intrinsics of the rectilinear tile (pinhole, known FOV)

    solvePnP gives us the rotation R and translation t of the checker
    relative to the tile's virtual camera.  The checker normal in camera
    space is simply R @ [0, 0, 1] (the checker lies in Z=0 plane, normal = Z+).

    We then rotate that normal from the tile's camera space into world
    (ERP) space using the tile's own yaw/pitch rotation.
    """
    if len(swatch_centres_px) < 6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, out_w/2],
                  [0, f, out_h/2],
                  [0, 0, 1     ]], dtype=np.float64)
    dist_coeffs = np.zeros(4)

    pts_2d = np.array(swatch_centres_px, dtype=np.float64)  # (24,2)
    pts_3d = CC24_3D_POINTS.astype(np.float64)               # (24,3)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    R_cam, _ = cv2.Rodrigues(rvec)   # (3,3): checker→camera rotation

    # Checker plane normal in camera space (+Z in checker space)
    normal_cam = R_cam[:, 2]         # column 2 = Z axis of checker in cam space
    normal_cam = normal_cam / (np.linalg.norm(normal_cam) + 1e-8)

    # Rotate from tile camera space → world (ERP) space.
    # The tile camera's own orientation is yaw then pitch (same as erp_to_rectilinear).
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [ 0,             1, 0            ],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0,                0            ],
                   [0,  math.cos(pitch), math.sin(pitch)],
                   [0, -math.sin(pitch), math.cos(pitch)]])
    R_tile_to_world = Ry @ Rx

    normal_world = R_tile_to_world @ normal_cam
    normal_world /= np.linalg.norm(normal_world) + 1e-8
    return normal_world.astype(np.float32)


# ─── Main entry point ─────────────────────────────────────────────────────────

def find_colorchecker_in_erp(
    erp_linear: np.ndarray,
    fov_deg: float = 70.0,
    tile_w: int = 900,
    tile_h: int = 675,
    yaw_step_deg: float = 40.0,
    pitch_values: tuple = (-50.0, -25.0, 0.0),
    min_confidence: float = 0.05,
    debug_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Sweep the lower 2/3 of the ERP panorama with overlapping rectilinear tiles.

    The chart is always near the horizon or below (tripod, floor) — never sky.
    Default pitch_values cover 0° (horizon) down to -50° (floor).

    ALL tiles are searched. The detection with the highest confidence score
    wins, regardless of order. min_confidence is a floor to reject obvious
    noise (set very low — 0.05 — so a dim or partially visible checker still
    wins over nothing).

    Returns:
        swatches_linear : (24, 3) float32 linear RGB, or None
        info            : dict with detection metadata
    """
    if not HAVE_CCD:
        return None, {"error": "colour-checker-detection not installed. "
                                "pip install colour-checker-detection"}

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    tiles = _build_sweep_tiles(yaw_step_deg, pitch_values, fov_deg)
    print(f"[cc-erp] Sweeping {len(tiles)} rectilinear tiles "
          f"(FOV={fov_deg}°, {tile_w}×{tile_h}, "
          f"pitches={[int(p) for p in pitch_values]}°) ...")

    best: Optional[CheckerDetection] = None
    all_detections = []

    for idx, (yaw, pitch) in enumerate(tiles):
        tile_linear, map_uv = erp_to_rectilinear(
            erp_linear, yaw, pitch, fov_deg, tile_w, tile_h)

        det = _detect_in_tile(
            tile_linear, map_uv, erp_linear,
            yaw, pitch, debug_dir, idx)

        if det is None:
            continue

        all_detections.append(det)
        print(f"[cc-erp]  tile {idx:02d} yaw={yaw:.0f}° pitch={pitch:.0f}°  "
              f"confidence={det.confidence:.3f}")

        # Always keep the best — search ALL tiles, no early exit
        if det.confidence >= min_confidence:
            if best is None or det.confidence > best.confidence:
                best = det

    if best is None:
        print("[cc-erp] No ColorChecker found after full sweep.")
        return None, {"found": False, "tiles_searched": len(tiles)}

    print(f"[cc-erp] Best detection: tile yaw={best.tile_yaw:.0f}° "
          f"pitch={best.tile_pitch:.0f}° confidence={best.confidence:.3f}")
    print(f"[cc-erp] Checker face direction: "
          f"θ={best.checker_normal_theta_deg:.1f}° "
          f"φ={best.checker_normal_phi_deg:.1f}°")

    if debug_dir:
        _save_final_debug(debug_dir, best, erp_linear)

    info = {
        "found":                    True,
        "tiles_searched":           len(tiles),
        "best_tile_yaw_deg":        best.tile_yaw,
        "best_tile_pitch_deg":      best.tile_pitch,
        "confidence":               best.confidence,
        "checker_normal_world":     best.checker_normal_world.tolist(),
        "checker_normal_theta_deg": best.checker_normal_theta_deg,
        "checker_normal_phi_deg":   best.checker_normal_phi_deg,
        "swatch_centres_uv":        best.swatch_centres_uv.tolist(),
    }
    return best.swatches_linear, info


# ─── Colour correction matrix ─────────────────────────────────────────────────

def solve_color_matrix_from_swatches(
    measured_linear: np.ndarray,
    reference_linear: Optional[np.ndarray] = None,
    use_neutral_only: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Solve a 3×3 colour correction matrix M such that measured @ M ≈ reference.
    Operates in linear RGB space.  Least-squares over all 24 patches.

    use_neutral_only: if True, solve only on the 6 neutral patches (19–24).
      Useful when the checker has been exposed to strongly coloured light —
      the neutrals give a purer white-balance signal.

    Returns: (M 3×3, RMSE)
    """
    if reference_linear is None:
        reference_linear = CC24_LINEAR_SRGB

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

def _save_tile_debug(debug_dir, tile_idx, tile_bgr, centres,
                     swatches_bgr, yaw, pitch, confidence):
    vis = tile_bgr.copy()
    for i, (cx, cy) in enumerate(centres):
        cv2.circle(vis, (int(cx), int(cy)), 8, (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (int(cx)+5, int(cy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
    cv2.putText(vis, f"yaw={yaw:.0f} pitch={pitch:.0f} conf={confidence:.2f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(os.path.join(debug_dir, f"tile_{tile_idx:03d}_detected.jpg"), vis)


def _save_final_debug(debug_dir, det: CheckerDetection, erp_linear: np.ndarray):
    """Mark swatch centres on a tonemapped ERP preview."""
    from colorchecker_erp import _linear_to_u8_for_detection
    erp_u8 = cv2.cvtColor(_linear_to_u8_for_detection(erp_linear), cv2.COLOR_RGB2BGR)
    h, w = erp_u8.shape[:2]
    vis = erp_u8.copy()
    for i, (u, v) in enumerate(det.swatch_centres_uv):
        px = int(u * w)
        py = int(v * h)
        cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
        cv2.putText(vis, str(i+1), (px+4, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
    # Draw checker normal direction
    n = det.checker_normal_world
    phi_n = float(np.arctan2(n[0], n[2]))
    theta_n = float(np.arccos(np.clip(n[1], -1, 1)))
    nu = int((phi_n/(2*np.pi) + 0.5) * w)
    nv = int(theta_n/np.pi * h)
    cv2.drawMarker(vis, (nu, nv), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(os.path.join(debug_dir, "cc_erp_swatches.jpg"), vis)

    # Swatch comparison strip
    sw = 40
    strip = np.zeros((sw*2 + 4, sw*24, 3), dtype=np.float32)
    for i in range(24):
        strip[:sw, i*sw:(i+1)*sw] = det.swatches_linear[i]
        strip[sw+4:, i*sw:(i+1)*sw] = CC24_LINEAR_SRGB[i]
    strip_u8 = _linear_to_u8_for_detection(strip)
    cv2.imwrite(os.path.join(debug_dir, "cc_swatch_comparison.jpg"),
                cv2.cvtColor(strip_u8, cv2.COLOR_RGB2BGR))
