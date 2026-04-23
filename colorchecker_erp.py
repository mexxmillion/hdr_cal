"""
colorchecker_erp.py — ColorChecker detection inside equirectangular HDR maps.

Strategy:
  1. Gnomonic (rectilinear) projection of overlapping tiles covering the pano.
     Gnomonic is the only projection that preserves straight lines, so rectangle
     detectors see the checker as a rectangle (not a sinusoid).
  2. Run YOLO (colour-checker-detection inference API) on each tile.
     YOLO is fed display-mapped u8 sRGB — exactly what it was trained on.
  3. Sample the 24 swatches directly from the high-res linear HDR tile using the
     detected quadrilateral (grid geometry), then back-project swatch centres
     through the gnomonic inverse to get ERP (u,v) coordinates for debug.
  4. solvePnP pose estimate → checker face normal in world space.

Usage:
  from colorchecker_erp import find_colorchecker_in_erp

  swatches_linear, pose_info = find_colorchecker_in_erp(
      erp_linear,          # (H, W, 3) float32 linear HDR in working colorspace
      debug_dir="debug",   # optional
  )
"""

import os
import math
import shutil
from dataclasses import dataclass
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

HAVE_YOLO = False
_YOLO_MODEL_PATH = None
if HAVE_CCD:
    try:
        _ = ccd.detect_colour_checkers_inference
        HAVE_YOLO = True
        _repo = os.environ.get(
            "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY",
            os.path.join(os.path.expanduser("~"), ".colour-science", "colour-checker-detection"),
        )
        _cand = os.path.join(_repo, "colour-checker-detection-l-seg.pt")
        _YOLO_MODEL_PATH = _cand if os.path.isfile(_cand) else None
    except AttributeError:
        pass

try:
    import colour
    HAVE_COLOUR = True
except ImportError:
    HAVE_COLOUR = False


def _log_backends():
    print(f"[cc-erp] colour-checker-detection: {'yes' if HAVE_CCD else 'NO'}")
    print(f"[cc-erp] YOLO inference API     : {'yes' if HAVE_YOLO else 'no (ultralytics missing)'}")
    if HAVE_YOLO and _YOLO_MODEL_PATH:
        sz = os.path.getsize(_YOLO_MODEL_PATH) / (1024*1024)
        print(f"[cc-erp] YOLO model: {_YOLO_MODEL_PATH} ({sz:.1f} MB)")
    elif HAVE_YOLO:
        print(f"[cc-erp] YOLO model: will download on first use")


# ─── Colorspace conversions ──────────────────────────────────────────────────

def srgb_linear_to_acescg(img: np.ndarray) -> np.ndarray:
    if not HAVE_COLOUR:
        raise RuntimeError("colour-science required for colorspace conversion")
    return colour.RGB_to_RGB(
        np.asarray(img, dtype=np.float32), "sRGB", "ACEScg",
        apply_cctf_decoding=False, apply_cctf_encoding=False,
    ).astype(np.float32)


def acescg_to_srgb_linear(img: np.ndarray) -> np.ndarray:
    if not HAVE_COLOUR:
        raise RuntimeError("colour-science required for colorspace conversion")
    return colour.RGB_to_RGB(
        np.asarray(img, dtype=np.float32), "ACEScg", "sRGB",
        apply_cctf_decoding=False, apply_cctf_encoding=False,
    ).astype(np.float32)


# ─── CC24 reference values (linear sRGB / Rec.709 primaries, D65) ────────────
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
    [0.9412, 0.9412, 0.9412],  # 19 white
    [0.6196, 0.6196, 0.6196],  # 20 neutral 8
    [0.3647, 0.3647, 0.3647],  # 21 neutral 6.5
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5  ← WB reference
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black
], dtype=np.float32)

# Neutrals are equal R=G=B so the sRGB values are valid in any RGB colorspace.
# Chromatic patches differ slightly in ACEScg vs sRGB but WB only uses neutrals.
CC24_LINEAR = CC24_LINEAR_SRGB


def get_cc24_reference(colorspace: str = "acescg") -> np.ndarray:
    cs = colorspace.lower().replace("-", "").replace("_", "")
    if cs in ("acescg", "aces", "ap1", "srgb", "rec709", "linear"):
        return CC24_LINEAR_SRGB
    raise ValueError(f"Unknown colorspace '{colorspace}'")


# Physical layout for pose estimation
_PATCH_W = 24.0
_PATCH_H = 24.0
_GAP = 6.0
_STEP_X = _PATCH_W + _GAP
_STEP_Y = _PATCH_H + _GAP

CC24_3D_POINTS = np.array([
    [c * _STEP_X, r * _STEP_Y, 0.0]
    for r in range(4) for c in range(6)
], dtype=np.float32)


# ─── Gnomonic (rectilinear) projection ───────────────────────────────────────

def erp_to_rectilinear(erp_img: np.ndarray,
                       yaw_deg: float, pitch_deg: float,
                       fov_deg: float = 70.0,
                       out_w: int = 800, out_h: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a rectilinear view from an ERP panorama via gnomonic projection.

    Returns (tile, map_uv) where map_uv is (out_h, out_w, 2) of ERP (u,v) ∈ [0,1]
    for each output pixel — used to back-project tile pixels to ERP coords.
    """
    h_erp, w_erp = erp_img.shape[:2]
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)

    py_grid, px_grid = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    px = px_grid - (out_w - 1) / 2.0
    py = py_grid - (out_h - 1) / 2.0

    ray_cam = np.stack([px / f, -py / f, np.ones_like(px)], axis=-1)

    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [ 0,             1, 0            ],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0,               0              ],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]], dtype=np.float32)
    R = Ry @ Rx

    rays = ray_cam.reshape(-1, 3) @ R.T
    rays /= (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-8)
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]

    phi = np.arctan2(x, z)
    theta = np.arccos(np.clip(y, -1.0, 1.0))
    u = (phi / (2.0 * np.pi) + 0.5)
    v = theta / np.pi

    map_uv = np.stack([u, v], axis=-1).reshape(out_h, out_w, 2).astype(np.float32)
    map_x = np.mod((u * (w_erp - 1)).reshape(out_h, out_w).astype(np.float32), w_erp)
    map_y = (v * (h_erp - 1)).reshape(out_h, out_w).astype(np.float32)

    tile = cv2.remap(erp_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    return tile, map_uv


def backproject_pixel_to_erp(px: float, py: float, map_uv: np.ndarray) -> Tuple[float, float]:
    out_h, out_w = map_uv.shape[:2]
    pxc = float(np.clip(px, 0, out_w - 1))
    pyc = float(np.clip(py, 0, out_h - 1))
    x0, y0 = int(pxc), int(pyc)
    x1, y1 = min(x0 + 1, out_w - 1), min(y0 + 1, out_h - 1)
    fx, fy = pxc - x0, pyc - y0
    uv = (map_uv[y0, x0] * (1-fx)*(1-fy) + map_uv[y0, x1] * fx*(1-fy) +
          map_uv[y1, x0] * (1-fx)*fy    + map_uv[y1, x1] * fx*fy)
    return float(uv[0]), float(uv[1])


def sample_erp_bilinear(erp: np.ndarray, u: float, v: float) -> np.ndarray:
    h, w = erp.shape[:2]
    px = (u % 1.0) * (w - 1)
    py = np.clip(v, 0.0, 1.0) * (h - 1)
    x0, y0 = int(px), int(py)
    x1 = (x0 + 1) % w
    y1 = min(y0 + 1, h - 1)
    fx, fy = px - x0, py - y0
    return (erp[y0, x0] * (1-fx)*(1-fy) + erp[y0, x1] * fx*(1-fy) +
            erp[y1, x0] * (1-fx)*fy    + erp[y1, x1] * fx*fy)


# ─── Display mapping for YOLO input ──────────────────────────────────────────

def _linear_to_u8_for_display(img_linear: np.ndarray) -> np.ndarray:
    """ACEScg linear HDR → sRGB u8 for YOLO input / debug display.
    Median luminance → 0.18 (like a camera meter), then sRGB gamma.
    """
    img = np.clip(np.asarray(img_linear, dtype=np.float32), 0.0, None)
    if HAVE_COLOUR:
        img = np.clip(acescg_to_srgb_linear(img), 0.0, None)
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    valid = lum[lum > 1e-6]
    scale = 0.18 / max(float(np.median(valid)), 1e-6) if valid.size else 1.0
    disp = np.clip(img * scale, 0.0, 1.0)
    srgb = np.where(disp <= 0.0031308,
                    disp * 12.92,
                    1.055 * np.power(np.clip(disp, 1e-9, None), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)

# Alias for external callers
_linear_to_u8_for_detection = _linear_to_u8_for_display


# ─── Detection result ────────────────────────────────────────────────────────

@dataclass
class CheckerDetection:
    swatches_linear: np.ndarray          # (24, 3) float32, in working colorspace
    swatch_centres_uv: np.ndarray        # (24, 2) float32, ERP coords [0,1]
    swatch_centres_tile: np.ndarray      # (24, 2) float32, tile pixel coords
    quad_tile: np.ndarray                # (4, 2) float32, tile pixel coords (TL,TR,BR,BL)
    quad_center_uv: np.ndarray           # (2,) float32
    tile_yaw: float
    tile_pitch: float
    tile_fov_deg: float
    checker_normal_world: np.ndarray
    checker_normal_theta_deg: float
    checker_normal_phi_deg: float
    confidence: float
    raw_swatches_bgr: np.ndarray
    detection_method: str = "yolo"
    stage_label: str = "coarse"
    cc_rectified: Optional[np.ndarray] = None  # rectified chart image (for debug)


# ─── Geometry helpers ────────────────────────────────────────────────────────

def _order_quad_tl_tr_br_bl(poly: np.ndarray) -> np.ndarray:
    q = np.array(poly, dtype=np.float32).reshape(4, 2)
    s = q.sum(axis=1)
    d = q[:, 0] - q[:, 1]
    return np.array([q[np.argmin(s)], q[np.argmin(d)],
                     q[np.argmax(s)], q[np.argmax(d)]], dtype=np.float32)


def _sample_swatches_at_centres(tile_linear: np.ndarray,
                                centres: np.ndarray,
                                quad: np.ndarray,
                                sample_frac: float = 0.30,
                                K: int = 5) -> np.ndarray:
    """Sample 24 swatches as K×K block averages around each centre.
    Block size scales with the local cell footprint derived from the quad.
    Returns (24, 3) float in tile colorspace.
    """
    tl, tr, br, bl = _order_quad_tl_tr_br_bl(quad)
    cell_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) * 0.5 / 6.0
    cell_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) * 0.5 / 4.0
    half_w = cell_w * sample_frac
    half_h = cell_h * sample_frac
    offs = np.linspace(-1.0, 1.0, K, dtype=np.float32)

    # Build sample grid per centre along quad-aligned axes
    ux = (tr - tl)
    ux /= max(float(np.linalg.norm(ux)), 1e-8)
    uy = (bl - tl)
    uy /= max(float(np.linalg.norm(uy)), 1e-8)

    pts_x = np.empty(24 * K * K, dtype=np.float32)
    pts_y = np.empty(24 * K * K, dtype=np.float32)
    idx = 0
    for cx, cy in centres:
        for oy in offs:
            for ox in offs:
                dx = ux[0] * (ox * half_w) + uy[0] * (oy * half_h)
                dy = ux[1] * (ox * half_w) + uy[1] * (oy * half_h)
                pts_x[idx] = cx + dx
                pts_y[idx] = cy + dy
                idx += 1

    samples = cv2.remap(tile_linear,
                        pts_x.reshape(1, -1), pts_y.reshape(1, -1),
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    samples = samples.reshape(24, K * K, 3)
    return samples.mean(axis=1).astype(np.float32)


def _compute_confidence(swatches_linear: np.ndarray,
                        cc24_ref: np.ndarray) -> float:
    """Score via neutral-ramp chroma agreement (illuminant-invariant)."""
    if swatches_linear.shape != (24, 3):
        return 0.0
    meas = swatches_linear[18:24]
    ref = cc24_ref[18:24]
    lm = 0.2126 * meas[:, 0] + 0.7152 * meas[:, 1] + 0.0722 * meas[:, 2]
    lr = 0.2126 * ref[:, 0] + 0.7152 * ref[:, 1] + 0.0722 * ref[:, 2]
    scale = np.clip(lr / (lm + 1e-8), 0.0, 20.0)[:, None]
    err = float(np.mean(np.abs(meas * scale - ref)))
    return float(np.clip(1.0 - err * 5.0, 0.0, 1.0))


# ─── Pose estimation ─────────────────────────────────────────────────────────

def _estimate_checker_pose(quad_px: np.ndarray,
                           out_w: int, out_h: int,
                           yaw_deg: float, pitch_deg: float,
                           fov_deg: float) -> np.ndarray:
    """solvePnP on 4 quad corners (TL,TR,BR,BL) → checker face normal in world."""
    total_w = 5 * (_PATCH_W + _GAP) + _PATCH_W + 2 * _GAP
    total_h = 3 * (_PATCH_H + _GAP) + _PATCH_H + 2 * _GAP
    pts_3d = np.array([[0, 0, 0], [total_w, 0, 0],
                       [total_w, total_h, 0], [0, total_h, 0]], dtype=np.float64)

    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, out_w / 2.0],
                  [0, f, out_h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    try:
        ok, rvec, _ = cv2.solvePnP(pts_3d, np.array(quad_px, dtype=np.float64),
                                   K, np.zeros(4), flags=cv2.SOLVEPNP_IPPE)
    except Exception:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if not ok:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    R_cam, _ = cv2.Rodrigues(rvec)
    n_cam = R_cam[:, 2]
    n_cam /= np.linalg.norm(n_cam) + 1e-8

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [ 0,             1, 0            ],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0,                 0             ],
                   [0,  math.cos(pitch),  math.sin(pitch)],
                   [0, -math.sin(pitch),  math.cos(pitch)]])
    n_world = (Ry @ Rx) @ n_cam
    n_world /= np.linalg.norm(n_world) + 1e-8
    return n_world.astype(np.float32)


# ─── Core per-tile detector (YOLO only) ──────────────────────────────────────

def _detect_in_tile(tile_linear: np.ndarray,
                    map_uv: np.ndarray,
                    yaw: float, pitch: float,
                    tile_fov_deg: float,
                    cc24_ref: np.ndarray,
                    debug_dir: Optional[str] = None,
                    tile_label: str = "tile",
                    save_debug: bool = False) -> Optional[CheckerDetection]:
    """Run YOLO on a single rectilinear tile. Returns CheckerDetection or None."""
    if not HAVE_YOLO:
        return None

    out_h, out_w = tile_linear.shape[:2]

    # Display-map the HDR tile exactly like a JPG would look — that's YOLO's
    # training distribution. Feed float32 [0,1] with apply_cctf_decoding=True.
    tile_u8 = _linear_to_u8_for_display(tile_linear)
    tile_f = tile_u8.astype(np.float32) / 255.0

    try:
        results = ccd.detect_colour_checkers_inference(
            tile_f, additional_data=True, apply_cctf_decoding=True)
    except Exception as e:
        print(f"[cc-erp] YOLO error on {tile_label}: {e}")
        return None

    if not results:
        return None

    det = results[0]

    # Normalize quad coords to tile pixels. Newer lib versions return [0,1]
    # normalized; older versions return pixels in the detector's working
    # resolution. Detect which by inspecting magnitude.
    quad_raw = np.array(det.quadrilateral, dtype=np.float32).reshape(4, 2)
    qmax = float(np.max(np.abs(quad_raw)))
    if qmax <= 2.0:
        quad_tile = quad_raw * np.array([out_w, out_h], dtype=np.float32)
    else:
        # Assume detector reformatted space — rescale by max-dim ratio.
        det_scale = qmax / max(float(out_w), float(out_h), 1.0)
        quad_tile = quad_raw / max(det_scale, 1e-6)
    quad_tile = _order_quad_tl_tr_br_bl(quad_tile)

    # Use the detector's per-swatch geometry (swatch_masks in rectified-chart
    # space) mapped back to tile pixels via the rectified→tile homography.
    # The library already returns masks in CC24 row-major order.
    try:
        cc_rectified = np.array(det.colour_checker, dtype=np.float32)
        masks = np.array(det.swatch_masks, dtype=np.float32)  # (24, 4) [y0,y1,x0,x1]
    except Exception as e:
        print(f"[cc-erp] {tile_label}: missing additional_data ({e})")
        return None

    if masks.shape != (24, 4) or cc_rectified.ndim != 3:
        return None

    H_cc, W_cc = cc_rectified.shape[:2]
    cx_rect = (masks[:, 2] + masks[:, 3]) * 0.5
    cy_rect = (masks[:, 0] + masks[:, 1]) * 0.5
    rect_corners = np.array([[0, 0], [W_cc, 0], [W_cc, H_cc], [0, H_cc]], dtype=np.float32)
    H_rect_to_tile, _ = cv2.findHomography(rect_corners, quad_tile)
    if H_rect_to_tile is None:
        return None

    pts = np.column_stack([cx_rect, cy_rect, np.ones(24, dtype=np.float32)])
    mapped = (H_rect_to_tile @ pts.T).T
    centres_tile = (mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-8, None)).astype(np.float32)

    # Sample HDR scene-linear values at the detector-provided centres.
    # (The lib's det.swatch_colours is in the tonemapped u8 input space, not
    # useful for HDR calibration — that's why we re-sample the linear tile.)
    swatches_wcs = _sample_swatches_at_centres(tile_linear, centres_tile, quad_tile)

    centres_uv = np.array([backproject_pixel_to_erp(cx, cy, map_uv)
                           for cx, cy in centres_tile], dtype=np.float32)

    confidence = _compute_confidence(swatches_wcs, cc24_ref)

    normal_world = _estimate_checker_pose(quad_tile, out_w, out_h,
                                          yaw, pitch, tile_fov_deg)
    theta_n = float(np.degrees(np.arccos(np.clip(normal_world[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(normal_world[0], normal_world[2])))

    qc = quad_tile.mean(axis=0)
    quad_center_uv = np.array(backproject_pixel_to_erp(float(qc[0]), float(qc[1]), map_uv),
                              dtype=np.float32)

    cc_rectified = None
    try:
        cc_rectified = np.array(det.colour_checker, dtype=np.float32)
    except Exception:
        pass

    if save_debug and debug_dir:
        vis = cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR)
        cv2.polylines(vis, [np.round(quad_tile).astype(np.int32)], True, (0, 200, 255), 2)
        for i, (cx, cy) in enumerate(centres_tile):
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i+1), (int(round(cx))+4, int(round(cy))-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
        cv2.putText(vis, f"{tile_label} yaw={yaw:.0f} pitch={pitch:.0f} conf={confidence:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, f"tile_{tile_label}_detected.jpg"), vis)

    return CheckerDetection(
        swatches_linear=swatches_wcs.astype(np.float32),
        swatch_centres_uv=centres_uv.astype(np.float32),
        swatch_centres_tile=centres_tile.astype(np.float32),
        quad_tile=quad_tile.astype(np.float32),
        quad_center_uv=quad_center_uv,
        tile_yaw=yaw, tile_pitch=pitch, tile_fov_deg=tile_fov_deg,
        checker_normal_world=normal_world,
        checker_normal_theta_deg=theta_n,
        checker_normal_phi_deg=phi_n,
        confidence=confidence,
        raw_swatches_bgr=np.clip(swatches_wcs * 255, 0, 255).astype(np.uint8),
        detection_method="yolo",
        stage_label=tile_label,
        cc_rectified=cc_rectified,
    )


# ─── Main entry point ────────────────────────────────────────────────────────

def find_colorchecker_in_erp(
    erp_linear: np.ndarray,
    colorspace: str = "acescg",
    debug_dir: Optional[str] = None,
    sweep_fov: float = 50.0,
    sweep_overlap: float = 10.0,
    sweep_min_pitch: float = -30.0,
    sweep_max_pitch: float = 90.0,
    tile_size: int = 1024,
    min_confidence: float = 0.05,
    early_exit_confidence: float = 0.35,
    # kept for API compatibility (no-ops in YOLO-only mode):
    read_backend: str = "yolo",
    compare_backends: bool = False,
    **_unused,
) -> Tuple[Optional[np.ndarray], dict]:
    """Find a CC24 inside an ERP panorama via YOLO on a gnomonic tile sweep.

    Returns (swatches_linear (24,3) in working colorspace, info dict) or (None, info).
    """
    if not HAVE_CCD:
        return None, {"error": "colour-checker-detection not installed",
                      "found": False, "tiles_searched": 0}
    if not HAVE_YOLO:
        return None, {"error": "YOLO inference API unavailable (install ultralytics)",
                      "found": False, "tiles_searched": 0}

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    _log_backends()
    cc24_ref = get_cc24_reference(colorspace)

    # Build sweep grid: nadir-first, 360° yaw, configurable pitch band.
    fov = float(np.clip(sweep_fov, 30.0, 140.0))
    step = max(10.0, fov - sweep_overlap)
    p_lo = float(np.clip(sweep_min_pitch, -90.0, 89.0))
    p_hi = float(np.clip(sweep_max_pitch, p_lo + 1.0, 90.0))

    pitches = []
    p = p_hi
    while p >= p_lo:
        pitches.append(p)
        p -= step
    if pitches[-1] > p_lo:
        pitches.append(p_lo)

    yaws = []
    y = 0.0
    while y < 360.0:
        yaws.append(y)
        y += step

    tiles = [(yy, pp) for pp in pitches for yy in yaws]
    print(f"[cc-erp] Sweep: {len(tiles)} tiles  fov={fov:.0f}° step={step:.0f}° "
          f"pitch=[{p_lo:.0f},{p_hi:.0f}]  yaws={len(yaws)} pitches={len(pitches)}")

    best: Optional[CheckerDetection] = None
    searched = 0

    for idx, (yaw, pitch) in enumerate(tiles):
        searched += 1
        tile, map_uv = erp_to_rectilinear(erp_linear, yaw, pitch, fov, tile_size, tile_size)
        label = f"sweep_{idx:03d}"

        if debug_dir:
            # Save raw sweep tile for visual inspection / for cc_debug re-testing.
            u8 = _linear_to_u8_for_display(tile)
            bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"sweep yaw={yaw:.0f} pitch={pitch:.0f} fov={fov:.0f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(debug_dir, f"sweep_y{int(yaw):03d}_p{int(pitch):+03d}.jpg"), bgr)

        det = _detect_in_tile(tile, map_uv, yaw, pitch, fov, cc24_ref,
                              debug_dir=debug_dir, tile_label=label,
                              save_debug=bool(debug_dir))
        if det is None or det.confidence < min_confidence:
            continue
        if best is None or det.confidence > best.confidence:
            best = det
            print(f"[cc-erp]  hit  tile={idx} yaw={yaw:.0f}° pitch={pitch:.0f}° "
                  f"conf={det.confidence:.3f}")
        # Early exit: a confident detection beats finishing the sweep.
        # The recenter passes will refine geometry anyway.
        if best.confidence >= early_exit_confidence:
            print(f"[cc-erp] Early exit at tile {idx}/{len(tiles)} "
                  f"(conf={best.confidence:.3f} >= {early_exit_confidence:.2f})")
            break

    if best is None:
        print(f"[cc-erp] No chart found after {searched} tiles")
        return None, {"found": False, "tiles_searched": searched}

    print(f"[cc-erp] Coarse best: yaw={best.tile_yaw:.0f}° pitch={best.tile_pitch:.0f}° "
          f"conf={best.confidence:.3f}")

    # Refinement passes: recenter on the detection and re-run at tighter FOV
    # to get a larger chart image for more accurate swatch sampling.
    def _uv_to_yp(uv: np.ndarray) -> Tuple[float, float]:
        u, v = float(uv[0]), float(uv[1])
        yp = float((((u - 0.5) * 360.0) + 180.0) % 360.0 - 180.0)
        pt = float(np.clip((0.5 - v) * 180.0, -85.0, 85.0))
        return yp, pt

    refined_stage = "coarse"
    ry, rp = _uv_to_yp(best.quad_center_uv)

    # Wide recenter at 90° FOV
    tile_w, mu_w = erp_to_rectilinear(erp_linear, ry, rp, 90.0, tile_size, tile_size)
    det_w = _detect_in_tile(tile_w, mu_w, ry, rp, 90.0, cc24_ref,
                            debug_dir=debug_dir, tile_label="recenter_wide",
                            save_debug=bool(debug_dir))
    if det_w is not None:
        best = det_w
        refined_stage = "recenter_wide"
        print(f"[cc-erp] Recentered wide: conf={det_w.confidence:.3f}")

    # Tight recenter: shrink FOV so the chart covers ~50% of the tile.
    q = best.quad_tile
    frac = max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))) / float(tile_size)
    tight_fov = float(np.clip(best.tile_fov_deg * frac / 0.50, 35.0, 70.0)) if frac > 1e-4 else 50.0
    ty, tp = _uv_to_yp(best.quad_center_uv)
    tile_t, mu_t = erp_to_rectilinear(erp_linear, ty, tp, tight_fov, tile_size, tile_size)
    det_t = _detect_in_tile(tile_t, mu_t, ty, tp, tight_fov, cc24_ref,
                            debug_dir=debug_dir, tile_label="recenter_tight",
                            save_debug=bool(debug_dir))
    if det_t is not None:
        best = det_t
        refined_stage = "recenter_tight"
        print(f"[cc-erp] Recentered tight fov={tight_fov:.0f}°: conf={det_t.confidence:.3f}")

    print(f"[cc-erp] Final: conf={best.confidence:.3f}  θ={best.checker_normal_theta_deg:.1f}° "
          f"φ={best.checker_normal_phi_deg:.1f}°")

    if debug_dir:
        _save_gui_debug(debug_dir, best, erp_linear, cc24_ref)

    info = {
        "found": True,
        "tiles_searched": searched,
        "refinement_stage": refined_stage,
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
        "read_backend": "yolo",
    }
    return best.swatches_linear, info


# ─── Colour correction matrix ────────────────────────────────────────────────

def solve_color_matrix_from_swatches(
    measured_linear: np.ndarray,
    reference_linear: Optional[np.ndarray] = None,
    use_neutral_only: bool = False,
    colorspace: str = "acescg",
) -> Tuple[np.ndarray, float]:
    """Solve 3×3 M such that measured @ M ≈ reference (least-squares)."""
    if reference_linear is None:
        reference_linear = get_cc24_reference(colorspace)
    A = measured_linear[18:] if use_neutral_only else measured_linear
    B = reference_linear[18:] if use_neutral_only else reference_linear
    M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    pred = measured_linear @ M
    rmse = float(np.sqrt(np.mean((pred - reference_linear) ** 2)))
    return M.astype(np.float32), rmse


def apply_color_matrix(img_linear: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = img_linear.shape[:2]
    corrected = img_linear.reshape(-1, 3).astype(np.float32) @ M
    return np.clip(corrected, 0.0, None).reshape(h, w, 3).astype(np.float32)


# ─── Debug images for the GUI ────────────────────────────────────────────────

def _save_gui_debug(debug_dir: str,
                    det: CheckerDetection,
                    erp_linear: np.ndarray,
                    cc24_ref: np.ndarray):
    """Write the three files the GUI expects:
      cc_detected_tile.jpg, cc_rectified_final.jpg, cc_swatch_comparison.jpg
    Plus cc_erp_swatches.jpg (ERP overlay, handy for diagnosis).
    """
    # 1. Detected tile — already saved by _detect_in_tile as tile_<stage>_detected.jpg
    src = os.path.join(debug_dir, f"tile_{det.stage_label}_detected.jpg")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(debug_dir, "cc_detected_tile.jpg"))

    # 2. Rectified chart image from YOLO
    if det.cc_rectified is not None:
        rect = np.clip(det.cc_rectified * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "cc_rectified_final.jpg"),
                    cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))

    # 3. ERP overlay with swatch positions
    erp_u8 = _linear_to_u8_for_display(erp_linear)
    erp_bgr = cv2.cvtColor(erp_u8, cv2.COLOR_RGB2BGR)
    h, w = erp_bgr.shape[:2]
    for i, (u, v) in enumerate(det.swatch_centres_uv):
        px = int(np.clip(u * w, 0, w - 1))
        py = int(np.clip(v * h, 0, h - 1))
        cv2.circle(erp_bgr, (px, py), 6, (0, 255, 0), -1)
        cv2.circle(erp_bgr, (px, py), 7, (0, 0, 0), 1)
        cv2.putText(erp_bgr, str(i+1), (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    n = det.checker_normal_world
    phi = float(np.arctan2(n[0], n[2]))
    th = float(np.arccos(np.clip(n[1], -1, 1)))
    nu = int(np.clip((phi / (2*np.pi) + 0.5) * w, 0, w-1))
    nv = int(np.clip(th / np.pi * h, 0, h-1))
    cv2.drawMarker(erp_bgr, (nu, nv), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.imwrite(os.path.join(debug_dir, "cc_erp_swatches.jpg"), erp_bgr)

    # 4. Swatch comparison strip: measured / reference / post-WB
    sw = 48
    gap = 2
    n_rows = 3
    strip_h = sw * n_rows + gap * (n_rows - 1)
    p22 = det.swatches_linear[21]
    wb = (cc24_ref[21] / np.clip(p22, 1e-8, None)).astype(np.float32)

    strip_lin = np.zeros((strip_h, sw * 24, 3), dtype=np.float32)
    for i in range(24):
        m = det.swatches_linear[i]
        r0 = 0
        r1 = sw + gap
        r2 = (sw + gap) * 2
        strip_lin[r0:r0+sw, i*sw:(i+1)*sw] = m
        strip_lin[r1:r1+sw, i*sw:(i+1)*sw] = cc24_ref[i]
        strip_lin[r2:r2+sw, i*sw:(i+1)*sw] = m * wb

    strip_u8 = _linear_to_u8_for_display(strip_lin)
    strip_bgr = cv2.cvtColor(strip_u8, cv2.COLOR_RGB2BGR)
    for i in range(24):
        x = i * sw
        cv2.putText(strip_bgr, str(i+1), (x+2, sw-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        cv2.putText(strip_bgr, "M", (x+2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        cv2.putText(strip_bgr, "R", (x+2, sw+gap+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        cv2.putText(strip_bgr, "W", (x+2, (sw+gap)*2+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
    p22_x = 21 * sw
    cv2.rectangle(strip_bgr, (p22_x, 0), (p22_x+sw-1, strip_h-1), (0, 0, 255), 2)
    cv2.putText(strip_bgr, "#22 WB", (p22_x, strip_h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_dir, "cc_swatch_comparison.jpg"), strip_bgr)

    # Per-patch console printout (handy for debugging)
    print(f"[cc-erp] ── Swatch values (linear, working colorspace) ──")
    print(f"[cc-erp]   {'#':>3}  {'R':>8}  {'G':>8}  {'B':>8}  {'luma':>8}  note")
    for i in range(24):
        v = det.swatches_linear[i]
        luma = 0.2126*v[0] + 0.7152*v[1] + 0.0722*v[2]
        note = "  ← PATCH 22 (WB ref)" if i == 21 else ("  ← neutral" if i >= 18 else "")
        print(f"[cc-erp]   {i+1:>3}  {v[0]:>8.5f}  {v[1]:>8.5f}  {v[2]:>8.5f}  {luma:>8.5f}{note}")

    scale = cc24_ref[21] / np.clip(p22, 1e-8, None)
    sgn = scale / max(float(scale[1]), 1e-8)
    print(f"[cc-erp] Patch22 measured: R={p22[0]:.5f} G={p22[1]:.5f} B={p22[2]:.5f}")
    print(f"[cc-erp] WB scale (ref/meas): R={scale[0]:.5f} G={scale[1]:.5f} B={scale[2]:.5f}")
    print(f"[cc-erp] G-normalised WB  : R={sgn[0]:.5f} G={sgn[1]:.5f} B={sgn[2]:.5f}")
