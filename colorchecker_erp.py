"""
colorchecker_erp.py — ColorChecker detection inside equirectangular HDR maps.

Strategy:
  1. The user draws a search rectangle (UV bbox) on the latlong preview.
  2. Gnomonic (rectilinear) projection of that single region.
  3. YOLO + OpenCV rectification on the rectilinear tile.
     If auto-detect fails, the user places 4 corners on the same tile.
  4. Sample 24 swatches from the high-res linear HDR via perspective warp.

Usage:
  from colorchecker_erp import find_colorchecker_in_rect, find_colorchecker_manual
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

# Lazy-loaded ultralytics model — loaded once per process, reused across all tiles.
# Avoids the ~15 s subprocess-per-tile overhead from ccd.detect_colour_checkers_inference.
_YOLO_MODEL = None


def _get_yolo_model():
    """Return the cached ultralytics YOLO model, loading it on first call."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if not _YOLO_MODEL_PATH:
        return None
    try:
        from ultralytics import YOLO as _YOLO
        _YOLO_MODEL = _YOLO(_YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[cc-erp] YOLO direct-load failed: {e}")
    return _YOLO_MODEL


def _yolo_detect(image_u8_rgb: np.ndarray, conf: float = 0.25) -> list:
    """Run YOLO directly (no subprocess). Returns [(conf, cls_id, mask_np), ...]."""
    model = _get_yolo_model()
    if model is None:
        return []
    # ultralytics expects BGR numpy arrays (OpenCV convention)
    image_bgr = cv2.cvtColor(image_u8_rgb, cv2.COLOR_RGB2BGR)
    raw = model(image_bgr, conf=conf, verbose=False, imgsz=1280)
    if not raw or raw[0].masks is None:
        return []
    results = []
    for i in range(len(raw[0].boxes)):
        c = float(raw[0].boxes.conf[i])
        cls = int(raw[0].boxes.cls[i])
        m = raw[0].masks.data[i].cpu().numpy().astype(np.float32)
        results.append((c, cls, m))
    return results

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
        print(f"[cc-erp] YOLO model: {_YOLO_MODEL_PATH} ({sz:.1f} MB)"
              f"  [direct ultralytics — model loads once per process]")
    elif HAVE_YOLO:
        print(f"[cc-erp] YOLO model: will download on first use")


# ─── Colorspace conversions ──────────────────────────────────────────────────

# Precomputed 3×3 matrices from colour.RGB_to_RGB (sRGB↔ACEScg, D65 reference
# with Bradford CAT to D60 for ACEScg). Direct matmul is ~100× faster than
# colour.RGB_to_RGB which re-looks-up colorspaces and runs validation each call.
_M_SRGB_TO_ACESCG = np.array([
    [ 0.6131324221, 0.3395947780, 0.0472728000],
    [ 0.0701243812, 0.9163940113, 0.0134816074],
    [ 0.0205876659, 0.1095745646, 0.8698377695],
], dtype=np.float32)
_M_ACESCG_TO_SRGB = np.linalg.inv(_M_SRGB_TO_ACESCG).astype(np.float32)


def srgb_linear_to_acescg(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return (a.reshape(-1, 3) @ _M_SRGB_TO_ACESCG.T).reshape(a.shape).astype(np.float32)


def acescg_to_srgb_linear(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return (a.reshape(-1, 3) @ _M_ACESCG_TO_SRGB.T).reshape(a.shape).astype(np.float32)


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


def _compute_confidence(swatches_linear: np.ndarray,
                        cc24_ref: np.ndarray,
                        quad_tile: Optional[np.ndarray] = None) -> Tuple[float, dict]:
    """Score a detection via geometry + neutral-ramp sanity.

    A partial-chart YOLO hit produces a plausible-looking quad but maps the
    rectified neutrals onto wrong tile regions — so the neutrals come back
    non-monotonic or chromatic. Multiple gates catch that:

      1. Quad aspect ratio vs CC24's 1.5:1 (hard gate).
      2. Neutral luminance monotonicity (hard gate).
      3. Neutral chroma error after grey-card WB (score).

    Returns (confidence in [0,1], diag dict). confidence=0 means a hard gate
    failed and the detection should be discarded.
    """
    if swatches_linear.shape != (24, 3):
        return 0.0, {"reason": "wrong shape"}

    diag = {}

    # Aspect gate is now done pre-sampling in _detect_in_tile (orientation
    # invariant). Record it for diagnostics only.
    if quad_tile is not None and quad_tile.shape == (4, 2):
        s01 = float(np.linalg.norm(quad_tile[1] - quad_tile[0]))
        s12 = float(np.linalg.norm(quad_tile[2] - quad_tile[1]))
        s23 = float(np.linalg.norm(quad_tile[3] - quad_tile[2]))
        s30 = float(np.linalg.norm(quad_tile[0] - quad_tile[3]))
        pair_a = 0.5 * (s01 + s23)
        pair_b = 0.5 * (s12 + s30)
        diag["aspect"] = max(pair_a, pair_b) / max(min(pair_a, pair_b), 1e-6)

    # Gate 2: the 6 neutral patches should correlate strongly with the
    # reference neutral ramp (white -> black). Use Pearson correlation on
    # log-luma — robust to HDR clipping and sample noise while still catching
    # mappings that landed on wrong tiles (random colors don't correlate).
    meas = swatches_linear[18:24]
    ref = cc24_ref[18:24]
    lm = 0.2126 * meas[:, 0] + 0.7152 * meas[:, 1] + 0.0722 * meas[:, 2]
    lr = 0.2126 * ref[:, 0] + 0.7152 * ref[:, 1] + 0.0722 * ref[:, 2]
    lm_log = np.log(np.clip(lm, 1e-6, None))
    lr_log = np.log(np.clip(lr, 1e-6, None))
    lm_c = lm_log - lm_log.mean()
    lr_c = lr_log - lr_log.mean()
    denom = float(np.sqrt((lm_c * lm_c).sum() * (lr_c * lr_c).sum()))
    corr = float((lm_c * lr_c).sum() / denom) if denom > 1e-8 else 0.0
    diag["neutral_lum"] = [float(x) for x in lm]
    diag["neutral_corr"] = corr
    if corr < 0.80:
        return 0.0, {**diag,
                     "reason": f"neutral ramp correlation {corr:.2f} < 0.80"}

    # Score: neutral chroma agreement after per-row grey-card scale.
    scale = np.clip(lr / (lm + 1e-8), 0.0, 20.0)[:, None]
    err = float(np.mean(np.abs(meas * scale - ref)))
    score = float(np.clip(1.0 - err * 5.0, 0.0, 1.0))
    # Correlation between 0.80 and 0.95 gets a proportional penalty.
    score *= float(np.clip((corr - 0.80) / 0.15, 0.0, 1.0)) * 0.3 + 0.7
    diag["chroma_err"] = err
    diag["score"] = score
    return score, diag


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


# ─── Core per-tile detector (two-pass YOLO + OpenCV rectification) ───────────
#
# Strategy:
#   Pass 1 — YOLO on the full tile at low conf → rough bbox (just location).
#   Crop + zoom so the chart fills the frame.
#   Pass 2 — YOLO on the zoomed crop → segmentation mask.
#   Mask contour → minAreaRect → 4 corners (CC24 is rectangular).
#   Try all 4 rotations of the quad, warp HDR data, sample fixed 6×4 grid,
#   pick the rotation with best match to CC24 reference (same as the library).

# CC24 rectified target: 6 cols × 4 rows, aspect 1.5:1
_RECT_W = 600
_RECT_H = 400
_CC_COLS = 6
_CC_ROWS = 4
_SWATCH_SAMPLES = 16  # half-width of sampling region in rectified space (fallback)


def _pick_rect_dims(quad: np.ndarray) -> Tuple[int, int, int]:
    """Choose rectified (w, h, swatch_half_size) based on the chart's
    actual size in source pixels.

    The rectified target follows the source quad so we don't over- or
    under-sample swatches relative to the input resolution. Swatch half-size
    is ~1/3 of a single cell, clamped to a sane range.
    """
    span_x = float(max(np.ptp(quad[:, 0]), 1.0))
    span_y = float(max(np.ptp(quad[:, 1]), 1.0))
    # Use 1.5:1 aspect anchored to the larger source span.
    long_src = max(span_x, span_y)
    rect_w = int(np.clip(round(long_src), 240, 1200))
    rect_h = int(round(rect_w / 1.5))
    cell = min(rect_w / _CC_COLS, rect_h / _CC_ROWS)
    samples = int(np.clip(round(cell / 3.0), 5, 25))
    return rect_w, rect_h, samples


def _cc24_swatch_masks(w: int, h: int, cols: int, rows: int,
                       samples: int) -> np.ndarray:
    """Build (24, 4) swatch masks [y0, y1, x0, x1] for a cols×rows grid."""
    masks = []
    off_x = w / cols / 2
    off_y = h / rows / 2
    for j in np.linspace(off_y, h - off_y, rows):
        for i in np.linspace(off_x, w - off_x, cols):
            masks.append([int(j - samples), int(j + samples),
                          int(i - samples), int(i + samples)])
    return np.array(masks, dtype=np.int32)


def _quad_from_mask(mask: np.ndarray, img_u8_rgb: np.ndarray
                    ) -> Optional[np.ndarray]:
    """Extract an oriented rectangle from Canny edges within the YOLO mask.
    Returns (4, 2) float32 ordered TL, TR, BR, BL or None."""
    h, w = img_u8_rgb.shape[:2]
    gray = cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2GRAY)

    # Resize mask, threshold, dilate slightly to include chart border
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask_full > 0.3).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dilated = cv2.dilate(mask_bin, kernel)

    # Canny on the image, masked to YOLO region
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, mask_dilated)

    # Erode the mask to find inner edges only (chart boundary, not background)
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask_bin, kernel_sm)
    # Keep edges near the mask border (within dilated but outside eroded)
    border_band = cv2.subtract(mask_dilated, mask_eroded)
    edges_border = cv2.bitwise_and(edges, border_band)

    # Use border edges if enough, otherwise fall back to all masked edges
    contours_b, _ = cv2.findContours(edges_border, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_SIMPLE)
    all_pts_b = np.vstack(contours_b) if contours_b else np.empty((0, 1, 2), np.int32)

    contours_a, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_SIMPLE)
    all_pts_a = np.vstack(contours_a) if contours_a else np.empty((0, 1, 2), np.int32)

    pts = all_pts_b if len(all_pts_b) >= 50 else all_pts_a
    if len(pts) < 10:
        return None

    mar = cv2.minAreaRect(pts)
    ww, hh = mar[1]
    if min(ww, hh) < 1:
        return None
    aspect = max(ww, hh) / min(ww, hh)
    print(f"[cc-erp] quad_from_mask: aspect={aspect:.2f} "
          f"border_pts={len(all_pts_b)} all_pts={len(all_pts_a)}")
    if not (1.1 < aspect < 2.5):
        return None
    box = cv2.boxPoints(mar).astype(np.float32)

    # Refine corners with subpixel accuracy
    ordered = _order_quad_tl_tr_br_bl(box)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    refined = cv2.cornerSubPix(gray, ordered.reshape(-1, 1, 2),
                               (11, 11), (-1, -1), criteria)
    return refined.reshape(4, 2)


def _rectify_and_sample(quad: np.ndarray, hdr_img: np.ndarray,
                        rect_w: int, rect_h: int,
                        masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Warp hdr_img using quad → rect, sample swatches.
    Returns (swatches (24,3), rectified_hdr)."""
    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    rect_hdr = cv2.warpPerspective(hdr_img, M, (rect_w, rect_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    swatches = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        y0 = max(0, y0); x0 = max(0, x0)
        y1 = min(rect_h, y1); x1 = min(rect_w, x1)
        if y1 > y0 and x1 > x0:
            swatches[i] = rect_hdr[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return swatches, rect_hdr


def _best_rotation(quad: np.ndarray, hdr_img: np.ndarray,
                   cc24_ref: np.ndarray, rect_w: int, rect_h: int,
                   masks: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Try all 4 rotations of quad, return (best_quad, swatches, rectified, score)."""
    best_score = -1.0
    best_quad = quad
    best_sw = np.zeros((24, 3), dtype=np.float32)
    best_rect = np.zeros((rect_h, rect_w, 3), dtype=np.float32)

    for rot in range(4):
        q = np.roll(quad, rot, axis=0)
        sw, rect = _rectify_and_sample(q, hdr_img, rect_w, rect_h, masks)
        # Score: try forward and reversed swatch order
        for swatches in (sw, sw[::-1]):
            lum_meas = swatches[:, 1]  # green channel as luminance proxy
            lum_ref = cc24_ref[:, 1]
            if np.std(lum_meas) < 1e-6 or np.std(lum_ref) < 1e-6:
                continue
            corr = float(np.corrcoef(lum_meas, lum_ref)[0, 1])
            if corr > best_score:
                best_score = corr
                best_quad = q
                best_sw = swatches
                best_rect = rect

    return best_quad, best_sw, best_rect, best_score


def _finalize_detection_from_quad(
        quad_tile: np.ndarray,
        tile_linear: np.ndarray,
        map_uv: np.ndarray,
        yaw: float, pitch: float, tile_fov_deg: float,
        cc24_ref: np.ndarray,
        detection_method: str,
        tile_label: str,
        debug_dir: Optional[str] = None) -> Optional[CheckerDetection]:
    """Shared post-processing: given a quadrilateral in tile pixel space,
    sample swatches from the LINEAR HDR, score confidence, compute pose,
    backproject swatch centres to ERP UV, return a CheckerDetection.

    Reused by both the segmentation and YOLO detectors so the downstream
    swatch/pose/confidence behaviour is identical regardless of how the
    quad was found.
    """
    out_h, out_w = tile_linear.shape[:2]

    rect_w, rect_h, samples = _pick_rect_dims(quad_tile)
    masks = _cc24_swatch_masks(rect_w, rect_h, _CC_COLS, _CC_ROWS, samples)
    quad_ordered, swatches_wcs, cc_rectified, rot_corr = _best_rotation(
        quad_tile, tile_linear, cc24_ref, rect_w, rect_h, masks)

    confidence, conf_diag = _compute_confidence(swatches_wcs, cc24_ref,
                                                quad_ordered)
    if confidence <= 0.0:
        print(f"[cc-erp] {tile_label}: {detection_method} rejected conf=0 — "
              f"{conf_diag.get('reason', 'score=0')}  "
              f"aspect={conf_diag.get('aspect', 0):.2f} "
              f"corr={conf_diag.get('neutral_corr', 0):.2f}")
        return None

    # Backproject swatch centres in rectified space → tile px → ERP UV.
    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]],
                   dtype=np.float32)
    M_fwd = cv2.getPerspectiveTransform(quad_ordered, dst)
    try:
        M_inv = np.linalg.inv(M_fwd)
    except np.linalg.LinAlgError:
        print(f"[cc-erp] {tile_label}: singular perspective matrix")
        return None
    centres_rect = np.zeros((24, 2), dtype=np.float32)
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        centres_rect[i] = [(x0 + x1) * 0.5, (y0 + y1) * 0.5]
    pts_h = np.column_stack([centres_rect[:, 0], centres_rect[:, 1],
                             np.ones(24, np.float32)])
    mapped = (M_inv @ pts_h.T).T
    centres_tile = (mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-8, None))
    centres_uv = np.array(
        [backproject_pixel_to_erp(cx, cy, map_uv) for cx, cy in centres_tile],
        dtype=np.float32)

    # Pose
    quad_tlbr = _order_quad_tl_tr_br_bl(quad_ordered)
    normal_world = _estimate_checker_pose(quad_tlbr, out_w, out_h,
                                          yaw, pitch, tile_fov_deg)
    if np.any(np.isnan(normal_world)):
        normal_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    theta_n = float(np.degrees(np.arccos(np.clip(normal_world[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(normal_world[0], normal_world[2])))

    qc = quad_tlbr.mean(axis=0)
    quad_center_uv = np.array(
        backproject_pixel_to_erp(float(qc[0]), float(qc[1]), map_uv),
        dtype=np.float32)

    if debug_dir:
        u8 = _linear_to_u8_for_display(tile_linear)
        vis = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.polylines(vis, [np.round(quad_tlbr).astype(np.int32)], True,
                      (0, 200, 255), 2)
        for cx, cy in centres_tile:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 4,
                       (0, 255, 0), -1)
        cv2.putText(vis,
                    f"{tile_label} {detection_method} conf={confidence:.2f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir,
                                 f"{tile_label}_{detection_method}_quad.jpg"),
                    vis)

    print(f"[cc-erp] {tile_label}: {detection_method} HIT  "
          f"conf={confidence:.3f}  rot_corr={rot_corr:.2f}  "
          f"aspect={conf_diag.get('aspect', 0):.2f}")

    return CheckerDetection(
        swatches_linear=swatches_wcs.astype(np.float32),
        swatch_centres_uv=centres_uv.astype(np.float32),
        swatch_centres_tile=centres_tile.astype(np.float32),
        quad_tile=quad_tlbr.astype(np.float32),
        quad_center_uv=quad_center_uv,
        tile_yaw=yaw, tile_pitch=pitch, tile_fov_deg=tile_fov_deg,
        checker_normal_world=normal_world,
        checker_normal_theta_deg=theta_n,
        checker_normal_phi_deg=phi_n,
        confidence=confidence,
        raw_swatches_bgr=np.clip(swatches_wcs * 255, 0, 255).astype(np.uint8),
        detection_method=detection_method,
        stage_label=tile_label,
        cc_rectified=cc_rectified,
    )


def _detect_in_tile_segmentation(tile_linear: np.ndarray,
                                 map_uv: np.ndarray,
                                 yaw: float, pitch: float,
                                 tile_fov_deg: float,
                                 cc24_ref: np.ndarray,
                                 debug_dir: Optional[str] = None,
                                 tile_label: str = "tile",
                                 ) -> Optional[CheckerDetection]:
    """Classical-CV (colour-checker-detection segmentation) chart finder.
    Fast, no model load. Tried first; YOLO is the fallback."""
    if not HAVE_CCD:
        return None
    try:
        import colour_checker_detection as _ccd
        seg_fn = _ccd.detect_colour_checkers_segmentation
    except Exception as e:
        print(f"[cc-erp] {tile_label}: segmentation unavailable: {e}")
        return None

    tile_u8 = _linear_to_u8_for_display(tile_linear)
    img_f = tile_u8.astype(np.float32) / 255.0
    try:
        results = seg_fn(img_f, additional_data=True, apply_cctf_decoding=True)
    except Exception as e:
        print(f"[cc-erp] {tile_label}: segmentation raised: {e}")
        return None

    print(f"[cc-erp] {tile_label}: segmentation hits={len(results)}")
    if not results:
        return None

    h, w = tile_linear.shape[:2]
    best: Optional[CheckerDetection] = None
    for i, d in enumerate(results):
        try:
            quad_norm = np.asarray(d.quadrilateral, dtype=np.float32)
        except Exception:
            continue
        if quad_norm.shape != (4, 2):
            continue
        # ccd returns normalised [0,1] coords — scale to tile pixels.
        if quad_norm.max() <= 1.5:
            quad_px = quad_norm * np.array([w, h], dtype=np.float32)
        else:
            quad_px = quad_norm
        det = _finalize_detection_from_quad(
            quad_px, tile_linear, map_uv, yaw, pitch, tile_fov_deg,
            cc24_ref, detection_method="segmentation",
            tile_label=f"{tile_label}_seg{i}", debug_dir=debug_dir)
        if det is not None and (best is None or det.confidence > best.confidence):
            best = det
    return best


def _detect_in_tile(tile_linear: np.ndarray,
                    map_uv: np.ndarray,
                    yaw: float, pitch: float,
                    tile_fov_deg: float,
                    cc24_ref: np.ndarray,
                    debug_dir: Optional[str] = None,
                    tile_label: str = "tile",
                    save_debug: bool = False) -> Optional[CheckerDetection]:
    """Two-pass YOLO + OpenCV-native rectification (no CCD extractor)."""
    if not HAVE_YOLO:
        return None

    import time as _time
    _t = {}
    _t0 = _time.perf_counter()

    out_h, out_w = tile_linear.shape[:2]
    tile_u8 = _linear_to_u8_for_display(tile_linear)
    _t["display_map"] = _time.perf_counter() - _t0; _tm = _time.perf_counter()

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{tile_label}_yolo_input.jpg"),
                    cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR))

    # ── Pass 1: locate at low confidence ──────────────────────────────────────
    hits1 = _yolo_detect(tile_u8, conf=0.10)
    _t["yolo_p1"] = _time.perf_counter() - _tm; _tm = _time.perf_counter()
    print(f"[cc-erp] {tile_label}: pass-1 hits={len(hits1)} "
          f"top_conf={max([h[0] for h in hits1], default=0.0):.3f}")
    if not hits1:
        return None

    p1_conf, _, p1_mask = max(hits1, key=lambda x: x[0])
    mask_tile = cv2.resize(p1_mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(mask_tile > 0.5)
    if len(xs) == 0:
        return None
    bx0, bx1 = int(xs.min()), int(xs.max())
    by0, by1 = int(ys.min()), int(ys.max())
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 4 or bh < 4:
        return None

    # ── Crop + zoom so the chart fills the frame ─────────────────────────────
    pad_x = max(int(bw * 0.6), 20)
    pad_y = max(int(bh * 0.6), 20)
    cx0 = max(0, bx0 - pad_x)
    cy0 = max(0, by0 - pad_y)
    cx1 = min(out_w, bx1 + pad_x)
    cy1 = min(out_h, by1 + pad_y)

    crop_u8 = tile_u8[cy0:cy1, cx0:cx1]
    crop_linear = tile_linear[cy0:cy1, cx0:cx1]
    crop_h, crop_w = crop_u8.shape[:2]
    zoom_scale = 1024.0 / max(crop_h, crop_w, 1)
    zh = max(1, int(round(crop_h * zoom_scale)))
    zw = max(1, int(round(crop_w * zoom_scale)))
    zoom_u8 = cv2.resize(crop_u8, (zw, zh), interpolation=cv2.INTER_LINEAR)
    zoom_linear = cv2.resize(crop_linear, (zw, zh), interpolation=cv2.INTER_LINEAR)

    # ── Pass 2: YOLO on zoomed crop → mask → oriented rect corners ───────────
    hits2 = _yolo_detect(zoom_u8, conf=0.10)
    _t["yolo_p2"] = _time.perf_counter() - _tm; _tm = _time.perf_counter()

    if not hits2:
        print(f"[cc-erp] {tile_label}: p1 conf={p1_conf:.2f} — pass-2 no hits")
        return None

    best_conf, _, best_mask = max(hits2, key=lambda x: x[0])
    quad_zoom = _quad_from_mask(best_mask, zoom_u8)
    if quad_zoom is None:
        print(f"[cc-erp] {tile_label}: p1 conf={p1_conf:.2f} — "
              f"mask→rect failed")
        return None

    _t["quad"] = _time.perf_counter() - _tm; _tm = _time.perf_counter()

    # ── Rectify + sample all 4 rotations, pick best match ────────────────────
    rect_w, rect_h, samples = _pick_rect_dims(quad_zoom)
    masks = _cc24_swatch_masks(rect_w, rect_h, _CC_COLS, _CC_ROWS, samples)
    quad_zoom, swatches_wcs, cc_rectified, rot_corr = _best_rotation(
        quad_zoom, zoom_linear, cc24_ref, rect_w, rect_h, masks)

    _t["rectify"] = _time.perf_counter() - _tm; _tm = _time.perf_counter()

    # ── Confidence scoring ───────────────────────────────────────────────────
    confidence, conf_diag = _compute_confidence(swatches_wcs, cc24_ref, quad_zoom)

    if debug_dir:
        dbg_zoom = zoom_u8.copy()
        pts = quad_zoom.astype(np.int32)
        for j in range(4):
            cv2.line(dbg_zoom, tuple(pts[j]), tuple(pts[(j+1)%4]),
                     (0, 255, 0), 2)
            cv2.putText(dbg_zoom, str(j), tuple(pts[j]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{tile_label}_zoom_quad.png"),
                    cv2.cvtColor(dbg_zoom, cv2.COLOR_RGB2BGR))
        rect_u8 = np.clip(cc_rectified * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"{tile_label}_rectified.png"),
                    cv2.cvtColor(rect_u8, cv2.COLOR_RGB2BGR))

    # ── Swatch centres in tile pixel space ───────────────────────────────────
    centres_rect = np.zeros((24, 2), dtype=np.float32)
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        centres_rect[i] = [(x0 + x1) * 0.5, (y0 + y1) * 0.5]

    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]],
                   dtype=np.float32)
    M_fwd = cv2.getPerspectiveTransform(quad_zoom, dst)
    try:
        M_inv = np.linalg.inv(M_fwd)
    except np.linalg.LinAlgError:
        print(f"[cc-erp] {tile_label}: singular perspective matrix")
        return None
    pts_h = np.column_stack([centres_rect[:, 0], centres_rect[:, 1],
                             np.ones(24, np.float32)])
    mapped = (M_inv @ pts_h.T).T
    centres_zoom = (mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-8, None))
    centres_tile = (centres_zoom / zoom_scale
                    + np.array([cx0, cy0], dtype=np.float32)).astype(np.float32)
    centres_uv = np.array([backproject_pixel_to_erp(cx, cy, map_uv)
                           for cx, cy in centres_tile], dtype=np.float32)

    _t["map_back"] = _time.perf_counter() - _tm
    if confidence <= 0.0:
        print(f"[cc-erp] {tile_label}: rejected conf=0 — "
              f"{conf_diag.get('reason', 'score=0')}  "
              f"(aspect={conf_diag.get('aspect', 0):.2f} "
              f"corr={conf_diag.get('neutral_corr', 0):.2f} "
              f"chroma_err={conf_diag.get('chroma_err', float('nan')):.3f})")
        return None

    t_total = _time.perf_counter() - _t0
    print(f"[cc-erp] {tile_label}: conf={confidence:.3f} "
          f"rot_corr={rot_corr:.2f} "
          f"aspect={conf_diag.get('aspect', 0):.2f} "
          f"corr={conf_diag.get('neutral_corr', 0):.2f} "
          f"chroma_err={conf_diag.get('chroma_err', float('nan')):.3f} "
          f"[{t_total*1000:.0f}ms]")

    # ── Pose estimation ──────────────────────────────────────────────────────
    quad_tile = quad_zoom / zoom_scale + np.array([cx0, cy0], dtype=np.float32)
    quad_ordered = _order_quad_tl_tr_br_bl(quad_tile)
    normal_world = _estimate_checker_pose(quad_ordered, out_w, out_h,
                                          yaw, pitch, tile_fov_deg)
    if np.any(np.isnan(normal_world)):
        normal_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    theta_n = float(np.degrees(np.arccos(np.clip(normal_world[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(normal_world[0], normal_world[2])))

    qc = quad_tile.mean(axis=0)
    quad_center_uv = np.array(backproject_pixel_to_erp(float(qc[0]), float(qc[1]),
                                                        map_uv), dtype=np.float32)

    if save_debug and debug_dir:
        vis = cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR)
        cv2.polylines(vis, [np.round(quad_ordered).astype(np.int32)], True,
                      (0, 200, 255), 2)
        for i, (cx, cy) in enumerate(centres_tile):
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i+1), (int(round(cx))+4, int(round(cy))-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
        cv2.putText(vis, f"{tile_label} yaw={yaw:.0f} pitch={pitch:.0f} "
                    f"conf={confidence:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, f"tile_{tile_label}_detected.jpg"), vis)

    return CheckerDetection(
        swatches_linear=swatches_wcs.astype(np.float32),
        swatch_centres_uv=centres_uv.astype(np.float32),
        swatch_centres_tile=centres_tile.astype(np.float32),
        quad_tile=quad_ordered.astype(np.float32),
        quad_center_uv=quad_center_uv,
        tile_yaw=yaw, tile_pitch=pitch, tile_fov_deg=tile_fov_deg,
        checker_normal_world=normal_world,
        checker_normal_theta_deg=theta_n,
        checker_normal_phi_deg=phi_n,
        confidence=confidence,
        raw_swatches_bgr=np.clip(swatches_wcs * 255, 0, 255).astype(np.uint8),
        detection_method="yolo_opencv",
        stage_label=tile_label,
        cc_rectified=cc_rectified,
    )


# ─── Manual corner-based detection ───────────────────────────────────────────

def find_colorchecker_manual(
    erp_linear: np.ndarray,
    corners_uv: List[Tuple[float, float]],
    colorspace: str = "acescg",
    debug_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Sample a CC24 from user-specified 4 corners on the ERP latlong.

    corners_uv: list of 4 (u,v) tuples in [0,1] space (TL, TR, BR, BL order).
    Returns (swatches_linear (24,3), info dict) or (None, info).
    """
    if len(corners_uv) != 4:
        return None, {"error": "need exactly 4 corners", "found": False}

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    cc24_ref = get_cc24_reference(colorspace)
    h, w = erp_linear.shape[:2]

    # Convert UV to pixel coords
    corners_px = np.array([[u * w, v * h] for u, v in corners_uv],
                          dtype=np.float32)

    # Centre of the quad in UV and yaw/pitch
    centre_uv = corners_px.mean(axis=0) / np.array([w, h], dtype=np.float32)
    cu, cv = float(centre_uv[0]), float(centre_uv[1])
    yaw = (cu - 0.5) * 360.0
    # erp_to_rectilinear: +pitch tilts camera DOWN; ERP v=1 is the nadir.
    pitch = (cv - 0.5) * 180.0

    # FOV: pick a tile that covers the quad with margin
    span_u = float(np.ptp(corners_px[:, 0])) / w * 360.0
    span_v = float(np.ptp(corners_px[:, 1])) / h * 180.0
    fov = float(np.clip(max(span_u, span_v) * 2.5, 40.0, 120.0))

    tile_size = 1024
    tile_linear, map_uv = erp_to_rectilinear(erp_linear, yaw, pitch, fov,
                                              tile_size, tile_size)
    tile_h, tile_w = tile_linear.shape[:2]

    # Map the 4 UV corners to tile pixel space via the UV map
    quad_tile = np.zeros((4, 2), dtype=np.float32)
    for i, (u, v) in enumerate(corners_uv):
        # Find nearest pixel in tile whose map_uv matches this UV
        dist = (map_uv[..., 0] - u) ** 2 + (map_uv[..., 1] - v) ** 2
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        quad_tile[i] = [min_idx[1], min_idx[0]]  # (x, y)

    print(f"[cc-erp] Manual: quad_tile={quad_tile.tolist()} "
          f"yaw={yaw:.1f} pitch={pitch:.1f} fov={fov:.1f}")

    # Rectify + sample — adaptive sizing based on chart's pixel span.
    rect_w, rect_h, samples = _pick_rect_dims(quad_tile)
    masks = _cc24_swatch_masks(rect_w, rect_h, _CC_COLS, _CC_ROWS, samples)
    quad_tile_ordered, swatches_wcs, cc_rectified, rot_corr = _best_rotation(
        quad_tile, tile_linear, cc24_ref, rect_w, rect_h, masks)

    confidence, conf_diag = _compute_confidence(swatches_wcs, cc24_ref,
                                                quad_tile_ordered)

    print(f"[cc-erp] Manual: conf={confidence:.3f} rot_corr={rot_corr:.2f} "
          f"corr={conf_diag.get('neutral_corr', 0):.2f} "
          f"chroma_err={conf_diag.get('chroma_err', float('nan')):.3f}")

    if debug_dir:
        tile_u8 = _linear_to_u8_for_display(tile_linear)
        vis = cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR)
        cv2.polylines(vis, [np.round(quad_tile_ordered).astype(np.int32)], True,
                      (0, 200, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "manual_tile_detected.jpg"), vis)
        rect_u8 = np.clip(cc_rectified * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, "manual_rectified.png"),
                    cv2.cvtColor(rect_u8, cv2.COLOR_RGB2BGR))

    # Swatch centres back to UV space
    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]],
                   dtype=np.float32)
    M_fwd = cv2.getPerspectiveTransform(quad_tile_ordered, dst)
    try:
        M_inv = np.linalg.inv(M_fwd)
    except np.linalg.LinAlgError:
        return None, {"error": "singular perspective matrix", "found": False}
    centres_rect = np.zeros((24, 2), dtype=np.float32)
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        centres_rect[i] = [(x0 + x1) * 0.5, (y0 + y1) * 0.5]
    pts_h = np.column_stack([centres_rect[:, 0], centres_rect[:, 1],
                             np.ones(24, np.float32)])
    mapped = (M_inv @ pts_h.T).T
    centres_tile = (mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-8, None))
    centres_uv_out = np.array([backproject_pixel_to_erp(cx, cy, map_uv)
                               for cx, cy in centres_tile], dtype=np.float32)

    # Pose
    normal_world = _estimate_checker_pose(
        _order_quad_tl_tr_br_bl(quad_tile_ordered),
        tile_w, tile_h, yaw, pitch, fov)
    if np.any(np.isnan(normal_world)):
        normal_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    theta_n = float(np.degrees(np.arccos(np.clip(normal_world[1], -1, 1))))
    phi_n = float(np.degrees(np.arctan2(normal_world[0], normal_world[2])))

    if debug_dir:
        _save_gui_debug(debug_dir,
                        CheckerDetection(
                            swatches_linear=swatches_wcs,
                            swatch_centres_uv=centres_uv_out,
                            swatch_centres_tile=centres_tile.astype(np.float32),
                            quad_tile=quad_tile_ordered,
                            quad_center_uv=centre_uv,
                            tile_yaw=yaw, tile_pitch=pitch, tile_fov_deg=fov,
                            checker_normal_world=normal_world,
                            checker_normal_theta_deg=theta_n,
                            checker_normal_phi_deg=phi_n,
                            confidence=confidence,
                            raw_swatches_bgr=np.clip(swatches_wcs * 255, 0, 255).astype(np.uint8),
                            detection_method="manual",
                            stage_label="manual",
                            cc_rectified=cc_rectified,
                        ), erp_linear, cc24_ref)

    info = {
        "found": True,
        "confidence": confidence,
        "detection_method": "manual",
        "checker_normal_theta_deg": theta_n,
        "checker_normal_phi_deg": phi_n,
        "quad_center_uv": [cu, cv],
        "swatch_centres_uv": centres_uv_out.tolist(),
    }
    return swatches_wcs.astype(np.float32), info


# ─── Auto sweep entry point ─────────────────────────────────────────────────

def _rect_uv_to_tile_params(rect_uv: Tuple[float, float, float, float]
                            ) -> Tuple[float, float, float]:
    """Convert a UV bbox (u0, v0, u1, v1) on the ERP to (yaw, pitch, fov_deg)
    for a single rectilinear tile that frames it with a small margin."""
    u0, v0, u1, v1 = rect_uv
    u0, u1 = float(min(u0, u1)), float(max(u0, u1))
    v0, v1 = float(min(v0, v1)), float(max(v0, v1))
    cu, cv = (u0 + u1) * 0.5, (v0 + v1) * 0.5
    yaw = (cu - 0.5) * 360.0
    # erp_to_rectilinear convention: +pitch_deg tilts camera DOWN (nadir).
    # In ERP, v=0 is up / v=1 is down, so pitch = (cv - 0.5) * 180.
    pitch = (cv - 0.5) * 180.0
    # FOV = exact angular span of the drawn rect, no margin, no clamp.
    # The tile is rendered at the rect's aspect so the chart isn't stretched.
    span_u_deg = (u1 - u0) * 360.0
    span_v_deg = (v1 - v0) * 180.0
    fov = float(max(span_u_deg, span_v_deg))
    return yaw, pitch, fov


def find_colorchecker_in_rect(
    erp_linear: np.ndarray,
    rect_uv: Tuple[float, float, float, float],
    colorspace: str = "acescg",
    debug_dir: Optional[str] = None,
    tile_size: int = 1024,
    min_confidence: float = 0.50,
    **_unused,
) -> Tuple[Optional[np.ndarray], dict]:
    """Detect a CC24 inside a user-drawn ERP rectangle.

    rect_uv: (u0, v0, u1, v1) in [0,1] coordinates on the latlong image.
    Runs the YOLO+OpenCV detector once on a single rectilinear tile centred
    on the rectangle. No sphere sweep.
    """
    # Segmentation runs first now, so don't gate on HAVE_YOLO upfront.
    if not (HAVE_CCD or HAVE_YOLO):
        return None, {"error": "no detection backend available "
                               "(install colour-checker-detection / ultralytics)",
                      "found": False}

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    _log_backends()
    cc24_ref = get_cc24_reference(colorspace)

    yaw, pitch, fov = _rect_uv_to_tile_params(rect_uv)
    # Tile size in the longer dimension; the shorter dim follows the rect aspect.
    u0, v0, u1, v1 = rect_uv
    span_u_deg = abs(u1 - u0) * 360.0
    span_v_deg = abs(v1 - v0) * 180.0
    if span_u_deg >= span_v_deg:
        out_w = tile_size
        out_h = max(64, int(round(tile_size * (span_v_deg / max(span_u_deg, 1e-6)))))
    else:
        out_h = tile_size
        out_w = max(64, int(round(tile_size * (span_u_deg / max(span_v_deg, 1e-6)))))
    print(f"[cc-erp] Rect detect: yaw={yaw:.1f}° pitch={pitch:.1f}° "
          f"fov={fov:.1f}° rect_uv={rect_uv}  tile={out_w}x{out_h}")

    tile, map_uv = erp_to_rectilinear(erp_linear, yaw, pitch, fov, out_w, out_h)

    if debug_dir:
        u8 = _linear_to_u8_for_display(tile)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"rect yaw={yaw:.0f} pitch={pitch:.0f} fov={fov:.0f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, "rect_tile.jpg"), bgr)

    # 1) Segmentation first — fast, no model load, often enough for charts
    # that fill a healthy fraction of the rect.
    det = _detect_in_tile_segmentation(
        tile, map_uv, yaw, pitch, fov, cc24_ref,
        debug_dir=debug_dir, tile_label="rect")

    # 2) YOLO two-pass fallback when segmentation can't lock on.
    if det is None or det.confidence < min_confidence:
        if det is not None:
            print(f"[cc-erp] Rect detect: segmentation conf={det.confidence:.3f} "
                  f"< {min_confidence:.2f} — falling back to YOLO")
        else:
            print(f"[cc-erp] Rect detect: segmentation found nothing — "
                  f"falling back to YOLO")
        yolo_det = _detect_in_tile(tile, map_uv, yaw, pitch, fov, cc24_ref,
                                   debug_dir=debug_dir, tile_label="rect",
                                   save_debug=bool(debug_dir))
        if yolo_det is not None and (det is None
                                     or yolo_det.confidence > det.confidence):
            det = yolo_det

    if det is None or det.confidence < min_confidence:
        conf = det.confidence if det else 0.0
        print(f"[cc-erp] Rect detect: no chart "
              f"(conf={conf:.3f} < {min_confidence:.2f})")
        return None, {"found": False, "confidence": conf,
                      "rect_uv": list(rect_uv)}

    print(f"[cc-erp] Rect detect: conf={det.confidence:.3f}  "
          f"θ={det.checker_normal_theta_deg:.1f}° "
          f"φ={det.checker_normal_phi_deg:.1f}°")

    if debug_dir:
        _save_gui_debug(debug_dir, det, erp_linear, cc24_ref)

    info = {
        "found": True,
        "confidence": det.confidence,
        "detection_method": det.detection_method,
        "rect_uv": list(rect_uv),
        "best_tile_yaw_deg": det.tile_yaw,
        "best_tile_pitch_deg": det.tile_pitch,
        "best_tile_fov_deg": det.tile_fov_deg,
        "checker_normal_world": det.checker_normal_world.tolist(),
        "checker_normal_theta_deg": det.checker_normal_theta_deg,
        "checker_normal_phi_deg": det.checker_normal_phi_deg,
        "swatch_centres_uv": det.swatch_centres_uv.tolist(),
        "quad_center_uv": det.quad_center_uv.tolist(),
    }
    return det.swatches_linear, info


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
    if not (np.isnan(phi) or np.isnan(th)):
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
