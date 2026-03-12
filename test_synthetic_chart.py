"""
test_synthetic_chart.py
=======================
Generates a synthetic ERP panorama with a perfect CC24 chart baked in,
then runs the detection pipeline and prints every swatch value the library
returns vs what we injected.

This isolates whether the detection + swatch extraction is correct
independent of any real-scene ambiguity.

Usage:
    python test_synthetic_chart.py
    python test_synthetic_chart.py --tint 1.5 0.9 0.4   # apply warm tint to scene
    python test_synthetic_chart.py --out-dir my_debug
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# CC24 reference — sRGB display values (NOT linear)
# ---------------------------------------------------------------------------
CC24_SRGB = np.array([
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


def srgb_to_linear(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v**(1/2.4) - 0.055)


CC24_LINEAR = srgb_to_linear(CC24_SRGB).astype(np.float32)


def make_chart_image(patch_px=60, gap_px=8, border_px=12, tint=None):
    """
    Render CC24 as a flat rectilinear image (4 rows × 6 cols) in LINEAR space.
    tint: (3,) RGB multiplier applied to every patch (simulates coloured illuminant).
    Returns float32 linear RGB image.
    """
    rows, cols = 4, 6
    w = cols * patch_px + (cols - 1) * gap_px + 2 * border_px
    h = rows * patch_px + (rows - 1) * gap_px + 2 * border_px

    img = np.ones((h, w, 3), dtype=np.float32) * 0.02  # dark border background

    for idx in range(24):
        r = idx // cols
        c = idx % cols
        x0 = border_px + c * (patch_px + gap_px)
        y0 = border_px + r * (patch_px + gap_px)
        x1 = x0 + patch_px
        y1 = y0 + patch_px
        colour = CC24_LINEAR[idx].copy()
        if tint is not None:
            colour = colour * tint
        img[y0:y1, x0:x1] = colour

    return img, (w, h, border_px, patch_px, gap_px)


def embed_chart_in_erp(chart_linear, erp_w=2048, erp_h=1024,
                        chart_yaw_deg=200.0, chart_pitch_deg=45.0,
                        chart_scale=0.25, sky_tint=None):
    """
    Embed the chart into a simple ERP panorama.
    The panorama is a gradient sky (top) + ground (bottom).
    The chart is perspective-projected into the ERP at the given yaw/pitch.
    """
    # Build simple background: sky blue-grey top, warm brown ground bottom
    erp = np.zeros((erp_h, erp_w, 3), dtype=np.float32)
    for y in range(erp_h):
        t = y / erp_h
        if t < 0.5:  # sky
            sky = np.array([0.15, 0.20, 0.35], dtype=np.float32)
            erp[y] = sky * (1.0 - t * 0.4)
        else:  # ground
            ground = np.array([0.08, 0.07, 0.04], dtype=np.float32)
            erp[y] = ground

    if sky_tint is not None:
        erp *= sky_tint

    # Add a bright sun disc at yaw=0, pitch=40
    for y in range(erp_h):
        for_lat = (0.5 - y / erp_h) * np.pi          # pitch
        for x in range(erp_w):
            lon = (x / erp_w - 0.5) * 2 * np.pi      # yaw
            # sun at yaw=0, pitch=40deg
            sun_yaw   = 0.0
            sun_pitch = np.radians(40)
            d = np.sqrt((lon - sun_yaw)**2 + (for_lat - sun_pitch)**2)
            if d < 0.05:
                erp[y, x] = np.array([8.0, 7.0, 5.0], dtype=np.float32)
            elif d < 0.15:
                erp[y, x] += np.array([0.5, 0.45, 0.3], dtype=np.float32) * (0.15 - d) / 0.15

    # ── Place chart via gnomonic projection ──────────────────────────────
    ch, cw = chart_linear.shape[:2]

    # How large to make the chart in the tile (fraction of tile size)
    tile_w, tile_h = 900, 675
    fov_deg = 70.0

    # Chart occupies chart_scale fraction of tile width
    chart_tile_w = int(tile_w * chart_scale)
    chart_tile_h = int(chart_tile_w * ch / cw)

    # Resize chart to tile size
    chart_resized = cv2.resize(chart_linear, (chart_tile_w, chart_tile_h),
                               interpolation=cv2.INTER_LINEAR)

    # Offset within tile (centre it)
    ox = (tile_w - chart_tile_w) // 2
    oy = (tile_h - chart_tile_h) // 2

    # Build gnomonic → ERP mapping
    f = (tile_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    yaw_r   = np.radians(chart_yaw_deg)
    pitch_r = np.radians(chart_pitch_deg)

    Ry = np.array([[ np.cos(yaw_r), 0, np.sin(yaw_r)],
                   [ 0,             1, 0            ],
                   [-np.sin(yaw_r), 0, np.cos(yaw_r)]], dtype=np.float64)
    Rx = np.array([[1, 0,              0             ],
                   [0, np.cos(pitch_r), -np.sin(pitch_r)],
                   [0, np.sin(pitch_r),  np.cos(pitch_r)]], dtype=np.float64)
    R  = Ry @ Rx

    # For each chart pixel in the tile, back-project to ERP
    for cy_t in range(chart_tile_h):
        for cx_t in range(chart_tile_w):
            # Tile pixel position
            tx = cx_t + ox
            ty = cy_t + oy
            # Gnomonic ray
            dx = (tx - tile_w / 2.0) / f
            dy = (ty - tile_h / 2.0) / f
            ray_cam = np.array([dx, -dy, 1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = R @ ray_cam

            # World ray → ERP uv
            lon = np.arctan2(ray_world[0], ray_world[2])
            lat = np.arcsin(np.clip(-ray_world[1], -1, 1))
            u = (lon / (2 * np.pi) + 0.5) % 1.0
            v = 0.5 - lat / np.pi

            ex = int(np.clip(u * erp_w, 0, erp_w - 1))
            ey = int(np.clip(v * erp_h, 0, erp_h - 1))
            erp[ey, ex] = chart_resized[cy_t, cx_t]

    return erp


def run_detection(erp_linear, debug_dir, tint):
    """Run the colorchecker_erp detection pipeline and print full diagnostics."""
    try:
        from colorchecker_erp import find_colorchecker_in_erp
    except ImportError:
        print("ERROR: colorchecker_erp.py not found. Run from the same directory.")
        sys.exit(1)

    print("\n" + "="*70)
    print("RUNNING DETECTION")
    print("="*70)

    swatches_linear, info = find_colorchecker_in_erp(
        erp_linear,
        debug_dir=debug_dir,
    )

    if swatches_linear is None:
        print("ERROR: No checker detected.")
        return

    print(f"\nDetection confidence: {info.get('confidence', '?'):.3f}")
    print(f"Tile: yaw={info.get('best_tile_yaw_deg','?')}° pitch={info.get('best_tile_pitch_deg','?')}°")

    # ── Full patch comparison table ───────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'#':>3}  {'INJECTED R':>10} {'INJECTED G':>10} {'INJECTED B':>10}  "
          f"{'DETECTED R':>10} {'DETECTED G':>10} {'DETECTED B':>10}  "
          f"{'ERR R':>7} {'ERR G':>7} {'ERR B':>7}  NOTE")
    print("-"*70)

    # What we actually injected (linear, possibly tinted)
    tint_arr = np.array(tint, dtype=np.float32) if tint else np.ones(3, dtype=np.float32)
    injected = CC24_LINEAR * tint_arr[None, :]  # (24,3) linear, tinted

    for i in range(24):
        inj = injected[i]
        det = swatches_linear[i]
        err = det - inj
        note = ""
        if i == 21:
            note = " ← PATCH 22 WB REF"
        elif i >= 18:
            note = " ← neutral"
        print(f"{i+1:>3}  {inj[0]:>10.5f} {inj[1]:>10.5f} {inj[2]:>10.5f}  "
              f"{det[0]:>10.5f} {det[1]:>10.5f} {det[2]:>10.5f}  "
              f"{err[0]:>+7.4f} {err[1]:>+7.4f} {err[2]:>+7.4f}{note}")

    # ── Patch 22 WB derivation ────────────────────────────────────────────
    print("\n" + "="*70)
    print("PATCH 22 WB DERIVATION")
    p22_det  = swatches_linear[21]
    p22_inj  = injected[21]
    p22_ref  = CC24_LINEAR[21]

    scale_raw = p22_ref / np.clip(p22_det, 1e-8, None)
    scale_gn  = scale_raw / max(float(scale_raw[1]), 1e-8)

    print(f"  Injected (tinted)  : R={p22_inj[0]:.5f}  G={p22_inj[1]:.5f}  B={p22_inj[2]:.5f}")
    print(f"  Detected           : R={p22_det[0]:.5f}  G={p22_det[1]:.5f}  B={p22_det[2]:.5f}")
    print(f"  Reference (D65)    : R={p22_ref[0]:.5f}  G={p22_ref[1]:.5f}  B={p22_ref[2]:.5f}")
    print(f"  scale = ref/det    : R={scale_raw[0]:.5f}  G={scale_raw[1]:.5f}  B={scale_raw[2]:.5f}")
    print(f"  G-normalised WB    : R={scale_gn[0]:.5f}  G={scale_gn[1]:.5f}  B={scale_gn[2]:.5f}")

    if tint:
        expected_scale = np.ones(3, dtype=np.float32) / tint_arr
        expected_gn = expected_scale / max(float(expected_scale[1]), 1e-8)
        print(f"\n  EXPECTED WB (1/tint): R={expected_gn[0]:.5f}  G={expected_gn[1]:.5f}  B={expected_gn[2]:.5f}")
        err_scale = scale_gn - expected_gn
        print(f"  Scale error         : R={err_scale[0]:+.5f}  G={err_scale[1]:+.5f}  B={err_scale[2]:+.5f}")

    # ── Swatch comparison PNG ─────────────────────────────────────────────
    save_comparison_png(debug_dir, injected, swatches_linear, scale_gn)


def save_comparison_png(debug_dir, injected, detected, wb_scale):
    """
    4-row strip:
      Row 1 (I): injected colours (what we put in)
      Row 2 (D): detected colours (what library returned)
      Row 3 (R): CC24 reference D65
      Row 4 (W): detected × WB scale (should match reference after correction)
    """
    sw = 64
    gap = 3
    n_rows = 4
    h = sw * n_rows + gap * (n_rows - 1)
    w = sw * 24

    def to_u8(linear_val):
        """Linear → sRGB gamma → uint8"""
        srgb = linear_to_srgb(np.clip(linear_val, 0, 1))
        return np.clip(srgb * 255, 0, 255).astype(np.uint8)

    strip = np.zeros((h, w, 3), dtype=np.uint8)

    rows_data = [
        (injected,                   "I", "injected"),
        (detected,                   "D", "detected"),
        (CC24_LINEAR,                "R", "reference D65"),
        (detected * wb_scale[None,:], "W", "detected x WB"),
    ]

    for row_i, (data, label, _) in enumerate(rows_data):
        y0 = row_i * (sw + gap)
        for i in range(24):
            x0 = i * sw
            col_u8 = to_u8(data[i])           # (3,) uint8 RGB
            col_bgr = col_u8[[2, 1, 0]]        # RGB → BGR for OpenCV
            strip[y0:y0+sw, x0:x0+sw] = col_bgr

            # Patch number on row 0 only
            if row_i == 0:
                cv2.putText(strip, str(i+1), (x0+2, y0+sw-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
            # Row label
            cv2.putText(strip, label, (x0+2, y0+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

    # Red border on patch 22 (index 21)
    p22_x = 21 * sw
    cv2.rectangle(strip, (p22_x, 0), (p22_x+sw-1, h-1), (0,0,255), 2)
    cv2.putText(strip, "#22", (p22_x+2, h-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

    path = os.path.join(debug_dir, "synthetic_swatch_comparison.jpg")
    cv2.imwrite(path, strip)
    print(f"\nSaved: {path}")
    print("  Row I = injected (ground truth)")
    print("  Row D = detected by library")
    print("  Row R = CC24 reference D65")
    print("  Row W = detected × WB scale  (should match R after correction)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tint",    nargs=3, type=float, default=None,
                    metavar=("R","G","B"),
                    help="Multiply all chart patches by this RGB tint "
                         "(e.g. 1.5 0.9 0.4 for warm golden-hour cast)")
    ap.add_argument("--chart-yaw",   type=float, default=200.0)
    ap.add_argument("--chart-pitch", type=float, default=45.0)
    ap.add_argument("--chart-scale", type=float, default=0.30,
                    help="Chart width as fraction of tile width (0.1–0.5)")
    ap.add_argument("--erp-w",  type=int, default=2048)
    ap.add_argument("--erp-h",  type=int, default=1024)
    ap.add_argument("--out-dir", default="debug_synthetic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tint = np.array(args.tint, dtype=np.float32) if args.tint else None

    print(f"Generating synthetic chart...")
    print(f"  tint       : {tint}")
    print(f"  chart pos  : yaw={args.chart_yaw}°  pitch={args.chart_pitch}°")
    print(f"  chart scale: {args.chart_scale}")

    chart_img, _ = make_chart_image(tint=tint)
    erp = embed_chart_in_erp(
        chart_img,
        erp_w=args.erp_w, erp_h=args.erp_h,
        chart_yaw_deg=args.chart_yaw,
        chart_pitch_deg=args.chart_pitch,
        chart_scale=args.chart_scale,
    )

    # Save the synthetic EXR for inspection
    try:
        import pyexr
        pyexr.write(os.path.join(args.out_dir, "synthetic_erp.exr"), erp)
        print(f"Saved EXR: {args.out_dir}/synthetic_erp.exr")
    except Exception:
        pass

    # Save a tonemapped preview
    preview = erp / (erp + 1.0)
    preview_u8 = np.clip(preview * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, "synthetic_erp_preview.jpg"),
                cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR))
    print(f"Saved preview: {args.out_dir}/synthetic_erp_preview.jpg")

    run_detection(erp, args.out_dir, tint)


if __name__ == "__main__":
    main()
