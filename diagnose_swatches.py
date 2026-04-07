"""
diagnose_swatches.py
====================
Run CC detection on a real EXR and print exactly what the library returns
vs CC24 reference, then save a comparison PNG.

Usage:
    python diagnose_swatches.py input.exr
    python diagnose_swatches.py input.exr --debug-dir my_debug
"""
import argparse, os, sys
import numpy as np
import cv2

# ── CC24 reference (sRGB display values) ─────────────────────────────────────
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
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5  ← WB ref
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black
], dtype=np.float32)

def srgb_to_linear(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v/12.92, ((v+0.055)/1.055)**2.4).astype(np.float32)

def linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v*12.92, 1.055*v**(1/2.4)-0.055).astype(np.float32)

CC24_LINEAR = srgb_to_linear(CC24_SRGB)

PATCH_NAMES = [
    "dark skin","light skin","blue sky","foliage","blue flower","bluish green",
    "orange","purplish blue","moderate red","purple","yellow green","orange yellow",
    "blue","green","red","yellow","magenta","cyan",
    "white","neutral 8","neutral 6.5","neutral 5","neutral 3.5","black",
]

def load_exr(path):
    try:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(path)
        h = f.header()
        dw = h['dataWindow']
        w = dw.max.x - dw.min.x + 1
        hh = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(f.channel('R', pt), np.float32).reshape(hh, w)
        g = np.frombuffer(f.channel('G', pt), np.float32).reshape(hh, w)
        b = np.frombuffer(f.channel('B', pt), np.float32).reshape(hh, w)
        return np.stack([r,g,b], axis=-1)
    except Exception:
        pass
    try:
        import pyexr
        return pyexr.open(path).get().astype(np.float32)
    except Exception as e:
        print(f"EXR load failed: {e}")
        sys.exit(1)

def save_comparison_png(path, detected_linear, wb_scale_gn):
    """
    5-row strip:
      Row D : detected by library (tonemapped display)
      Row R : CC24 reference D65
      Row W : detected × WB scale (should match R)
      Row N : patch name label row
    """
    sw = 80
    gap = 4
    label_h = 14

    def row_pixels(values_linear):
        # values_linear: (24,3) linear
        # tonemap each patch independently so dark patches are visible
        strip = np.zeros((sw, sw*24, 3), dtype=np.uint8)
        for i, v in enumerate(values_linear):
            # normalise each patch to its own max for display clarity
            peak = max(float(v.max()), 1e-6)
            v_disp = linear_to_srgb(v / peak * 0.9)
            col_bgr = np.clip(v_disp[[2,1,0]] * 255, 0, 255).astype(np.uint8)
            strip[:, i*sw:(i+1)*sw] = col_bgr
        return strip

    def row_pixels_tm(values_linear):
        # global Reinhard tonemap — preserves relative brightness between patches
        strip = np.zeros((sw, sw*24, 3), dtype=np.uint8)
        all_vals = values_linear
        peak = max(float(all_vals.max()), 1e-6)
        for i, v in enumerate(all_vals):
            v_tm = v / peak  # simple global normalise
            v_srgb = linear_to_srgb(v_tm)
            col_bgr = np.clip(v_srgb[[2,1,0]] * 255, 0, 255).astype(np.uint8)
            strip[:, i*sw:(i+1)*sw] = col_bgr
        return strip

    detected_wb = detected_linear * wb_scale_gn[None, :]

    rows = [
        ("D detected (global tm)",   row_pixels_tm(detected_linear)),
        ("D detected (per-patch)",   row_pixels(detected_linear)),
        ("R reference D65",          row_pixels(CC24_LINEAR)),
        ("W det x WB (per-patch)",   row_pixels(detected_wb)),
    ]

    total_h = len(rows) * (sw + gap) + label_h
    canvas = np.zeros((total_h, sw*24, 3), dtype=np.uint8)

    for ri, (label, row_img) in enumerate(rows):
        y0 = ri * (sw + gap)
        canvas[y0:y0+sw] = row_img
        # row label on left
        cv2.putText(canvas, label, (2, y0+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220,220,220), 1)

    # patch numbers + names along bottom
    y_label = len(rows) * (sw + gap)
    for i in range(24):
        x = i * sw
        cv2.putText(canvas, f"{i+1}", (x+2, y_label+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

    # Red border on patch 22
    p22_x = 21 * sw
    cv2.rectangle(canvas, (p22_x, 0), (p22_x+sw-1, total_h-1), (0,0,255), 2)
    cv2.putText(canvas, "#22 WB", (p22_x+2, total_h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,255), 1)

    cv2.imwrite(path, canvas)
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exr", help="Input linear EXR panorama")
    ap.add_argument("--debug-dir", default="debug_diagnose")
    ap.add_argument("--fov",   type=float, default=70.0)
    ap.add_argument("--tile-w", type=int, default=900)
    ap.add_argument("--tile-h", type=int, default=675)
    ap.add_argument("--pitches", nargs="+", type=float,
                    default=[-45, -20, 0, 20, 45],
                    help="Pitch angles to sweep (default: -45 -20 0 20 45)")
    args = ap.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)

    print(f"Loading: {args.exr}")
    erp = load_exr(args.exr)
    print(f"  Shape: {erp.shape}  min={erp.min():.4f}  max={erp.max():.4f}")

    from colorchecker_erp import find_colorchecker_in_erp
    swatches_linear, info = find_colorchecker_in_erp(
        erp,
        fov_deg=args.fov,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        pitch_values=tuple(args.pitches),
        debug_dir=args.debug_dir,
    )

    if swatches_linear is None:
        print("ERROR: No checker detected.")
        sys.exit(1)

    print(f"\nDetection: confidence={info.get('confidence','?'):.3f}  "
          f"tile yaw={info.get('best_tile_yaw_deg','?')}°  "
          f"pitch={info.get('best_tile_pitch_deg','?')}°")

    # ── Full table ────────────────────────────────────────────────────────
    print(f"\n{'#':>3}  {'Name':<16}  "
          f"{'Det R':>8} {'Det G':>8} {'Det B':>8}  "
          f"{'Ref R':>8} {'Ref G':>8} {'Ref B':>8}  "
          f"{'Chroma R-G':>10} {'Chroma B-G':>10}")
    print("-"*100)

    for i in range(24):
        d  = swatches_linear[i]
        r  = CC24_LINEAR[i]
        luma_d = 0.2126*d[0] + 0.7152*d[1] + 0.0722*d[2]
        luma_r = 0.2126*r[0] + 0.7152*r[1] + 0.0722*r[2]
        # chroma = channel deviation from grey after luma normalisation
        if luma_d > 1e-6:
            dn = d / luma_d
            rg = dn[0] - dn[1]
            bg = dn[2] - dn[1]
        else:
            rg = bg = 0.0
        flag = "  ← WB REF" if i == 21 else ("  ← neutral" if i >= 18 else "")
        print(f"{i+1:>3}  {PATCH_NAMES[i]:<16}  "
              f"{d[0]:>8.4f} {d[1]:>8.4f} {d[2]:>8.4f}  "
              f"{r[0]:>8.4f} {r[1]:>8.4f} {r[2]:>8.4f}  "
              f"{rg:>+10.4f} {bg:>+10.4f}{flag}")

    # ── Patch 22 WB ───────────────────────────────────────────────────────
    p22 = swatches_linear[21]
    ref22 = CC24_LINEAR[21]
    scale_raw = ref22 / np.clip(p22, 1e-8, None)
    scale_gn  = (scale_raw / max(float(scale_raw[1]), 1e-8)).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"PATCH 22 (Neutral 5 ~18% grey) — WB reference")
    print(f"  detected linear : R={p22[0]:.5f}  G={p22[1]:.5f}  B={p22[2]:.5f}")
    print(f"  reference D65   : R={ref22[0]:.5f}  G={ref22[1]:.5f}  B={ref22[2]:.5f}")
    print(f"  chroma R-G      : {(p22[0]-p22[1])/(p22[1]+1e-8):+.4f}  "
          f"B-G: {(p22[2]-p22[1])/(p22[1]+1e-8):+.4f}")
    print(f"  scale ref/det   : R={scale_raw[0]:.5f}  G={scale_raw[1]:.5f}  B={scale_raw[2]:.5f}")
    print(f"  G-norm WB scale : R={scale_gn[0]:.5f}  G={scale_gn[1]:.5f}  B={scale_gn[2]:.5f}")

    # ── Also check neutral ramp monotonicity ─────────────────────────────
    print(f"\nNeutral ramp check (patches 19-24, should be monotonically decreasing):")
    for i in range(18, 24):
        d = swatches_linear[i]
        luma = 0.2126*d[0] + 0.7152*d[1] + 0.0722*d[2]
        rg = (d[0]-d[1])/(d[1]+1e-8)
        bg = (d[2]-d[1])/(d[1]+1e-8)
        flag = " ← WB ref" if i == 21 else ""
        print(f"  patch {i+1:02d}: luma={luma:.5f}  R-G={rg:+.4f}  B-G={bg:+.4f}{flag}")

    # ── PNG ───────────────────────────────────────────────────────────────
    png_path = os.path.join(args.debug_dir, "swatch_diagnose.png")
    save_comparison_png(png_path, swatches_linear, scale_gn)


if __name__ == "__main__":
    main()
