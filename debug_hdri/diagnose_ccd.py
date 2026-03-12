"""
diagnose_ccd.py — run this on the tile_029_crop.jpg to see exactly what
DataDetectionColourChecker gives us, then we can use the right attributes.

Usage:
    python diagnose_ccd.py path/to/tile_029_crop.jpg
    python diagnose_ccd.py path/to/tile_029_detected.jpg   # full tile also useful
"""
import sys
import numpy as np
import cv2
import colour_checker_detection as ccd

def tonemap(img_f32):
    """Simple Reinhard on float image for visualisation."""
    img = img_f32 / (img_f32 + 1.0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def load_as_float_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tile_029_crop.jpg"
    img_f32 = load_as_float_rgb(path)
    h, w = img_f32.shape[:2]
    print(f"Image: {path}  shape={img_f32.shape}")

    print("\n=== Detection with additional_data=True ===")
    try:
        results = ccd.detect_colour_checkers_segmentation(
            img_f32,
            show=False,
            additional_data=True,
            apply_cctf_decoding=False,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return

    print(f"Results count: {len(results)}")
    if not results:
        print("No checker detected.")
        return

    det = results[0]
    print(f"\nType: {type(det)}")
    print(f"\nAll attributes:")
    for attr in sorted(dir(det)):
        if attr.startswith('_'):
            continue
        val = getattr(det, attr)
        if callable(val):
            continue
        if isinstance(val, np.ndarray):
            print(f"  {attr}: ndarray shape={val.shape} dtype={val.dtype} "
                  f"min={val.min():.4f} max={val.max():.4f}")
            if val.ndim <= 2 and val.size <= 50:
                print(f"    values: {val}")
        elif isinstance(val, (list, tuple)):
            print(f"  {attr}: {type(val).__name__} len={len(val)}")
            for i, v in enumerate(val[:3]):
                if isinstance(v, np.ndarray):
                    print(f"    [{i}]: ndarray shape={v.shape}")
                else:
                    print(f"    [{i}]: {v}")
            if len(val) > 3:
                print(f"    ... ({len(val)} total)")
        else:
            print(f"  {attr}: {repr(val)}")

    # Specifically probe the most important ones
    print("\n=== Key attributes ===")
    
    # swatch_colours
    sc = getattr(det, 'swatch_colours', None)
    if sc is not None:
        sc = np.array(sc)
        print(f"swatch_colours shape={sc.shape}")
        print(f"  neutrals (patches 19-24): {sc[18:24]}")

    # rectangle — in what coord space?
    rect = getattr(det, 'rectangle', None)
    if rect is not None:
        print(f"\nrectangle: {np.array(rect)}")
        print(f"  (image is {w}x{h})")
    
    # quadrilateral
    quad = getattr(det, 'quadrilateral', None)
    if quad is not None:
        print(f"\nquadrilateral: {np.array(quad)}")

    # colour_checker sub-image
    cc = getattr(det, 'colour_checker', None)
    if cc is not None:
        cc = np.array(cc)
        print(f"\ncolour_checker: shape={cc.shape} min={cc.min():.3f} max={cc.max():.3f}")
        # Save it for inspection
        if cc.ndim == 3:
            cc_vis = tonemap(cc) if cc.max() > 1.5 else np.clip(cc * 255, 0, 255).astype(np.uint8)
            cv2.imwrite("debug_cc_subimage.jpg", cv2.cvtColor(cc_vis, cv2.COLOR_RGB2BGR))
            print("  Saved: debug_cc_subimage.jpg")

    # swatch_masks
    masks = getattr(det, 'swatch_masks', None)
    if masks is not None:
        print(f"\nswatch_masks: len={len(masks)}")
        for i, m in enumerate(masks[:4]):
            print(f"  [{i}]: {m}")

    # Draw rectangle/quad on image and save
    vis = cv2.cvtColor(np.clip(img_f32 * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    if rect is not None:
        r = np.array(rect, dtype=np.int32)
        if r.ndim == 2 and r.shape[1] == 2:
            cv2.polylines(vis, [r], True, (255, 0, 0), 2)  # blue
            print(f"\nDrawing rectangle (blue)")
    
    if quad is not None:
        q = np.array(quad, dtype=np.int32)
        if q.ndim == 2 and q.shape[1] == 2:
            cv2.polylines(vis, [q], True, (0, 255, 0), 2)  # green
            print(f"Drawing quadrilateral (green)")

    cv2.imwrite("debug_detection_overlay.jpg", vis)
    print(f"\nSaved: debug_detection_overlay.jpg")


if __name__ == '__main__':
    main()
