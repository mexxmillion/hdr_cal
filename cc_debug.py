#!/usr/bin/env python3
"""
cc_debug.py — ColorChecker detection debugger

Drag & drop a JPG/PNG image, hit Process, see detection results.
Shows segmentation and YOLO (if available) with highlighted quads,
swatch colours, and log messages.

Usage:
  python cc_debug.py
  python cc_debug.py image.jpg   (auto-load)
"""

import sys
import os
import traceback
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QScrollArea,
    QSplitter, QCheckBox,
)
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

# ── Bundled model path ──────────────────────────────────────────────────────
_BUNDLED_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if os.path.isdir(_BUNDLED_MODELS):
    os.environ.setdefault(
        "COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY", _BUNDLED_MODELS
    )

# ── Detection imports ───────────────────────────────────────────────────────
try:
    import colour_checker_detection as ccd
    HAVE_CCD = True
except ImportError:
    HAVE_CCD = False

HAVE_CCD_INFERENCE = False
if HAVE_CCD:
    try:
        _ = ccd.detect_colour_checkers_inference
        HAVE_CCD_INFERENCE = True
    except AttributeError:
        pass

# ── CC24 reference (sRGB linear) ───────────────────────────────────────────
CC24_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage",
    "blue flower", "bluish green", "orange", "purplish blue",
    "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow",
    "magenta", "cyan", "white", "neutral 8",
    "neutral 6.5", "neutral 5", "neutral 3.5", "black",
]


def _np_to_qpixmap(img_rgb_u8: np.ndarray, max_w: int = 800) -> QPixmap:
    h, w = img_rgb_u8.shape[:2]
    if w > max_w:
        scale = max_w / w
        img_rgb_u8 = cv2.resize(img_rgb_u8, (max_w, int(h * scale)))
        h, w = img_rgb_u8.shape[:2]
    qimg = QImage(img_rgb_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _draw_quad(img_bgr, quad, colour=(0, 255, 0), thickness=2):
    pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], True, colour, thickness)


def _draw_swatches(img_bgr, det, label=""):
    """Draw swatch rectangles on the image."""
    if not hasattr(det, "swatch_masks") or det.swatch_masks is None:
        return
    masks = np.array(det.swatch_masks, dtype=np.float32)
    sw = np.array(det.swatch_colours, dtype=np.float32)
    if masks.shape[0] < 24 or sw.shape != (24, 3):
        return
    h, w = img_bgr.shape[:2]
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        x0i, x1i = int(x0 * w), int(x1 * w)
        y0i, y1i = int(y0 * h), int(y1 * h)
        c = tuple(int(v * 255) for v in sw[i][::-1])  # RGB→BGR
        cv2.rectangle(img_bgr, (x0i, y0i), (x1i, y1i), c, 2)
        cv2.putText(img_bgr, str(i + 1), (x0i + 2, y1i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def _swatch_strip(sw_colours, label=""):
    """Create a 24-patch colour strip image."""
    pw, ph = 50, 50
    strip = np.zeros((ph + 20, pw * 24, 3), dtype=np.uint8)
    for i in range(min(24, len(sw_colours))):
        c = np.clip(sw_colours[i] * 255, 0, 255).astype(np.uint8)
        x0 = i * pw
        strip[20:, x0:x0 + pw] = c[None, None, :]
        cv2.putText(strip, str(i + 1), (x0 + 15, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if label:
        cv2.putText(strip, label, (5, strip.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
    return strip


class DropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & drop an image here\nor click Browse")
        self.setMinimumSize(400, 300)
        self.setStyleSheet("QLabel { border: 2px dashed #555; background: #1a1a2e; color: #666; }")
        self._file_callback = None

    def set_file_callback(self, cb):
        self._file_callback = cb

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls and self._file_callback:
            self._file_callback(urls[0].toLocalFile())


class CCDebugWindow(QMainWindow):
    def __init__(self, initial_file=None):
        super().__init__()
        self.setWindowTitle("ColorChecker Detection Debugger")
        self.setMinimumSize(900, 700)
        self._image_path = None
        self._image_rgb = None

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)

        # Top: buttons
        btn_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse)
        self._process_btn = QPushButton("Process")
        self._process_btn.clicked.connect(self._process)
        self._process_btn.setEnabled(False)
        self._use_inference = QCheckBox("Try YOLO inference")
        self._use_inference.setChecked(HAVE_CCD_INFERENCE)
        self._use_inference.setEnabled(HAVE_CCD_INFERENCE)
        self._apply_decode = QCheckBox("apply_cctf_decoding")
        self._apply_decode.setChecked(True)
        self._apply_decode.setToolTip("Decode sRGB gamma on input (enable for JPG/PNG photos)")
        btn_row.addWidget(self._browse_btn)
        btn_row.addWidget(self._process_btn)
        btn_row.addWidget(self._use_inference)
        btn_row.addWidget(self._apply_decode)
        btn_row.addStretch()
        main_lay.addLayout(btn_row)

        # Middle: image + log splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Image area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._image_label = DropLabel()
        self._image_label.set_file_callback(self._load_file)
        scroll.setWidget(self._image_label)
        splitter.addWidget(scroll)

        # Log area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("QTextEdit { background: #0d0d1a; color: #c0c0c0; font-family: monospace; font-size: 12px; }")
        splitter.addWidget(self._log)

        splitter.setSizes([500, 200])
        main_lay.addWidget(splitter)

        # Status
        self._status = QLabel("Ready")
        main_lay.addWidget(self._status)

        self._print(f"colour-checker-detection: {'yes' if HAVE_CCD else 'NO'}")
        self._print(f"YOLO inference:           {'yes' if HAVE_CCD_INFERENCE else 'no'}")
        model_repo = os.environ.get("COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY", "~/.colour-science/...")
        model_path = os.path.join(model_repo, "colour-checker-detection-l-seg.pt")
        self._print(f"YOLO model:               {'FOUND' if os.path.isfile(model_path) else 'not found'} ({model_path})")
        self._print("")

        if initial_file:
            self._load_file(initial_file)

    def _print(self, msg):
        self._log.append(msg)
        QApplication.processEvents()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.exr *.hdr);;All (*)")
        if path:
            self._load_file(path)

    def _load_file(self, path):
        if not os.path.isfile(path):
            self._print(f"File not found: {path}")
            return
        self._image_path = path
        self._status.setText(f"Loaded: {os.path.basename(path)}")
        self._print(f"Loaded: {path}")

        # Read and display
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self._print("ERROR: cv2.imread failed")
            return
        self._image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = self._image_rgb.shape[:2]
        self._print(f"  Size: {w}x{h}  dtype={self._image_rgb.dtype}")
        self._image_label.setPixmap(_np_to_qpixmap(self._image_rgb))
        self._image_label.setStyleSheet("")
        self._process_btn.setEnabled(True)

    def _process(self):
        if self._image_rgb is None:
            return
        if not HAVE_CCD:
            self._print("ERROR: colour-checker-detection not installed")
            return

        self._print("\n" + "=" * 60)
        self._print("Processing...")
        self._status.setText("Detecting...")
        QApplication.processEvents()

        img = self._image_rgb.copy()
        h, w = img.shape[:2]
        apply_decode = self._apply_decode.isChecked()
        self._print(f"  apply_cctf_decoding={apply_decode}")

        # Prepare float input
        img_f = img.astype(np.float32) / 255.0
        self._print(f"  Input range: [{img_f.min():.3f}, {img_f.max():.3f}]")

        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        results_images = [img.copy()]  # start with original

        # ── Segmentation ────────────────────────────────────────────────
        self._print("\n--- Segmentation ---")
        try:
            seg_results = ccd.detect_colour_checkers_segmentation(
                img_f,
                additional_data=True,
                apply_cctf_decoding=apply_decode,
            )
            self._print(f"  Results: {len(seg_results)} detection(s)")
            for i, det in enumerate(seg_results):
                sw = np.array(det.swatch_colours, dtype=np.float32)
                quad = np.array(det.quadrilateral, dtype=np.float32)
                self._print(f"  Detection {i}: swatches={sw.shape} quad={quad.shape}")

                # Scale quad from [0,1] to pixels
                quad_px = quad.copy()
                quad_px[:, 0] *= w
                quad_px[:, 1] *= h

                _draw_quad(vis, quad_px, (0, 255, 0), 3)
                _draw_swatches(vis, det)

                # Log swatch values
                self._print(f"  Swatch colours (as returned, float [0,1]):")
                for j in range(min(24, sw.shape[0])):
                    self._print(f"    {j+1:2d} {CC24_NAMES[j]:16s}: "
                                f"R={sw[j,0]:.4f} G={sw[j,1]:.4f} B={sw[j,2]:.4f}")

                # Swatch strip
                strip = _swatch_strip(sw, f"seg #{i}")
                strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB) if strip.shape[2] == 3 else strip
                results_images.append(strip_rgb)

                # Checker image
                cc_img = np.array(det.colour_checker, dtype=np.float32)
                cc_u8 = np.clip(cc_img * 255, 0, 255).astype(np.uint8)
                results_images.append(cc_u8)

        except Exception as e:
            self._print(f"  ERROR: {e}")
            traceback.print_exc()

        # ── YOLO Inference ──────────────────────────────────────────────
        if self._use_inference.isChecked() and HAVE_CCD_INFERENCE:
            self._print("\n--- YOLO Inference ---")
            try:
                inf_results = ccd.detect_colour_checkers_inference(
                    img_f,
                    additional_data=True,
                    apply_cctf_decoding=apply_decode,
                )
                self._print(f"  Results: {len(inf_results)} detection(s)")
                for i, det in enumerate(inf_results):
                    sw = np.array(det.swatch_colours, dtype=np.float32)
                    quad = np.array(det.quadrilateral, dtype=np.float32)
                    self._print(f"  Detection {i}: swatches={sw.shape} quad={quad.shape}")

                    quad_px = quad.copy()
                    quad_px[:, 0] *= w
                    quad_px[:, 1] *= h
                    _draw_quad(vis, quad_px, (0, 200, 255), 3)

                    self._print(f"  Swatch colours:")
                    for j in range(min(24, sw.shape[0])):
                        self._print(f"    {j+1:2d} {CC24_NAMES[j]:16s}: "
                                    f"R={sw[j,0]:.4f} G={sw[j,1]:.4f} B={sw[j,2]:.4f}")

                    strip = _swatch_strip(sw, f"yolo #{i}")
                    strip_rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB) if strip.shape[2] == 3 else strip
                    results_images.append(strip_rgb)

            except Exception as e:
                self._print(f"  ERROR: {e}")
                traceback.print_exc()

        # ── Compose result image ────────────────────────────────────────
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        results_images[0] = vis_rgb  # replace original with annotated

        # Stack all result images vertically
        max_w = max(im.shape[1] for im in results_images)
        padded = []
        for im in results_images:
            if im.shape[1] < max_w:
                pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8)
                im = np.concatenate([im, pad], axis=1)
            padded.append(im)
        composite = np.concatenate(padded, axis=0)

        self._image_label.setPixmap(_np_to_qpixmap(composite, max_w=900))

        # Save debug output
        out_dir = os.path.dirname(self._image_path) or "."
        base = os.path.splitext(os.path.basename(self._image_path))[0]
        out_path = os.path.join(out_dir, f"{base}_cc_debug.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        self._print(f"\nSaved: {out_path}")

        self._status.setText(f"Done — saved {os.path.basename(out_path)}")
        self._print("Done.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    win = CCDebugWindow(initial_file=initial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
