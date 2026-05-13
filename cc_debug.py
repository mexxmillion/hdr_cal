#!/usr/bin/env python3
"""cc_debug.py — ColorChecker detection debugger. Drag & drop JPG/PNG, hit Process."""
import sys, os, traceback
import numpy as np, cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog, QScrollArea,
    QSplitter, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if os.path.isdir(_MODELS):
    os.environ.setdefault("COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY", _MODELS)

try:
    import colour_checker_detection as ccd
    HAVE_CCD = True
except ImportError:
    HAVE_CCD = False

HAVE_YOLO = False
_YOLO_MODEL = None
if HAVE_CCD:
    try:
        _ = ccd.detect_colour_checkers_inference
        HAVE_YOLO = True
        _repo = os.environ.get("COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY",
                               os.path.join(os.path.expanduser("~"), ".colour-science", "colour-checker-detection"))
        _mp = os.path.join(_repo, "colour-checker-detection-l-seg.pt")
        _YOLO_MODEL = _mp if os.path.isfile(_mp) else None
    except AttributeError:
        pass

CC24 = ["dark skin","light skin","blue sky","foliage","blue flower","bluish green",
        "orange","purplish blue","moderate red","purple","yellow green","orange yellow",
        "blue","green","red","yellow","magenta","cyan",
        "white","neutral 8","neutral 6.5","neutral 5","neutral 3.5","black"]

def _to_pix(img, max_w=800):
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w; img = cv2.resize(img, (max_w, int(h * s)))
        h, w = img.shape[:2]
    return QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888))

def _quad_px(q, w, h):
    qp = np.array(q, dtype=np.float32).copy(); qp[:, 0] *= w; qp[:, 1] *= h; return qp

def _draw_quad(img, q, c=(0,255,0), t=2):
    cv2.polylines(img, [np.int32(q).reshape(-1,1,2)], True, c, t)

def _strip(sw, label=""):
    pw, ph = 40, 40; s = np.zeros((ph+18, pw*24, 3), dtype=np.uint8)
    for i in range(min(24, len(sw))):
        c = np.clip(sw[i]*255, 0, 255).astype(np.uint8)
        s[18:, i*pw:(i+1)*pw] = c[None,None,:]
        cv2.putText(s, str(i+1), (i*pw+10, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    if label: cv2.putText(s, label, (4, s.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180,255,180), 1)
    return s

class DropLabel(QLabel):
    def __init__(self): super().__init__(); self.setAcceptDrops(True); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        u = e.mimeData().urls()
        if u and self._cb: self._cb(u[0].toLocalFile())
    _cb = None

class Win(QMainWindow):
    def __init__(self, init=None):
        super().__init__(); self.setWindowTitle("CC Detection Debugger"); self.setMinimumSize(800, 600)
        self._path = self._rgb = None
        c = QWidget(); self.setCentralWidget(c); lay = QVBoxLayout(c)
        row = QHBoxLayout()
        b = QPushButton("Browse"); b.clicked.connect(self._browse); row.addWidget(b)
        self._go = QPushButton("Process"); self._go.clicked.connect(self._run); self._go.setEnabled(False); row.addWidget(self._go)
        self._mode = QComboBox()
        modes = ["internal (two-pass)"] if HAVE_YOLO else []
        if HAVE_YOLO:
            modes.append("yolo (ccd wrapper)")
        modes += ["segmentation", "both"]
        self._mode.addItems(modes); row.addWidget(QLabel("Method:")); row.addWidget(self._mode)
        row.addStretch(); lay.addLayout(row)
        sp = QSplitter(Qt.Orientation.Vertical)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        self._img = DropLabel(); self._img.setText("Drop image here"); self._img._cb = self._load
        self._img.setMinimumSize(400, 250)
        self._img.setStyleSheet("QLabel{border:2px dashed #555;background:#1a1a2e;color:#666}")
        sc.setWidget(self._img); sp.addWidget(sc)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet("QTextEdit{background:#0d0d1a;color:#c0c0c0;font-family:monospace;font-size:11px}")
        sp.addWidget(self._log); sp.setSizes([450, 150]); lay.addWidget(sp)
        self._st = QLabel("Ready"); lay.addWidget(self._st)
        # Startup diagnostics
        self._p(f"colour-checker-detection : {'yes' if HAVE_CCD else 'NO — pip install colour-checker-detection'}")
        self._p(f"YOLO inference API       : {'yes' if HAVE_YOLO else 'no (ultralytics not installed)'}")
        if HAVE_YOLO:
            if _YOLO_MODEL:
                sz = os.path.getsize(_YOLO_MODEL) / (1024*1024)
                self._p(f"YOLO model FOUND         : {_YOLO_MODEL} ({sz:.1f} MB)")
            else:
                self._p(f"YOLO model NOT FOUND     : expected at {_mp}")
                self._p(f"  Will attempt download from HuggingFace on first run")
        self._p(f"Model repo               : {os.environ.get('COLOUR_SCIENCE__COLOUR_CHECKER_DETECTION__REPOSITORY', '(default)')}")
        self._p("")
        if init: self._load(init)

    def _p(self, m): self._log.append(m); QApplication.processEvents()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.jpg *.jpeg *.png *.tif *.bmp);;All (*)")
        if p: self._load(p)

    def _load(self, p):
        if not os.path.isfile(p): self._p(f"Not found: {p}"); return
        self._path = p; bgr = cv2.imread(p)
        if bgr is None: self._p("cv2.imread failed"); return
        self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = self._rgb.shape[:2]
        self._p(f"Loaded: {p}  ({w}x{h})")
        self._img.setPixmap(_to_pix(self._rgb)); self._img.setStyleSheet(""); self._go.setEnabled(True)
        self._st.setText(os.path.basename(p))

    def _run(self):
        if self._rgb is None or not HAVE_CCD: return
        self._p("\n" + "="*50)
        self._st.setText("Detecting..."); QApplication.processEvents()
        img = self._rgb.copy(); h, w = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        parts = [img.copy()]
        mode = self._mode.currentText()
        run_seg = mode in ("segmentation", "both")
        run_yolo = mode in ("yolo (ccd wrapper)", "both") and HAVE_YOLO
        run_internal = mode.startswith("internal") and HAVE_YOLO

        if run_internal:
            self._p("\n--- Internal two-pass (same code path as the GUI) ---")
            try:
                # The hdri_cal repo's own pipeline. Feeding a rectilinear JPEG
                # (e.g. rect_tile.jpg from cc_debug_out/colorchecker/) into this
                # mode tests *only* the detection logic, isolating it from the
                # ERP→rectilinear projection step.
                import colorchecker_erp as ce
                # Treat the image as already linear-ish for detector purposes:
                # the detector display-maps internally, so feeding it the
                # ERP debug JPEG (which is already display-mapped) just goes
                # through the tone-mapper a second time — that's still valid
                # for testing geometry. For best results, feed the *linear*
                # ERP crop saved by the pipeline (or any JPEG; YOLO is robust).
                tile_linear = img_f.copy()
                cc24_ref = ce.get_cc24_reference("acescg")
                # map_uv only matters for ERP-space backprojection — pass a
                # dummy identity map that just maps tile px → tile px / size.
                hh, ww = tile_linear.shape[:2]
                xs = np.linspace(0.0, 1.0, ww, dtype=np.float32)
                ys = np.linspace(0.0, 1.0, hh, dtype=np.float32)
                gu, gv = np.meshgrid(xs, ys)
                dummy_map = np.stack([gu, gv], axis=-1).astype(np.float32)
                # Match the GUI: segmentation first, YOLO fallback.
                det = ce._detect_in_tile_segmentation(
                    tile_linear, dummy_map,
                    yaw=0.0, pitch=0.0, tile_fov_deg=70.0,
                    cc24_ref=cc24_ref,
                    debug_dir=None, tile_label="ccdebug",
                )
                if det is None:
                    self._p("  segmentation found nothing — falling back to YOLO")
                    det = ce._detect_in_tile(
                        tile_linear, dummy_map,
                        yaw=0.0, pitch=0.0, tile_fov_deg=70.0,
                        cc24_ref=cc24_ref,
                        debug_dir=None, tile_label="ccdebug",
                        save_debug=False,
                    )
                if det is None:
                    self._p("  Internal pipeline: NO detection (see console "
                            "for [cc-erp] pass-1/pass-2 logs above).")
                else:
                    self._p(f"  Internal pipeline: HIT  conf={det.confidence:.3f}  "
                            f"method={det.detection_method}")
                    self._p(f"  Quad (TL,TR,BR,BL): "
                            f"{[(round(float(p[0]),1), round(float(p[1]),1)) for p in det.quad_tile]}")
                    _draw_quad(vis, det.quad_tile, (255, 100, 255), 3)
                    sw = det.swatches_linear
                    for j in range(min(24, sw.shape[0])):
                        self._p(f"    {j+1:2d} {CC24[j]:16s}  "
                                f"R={sw[j,0]:.4f} G={sw[j,1]:.4f} B={sw[j,2]:.4f}")
                    parts.append(_strip(sw, "internal two-pass"))
                    if det.cc_rectified is not None:
                        rect_u8 = np.clip(det.cc_rectified * 255, 0, 255).astype(np.uint8)
                        parts.append(rect_u8)
            except Exception as e:
                self._p(f"  ERROR: {e}"); traceback.print_exc()

        if run_yolo:
            self._p("\n--- YOLO Inference ---")
            self._p(f"  Model: {_YOLO_MODEL or 'will download'}")
            try:
                res = ccd.detect_colour_checkers_inference(img_f, additional_data=True, apply_cctf_decoding=True)
                self._p(f"  Detections: {len(res)}")
                for i, d in enumerate(res):
                    sw = np.array(d.swatch_colours, dtype=np.float32)
                    q = _quad_px(d.quadrilateral, w, h)
                    _draw_quad(vis, q, (0,200,255), 3)
                    self._p(f"  #{i} swatches={sw.shape}")
                    for j in range(min(24, sw.shape[0])):
                        self._p(f"    {j+1:2d} {CC24[j]:16s}  R={sw[j,0]:.4f} G={sw[j,1]:.4f} B={sw[j,2]:.4f}")
                    parts.append(_strip(sw, f"yolo #{i}"))
                    cc = np.array(d.colour_checker, dtype=np.float32)
                    parts.append(np.clip(cc*255, 0, 255).astype(np.uint8))
                if not res: self._p("  No chart detected by YOLO")
            except Exception as e:
                self._p(f"  ERROR: {e}"); traceback.print_exc()

        if run_seg:
            self._p("\n--- Segmentation ---")
            try:
                res = ccd.detect_colour_checkers_segmentation(img_f, additional_data=True, apply_cctf_decoding=True)
                self._p(f"  Detections: {len(res)}")
                for i, d in enumerate(res):
                    sw = np.array(d.swatch_colours, dtype=np.float32)
                    q = _quad_px(d.quadrilateral, w, h)
                    _draw_quad(vis, q, (0,255,0), 2)
                    self._p(f"  #{i} swatches={sw.shape}")
                    for j in range(min(24, sw.shape[0])):
                        self._p(f"    {j+1:2d} {CC24[j]:16s}  R={sw[j,0]:.4f} G={sw[j,1]:.4f} B={sw[j,2]:.4f}")
                    parts.append(_strip(sw, f"seg #{i}"))
                    cc = np.array(d.colour_checker, dtype=np.float32)
                    parts.append(np.clip(cc*255, 0, 255).astype(np.uint8))
                if not res: self._p("  No chart detected by segmentation")
            except Exception as e:
                self._p(f"  ERROR: {e}"); traceback.print_exc()

        # Compose
        parts[0] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        mw = max(p.shape[1] for p in parts)
        padded = []
        for p in parts:
            if p.shape[1] < mw: p = np.concatenate([p, np.zeros((p.shape[0], mw-p.shape[1], 3), np.uint8)], 1)
            padded.append(p)
        comp = np.concatenate(padded, 0)
        self._img.setPixmap(_to_pix(comp, 900))
        out = os.path.join(os.path.dirname(self._path) or ".", os.path.splitext(os.path.basename(self._path))[0] + "_cc_debug.jpg")
        cv2.imwrite(out, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        self._p(f"\nSaved: {out}"); self._st.setText(f"Done — {os.path.basename(out)}")

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    w = Win(sys.argv[1] if len(sys.argv) > 1 else None); w.show(); sys.exit(app.exec())
