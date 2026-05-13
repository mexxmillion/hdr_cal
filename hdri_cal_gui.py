"""
hdri_cal_gui.py  —  PySide6 front-end for the HDRI calibration pipeline.

Usage:
    python hdri_cal_gui.py

Requires:
    pip install PySide6
    (plus all hdri_cal.py dependencies)
"""

from __future__ import annotations
import sys, os, json, traceback, types
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QSlider,
    QGroupBox, QTextEdit, QFileDialog, QProgressBar,
    QScrollArea, QSizePolicy, QTabWidget, QFrame, QToolBar,
    QStatusBar, QMessageBox, QAbstractItemView, QDialog,
)
from PySide6.QtCore import Qt, QSize, QTimer, QObject, QRunnable, QThreadPool, Signal, Slot, QPoint, QPointF
from PySide6.QtGui import (
    QPixmap, QDragEnterEvent, QDropEvent, QColor,
    QPainter, QPen, QImage, QMouseEvent,
)

import numpy as np
import cv2

# ── Dark VFX palette ───────────────────────────────────────────────────────────
VFX_STYLE = """
QMainWindow, QWidget {
    background-color: #171720;
    color: #c8c8d8;
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 12px;
}
QSplitter::handle { background: #252530; width: 2px; height: 2px; }
QGroupBox {
    border: 1px solid #30303e;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
    color: #7878a0;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #252530;
    border: 1px solid #35354a;
    border-radius: 3px;
    padding: 5px 14px;
    color: #b0b0c8;
    font-size: 12px;
}
QPushButton:hover   { background-color: #2e2e3e; border-color: #505070; }
QPushButton:pressed { background-color: #1e1e2a; }
QPushButton:disabled { color: #404055; border-color: #252530; }
QPushButton#run_btn {
    background-color: #1a3d28;
    border-color: #286040;
    color: #70f0a0;
    font-weight: 700;
    font-size: 13px;
    padding: 7px 22px;
}
QPushButton#run_btn:hover    { background-color: #1e4a30; border-color: #347a50; }
QPushButton#run_btn:disabled { background-color: #16241c; color: #305040; }
QPushButton#validate_btn {
    background-color: #172240;
    border-color: #284870;
    color: #70b8f8;
    font-weight: 600;
}
QPushButton#validate_btn:hover { background-color: #1c2a50; }
QPushButton#abort_btn {
    background-color: #3a1818;
    border-color: #602828;
    color: #f88080;
    font-weight: 600;
}
QPushButton#abort_btn:hover { background-color: #481c1c; }
QPushButton#adv_toggle {
    background-color: #1a1a28;
    border: 1px solid #2a2a40;
    border-radius: 3px;
    padding: 5px 12px;
    color: #606090;
    font-size: 11px;
    text-align: left;
}
QPushButton#adv_toggle:hover  { border-color: #40408a; color: #9090c8; }
QPushButton#adv_toggle:checked { background-color: #1a1e38; border-color: #3a3a80; color: #8888d0; }
QComboBox {
    background-color: #1e1e2a;
    border: 1px solid #35354a;
    border-radius: 3px;
    padding: 3px 8px;
    color: #b8b8d0;
    min-width: 120px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #1e1e2a;
    border: 1px solid #45455a;
    selection-background-color: #253050;
    color: #b8b8d0;
}
QCheckBox { spacing: 6px; color: #a0a0c0; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #45455a; border-radius: 2px;
    background: #1a1a26;
}
QCheckBox::indicator:checked { background: #285a40; border-color: #3a8050; }
QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1a1a26;
    border: 1px solid #35354a;
    border-radius: 3px;
    padding: 3px 6px;
    color: #b8b8d0;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: #405090; }
QTextEdit {
    background-color: #11111a;
    border: 1px solid #252535;
    border-radius: 3px;
    color: #90c890;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
}
QListWidget {
    background-color: #14141e;
    border: 1px solid #252535;
    border-radius: 3px;
    color: #b0b0c8;
}
QListWidget::item { padding: 5px 8px; border-bottom: 1px solid #1c1c28; }
QListWidget::item:selected { background-color: #1c2a50; color: #80b0f8; }
QListWidget::item:hover    { background-color: #18182a; }
QProgressBar {
    background-color: #1a1a26;
    border: 1px solid #252535;
    border-radius: 2px;
    height: 5px;
}
QProgressBar::chunk { background-color: #287840; border-radius: 2px; }
QTabWidget::pane { border: 1px solid #252535; }
QTabBar::tab {
    background: #1a1a26; border: 1px solid #252535;
    padding: 5px 14px; color: #606080; border-bottom: none;
}
QTabBar::tab:selected { background: #171720; color: #c0c0d8; border-bottom: 1px solid #171720; }
QScrollBar:vertical { background: #14141e; width: 7px; border: none; }
QScrollBar::handle:vertical { background: #303048; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QLabel#header    { font-size: 17px; font-weight: 700; color: #e0e0f0; letter-spacing: -0.5px; }
QLabel#subheader { font-size: 11px; color: #505065; }
QLabel#section   { font-size: 10px; font-weight: 600; color: #606085; text-transform: uppercase; letter-spacing: 0.8px; }
QLabel#status_ok   { color: #58c080; font-size: 11px; }
QLabel#status_warn { color: #d89828; font-size: 11px; }
QLabel#status_err  { color: #d84848; font-size: 11px; }
QFrame#divider  { background-color: #252535; max-height: 1px; }
QFrame#drop_zone {
    border: 2px dashed #303050;
    border-radius: 8px;
    background-color: #12121c;
}
QFrame#drop_zone[dragover="true"] { border-color: #406080; background-color: #12181e; }
QStatusBar { background-color: #111118; color: #505065; font-size: 11px; }
QToolBar   { background-color: #14141e; border-bottom: 1px solid #252530; spacing: 4px; }
"""

ICON_WAIT    = "○"
ICON_RUNNING = "◉"
ICON_OK      = "✓"
ICON_WARN    = "⚠"
ICON_ERROR   = "✕"


# ── Pipeline config dataclass ──────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    input:              str = ""
    out:                str = ""
    debug_dir:          str = "debug_hdri"
    res:                str = "full"
    colorspace:         Optional[str] = None
    validate_only:      bool = False

    # WB / exposure
    calibration_mode:   str = "auto"
    wb_source:          str = "auto"
    exposure_source:    str = "auto"
    integration_mode:   str = "full_sphere"
    sphere_solve:       str = "auto"
    final_balance_target: str = "none"
    base_intensity:     float = 1.0
    base_temperature:   float = 6500.0
    base_tint:          float = 0.0

    # Hot lobe
    sun_threshold:      float = 0.1
    sun_upper_only:     bool = False

    # Centering
    center_hdri:        bool = True

    # Gain / energy
    lobe_neutralise:    float = 1.0
    albedo:             float = 0.18
    sphere_res:         int = 96
    sun_gain_ceiling:   float = 2000.0
    sun_gain_rolloff:   float = 500.0

    # Optional post-process validation suite (rendered sphere vs analytical,
    # 6-direction irradiance map, E=π check, grey-card predictions).
    # Off by default — adds ~1–2 s and is a debug aid, not a calibration step.
    validate_energy:    bool = False

    # Chart detection
    cc_min_confidence:  float = 0.50

    # User-drawn search rectangle on the latlong preview (u0, v0, u1, v1), or None.
    # When set, auto-detect runs inside this rect only — no sphere sweep.
    cc_search_rect:     Optional[tuple] = None

    # Manual chart corners (list of 4 (u,v) tuples in 0-1 space, or None for auto)
    cc_manual_corners:  Optional[list] = None



def config_to_namespace(cfg: PipelineConfig):
    ns = types.SimpleNamespace()
    for k, v in cfg.__dict__.items():
        setattr(ns, k, v)

    # Aliases / derived attrs expected by hdri_cal._run_pipeline
    ns.hot_threshold        = cfg.sun_threshold
    ns.upper_only           = cfg.sun_upper_only
    ns.sun_blur_px          = 0

    if getattr(ns, "sphere_solve", "auto") == "auto":
        ns.sphere_solve = "energy_conservation"

    # Attrs not in PipelineConfig but referenced in pipeline
    _defaults = {
        "kelvin":               None,
        "rgb_scale":            None,
        "dome_wb":              None,
        "metering_mode":        "upper_hemi_irradiance",
        "meter_stat":           "median",
        "meter_target":         1.0,
        "swatch_size":          5,
        "swatch":               None,
        "wb_swatch":            None,
        "sphere_target":        "irradiance",
        "exposure_scale":       None,
        "chart_facing":         "auto",
        "sun_upper_only":       False,
        "sun_blur_px":          0,
        "center_elevation":     False,
        "sphere_res":           96,
        "direct_highlight_target": 0.32,
        "final_balance_target": "none",
        "target_peak_ratio":    2.5,
        "ref_sphere":           None,
        "ref_sphere_cx":        None,
        "ref_sphere_cy":        None,
        "ref_sphere_r":         None,
        "ref_sphere_albedo":    0.18,
        "colorchecker":         None,
        "colorchecker_in_hdri": False,
        "cc_read_backend":      "colour",
        "cc_compare_backends":  False,
        "cc_min_confidence":    0.50,
        "cc_search_rect":       None,
        "cc_manual_corners":    None,
        "validate_only":        False,
        "validate_energy":      False,
    }
    for attr, default in _defaults.items():
        if not hasattr(ns, attr):
            setattr(ns, attr, default)

    return ns


# ── Worker thread ──────────────────────────────────────────────────────────────
class PipelineSignals(QObject):
    log      = Signal(str)
    warning  = Signal(str)
    preview  = Signal(str)
    done     = Signal(dict)
    error    = Signal(str)


class PipelineWorker(QRunnable):
    def __init__(self, cfg: PipelineConfig, signals: PipelineSignals):
        super().__init__()
        self.cfg = cfg; self.signals = signals; self._abort = False

    def abort(self): self._abort = True

    @Slot()
    def run(self):
        import hdri_cal as hc
        result = {"warnings": [], "report": {}, "previews": []}
        orig_log = hc.log; orig_warn = hc.warn

        def _log(msg):
            self.signals.log.emit(str(msg)); orig_log(msg)
        def _warn(msg):
            self.signals.warning.emit(str(msg))
            result["warnings"].append(str(msg)); orig_warn(msg)

        hc.log = _log; hc.warn = _warn
        try:
            hc._run_pipeline(config_to_namespace(self.cfg))
            dd = self.cfg.debug_dir
            for fname in [
                "01_wb_preview.png", "02_exposed_preview.png",
                "03_hot_mask.png",   "07_corrected_preview.png",
                "08_verify_sphere_final.png",
                "07b_final_balanced_preview.png",
                "colorchecker/cc_detected_tile.jpg",
                "colorchecker/cc_rectified_final.jpg",
                "colorchecker/cc_swatch_comparison.jpg",
            ]:
                p = os.path.join(dd, fname)
                if os.path.exists(p):
                    result["previews"].append(p); self.signals.preview.emit(p)
            rp = os.path.join(dd, "report.json")
            if os.path.exists(rp):
                with open(rp) as f: result["report"] = json.load(f)
            self.signals.done.emit(result)
        except Exception as e:
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            hc.log = orig_log; hc.warn = orig_warn


# ── Drop zone ──────────────────────────────────────────────────────────────────
class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone"); self.setAcceptDrops(True)
        self.setMinimumHeight(78); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignCenter)
        icon = QLabel("⊕"); icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 22px; color: #303050; border: none;")
        text = QLabel("Drop EXR / HDR / PNG / WebP  ·  or click to browse")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("color: #404060; font-size: 11px; border: none;")
        lay.addWidget(icon); lay.addWidget(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDRI files", "",
            "HDRI Images (*.exr *.hdr *.jpg *.jpeg *.png *.webp);;All Files (*.*)")
        if paths: self.files_dropped.emit(paths)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("dragover", "true")
            self.style().unpolish(self); self.style().polish(self)

    def dragLeaveEvent(self, e):
        self.setProperty("dragover", "false")
        self.style().unpolish(self); self.style().polish(self)

    def dropEvent(self, e: QDropEvent):
        self.setProperty("dragover", "false")
        self.style().unpolish(self); self.style().polish(self)
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith((".exr",".hdr",".jpg",".jpeg",".png",".webp"))]
        if paths: self.files_dropped.emit(paths)


# ── Rubber-band search rect for the Source preview ────────────────────────────

class RectDrawLabel(QLabel):
    """QLabel that lets the user drag-out a rectangle. Emits rect_uv on release
    as a normalised (u0, v0, u1, v1) tuple in image coords [0,1]."""
    rect_drawn = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pm: Optional[QPixmap] = None
        self._dragging = False
        self._start: Optional[QPointF] = None  # image coords
        self._end: Optional[QPointF] = None    # image coords
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)

    def set_base_pixmap(self, pm: QPixmap):
        self._base_pm = pm
        self._dragging = False
        self._start = None
        self._end = None
        self._redraw()

    def clear_rect(self):
        self._start = None
        self._end = None
        self._dragging = False
        self._redraw()

    def set_rect_uv(self, rect_uv: Optional[tuple]):
        if rect_uv is None or self._base_pm is None or self._base_pm.isNull():
            self.clear_rect()
            return
        w, h = self._base_pm.width(), self._base_pm.height()
        u0, v0, u1, v1 = rect_uv
        self._start = QPointF(u0 * w, v0 * h)
        self._end = QPointF(u1 * w, v1 * h)
        self._dragging = False
        self._redraw()

    def _widget_to_image(self, widget_pos: QPointF) -> Optional[QPointF]:
        if self._base_pm is None or self._base_pm.isNull():
            return None
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        lbl_w, lbl_h = self.width(), self.height()
        pm_w, pm_h = pm.width(), pm.height()
        ox = (lbl_w - pm_w) / 2.0
        oy = (lbl_h - pm_h) / 2.0
        ix = (widget_pos.x() - ox) * self._base_pm.width() / pm_w
        iy = (widget_pos.y() - oy) * self._base_pm.height() / pm_h
        ix = max(0, min(ix, self._base_pm.width() - 1))
        iy = max(0, min(iy, self._base_pm.height() - 1))
        return QPointF(ix, iy)

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return
        pos = self._widget_to_image(ev.position())
        if pos is None:
            return
        self._dragging = True
        self._start = pos
        self._end = pos
        self._redraw()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if not self._dragging:
            return
        pos = self._widget_to_image(ev.position())
        if pos is None:
            return
        self._end = pos
        self._redraw()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if not self._dragging or self._start is None or self._end is None:
            return
        self._dragging = False
        self._redraw()
        if self._base_pm is None:
            return
        w, h = self._base_pm.width(), self._base_pm.height()
        u0 = min(self._start.x(), self._end.x()) / w
        v0 = min(self._start.y(), self._end.y()) / h
        u1 = max(self._start.x(), self._end.x()) / w
        v1 = max(self._start.y(), self._end.y()) / h
        if (u1 - u0) * w < 4 or (v1 - v0) * h < 4:
            self._start = None
            self._end = None
            self._redraw()
            return
        self.rect_drawn.emit((u0, v0, u1, v1))

    def _redraw(self):
        if self._base_pm is None or self._base_pm.isNull():
            self.setPixmap(QPixmap())
            return
        pm = self._base_pm.copy()
        if self._start is not None and self._end is not None:
            p = QPainter(pm)
            pen = QPen(QColor(255, 200, 64), 2, Qt.DashLine)
            p.setPen(pen)
            x0 = int(min(self._start.x(), self._end.x()))
            y0 = int(min(self._start.y(), self._end.y()))
            x1 = int(max(self._start.x(), self._end.x()))
            y1 = int(max(self._start.y(), self._end.y()))
            p.drawRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            p.end()
        # Scale to widget if needed
        scaled = pm.scaled(
            max(self.width(), 1), max(self.height(), 1),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._redraw()


# ── ColorChecker corner picker dialog ─────────────────────────────────────────

class _PickerCanvas(QLabel):
    """Image label that lets the user click up to 4 corners. Once 4 corners
    are placed, draws a CC24 reference-swatch overlay registered to the quad."""
    corner_changed = Signal()

    # CC24 layout shared with colorchecker_erp.
    _CC_COLS = 6
    _CC_ROWS = 4

    # Hit-test radius for grabbing an existing corner (image-pixel space).
    _GRAB_RADIUS = 14.0

    def __init__(self, pixmap: QPixmap, ref_swatches_u8: Optional[np.ndarray] = None,
                 parent=None):
        super().__init__(parent)
        self._base_pm = pixmap
        self._corners: list[QPointF] = []
        # ref_swatches_u8: (24, 3) uint8 RGB values for overlay drawing.
        self._ref = ref_swatches_u8
        self._drag_idx: Optional[int] = None  # which corner is being dragged
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def corners_xy(self) -> list[tuple[float, float]]:
        """Return corners in image pixel coordinates."""
        return [(float(p.x()), float(p.y())) for p in self._corners]

    def reset_corners(self):
        self._corners.clear()
        self._drag_idx = None
        self._redraw()
        self.corner_changed.emit()

    def _nearest_corner(self, pos: QPointF) -> Optional[int]:
        """Return the index of the closest corner within grab radius, or None."""
        best_i = None
        best_d2 = self._GRAB_RADIUS ** 2
        for i, c in enumerate(self._corners):
            dx = c.x() - pos.x()
            dy = c.y() - pos.y()
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return
        pos = self._widget_to_image(ev.position())
        if pos is None:
            return
        # If clicking on an existing corner, start dragging it.
        idx = self._nearest_corner(pos)
        if idx is not None:
            self._drag_idx = idx
            self.setCursor(Qt.ClosedHandCursor)
            return
        # Otherwise add a new corner (up to 4).
        if len(self._corners) < 4:
            self._corners.append(pos)
            self._drag_idx = len(self._corners) - 1
            self.setCursor(Qt.ClosedHandCursor)
            self._redraw()
            self.corner_changed.emit()

    def mouseMoveEvent(self, ev: QMouseEvent):
        pos = self._widget_to_image(ev.position())
        if pos is None:
            return
        if self._drag_idx is not None:
            self._corners[self._drag_idx] = pos
            self._redraw()
            self.corner_changed.emit()
            return
        # Hover: change cursor when over a corner.
        if self._nearest_corner(pos) is not None:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return
        if self._drag_idx is not None:
            self._drag_idx = None
            self.setCursor(Qt.CrossCursor)

    def _widget_to_image(self, widget_pos: QPointF) -> Optional[QPointF]:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        lbl_w, lbl_h = self.width(), self.height()
        pm_w, pm_h = pm.width(), pm.height()
        ox = (lbl_w - pm_w) / 2.0
        oy = (lbl_h - pm_h) / 2.0
        ix = (widget_pos.x() - ox) * self._base_pm.width() / pm_w
        iy = (widget_pos.y() - oy) * self._base_pm.height() / pm_h
        ix = max(0, min(ix, self._base_pm.width() - 1))
        iy = max(0, min(iy, self._base_pm.height() - 1))
        return QPointF(ix, iy)

    def _draw_swatch_overlay(self, painter: QPainter):
        """Once 4 corners are placed, draw 24 swatch outlines coloured from
        the CC24 reference, mapped through the user's quad."""
        if len(self._corners) != 4 or self._ref is None:
            return
        src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                       dtype=np.float32)
        dst = np.array([[p.x(), p.y()] for p in self._corners], dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(src, dst)
        except Exception:
            return
        cols, rows = self._CC_COLS, self._CC_ROWS
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                # cell centre and half-extent in normalised quad space
                cx = (c + 0.5) / cols
                cy = (r + 0.5) / rows
                # Inner sample region ~ 33% of cell
                hx = 0.33 / cols * 0.5
                hy = 0.33 / rows * 0.5
                corners_n = np.array([
                    [cx - hx, cy - hy],
                    [cx + hx, cy - hy],
                    [cx + hx, cy + hy],
                    [cx - hx, cy + hy],
                ], dtype=np.float32).reshape(-1, 1, 2)
                warped = cv2.perspectiveTransform(corners_n, M).reshape(-1, 2)
                rcol = self._ref[idx]
                fill = QColor(int(rcol[0]), int(rcol[1]), int(rcol[2]), 200)
                painter.setBrush(fill)
                painter.setPen(QPen(QColor(0, 0, 0, 220), 1))
                poly = [QPoint(int(round(x)), int(round(y))) for x, y in warped]
                painter.drawPolygon(poly)

    def _redraw(self):
        pm = self._base_pm.copy()
        painter = QPainter(pm)
        # Swatch overlay first (under the quad outline)
        self._draw_swatch_overlay(painter)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for i, pt in enumerate(self._corners):
            x, y = int(pt.x()), int(pt.y())
            painter.drawEllipse(QPoint(x, y), 6, 6)
            painter.drawText(x + 10, y - 4, f"{i+1}")
        if len(self._corners) >= 2:
            for i in range(len(self._corners)):
                j = (i + 1) % len(self._corners)
                if j < len(self._corners):
                    painter.drawLine(
                        int(self._corners[i].x()), int(self._corners[i].y()),
                        int(self._corners[j].x()), int(self._corners[j].y()))
            if len(self._corners) == 4:
                painter.drawLine(
                    int(self._corners[3].x()), int(self._corners[3].y()),
                    int(self._corners[0].x()), int(self._corners[0].y()))
        painter.end()
        self.setPixmap(pm)


def _cc24_reference_u8() -> np.ndarray:
    """24×3 uint8 sRGB values for the CC24, for overlay drawing only."""
    try:
        from colorchecker_erp import CC24_LINEAR_SRGB
        ref_lin = CC24_LINEAR_SRGB
    except Exception:
        # Fallback: tiny inline grey ramp so the overlay still works.
        ref_lin = np.tile(np.linspace(0.1, 0.9, 24, dtype=np.float32)[:, None],
                          (1, 3))
    # Apply sRGB encoding for display.
    a = np.clip(ref_lin, 0.0, 1.0).astype(np.float32)
    enc = np.where(a <= 0.0031308,
                   12.92 * a,
                   1.055 * np.power(a, 1.0 / 2.4) - 0.055)
    return np.clip(enc * 255.0, 0, 255).astype(np.uint8)


class ChartCornerPicker(QDialog):
    """Modal dialog: shows a rectilinear crop of the user-drawn search rect
    on the latlong, lets the user click 4 chart corners, and returns the
    corners back in ERP UV space.

    Pass either:
      - (erp_linear, rect_uv) to compute a rectilinear crop, OR
      - image_path to a pre-rendered image (legacy path; returns corners in
        that image's own UV space).
    """

    def __init__(self,
                 erp_linear: Optional[np.ndarray] = None,
                 rect_uv: Optional[tuple] = None,
                 image_path: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Place ColorChecker Corners (TL → TR → BR → BL)")
        self.setMinimumSize(900, 500)
        self.resize(1200, 750)
        self.corners_uv: Optional[list[tuple[float, float]]] = None
        self._map_uv: Optional[np.ndarray] = None  # for ERP backprojection
        self._crop_w = 0
        self._crop_h = 0

        lay = QVBoxLayout(self)

        pm: Optional[QPixmap] = None
        if erp_linear is not None and rect_uv is not None:
            pm = self._build_crop_pixmap(erp_linear, rect_uv)
            if pm is None:
                lay.addWidget(QLabel("Failed to build rectilinear crop."))
                return
        elif image_path is not None:
            pm = QPixmap(image_path)
            if pm.isNull():
                lay.addWidget(QLabel("Failed to load preview image."))
                return
        else:
            lay.addWidget(QLabel("ChartCornerPicker: no input image provided."))
            return

        self._crop_w = pm.width()
        self._crop_h = pm.height()

        ref_u8 = _cc24_reference_u8()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._canvas = _PickerCanvas(pm, ref_swatches_u8=ref_u8)
        self._canvas.corner_changed.connect(self._on_corners_changed)
        scroll.setWidget(self._canvas)
        lay.addWidget(scroll)

        info = QLabel(
            "Click to place 4 corners in order: TL → TR → BR → BL.  "
            "Then drag any corner to fine-tune. Swatch overlay updates live.")
        info.setAlignment(Qt.AlignCenter)
        lay.addWidget(info)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._canvas.reset_corners)
        self._ok_btn = QPushButton("Accept")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._ok_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

    def _build_crop_pixmap(self, erp_linear: np.ndarray,
                           rect_uv: tuple) -> Optional[QPixmap]:
        try:
            from colorchecker_erp import (
                _rect_uv_to_tile_params, erp_to_rectilinear,
                _linear_to_u8_for_display,
            )
        except Exception as e:
            print(f"[picker] failed to import colorchecker_erp helpers: {e}")
            return None
        yaw, pitch, fov = _rect_uv_to_tile_params(rect_uv)
        # Crop size matches the rect's aspect — longer side gets 1024 px.
        u0, v0, u1, v1 = rect_uv
        span_u = abs(u1 - u0) * 360.0
        span_v = abs(v1 - v0) * 180.0
        if span_u >= span_v:
            out_w = 1024
            out_h = max(64, int(round(1024 * (span_v / max(span_u, 1e-6)))))
        else:
            out_h = 1024
            out_w = max(64, int(round(1024 * (span_u / max(span_v, 1e-6)))))
        tile_linear, map_uv = erp_to_rectilinear(erp_linear, yaw, pitch, fov,
                                                  out_w, out_h)
        self._map_uv = map_uv
        u8 = _linear_to_u8_for_display(tile_linear)
        h, w = u8.shape[:2]
        qimg = QImage(u8.data, w, h, w * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _on_corners_changed(self):
        n = len(self._canvas._corners)
        self._ok_btn.setEnabled(n == 4)

    def _on_accept(self):
        xys = self._canvas.corners_xy()
        if len(xys) != 4:
            return
        if self._map_uv is not None:
            mu = self._map_uv
            mh, mw = mu.shape[:2]
            uvs: list[tuple[float, float]] = []
            for cx, cy in xys:
                ix = int(np.clip(round(cx), 0, mw - 1))
                iy = int(np.clip(round(cy), 0, mh - 1))
                u, v = float(mu[iy, ix, 0]), float(mu[iy, ix, 1])
                uvs.append((u, v))
            self.corners_uv = uvs
        else:
            # Legacy: corners were placed on the full latlong preview.
            uvs = [(x / max(self._crop_w, 1), y / max(self._crop_h, 1))
                   for x, y in xys]
            self.corners_uv = uvs
        self.accept()


# ── File item ──────────────────────────────────────────────────────────────────
class FileItem:
    def __init__(self, path):
        self.path = path; self.name = Path(path).name
        self.status = "waiting"; self.warnings = []; self.report = {}; self.previews = []
        self.source_preview = None
        self.config = None

    def icon(self):
        return {"waiting": ICON_WAIT, "running": ICON_RUNNING, "ok": ICON_OK,
                "warn": ICON_WARN, "error": ICON_ERROR}.get(self.status, "?")

    def color(self):
        return {"waiting": "#505065", "running": "#6090d8", "ok": "#48a870",
                "warn": "#b88820", "error": "#b84040"}.get(self.status, "#505065")


# ── Preview panel ──────────────────────────────────────────────────────────────
class PreviewPanel(QWidget):
    # Emitted when the user drags out a search rect on the Source tab.
    search_rect_drawn = Signal(tuple)  # (u0, v0, u1, v1)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._tabs = QTabWidget(); lay.addWidget(self._tabs)
        self._labels: dict[str, QLabel] = {}

        # Source tab uses a RectDrawLabel so the user can drag out a search rect.
        self._source_label = RectDrawLabel()
        self._source_label.setStyleSheet("background: #10101a;")
        self._source_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source_label.rect_drawn.connect(self.search_rect_drawn.emit)
        sc_src = QScrollArea(); sc_src.setWidget(self._source_label)
        sc_src.setWidgetResizable(True)
        self._tabs.addTab(sc_src, "Source")
        self._labels["source_preview.png"] = self._source_label

        for name, key in [
            ("WB",         "01_wb_preview.png"),
            ("Exposed",    "02_exposed_preview.png"),
            ("Lobe",       "03_hot_mask.png"),
            ("Chart Tile", "cc_detected_tile.jpg"),
            ("Rectified",  "cc_rectified_final.jpg"),
            ("Swatches",   "cc_swatch_comparison.jpg"),
            ("Solved",     "07_corrected_preview.png"),
            ("Balanced",   "07b_final_balanced_preview.png"),
            ("Final",      "08_verify_sphere_final.png"),
        ]:
            lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #10101a;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            sc = QScrollArea(); sc.setWidget(lbl); sc.setWidgetResizable(True)
            self._tabs.addTab(sc, name); self._labels[key] = lbl
        self.clear()

    def set_search_rect(self, rect_uv: Optional[tuple]):
        self._source_label.set_rect_uv(rect_uv)

    def clear(self):
        for key, lbl in self._labels.items():
            if isinstance(lbl, RectDrawLabel):
                lbl.set_base_pixmap(QPixmap())
                lbl.setText("─")
                lbl.setStyleSheet("background:#10101a; color:#282840; font-size:28px;")
            else:
                lbl.setText("─")
                lbl.setStyleSheet("background:#10101a; color:#282840; font-size:28px;")
                lbl.setPixmap(QPixmap())

    def update_preview(self, path):
        fname = os.path.basename(path)
        for key, lbl in self._labels.items():
            if fname == os.path.basename(key) or path.endswith(key):
                px = QPixmap(path)
                if not px.isNull():
                    if isinstance(lbl, RectDrawLabel):
                        lbl.set_base_pixmap(px)
                        lbl.setStyleSheet("background:#10101a;")
                        lbl.setText("")
                    else:
                        lbl.setPixmap(px.scaled(
                            lbl.width() or 600, lbl.height() or 400,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        lbl.setStyleSheet("background:#10101a;"); lbl.setText("")
                break


# ── Report panel ───────────────────────────────────────────────────────────────
class ReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._warn_list = QListWidget(); self._warn_list.setMaximumHeight(100)
        lay.addWidget(QLabel("Warnings")); lay.addWidget(self._warn_list)
        self._grid = QFormLayout()
        grid_w = QWidget(); grid_w.setLayout(self._grid)
        lay.addWidget(QLabel("Energy")); lay.addWidget(grid_w); lay.addStretch()

    def clear(self):
        self._warn_list.clear()
        while self._grid.rowCount(): self._grid.removeRow(0)

    def update(self, report: dict, warnings: list):
        self.clear()
        for w in warnings:
            it = QListWidgetItem(f"⚠  {w}"); it.setForeground(QColor("#b88820"))
            self._warn_list.addItem(it)
        if not warnings:
            it = QListWidgetItem("✓  No warnings"); it.setForeground(QColor("#48a870"))
            self._warn_list.addItem(it)

        def row(lbl, val, ok=None):
            l = QLabel(str(val))
            if ok is True:    l.setObjectName("status_ok")
            elif ok is False: l.setObjectName("status_err")
            self._grid.addRow(lbl, l)

        ev = report.get("energy_validation", {})
        if ev:
            row("E_upper",     f"{ev.get('E_upper',0):.4f}")
            row("Flat card ↑", f"{ev.get('pred_flat_card_up',0):.5f}")
            row("Sphere mean", f"{ev.get('pred_sphere_mean',0):.5f}")
            err = ev.get("rendered_vs_analytical_err", 1.0)
            row("Rendered vs analytical", f"{err:.1%}", ok=(err < 0.10))
        energy = report.get("energy", {}); clamp = report.get("clamping", {})
        if energy:
            row("E_upper", f"{energy.get('E_upper',0):.4f}")
            ci = energy.get("chroma_imbalance", 0)
            row("Chroma imbalance", f"{ci:.4f}", ok=(ci < 0.05))
        if clamp:
            cf = clamp.get("clip_fraction", 0)
            row("Clip fraction", f"{cf:.4%}", ok=(cf < 0.001))
            row("Likely clipped", "YES ⚠" if clamp.get("likely_clipped") else "No",
                ok=(not clamp.get("likely_clipped")))


class SliderField(QWidget):
    def __init__(self, minimum: int, maximum: int, initial: int, formatter,
                 *, decimals: int = 0, scale: float = 1.0, suffix: str = "", parent=None):
        super().__init__(parent)
        self._formatter = formatter
        self._scale = float(scale)
        self._syncing = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(initial)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum / self._scale, maximum / self._scale)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(max(1.0 / self._scale, 10 ** (-decimals)))
        if suffix:
            self.spin.setSuffix(suffix)
        self.spin.setValue(initial / self._scale)
        self.label = QLabel()
        self.label.setMinimumWidth(72)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self._refresh_label(self.slider.value())

    def _refresh_label(self, value: int):
        self.label.setText(self._formatter(value))

    def _on_slider_changed(self, value: int):
        self._refresh_label(value)
        if self._syncing:
            return
        self._syncing = True
        try:
            self.spin.setValue(value / self._scale)
        finally:
            self._syncing = False

    def _on_spin_changed(self, value: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.slider.setValue(int(round(value * self._scale)))
        finally:
            self._syncing = False

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, value: int):
        self.slider.setValue(int(value))

    @property
    def valueChanged(self):
        return self.slider.valueChanged

    def setToolTip(self, text: str):
        self.slider.setToolTip(text)
        self.spin.setToolTip(text)
        self.label.setToolTip(text)
        super().setToolTip(text)


# ── Settings panel ─────────────────────────────────────────────────────────────
class SettingsPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        w = QWidget(); self.setWidget(w)
        self._lay = QVBoxLayout(w)
        self._lay.setSpacing(8); self._lay.setContentsMargins(8,8,8,8)

        self._build_output()
        self._build_simple()
        self._build_advanced_groups()

        self._lay.addStretch()
        self._sync_calibration_mode()

    def _build_output(self):
        grp = QGroupBox("Output"); f = QFormLayout(grp)
        self.out_suffix = QLineEdit("_cal"); self.out_suffix.setMaximumWidth(80)
        self.out_dir    = QLineEdit(); self.out_dir.setPlaceholderText("Same folder as input")
        self.debug_dir  = QLineEdit("debug_hdri")
        self.res        = QComboBox(); self.res.addItems(["full","half","quarter"])
        f.addRow("Suffix",     self.out_suffix)
        f.addRow("Output dir", self.out_dir)
        f.addRow("Debug dir",  self.debug_dir)
        f.addRow("Resolution", self.res)
        self._lay.addWidget(grp)

    def _build_simple(self):
        grp = QGroupBox("Calibration")
        f = QFormLayout(grp)

        self.calibration_mode = QComboBox()
        self.calibration_mode.addItems(["auto", "advanced"])
        self.calibration_mode.setToolTip(
            "auto     - if chart is found, use chart WB/exposure; otherwise leave WB/exposure untouched\n"
            "advanced - reveal manual source and sun-solve controls below")
        self.calibration_mode.currentTextChanged.connect(self._sync_calibration_mode)

        self.input_colorspace = QComboBox()
        self.input_colorspace.addItems(["auto", "acescg", "srgb"])
        self.input_colorspace.setToolTip(
            "Input primaries of the source image.\n"
            "auto   - EXR/HDR defaults to ACEScg, JPG/PNG to sRGB\n"
            "acescg - treat input as linear ACEScg\n"
            "srgb   - treat input as linear sRGB and convert to ACEScg before processing"
        )

        self.center_hdri = QCheckBox("Centre HDRI on sun")
        self.center_hdri.setChecked(True)
        self.center_hdri.setToolTip("Shift azimuth so the sun sits at the centre column (phi=0)")

        self.base_intensity = SliderField(1, 1600, 100, lambda v: f"{v / 100.0:.2f}x", decimals=2, scale=100.0)
        self.base_intensity.setToolTip("Base input intensity multiplier for preview and optional manual override")

        self.base_temperature = SliderField(2000, 15000, 6500, lambda v: f"{v:d} K", decimals=0, scale=1.0, suffix=" K")
        self.base_temperature.setToolTip("Photographic white balance temperature")

        self.base_tint = SliderField(-100, 100, 0, lambda v: f"{v / 100.0:+.2f}", decimals=2, scale=100.0)
        self.base_tint.setToolTip("Tint adjustment: +1.00 = magenta, -1.00 = green")

        f.addRow("Mode",            self.calibration_mode)
        f.addRow("Input primaries", self.input_colorspace)
        f.addRow("Intensity",       self.base_intensity)
        f.addRow("Temperature",     self.base_temperature)
        f.addRow("Tint",            self.base_tint)
        reset_row = QHBoxLayout()
        self.reset_base_btn = QPushButton("Reset Photographic")
        self.reset_base_btn.setMaximumWidth(140)
        self.reset_base_btn.clicked.connect(self._reset_photographic_controls)
        reset_row.addWidget(self.reset_base_btn)
        reset_row.addStretch()
        f.addRow("",                reset_row)
        f.addRow("",                self.center_hdri)
        self._lay.addWidget(grp)

    def _build_advanced_groups(self):
        self._adv_groups: list[QWidget] = []

        grp = QGroupBox("Advanced Calibration")
        f = QFormLayout(grp)

        self.wb_source = QComboBox()
        self.wb_source.addItems(["auto", "chart", "sphere", "meter", "none"])
        self.wb_source.setToolTip(
            "auto   - chart if found, else pixel-average meter\n"
            "chart  - require ColorChecker for WB\n"
            "sphere - neutralise rendered grey sphere\n"
            "meter  - pixel-average neutralisation (camera meter)\n"
            "none   - skip WB")

        self.exp_source = QComboBox()
        self.exp_source.addItems(["auto", "chart", "sphere", "meter", "none"])
        self.exp_source.setToolTip(
            "auto   - chart if found, else pixel-average meter\n"
            "chart  - use patch 22 for exposure\n"
            "sphere - normalise sphere render to 18%% grey\n"
            "meter  - pixel-average exposure to 18%% grey\n"
            "none   - skip exposure correction")

        self.integration_mode = QComboBox()
        self.integration_mode.addItems(["full_sphere", "upper_dome", "sun_facing"])
        self.integration_mode.setToolTip(
            "Region used for meter/sphere WB and exposure:\n"
            "full_sphere - all pixels\n"
            "upper_dome  - sky hemisphere only\n"
            "sun_facing  - hemisphere facing the dominant light")

        self.solver_mode = QComboBox()
        self.solver_mode.addItems(["auto", "energy_conservation", "sun_facing_card", "sun_facing_vertical", "none"])
        self.solver_mode.setCurrentText("auto")
        self.solver_mode.setToolTip(
            "auto                 - default sun solve\n"
            "energy_conservation  - target an upward-facing grey card\n"
            "sun_facing_card      - target a grey card whose normal points at the sun\n"
            "sun_facing_vertical  - target a vertical card rotated toward the sun azimuth\n"
            "none                 - disable sun solve")

        self.final_balance_target = QComboBox()
        self.final_balance_target.addItems(["none", "auto"])
        self.final_balance_target.setToolTip(
            "none - no final trim after sun solve\n"
            "auto - measure the imaginary target card and apply one final RGB balance to make it neutral at albedo")

        self.albedo = QDoubleSpinBox()
        self.albedo.setRange(0.01, 1.0)
        self.albedo.setValue(0.18)
        self.albedo.setDecimals(4)
        self.albedo.setToolTip("Reflectance of the target grey card / swatch")

        self.validate_energy = QCheckBox("Run energy & calibration validation")
        self.validate_energy.setChecked(False)
        self.validate_energy.setToolTip(
            "Optional post-process: renders a grey sphere, integrates the 6-axis "
            "irradiance map, prints E=π / grey-card predictions, and writes "
            "meta['energy_validation'] in the report.\n"
            "Adds ~1–2 s; off by default — debug aid, not a calibration step.")

        f.addRow("WB source",        self.wb_source)
        f.addRow("Exposure source",  self.exp_source)
        f.addRow("Integration mode", self.integration_mode)
        f.addRow("Sun solve",        self.solver_mode)
        f.addRow("Final balance",   self.final_balance_target)
        f.addRow("Albedo",          self.albedo)
        f.addRow("Validation",       self.validate_energy)
        self._adv_groups.append(grp)

        grp = QGroupBox("Sun / Hot Lobe")
        f = QFormLayout(grp)
        self.sun_threshold = QDoubleSpinBox()
        self.sun_threshold.setRange(0.01, 0.99)
        self.sun_threshold.setValue(0.1)
        self.sun_threshold.setDecimals(3)
        self.sun_threshold.setToolTip("Fraction below peak defining the lobe")
        self.lobe_neutralise = QDoubleSpinBox()
        self.lobe_neutralise.setRange(0.0, 1.0)
        self.lobe_neutralise.setValue(1.0)
        self.lobe_neutralise.setDecimals(2)
        self.lobe_neutralise.setToolTip("0 = keep sun colour, 1 = fully desaturate to white before boost")
        self.sun_gain_ceiling = QDoubleSpinBox()
        self.sun_gain_ceiling.setRange(10, 20000)
        self.sun_gain_ceiling.setValue(2000)
        self.sun_gain_ceiling.setDecimals(0)
        self.sun_gain_ceiling.setSuffix("×")
        self.sun_gain_ceiling.setToolTip("Hard cap on per-channel sun gain")
        self.sun_gain_rolloff = QDoubleSpinBox()
        self.sun_gain_rolloff.setRange(10, 10000)
        self.sun_gain_rolloff.setValue(500)
        self.sun_gain_rolloff.setDecimals(0)
        self.sun_gain_rolloff.setSuffix("×")
        self.sun_gain_rolloff.setToolTip("Soft rolloff start before the gain ceiling")
        f.addRow("Sun threshold",   self.sun_threshold)
        f.addRow("Lobe neutralise", self.lobe_neutralise)
        f.addRow("Gain ceiling",    self.sun_gain_ceiling)
        f.addRow("Gain rolloff",    self.sun_gain_rolloff)
        self._adv_groups.append(grp)

        grp = QGroupBox("Chart Detection")
        v = QVBoxLayout(grp)
        v.setContentsMargins(8, 14, 8, 8)
        v.setSpacing(6)

        # min-confidence is only used by the hidden auto-detect path
        self.cc_min_confidence = QDoubleSpinBox()
        self.cc_min_confidence.setRange(0.0, 1.0)
        self.cc_min_confidence.setSingleStep(0.05)
        self.cc_min_confidence.setDecimals(2)
        self.cc_min_confidence.setValue(0.50)
        self.cc_min_confidence.setVisible(False)

        # Step 1 — search rect.  Clear button rides on the step header so
        # nothing depends on the rightmost column having room for a button.
        self.cc_search_rect: Optional[tuple] = None
        step1_row = QHBoxLayout()
        step1_row.setSpacing(4)
        step1 = QLabel("1.  Drag a rectangle on the Source preview")
        step1.setStyleSheet("color: #8888a8; font-size: 11px;")
        step1_row.addWidget(step1, 1)
        clear_rect_btn = QPushButton("✕")
        clear_rect_btn.setFixedSize(20, 20)
        clear_rect_btn.setToolTip("Clear search rectangle")
        clear_rect_btn.clicked.connect(self._clear_cc_rect)
        step1_row.addWidget(clear_rect_btn, 0)
        v.addLayout(step1_row)

        self._cc_rect_label = QLabel("(none)")
        self._cc_rect_label.setStyleSheet(
            "color: #606080; font-size: 10px; padding-left: 14px;")
        self._cc_rect_label.setWordWrap(True)
        v.addWidget(self._cc_rect_label)

        # Spacer divider
        div = QFrame(); div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #303040;")
        div.setFixedHeight(1)
        v.addWidget(div)

        # Step 2 — place corners.
        self.cc_manual_corners: Optional[list] = None
        step2_row = QHBoxLayout()
        step2_row.setSpacing(4)
        step2 = QLabel("2.  Place the 4 chart corners")
        step2.setStyleSheet("color: #8888a8; font-size: 11px;")
        step2.setToolTip("Order: TL → TR → BR → BL")
        step2_row.addWidget(step2, 1)
        clear_btn = QPushButton("✕")
        clear_btn.setFixedSize(20, 20)
        clear_btn.setToolTip("Clear placed corners")
        clear_btn.clicked.connect(self._clear_cc_corners)
        step2_row.addWidget(clear_btn, 0)
        v.addLayout(step2_row)

        self.pick_chart_btn = QPushButton("Place Chart Corners")
        self.pick_chart_btn.setMinimumHeight(28)
        self.pick_chart_btn.setToolTip(
            "Open the rectilinear crop of the search rect and click 4 corners "
            "(TL → TR → BR → BL). Drag any corner afterward to fine-tune.")
        v.addWidget(self.pick_chart_btn)

        self._cc_corners_label = QLabel("(not set)")
        self._cc_corners_label.setStyleSheet(
            "color: #606080; font-size: 10px; padding-left: 14px;")
        v.addWidget(self._cc_corners_label)

        # Hidden auto-detect (dev only — preserved code path)
        self.detect_chart_btn = QPushButton("Detect Chart In Rect")
        self.detect_chart_btn.setVisible(False)

        # Chart Detection is always visible — primary workflow, not "advanced".
        self._chart_detection_group = grp

        for grp in self._adv_groups:
            self._lay.addWidget(grp)
        # Add chart detection last so it sits at the bottom of the settings
        # column, near the action buttons.
        self._lay.addWidget(self._chart_detection_group)

    def _sync_calibration_mode(self):
        show_advanced = (self.calibration_mode.currentText() == "advanced")
        for grp in self._adv_groups:
            grp.setVisible(show_advanced)

    def _clear_cc_corners(self):
        self.cc_manual_corners = None
        self._cc_corners_label.setText("(not set)")
        self._cc_corners_label.setStyleSheet("color: #606080; font-size: 10px;")

    def _set_cc_corners(self, corners: list):
        self.cc_manual_corners = corners
        self._cc_corners_label.setText(f"4 corners set")
        self._cc_corners_label.setStyleSheet("color: #58c080; font-size: 10px;")

    def _clear_cc_rect(self):
        self.cc_search_rect = None
        self._cc_rect_label.setText("(none — draw on Source preview)")
        self._cc_rect_label.setStyleSheet("color: #606080; font-size: 10px;")

    def _set_cc_rect(self, rect_uv: Optional[tuple]):
        if rect_uv is None:
            self._clear_cc_rect()
            return
        u0, v0, u1, v1 = rect_uv
        self.cc_search_rect = (float(u0), float(v0), float(u1), float(v1))
        self._cc_rect_label.setText(
            f"u=[{u0:.3f}, {u1:.3f}]  v=[{v0:.3f}, {v1:.3f}]")
        self._cc_rect_label.setStyleSheet("color: #58c080; font-size: 10px;")

    def _reset_photographic_controls(self):
        self._set_base_intensity_value(1.0)
        self._set_base_temperature_value(6500.0)
        self._set_base_tint_value(0.0)

    def _base_intensity_value(self) -> float:
        return self.base_intensity.value() / 100.0

    def _base_temperature_value(self) -> float:
        return float(self.base_temperature.value())

    def _base_tint_value(self) -> float:
        return self.base_tint.value() / 100.0

    def _set_base_intensity_value(self, value: float):
        self.base_intensity.setValue(int(round(max(0.01, min(16.0, value)) * 100.0)))

    def _set_base_temperature_value(self, value: float):
        self.base_temperature.setValue(int(round(max(2000.0, min(15000.0, value)))))

    def _set_base_tint_value(self, value: float):
        self.base_tint.setValue(int(round(max(-1.0, min(1.0, value)) * 100.0)))

    def _photographic_rgb_scale(self, cfg: PipelineConfig):
        import hdri_cal as hc
        kelvin_scale = hc.kelvin_to_rgb_scale(float(cfg.base_temperature))
        tint = float(cfg.base_tint)
        tint_scale = np.array([
            1.0 + 0.35 * tint,
            1.0 - 0.70 * tint,
            1.0 + 0.35 * tint,
        ], dtype=np.float32)
        tint_scale = np.clip(tint_scale, 0.05, None)
        scale = kelvin_scale * tint_scale
        scale /= max(float(np.mean(scale)), 1e-8)
        return scale.astype(np.float32)

    def _photographic_rgb_scale_string(self, cfg: PipelineConfig):
        scale = self._photographic_rgb_scale(cfg)
        return f"{scale[0]:.6f},{scale[1]:.6f},{scale[2]:.6f}"

    def _browse(self, line: QLineEdit, filt: str):
        p, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if p: line.setText(p)

    def build_config(self, input_path: str) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.input = input_path
        p = Path(input_path)
        out_dir  = self.out_dir.text().strip() or str(p.parent)
        suffix   = self.out_suffix.text().strip() or "_cal"
        cfg.out  = str(Path(out_dir) / (p.stem + suffix + ".exr"))
        cfg.debug_dir   = str(Path(out_dir) / (p.stem + "_debug"))
        cfg.res         = self.res.currentText()

        # Simple
        cfg.calibration_mode = self.calibration_mode.currentText()
        cfg.center_hdri = self.center_hdri.isChecked()
        cs = self.input_colorspace.currentText()
        cfg.colorspace = None if cs == "auto" else cs

        # Advanced calibration choices
        cfg.integration_mode = self.integration_mode.currentText()
        if cfg.calibration_mode == "auto":
            cfg.wb_source = "auto"
            cfg.exposure_source = "auto"
            cfg.sphere_solve = "auto"
            cfg.final_balance_target = "none"
        else:
            cfg.wb_source = self.wb_source.currentText()
            cfg.exposure_source = self.exp_source.currentText()
            cfg.sphere_solve = self.solver_mode.currentText()
            cfg.final_balance_target = self.final_balance_target.currentText()

        cfg.base_intensity = self._base_intensity_value()
        cfg.base_temperature = self._base_temperature_value()
        cfg.base_tint = self._base_tint_value()

        if cfg.calibration_mode == "advanced":
            temp_override = abs(self._base_temperature_value() - 6500.0) > 1e-6
            tint_override = abs(self._base_tint_value()) > 1e-6
            intensity_override = abs(self._base_intensity_value() - 1.0) > 1e-6
            if cfg.wb_source == "none" and (temp_override or tint_override):
                cfg.wb_source = "manual"
                cfg.rgb_scale = self._photographic_rgb_scale_string(cfg)
            if cfg.exposure_source == "none" and intensity_override:
                cfg.exposure_source = "manual"
                cfg.exposure_scale = cfg.base_intensity

        cfg.albedo = self.albedo.value()
        cfg.sun_threshold = self.sun_threshold.value()
        cfg.lobe_neutralise = self.lobe_neutralise.value()
        cfg.sun_gain_ceiling = self.sun_gain_ceiling.value()
        cfg.sun_gain_rolloff = self.sun_gain_rolloff.value()
        cfg.cc_min_confidence = self.cc_min_confidence.value()
        cfg.cc_search_rect = self.cc_search_rect
        cfg.cc_manual_corners = self.cc_manual_corners
        cfg.validate_energy = bool(self.validate_energy.isChecked())

        return cfg


# ── Log panel ──────────────────────────────────────────────────────────────────
class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(3)
        hdr = QHBoxLayout(); hdr.addWidget(QLabel("Pipeline Log"))
        btn = QPushButton("Clear"); btn.setMaximumWidth(52); btn.clicked.connect(self.clear)
        hdr.addStretch(); hdr.addWidget(btn); lay.addLayout(hdr)
        self._text = QTextEdit(); self._text.setReadOnly(True); lay.addWidget(self._text)

    def append(self, msg, color="#90c890"):
        self._text.setTextColor(QColor(color))
        self._text.append(str(msg))
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())

    def append_warn(self, msg):  self.append(f"⚠  {msg}", "#d89828")
    def append_error(self, msg): self.append(f"✕  {msg}", "#d84848")
    def clear(self):             self._text.clear()


# ── Validate helper ────────────────────────────────────────────────────────────
def run_validate(path: str, cfg: PipelineConfig) -> dict:
    try:
        import hdri_cal as hc
        args = config_to_namespace(cfg)
        args.input = path; args.validate_only = True
        os.makedirs(args.debug_dir, exist_ok=True)
        hc._run_pipeline(args)
        rp = os.path.join(args.debug_dir, "report.json")
        report = json.load(open(rp)) if os.path.exists(rp) else {}
        energy = report.get("energy", {}); clamp = report.get("clamping", {})
        h, w_ = report.get("working_resolution", [0, 0])
        return {
            "path": path, "ok": True,
            "resolution":        f"{w_}×{h}",
            "orientation_ratio": report.get("orientation_energy_ratio", 0),
            "E_upper":           energy.get("E_upper", 0),
            "clip_fraction":     clamp.get("clip_fraction", 0),
            "lum_max":           clamp.get("lum_max", 0),
            "likely_clipped":    clamp.get("likely_clipped", False),
            "chroma_imbalance":  energy.get("chroma_imbalance", 0),
            "warnings":          report.get("warnings", []),
            "report":            report,
        }
    except Exception as e:
        return {"path": path, "ok": False, "error": str(e), "warnings": [str(e)], "report": {}}


# ── Main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDRI Calibration  ·  VFX Pipeline Tool")
        self.resize(1280, 860)
        # _file_items kept as a 0-or-1 length list so the rest of the code can
        # still index it; batch UI was removed but the data model is unchanged.
        self._file_items:   List[FileItem]           = []
        self._current_item: Optional[FileItem]       = None
        self._worker:       Optional[PipelineWorker] = None
        self._signals:      Optional[PipelineSignals] = None
        self._pool          = QThreadPool(); self._pool.setMaxThreadCount(1)
        self._running       = False; self._abort_flag = False
        self._syncing_settings = False
        self.setAcceptDrops(True)  # window-wide drag/drop
        self._build_ui(); self._check_hdri_cal()

    # Drag-and-drop on the whole window ──
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(
                        (".exr", ".hdr", ".jpg", ".jpeg", ".png", ".webp")):
                    e.acceptProposedAction()
                    return

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith(
                     (".exr", ".hdr", ".jpg", ".jpeg", ".png", ".webp"))]
        if paths:
            self._on_files_dropped(paths[:1])  # single-file workflow

    def _build_ui(self):
        self.setStyleSheet(VFX_STYLE)

        # Action buttons live in the right settings column now — the top
        # toolbar is gone. Build the buttons here so the right panel can mount
        # them at the bottom.
        self._run_btn = QPushButton("▶  Process")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.clicked.connect(self._on_run)
        self._val_btn = QPushButton("⚡  Validate")
        self._val_btn.setObjectName("validate_btn")
        self._val_btn.setMinimumHeight(28)
        self._val_btn.clicked.connect(self._on_validate)
        self._abort_btn = QPushButton("■  Abort")
        self._abort_btn.setObjectName("abort_btn")
        self._abort_btn.setMinimumHeight(28)
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        self._open_btn = QPushButton("📂  Open")
        self._open_btn.setMinimumHeight(28)
        self._open_btn.clicked.connect(self._on_browse_open)
        self._clear_btn = QPushButton("✕  Clear")
        self._clear_btn.setMinimumHeight(28)
        self._clear_btn.setToolTip("Unload the current file")
        self._clear_btn.clicked.connect(self._clear_queue)
        self._out_btn = QPushButton("📁  Output Folder")
        self._out_btn.setMinimumHeight(28)
        self._out_btn.clicked.connect(self._open_output_folder)
        self._progress = QProgressBar()
        self._progress.setMaximumHeight(10)
        self._progress.setVisible(False)

        self._file_label = QLabel("No file loaded — drag & drop EXR / HDR / PNG / WebP")
        self._file_label.setStyleSheet("color: #808098; padding: 4px 0px;")
        self._file_label.setWordWrap(True)

        sp = QSplitter(Qt.Horizontal); self.setCentralWidget(sp)

        # LEFT (preview + log/report)
        centre = QWidget(); cl = QVBoxLayout(centre); cl.setContentsMargins(4,4,4,4); cl.setSpacing(4)
        vs = QSplitter(Qt.Vertical)
        self._preview = PreviewPanel(); vs.addWidget(self._preview)
        self._preview.search_rect_drawn.connect(self._on_search_rect_drawn)
        bot = QTabWidget(); self._log = LogPanel(); self._report = ReportPanel()
        bot.addTab(self._log, "Log"); bot.addTab(self._report, "Report"); vs.addWidget(bot)
        vs.setSizes([540, 200]); cl.addWidget(vs); sp.addWidget(centre)

        # RIGHT (current file + settings + action buttons all together)
        right = QWidget(); right.setMinimumWidth(340); right.setMaximumWidth(440)
        rl = QVBoxLayout(right); rl.setContentsMargins(6, 6, 6, 6); rl.setSpacing(4)

        # Header: current file
        rl.addWidget(self._file_label)

        # Settings (scrollable)
        self._settings = SettingsPanel()
        rl.addWidget(self._settings, 1)  # stretch — takes remaining space

        # Action buttons block, anchored at the bottom
        actions = QFrame()
        actions.setObjectName("actions_block")
        actions.setStyleSheet(
            "#actions_block { border-top: 1px solid #303040; padding-top: 6px; }")
        al = QVBoxLayout(actions); al.setContentsMargins(0, 6, 0, 0); al.setSpacing(4)

        # Row 1: Open + Clear
        r1 = QHBoxLayout(); r1.setSpacing(4)
        r1.addWidget(self._open_btn); r1.addWidget(self._clear_btn)
        al.addLayout(r1)

        # Row 2: Process (full width, prominent)
        al.addWidget(self._run_btn)

        # Row 3: Validate + Abort
        r3 = QHBoxLayout(); r3.setSpacing(4)
        r3.addWidget(self._val_btn); r3.addWidget(self._abort_btn)
        al.addLayout(r3)

        # Row 4: Output folder
        al.addWidget(self._out_btn)

        # Progress bar
        al.addWidget(self._progress)

        rl.addWidget(actions)

        sp.addWidget(right)
        sp.setSizes([940, 380])

        self._source_preview_timer = QTimer(self)
        self._source_preview_timer.setSingleShot(True)
        self._source_preview_timer.timeout.connect(self._refresh_selected_source_preview)
        self._connect_settings_signals()

        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage("Ready  ·  Drop EXR / HDR / PNG / WebP into the window to begin")

    def _connect_settings_signals(self):
        for signal in [
            self._settings.calibration_mode.currentTextChanged,
            self._settings.input_colorspace.currentTextChanged,
            self._settings.center_hdri.toggled,
            self._settings.base_intensity.valueChanged,
            self._settings.base_temperature.valueChanged,
            self._settings.base_tint.valueChanged,
            self._settings.wb_source.currentTextChanged,
            self._settings.exp_source.currentTextChanged,
            self._settings.integration_mode.currentTextChanged,
            self._settings.solver_mode.currentTextChanged,
            self._settings.final_balance_target.currentTextChanged,
            self._settings.albedo.valueChanged,
            self._settings.sun_threshold.valueChanged,
            self._settings.lobe_neutralise.valueChanged,
            self._settings.sun_gain_ceiling.valueChanged,
            self._settings.sun_gain_rolloff.valueChanged,
            self._settings.cc_min_confidence.editingFinished,
            self._settings.validate_energy.toggled,
        ]:
            signal.connect(self._on_settings_changed)
        self._settings.pick_chart_btn.clicked.connect(self._on_pick_chart_corners)
        self._settings.detect_chart_btn.clicked.connect(self._on_detect_chart_in_rect)

    def _on_search_rect_drawn(self, rect_uv: tuple):
        """User dragged out a search rect on the Source preview. Only records
        the rect — no detection, no preview rebuild."""
        self._settings._set_cc_rect(rect_uv)
        # Clear any stale manual corners so they don't override a fresh rect.
        self._settings._clear_cc_corners()
        # Persist on the current FileItem without triggering a source preview
        # refresh (which reloads the EXR/PNG and takes seconds).
        idx = self._selected_index()
        if idx is not None:
            fi = self._file_items[idx]
            fi.config = self._settings.build_config(fi.path)
        u0, v0, u1, v1 = rect_uv
        self._status.showMessage(
            f"Search rect set:  u=[{u0:.3f}, {u1:.3f}]  v=[{v0:.3f}, {v1:.3f}]  "
            f"— click 'Place Chart Corners' to set the chart.")

    def _load_selected_erp(self) -> Optional[np.ndarray]:
        """Load the currently selected HDRI as a linear float32 latlong."""
        idx = self._selected_index()
        if idx is None:
            QMessageBox.warning(self, "No file selected",
                                "Select an HDRI file first.")
            return None
        fi = self._file_items[idx]
        try:
            import hdri_cal as hc
            img, _meta = hc.load_image_any(fi.path, target_colorspace="acescg")
            return np.asarray(img, dtype=np.float32)
        except Exception as e:
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load HDRI:\n{e}")
            return None

    def _on_detect_chart_in_rect(self):
        rect_uv = self._settings.cc_search_rect
        if rect_uv is None:
            QMessageBox.information(
                self, "No search rectangle",
                "Draw a search rectangle on the Source preview first, then "
                "click 'Detect Chart In Rect'.")
            return
        erp = self._load_selected_erp()
        if erp is None:
            return
        try:
            from colorchecker_erp import find_colorchecker_in_rect
        except Exception as e:
            QMessageBox.critical(self, "Detector unavailable", str(e))
            return
        # Always write debug JPEGs for the button-triggered detect so the user
        # can isolate projection vs detection issues with cc_debug.py.
        fi = self._file_items[self._selected_index()]
        debug_dir = os.path.abspath(os.path.join(
            os.path.dirname(fi.path) or ".",
            "cc_debug_out", "detect_" + Path(fi.path).stem))
        os.makedirs(debug_dir, exist_ok=True)
        self._log.append(f"[detect] debug dir: {debug_dir}")
        # Also dump the linear ERP crop (before display-map) so cc_debug.py
        # can compare what the detector sees vs the source.
        try:
            from colorchecker_erp import (
                erp_to_rectilinear, _rect_uv_to_tile_params,
                _linear_to_u8_for_display,
            )
            yaw, pitch, fov = _rect_uv_to_tile_params(tuple(rect_uv))
            u0, v0, u1, v1 = rect_uv
            span_u = abs(u1 - u0) * 360.0
            span_v = abs(v1 - v0) * 180.0
            if span_u >= span_v:
                out_w = 1024
                out_h = max(64, int(round(1024 * (span_v / max(span_u, 1e-6)))))
            else:
                out_h = 1024
                out_w = max(64, int(round(1024 * (span_u / max(span_v, 1e-6)))))
            tile_linear, _ = erp_to_rectilinear(erp, yaw, pitch, fov, out_w, out_h)
            # Save a 32-bit-ish linear .exr too, for completeness
            try:
                cv2.imwrite(os.path.join(debug_dir, "rect_tile_linear.exr"),
                            cv2.cvtColor(tile_linear, cv2.COLOR_RGB2BGR))
            except Exception:
                pass
            disp = _linear_to_u8_for_display(tile_linear)
            # Also write under the name the preview pane picks up (Chart Tile tab).
            for name in ("rect_tile_display.jpg", "cc_detected_tile.jpg"):
                cv2.imwrite(os.path.join(debug_dir, name),
                            cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            # Surface the tile in the GUI's Chart Tile preview immediately.
            self._preview.update_preview(
                os.path.join(debug_dir, "cc_detected_tile.jpg"))
        except Exception as e:
            self._log.append_warn(f"[detect] could not save extra debug: {e}")

        self._status.showMessage("Detecting chart in rect ...")
        QApplication.processEvents()
        swatches, info = find_colorchecker_in_rect(
            erp, rect_uv=tuple(rect_uv), colorspace="acescg",
            debug_dir=debug_dir,
            min_confidence=float(self._settings.cc_min_confidence.value()),
        )
        # List what actually landed on disk so the user knows immediately.
        try:
            files = sorted(os.listdir(debug_dir))
            self._log.append(f"[detect] wrote {len(files)} files to {debug_dir}:")
            for f in files:
                self._log.append(f"    {f}")
        except Exception:
            pass
        if info.get("found"):
            centres_uv = info.get("swatch_centres_uv") or []
            if len(centres_uv) == 24:
                # Approximate chart quad from outer swatch centres.
                pts = np.array(centres_uv, dtype=np.float32)
                tl = pts[0]; tr = pts[5]; br = pts[23]; bl = pts[18]
                self._settings._set_cc_corners([
                    (float(tl[0]), float(tl[1])),
                    (float(tr[0]), float(tr[1])),
                    (float(br[0]), float(br[1])),
                    (float(bl[0]), float(bl[1])),
                ])
                self._on_settings_changed()
            self._status.showMessage(
                f"Chart found  conf={info.get('confidence', 0):.2f}")
            self._log.append(
                f"[detect] found  conf={info.get('confidence', 0):.2f}  "
                f"yaw={info.get('best_tile_yaw_deg', 0):.1f}° "
                f"pitch={info.get('best_tile_pitch_deg', 0):.1f}° "
                f"fov={info.get('best_tile_fov_deg', 0):.1f}°")
        else:
            self._status.showMessage(
                "Chart not detected — try 'Place Chart Corners' to set "
                "them manually.")
            self._log.append_warn(
                f"[detect] no chart in rect  "
                f"conf={info.get('confidence', 0):.2f}")
            # Pop the debug folder open so the user can see what was written.
            try:
                os.startfile(debug_dir)
            except Exception:
                pass

    def _on_pick_chart_corners(self):
        rect_uv = self._settings.cc_search_rect
        if rect_uv is None:
            QMessageBox.information(
                self, "No search rectangle",
                "Draw a search rectangle on the Source preview first, then "
                "click 'Place Chart Corners'.")
            return
        erp = self._load_selected_erp()
        if erp is None:
            return
        dlg = ChartCornerPicker(erp_linear=erp, rect_uv=tuple(rect_uv),
                                parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.corners_uv is not None:
            self._settings._set_cc_corners(dlg.corners_uv)
            self._on_settings_changed()
            self._status.showMessage("Chart corners set manually (4 points)")

    def _apply_config_to_settings(self, cfg: PipelineConfig):
        self._syncing_settings = True
        try:
            self._settings.calibration_mode.setCurrentText(cfg.calibration_mode or "auto")
            cs = cfg.colorspace or "auto"
            self._settings.input_colorspace.setCurrentText(cs)
            self._settings.center_hdri.setChecked(bool(cfg.center_hdri))
            self._settings._set_base_intensity_value(float(getattr(cfg, "base_intensity", 1.0)))
            self._settings._set_base_temperature_value(float(getattr(cfg, "base_temperature", 6500.0)))
            self._settings._set_base_tint_value(float(getattr(cfg, "base_tint", 0.0)))
            self._settings.wb_source.setCurrentText(getattr(cfg, "wb_source", "auto"))
            self._settings.exp_source.setCurrentText(getattr(cfg, "exposure_source", "auto"))
            self._settings.integration_mode.setCurrentText(getattr(cfg, "integration_mode", "full_sphere"))
            self._settings.solver_mode.setCurrentText(getattr(cfg, "sphere_solve", "auto"))
            self._settings.final_balance_target.setCurrentText(getattr(cfg, "final_balance_target", "none"))
            self._settings.albedo.setValue(float(getattr(cfg, "albedo", 0.18)))
            self._settings.sun_threshold.setValue(float(getattr(cfg, "sun_threshold", 0.1)))
            self._settings.lobe_neutralise.setValue(float(getattr(cfg, "lobe_neutralise", 1.0)))
            self._settings.sun_gain_ceiling.setValue(float(getattr(cfg, "sun_gain_ceiling", 2000.0)))
            self._settings.sun_gain_rolloff.setValue(float(getattr(cfg, "sun_gain_rolloff", 500.0)))
            self._settings.cc_min_confidence.setValue(float(getattr(cfg, "cc_min_confidence", 0.50)))
            self._settings.validate_energy.setChecked(bool(getattr(cfg, "validate_energy", False)))
            rect = getattr(cfg, "cc_search_rect", None)
            self._settings._set_cc_rect(rect)
            corners = getattr(cfg, "cc_manual_corners", None)
            if corners:
                self._settings._set_cc_corners(corners)
            else:
                self._settings._clear_cc_corners()
            self._settings._sync_calibration_mode()
        finally:
            self._syncing_settings = False

    def _current_config_for(self, fi: FileItem) -> PipelineConfig:
        if fi.config is None:
            fi.config = self._settings.build_config(fi.path)
        return fi.config

    def _store_current_settings_into_selected(self):
        idx = self._selected_index()
        if idx is None:
            return None
        fi = self._file_items[idx]
        fi.config = self._settings.build_config(fi.path)
        return fi

    def _on_settings_changed(self, *_args):
        if self._syncing_settings:
            return
        fi = self._store_current_settings_into_selected()
        if fi is None:
            return
        self._request_source_preview_refresh()

    # ── Single-file load ──────────────────────────────────────────────────
    def _on_files_dropped(self, paths):
        if not paths:
            return
        if self._running:
            self._status.showMessage("Cannot load while a job is running")
            return
        p = paths[0]
        # Replace current file (single-file workflow).
        self._file_items.clear()
        self._current_item = None
        self._preview.clear(); self._report.clear()
        fi = FileItem(p)
        fi.config = self._settings.build_config(p)
        self._file_items.append(fi)
        self._ensure_source_preview(fi)
        self._update_q()
        self._on_file_selected(0)
        self._status.showMessage(f"Loaded: {fi.name}")

    def _on_browse_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HDRI", "",
            "HDRI Images (*.exr *.hdr *.jpg *.jpeg *.png *.webp);;All Files (*.*)")
        if path:
            self._on_files_dropped([path])

    def _request_source_preview_refresh(self, *_args):
        if not hasattr(self, "_source_preview_timer"):
            return
        self._source_preview_timer.start(120)

    def _refresh_selected_source_preview(self):
        idx = self._selected_index()
        if idx is None:
            return
        fi = self._file_items[idx]
        self._ensure_source_preview(fi, force=True)
        if self._current_item is fi:
            self._on_file_selected(idx)

    def _selected_index(self):
        return 0 if self._file_items else None

    def _open_output_folder(self):
        idx = self._selected_index()
        if idx is None:
            QMessageBox.information(self, "No file", "Load a file first.")
            return
        fi = self._file_items[idx]
        cfg = self._current_config_for(fi)
        out_dir = str(Path(cfg.out).parent)
        os.makedirs(out_dir, exist_ok=True)
        try:
            os.startfile(out_dir)
        except Exception as e:
            QMessageBox.warning(self, "Open Folder",
                                f"Could not open output folder:\n{e}")

    def _clear_queue(self):
        if self._running:
            return
        self._file_items.clear()
        self._current_item = None
        self._preview.clear()
        self._report.clear()
        self._update_q()
        self._settings._clear_cc_rect()
        self._settings._clear_cc_corners()
        self._status.showMessage("Cleared")

    def _photographic_rgb_scale(self, cfg: PipelineConfig):
        import hdri_cal as hc
        kelvin_scale = hc.kelvin_to_rgb_scale(float(cfg.base_temperature))
        tint = float(cfg.base_tint)
        tint_scale = np.array([
            1.0 + 0.35 * tint,
            1.0 - 0.70 * tint,
            1.0 + 0.35 * tint,
        ], dtype=np.float32)
        tint_scale = np.clip(tint_scale, 0.05, None)
        scale = kelvin_scale * tint_scale
        scale /= max(float(np.mean(scale)), 1e-8)
        return scale.astype(np.float32)

    def _photographic_rgb_scale_string(self, cfg: PipelineConfig):
        scale = self._photographic_rgb_scale(cfg)
        return f"{scale[0]:.6f},{scale[1]:.6f},{scale[2]:.6f}"

    def _source_preview_path(self, file_path: str, cfg: PipelineConfig) -> str:
        p = Path(file_path)
        cache_dir = Path.cwd() / ".gui_cache" / "source_previews"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            stamp = int(p.stat().st_mtime)
        except OSError:
            stamp = 0
        cs = cfg.colorspace or "auto"
        temp = f"{cfg.base_temperature:.0f}"
        tint = f"{cfg.base_tint:+.1f}".replace(".", "p")
        intensity = f"{cfg.base_intensity:.3f}".replace(".", "p")
        safe_name = f"{p.stem}_{stamp}_{cs}_int{intensity}_temp{temp}_tint{tint}_source_preview.png"
        return str(cache_dir / safe_name)

    def _ensure_source_preview(self, fi: FileItem, force: bool = False):
        try:
            import hdri_cal as hc
            cfg = self._current_config_for(fi)
            preview_path = self._source_preview_path(fi.path, cfg)
            if not force and fi.source_preview == preview_path and os.path.exists(preview_path):
                return fi.source_preview
            input_cs = cfg.colorspace
            img, _ = hc.load_image_any(fi.path, target_colorspace="acescg", input_colorspace=input_cs)
            base_img = np.clip(img, 0.0, None)
            rgb_scale = self._photographic_rgb_scale(cfg)
            img = base_img * rgb_scale[None, None, :] * float(cfg.base_intensity)
            h, w = img.shape[:2]
            max_w = 1400
            if w > max_w:
                scale = max_w / float(w)
                new_size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            base_lum = 0.2126 * base_img[..., 0] + 0.7152 * base_img[..., 1] + 0.0722 * base_img[..., 2]
            denom = max(float(np.percentile(base_lum, 99.5)), 1e-6)
            display_linear = hc._to_display_srgb_linear(np.clip(img / denom, 0.0, None), "acescg")
            view = hc.linear_to_srgb(np.clip(display_linear, 0.0, 1.0))
            out = np.clip(view * 255.0, 0, 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(preview_path, out_bgr)
            if fi.source_preview and fi.source_preview in fi.previews:
                fi.previews.remove(fi.source_preview)
            fi.source_preview = preview_path
            fi.previews.insert(0, preview_path)
            return preview_path
        except Exception:
            return None

    def _update_q(self):
        if not self._file_items:
            self._file_label.setText(
                "No file loaded — drag & drop EXR / HDR / PNG / WebP")
            self._file_label.setStyleSheet("color: #808098; padding-left: 12px;")
            return
        fi = self._file_items[0]
        self._file_label.setText(f"{fi.icon()}  {fi.name}")
        self._file_label.setStyleSheet(
            f"color: {fi.color()}; padding-left: 12px; font-weight: 600;")

    def _refresh_item(self, idx):
        # Single-file mode — the toolbar label is updated by _update_q().
        self._update_q()

    def _on_file_selected(self, row):
        if row < 0 or row >= len(self._file_items): return
        fi = self._file_items[row]; self._current_item = fi
        self._apply_config_to_settings(self._current_config_for(fi))
        self._ensure_source_preview(fi)
        self._preview.clear()
        for p in fi.previews: self._preview.update_preview(p)
        if fi.source_preview and os.path.exists(fi.source_preview):
            self._preview.update_preview(fi.source_preview)
        # Re-apply any persisted search rect overlay on top of the source.
        self._preview.set_search_rect(self._settings.cc_search_rect)
        self._report.update(fi.report, fi.warnings)

    # ── Run / Validate ────────────────────────────────────────────────────
    def _on_run(self):
        if not self._file_items:
            QMessageBox.information(self, "No file",
                                    "Drop or open an HDRI file first.")
            return
        if self._running:
            return
        if not self._prompt_chart_setup_if_needed():
            return
        self._run_single(validate_only=False)

    def _prompt_chart_setup_if_needed(self) -> bool:
        """If the user hasn't set up chart detection, ask what they want.
        Returns True if the run should proceed, False if it was cancelled."""
        if (self._settings.cc_manual_corners is not None
                or self._settings.cc_search_rect is not None):
            return True

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("ColorChecker not set up")
        msg.setText("No chart corners are set.\n\n"
                    "Place the chart now, or skip chart-based calibration?")
        manual_btn = msg.addButton("Place corners", QMessageBox.AcceptRole)
        skip_btn = msg.addButton("Skip chart", QMessageBox.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(manual_btn)
        msg.exec()
        clicked = msg.clickedButton()

        if clicked is cancel_btn:
            return False
        if clicked is skip_btn:
            self._log.append("Chart detection skipped — using base settings only.")
            return True
        # Manual placement needs a search rect first.
        if self._settings.cc_search_rect is None:
            QMessageBox.information(
                self, "Draw a search rectangle",
                "Drag a rectangle on the Source preview to mark where the "
                "ColorChecker is, then click 'Place Chart Corners'.")
            return False
        self._on_pick_chart_corners()
        return False

    def _on_validate(self):
        if not self._file_items:
            QMessageBox.information(self, "No file", "Load a file first.")
            return
        if self._running:
            return
        self._run_single(validate_only=True)

    def _on_abort(self):
        self._abort_flag = True
        if self._worker: self._worker.abort()
        self._status.showMessage("Aborting…")

    def _set_running(self, state):
        self._running = state
        self._run_btn.setEnabled(not state)
        self._val_btn.setEnabled(not state)
        self._abort_btn.setEnabled(state)
        self._progress.setVisible(state)
        self._open_btn.setEnabled(not state)
        self._clear_btn.setEnabled(not state)

    def _run_all(self, validate_only=False):
        # Kept for backwards compat — single-file path.
        self._run_single(validate_only=validate_only)

    def _run_single(self, validate_only: bool = False):
        if not self._file_items:
            return
        fi = self._file_items[0]
        if validate_only:
            self._set_running(True); self._log.clear()
            fi.status = "running"; self._refresh_item(0); QApplication.processEvents()
            result = run_validate(fi.path, self._current_config_for(fi))
            fi.warnings = result.get("warnings", []); fi.report = result
            if not result.get("ok"):
                fi.status = "error"
                self._log.append_error(f"{fi.name}: {result.get('error')}")
            else:
                fi.status = "warn" if fi.warnings else "ok"
                clipped = ("⚠ CLIPPED" if result.get("likely_clipped")
                           else f"clip={result.get('clip_fraction', 0):.3%}")
                self._log.append(
                    f"{fi.name}  {result.get('resolution', '?')}  "
                    f"E_upper={result.get('E_upper', 0):.3f}  {clipped}  "
                    f"chroma={result.get('chroma_imbalance', 0):.3f}",
                    "#80c0f0" if not fi.warnings else "#d8c040")
            self._refresh_item(0)
            self._set_running(False)
            self._status.showMessage("Validation complete")
            return

        self._abort_flag = False
        self._set_running(True); self._log.clear()
        self._progress.setRange(0, 1); self._progress.setValue(0)
        fi.status = "running"; fi.warnings = []
        fi.previews = [fi.source_preview] if fi.source_preview else []
        self._refresh_item(0)
        cfg = self._current_config_for(fi); cfg.validate_only = False
        os.makedirs(cfg.debug_dir, exist_ok=True)
        self._status.showMessage(f"Processing: {fi.name}")
        self._log.append(f"\n{'─'*60}\n▶  {fi.name}", "#6080b8")
        self._signals = PipelineSignals()
        self._signals.log.connect(self._log.append)
        self._signals.warning.connect(self._log.append_warn)
        self._signals.preview.connect(self._on_preview)
        self._signals.done.connect(lambda r, f=fi: self._on_done(r, f))
        self._signals.error.connect(lambda e, f=fi: self._on_error(e, f))
        self._worker = PipelineWorker(cfg, self._signals)
        self._pool.start(self._worker)

    def _on_preview(self, path):
        if self._current_item: self._current_item.previews.append(path)
        self._preview.update_preview(path)

    def _on_done(self, result, fi):
        fi.warnings = result.get("warnings", []); fi.report = result.get("report", {})
        fi.status = "warn" if fi.warnings else "ok"
        self._refresh_item(0); self._report.update(fi.report, fi.warnings)
        self._progress.setValue(1)
        self._log.append(f"✓  {fi.name}  complete", "#48c080")
        self._set_running(False)
        self._status.showMessage("Done")

    def _on_error(self, err, fi):
        fi.status = "error"; fi.warnings = [err[:200]]
        self._refresh_item(0); self._log.append_error(f"{fi.name}:\n{err}")
        self._progress.setValue(1)
        self._set_running(False)
        self._status.showMessage("Error")

    def _check_hdri_cal(self):
        try:
            import hdri_cal
            if not hasattr(hdri_cal, "_run_pipeline"):
                self._status.showMessage("⚠  hdri_cal.py missing _run_pipeline() — validate only")
                self._run_btn.setEnabled(False)
            else:
                self._status.showMessage("Ready  ·  hdri_cal loaded  ·  Drop files to begin")
        except ImportError:
            self._status.showMessage("⚠  hdri_cal.py not found — place alongside this script")
            self._run_btn.setEnabled(False); self._val_btn.setEnabled(False)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HDRI Cal")
    win = MainWindow(); win.show()
    cli = [a for a in sys.argv[1:] if a.lower().endswith((".exr",".hdr",".jpg",".jpeg",".png",".webp"))]
    if cli: win._on_files_dropped(cli)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
