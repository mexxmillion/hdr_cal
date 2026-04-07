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
    QStatusBar, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QSize, QTimer, QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QColor

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

    # Chart sweep
    sweep_fov:          float = 50.0
    sweep_overlap:      float = 10.0
    sweep_min_pitch:    float = -30.0
    sweep_max_pitch:    float = 90.0



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
        "validate_only":        False,
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
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self._tabs = QTabWidget(); lay.addWidget(self._tabs)
        self._labels: dict[str, QLabel] = {}
        for name, key in [
            ("Source",     "source_preview.png"),
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

    def clear(self):
        for lbl in self._labels.values():
            lbl.setText("─"); lbl.setStyleSheet("background:#10101a; color:#282840; font-size:28px;")
            lbl.setPixmap(QPixmap())

    def update_preview(self, path):
        fname = os.path.basename(path)
        for key, lbl in self._labels.items():
            if fname == os.path.basename(key) or path.endswith(key):
                px = QPixmap(path)
                if not px.isNull():
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

        f.addRow("WB source",        self.wb_source)
        f.addRow("Exposure source",  self.exp_source)
        f.addRow("Integration mode", self.integration_mode)
        f.addRow("Sun solve",        self.solver_mode)
        f.addRow("Final balance",   self.final_balance_target)
        f.addRow("Albedo",          self.albedo)
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

        grp = QGroupBox("Chart Sweep")
        f = QFormLayout(grp)
        self.sweep_fov = QDoubleSpinBox()
        self.sweep_fov.setRange(30.0, 140.0)
        self.sweep_fov.setValue(50.0)
        self.sweep_fov.setDecimals(0)
        self.sweep_fov.setSuffix("°")
        self.sweep_fov.setToolTip("Tile field of view for chart sweep")
        self.sweep_overlap = QDoubleSpinBox()
        self.sweep_overlap.setRange(0.0, 60.0)
        self.sweep_overlap.setValue(10.0)
        self.sweep_overlap.setDecimals(0)
        self.sweep_overlap.setSuffix("°")
        self.sweep_overlap.setToolTip("Overlap between adjacent sweep tiles")
        self.sweep_min_pitch = QDoubleSpinBox()
        self.sweep_min_pitch.setRange(-90.0, 45.0)
        self.sweep_min_pitch.setValue(-30.0)
        self.sweep_min_pitch.setDecimals(0)
        self.sweep_min_pitch.setSuffix("°")
        self.sweep_min_pitch.setToolTip("Lowest pitch (-90=zenith, 0=horizon, default -30)")
        self.sweep_max_pitch = QDoubleSpinBox()
        self.sweep_max_pitch.setRange(0.0, 90.0)
        self.sweep_max_pitch.setValue(90.0)
        self.sweep_max_pitch.setDecimals(0)
        self.sweep_max_pitch.setSuffix("°")
        self.sweep_max_pitch.setToolTip("Highest pitch (+90=nadir/ground, default 90)")
        f.addRow("Sweep FOV",       self.sweep_fov)
        f.addRow("Sweep overlap",   self.sweep_overlap)
        f.addRow("Min pitch",       self.sweep_min_pitch)
        f.addRow("Max pitch",       self.sweep_max_pitch)
        self._adv_groups.append(grp)

        for grp in self._adv_groups:
            self._lay.addWidget(grp)

    def _sync_calibration_mode(self):
        show_advanced = (self.calibration_mode.currentText() == "advanced")
        for grp in self._adv_groups:
            grp.setVisible(show_advanced)

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
        cfg.sweep_fov = self.sweep_fov.value()
        cfg.sweep_overlap = self.sweep_overlap.value()
        cfg.sweep_min_pitch = self.sweep_min_pitch.value()
        cfg.sweep_max_pitch = self.sweep_max_pitch.value()

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
        self.resize(1380, 880)
        self._file_items:   List[FileItem]           = []
        self._current_item: Optional[FileItem]       = None
        self._worker:       Optional[PipelineWorker] = None
        self._signals:      Optional[PipelineSignals] = None
        self._pool          = QThreadPool(); self._pool.setMaxThreadCount(1)
        self._running       = False; self._abort_flag = False
        self._syncing_settings = False
        self._build_ui(); self._check_hdri_cal()

    def _build_ui(self):
        self.setStyleSheet(VFX_STYLE)
        tb = QToolBar(); tb.setMovable(False); tb.setIconSize(QSize(16,16)); self.addToolBar(tb)

        self._run_btn = QPushButton("▶  Process"); self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._on_run)
        self._val_btn = QPushButton("⚡  Validate"); self._val_btn.setObjectName("validate_btn")
        self._val_btn.clicked.connect(self._on_validate)
        self._abort_btn = QPushButton("■  Abort"); self._abort_btn.setObjectName("abort_btn")
        self._abort_btn.setEnabled(False); self._abort_btn.clicked.connect(self._on_abort)
        tb.addWidget(self._run_btn); tb.addWidget(self._val_btn); tb.addWidget(self._abort_btn)
        tb.addSeparator()
        self._progress = QProgressBar(); self._progress.setMaximumWidth(180)
        self._progress.setMaximumHeight(12); self._progress.setVisible(False)
        tb.addWidget(self._progress)

        sp = QSplitter(Qt.Horizontal); self.setCentralWidget(sp)

        # LEFT
        left = QWidget(); left.setMinimumWidth(220); left.setMaximumWidth(320)
        ll = QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.setSpacing(6)
        hdr = QLabel("HDRI Cal"); hdr.setObjectName("header")
        sub = QLabel("IBL Calibration Pipeline"); sub.setObjectName("subheader")
        ll.addWidget(hdr); ll.addWidget(sub)
        div = QFrame(); div.setObjectName("divider"); div.setFixedHeight(1); ll.addWidget(div)
        self._drop = DropZone(); self._drop.files_dropped.connect(self._on_files_dropped)
        ll.addWidget(self._drop)
        qh = QHBoxLayout(); qh.addWidget(QLabel("Queue"))
        bc = QPushButton("Clear Files"); bc.setMinimumWidth(88); bc.setMaximumHeight(22)
        bc.clicked.connect(self._clear_queue); qh.addStretch(); qh.addWidget(bc); ll.addLayout(qh)

        qh2 = QHBoxLayout()
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setMaximumHeight(22)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._requeue_btn = QPushButton("Requeue Selected")
        self._requeue_btn.setMaximumHeight(22)
        self._requeue_btn.clicked.connect(self._requeue_selected)
        ll_btn = QPushButton("Open Output Folder")
        ll_btn.setMaximumHeight(22)
        ll_btn.clicked.connect(self._open_output_folder)
        qh2.addWidget(self._remove_btn)
        qh2.addWidget(self._requeue_btn)
        ll.addLayout(qh2)
        ll.addWidget(ll_btn)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        ll.addWidget(self._file_list)
        self._q_status = QLabel("0 files"); self._q_status.setObjectName("subheader")
        ll.addWidget(self._q_status); sp.addWidget(left)

        # CENTRE
        centre = QWidget(); cl = QVBoxLayout(centre); cl.setContentsMargins(4,4,4,4); cl.setSpacing(4)
        vs = QSplitter(Qt.Vertical)
        self._preview = PreviewPanel(); vs.addWidget(self._preview)
        bot = QTabWidget(); self._log = LogPanel(); self._report = ReportPanel()
        bot.addTab(self._log, "Log"); bot.addTab(self._report, "Report"); vs.addWidget(bot)
        vs.setSizes([540, 200]); cl.addWidget(vs); sp.addWidget(centre)

        # RIGHT
        right = QWidget(); right.setMinimumWidth(270); right.setMaximumWidth(350)
        rl = QVBoxLayout(right); rl.setContentsMargins(4,4,4,4)
        lbl = QLabel("Settings"); lbl.setObjectName("section"); rl.addWidget(lbl)
        self._settings = SettingsPanel(); rl.addWidget(self._settings); sp.addWidget(right)
        sp.setSizes([270, 780, 310])

        self._source_preview_timer = QTimer(self)
        self._source_preview_timer.setSingleShot(True)
        self._source_preview_timer.timeout.connect(self._refresh_selected_source_preview)
        self._connect_settings_signals()

        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage("Ready  ·  Drop EXR / HDR / PNG / WebP files to begin")

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
            self._settings.sweep_fov.editingFinished,
            self._settings.sweep_overlap.editingFinished,
            self._settings.sweep_min_pitch.editingFinished,
            self._settings.sweep_max_pitch.editingFinished,
        ]:
            signal.connect(self._on_settings_changed)

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
            self._settings.sweep_fov.setValue(float(getattr(cfg, "sweep_fov", 50.0)))
            self._settings.sweep_overlap.setValue(float(getattr(cfg, "sweep_overlap", 10.0)))
            self._settings.sweep_min_pitch.setValue(float(getattr(cfg, "sweep_min_pitch", -30.0)))
            self._settings.sweep_max_pitch.setValue(float(getattr(cfg, "sweep_max_pitch", 90.0)))
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

    # ── File queue ────────────────────────────────────────────────────────
    def _on_files_dropped(self, paths):
        added = 0
        for p in paths:
            if not any(fi.path == p for fi in self._file_items):
                fi = FileItem(p)
                fi.config = self._settings.build_config(p)
                self._file_items.append(fi)
                self._ensure_source_preview(fi)
                it = QListWidgetItem(f"{ICON_WAIT}  {fi.name}"); it.setForeground(QColor(fi.color()))
                self._file_list.addItem(it); added += 1
        self._update_q()
        if self._file_list.count() and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)
        if added: self._status.showMessage(f"Added {added} file(s)")

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
        row = self._file_list.currentRow()
        if row < 0 or row >= len(self._file_items):
            return None
        return row

    def _remove_selected(self):
        if self._running:
            return
        idx = self._selected_index()
        if idx is None:
            return
        self._file_items.pop(idx)
        self._file_list.takeItem(idx)
        if not self._file_items:
            self._current_item = None
            self._preview.clear()
            self._report.clear()
        else:
            self._file_list.setCurrentRow(min(idx, len(self._file_items) - 1))
        self._update_q()
        self._status.showMessage("Removed selected file")

    def _requeue_selected(self):
        idx = self._selected_index()
        if idx is None:
            return
        fi = self._file_items[idx]
        if fi.status == "running":
            return
        source_preview = fi.source_preview
        fi.status = "waiting"
        fi.warnings = []
        fi.report = {}
        fi.previews = [source_preview] if source_preview else []
        self._refresh_item(idx)
        if self._current_item is fi:
            self._on_file_selected(idx)
        self._update_q()
        self._status.showMessage(f"Requeued: {fi.name}")

    def _open_output_folder(self):
        idx = self._selected_index()
        if idx is None:
            return
        fi = self._file_items[idx]
        cfg = self._current_config_for(fi)
        out_dir = str(Path(cfg.out).parent)
        os.makedirs(out_dir, exist_ok=True)
        try:
            os.startfile(out_dir)
        except Exception as e:
            QMessageBox.warning(self, "Open Folder", f"Could not open output folder:\n{e}")

    def _clear_queue(self):
        if self._running: return
        self._file_items.clear(); self._file_list.clear()
        self._preview.clear(); self._report.clear(); self._update_q()

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
        n = len(self._file_items)
        ok = sum(1 for fi in self._file_items if fi.status == "ok")
        w  = sum(1 for fi in self._file_items if fi.status == "warn")
        e  = sum(1 for fi in self._file_items if fi.status == "error")
        parts = [f"{n} file{'s' if n!=1 else ''}"]
        if ok: parts.append(f"{ok} ✓")
        if w:  parts.append(f"{w} ⚠")
        if e:  parts.append(f"{e} ✕")
        self._q_status.setText("  ·  ".join(parts))

    def _refresh_item(self, idx):
        if idx < 0 or idx >= self._file_list.count(): return
        fi = self._file_items[idx]; it = self._file_list.item(idx)
        it.setText(f"{fi.icon()}  {fi.name}"); it.setForeground(QColor(fi.color()))

    def _on_file_selected(self, row):
        if row < 0 or row >= len(self._file_items): return
        fi = self._file_items[row]; self._current_item = fi
        self._apply_config_to_settings(self._current_config_for(fi))
        self._ensure_source_preview(fi)
        self._preview.clear()
        for p in fi.previews: self._preview.update_preview(p)
        self._report.update(fi.report, fi.warnings)

    # ── Run / Validate ────────────────────────────────────────────────────
    def _on_run(self):
        if not self._file_items: QMessageBox.information(self, "No files", "Add HDRI files first."); return
        if not self._running: self._run_all(False)

    def _on_validate(self):
        if not self._file_items: QMessageBox.information(self, "No files", "Add HDRI files first."); return
        if not self._running: self._run_validate_all()

    def _on_abort(self):
        self._abort_flag = True
        if self._worker: self._worker.abort()
        self._status.showMessage("Aborting…")

    def _set_running(self, state):
        self._running = state
        self._run_btn.setEnabled(not state); self._val_btn.setEnabled(not state)
        self._abort_btn.setEnabled(state); self._progress.setVisible(state)
        self._remove_btn.setEnabled(not state)
        self._requeue_btn.setEnabled(True)

    def _run_all(self, validate_only=False):
        self._abort_flag = False; self._set_running(True); self._log.clear()
        waiting = [i for i, fi in enumerate(self._file_items) if fi.status in ("waiting","warn","error")]
        if not waiting: self._set_running(False); return
        self._progress.setRange(0, len(waiting)); self._progress.setValue(0)
        self._run_queue = list(waiting); self._run_done = 0
        self._process_next(validate_only)

    def _run_validate_all(self):
        self._set_running(True); self._log.clear()
        for i, fi in enumerate(self._file_items):
            if self._abort_flag: break
            fi.status = "running"; self._refresh_item(i)
            self._file_list.setCurrentRow(i); QApplication.processEvents()
            result = run_validate(fi.path, self._current_config_for(fi))
            fi.warnings = result.get("warnings", []); fi.report = result
            if not result.get("ok"):
                fi.status = "error"; self._log.append_error(f"{fi.name}: {result.get('error')}")
            else:
                fi.status = "warn" if fi.warnings else "ok"
                clipped = "⚠ CLIPPED" if result.get("likely_clipped") else f"clip={result.get('clip_fraction',0):.3%}"
                self._log.append(
                    f"{fi.name}  {result.get('resolution','?')}  "
                    f"E_upper={result.get('E_upper',0):.3f}  {clipped}  "
                    f"chroma={result.get('chroma_imbalance',0):.3f}",
                    "#80c0f0" if not fi.warnings else "#d8c040")
            self._refresh_item(i)
        self._update_q(); self._set_running(False)
        self._status.showMessage("Validation complete")
        if self._file_items: self._file_list.setCurrentRow(0)

    def _process_next(self, validate_only=False):
        if self._abort_flag or not self._run_queue:
            self._set_running(False)
            self._status.showMessage("Aborted" if self._abort_flag else f"Done — {self._run_done} file(s)")
            self._update_q(); return
        idx = self._run_queue.pop(0); fi = self._file_items[idx]
        fi.status = "running"; fi.warnings = []; fi.previews = [fi.source_preview] if fi.source_preview else []
        self._refresh_item(idx); self._file_list.setCurrentRow(idx)
        cfg = self._current_config_for(fi); cfg.validate_only = validate_only
        os.makedirs(cfg.debug_dir, exist_ok=True)
        self._status.showMessage(f"Processing: {fi.name}")
        self._log.append(f"\n{'─'*60}\n▶  {fi.name}", "#6080b8")
        self._signals = PipelineSignals()
        self._signals.log.connect(self._log.append)
        self._signals.warning.connect(self._log.append_warn)
        self._signals.preview.connect(self._on_preview)
        self._signals.done.connect(lambda r, i=idx, f=fi: self._on_done(r, i, f, validate_only))
        self._signals.error.connect(lambda e, i=idx, f=fi: self._on_error(e, i, f, validate_only))
        self._worker = PipelineWorker(cfg, self._signals)
        self._pool.start(self._worker)

    def _on_preview(self, path):
        if self._current_item: self._current_item.previews.append(path)
        self._preview.update_preview(path)

    def _on_done(self, result, idx, fi, validate_only):
        fi.warnings = result.get("warnings", []); fi.report = result.get("report", {})
        fi.status = "warn" if fi.warnings else "ok"
        self._refresh_item(idx); self._report.update(fi.report, fi.warnings)
        self._run_done += 1; self._progress.setValue(self._progress.value() + 1)
        self._log.append(f"✓  {fi.name}  complete", "#48c080")
        QTimer.singleShot(50, lambda: self._process_next(validate_only))

    def _on_error(self, err, idx, fi, validate_only):
        fi.status = "error"; fi.warnings = [err[:200]]
        self._refresh_item(idx); self._log.append_error(f"{fi.name}:\n{err}")
        self._run_done += 1; self._progress.setValue(self._progress.value() + 1)
        QTimer.singleShot(50, lambda: self._process_next(validate_only))

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
