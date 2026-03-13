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
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QGroupBox, QTextEdit, QFileDialog, QProgressBar,
    QScrollArea, QSizePolicy, QTabWidget, QFrame, QToolBar,
    QStatusBar, QMessageBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QSize, QTimer, QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QColor

import numpy as np

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
    wb_source:          str = "auto"
    exposure_source:    str = "auto"
    kelvin:             Optional[float] = None
    rgb_scale:          Optional[str] = None
    dome_wb:            Optional[str] = None

    # Metering (used when exposure_source=metering)
    metering_mode:      str = "bottom_dome"
    meter_stat:         str = "median"
    meter_target:       float = 0.18
    swatch_size:        int = 5

    # Hot lobe
    sun_threshold:      float = 0.1
    sun_upper_only:     bool = False

    # Centering
    center_hdri:        bool = True
    center_elevation:   bool = False

    # Gain / energy
    lobe_neutralise:    float = 1.0
    albedo:             float = 0.18
    sphere_res:         int = 96
    sun_gain_ceiling:   float = 2000.0
    sun_gain_rolloff:   float = 500.0

    # Reference sphere plate
    ref_sphere:         Optional[str] = None
    ref_sphere_cx:      int = -1
    ref_sphere_cy:      int = -1
    ref_sphere_r:       int = -1
    ref_sphere_albedo:  float = 0.18


def config_to_namespace(cfg: PipelineConfig):
    ns = types.SimpleNamespace()
    for k, v in cfg.__dict__.items():
        setattr(ns, k, v)

    # Aliases / derived attrs expected by hdri_cal._run_pipeline
    ns.hot_threshold        = cfg.sun_threshold
    ns.upper_only           = cfg.sun_upper_only
    ns.sun_blur_px          = 0

    # Attrs not in PipelineConfig but referenced in pipeline
    _defaults = {
        "wb_swatch":            None,
        "swatch":               None,
        "colorchecker":         None,
        "colorchecker_in_hdri": False,
        "cc_read_backend":      "colour",
        "cc_compare_backends":  False,
        "sphere_solve":         "energy_conservation",
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
        text = QLabel("Drop EXR / HDR  ·  or click to browse")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("color: #404060; font-size: 11px; border: none;")
        lay.addWidget(icon); lay.addWidget(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDRI files", "",
            "HDRI Images (*.exr *.hdr *.jpg *.jpeg *.png);;All Files (*.*)")
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
                 if u.toLocalFile().lower().endswith((".exr",".hdr",".jpg",".jpeg",".png"))]
        if paths: self.files_dropped.emit(paths)


# ── File item ──────────────────────────────────────────────────────────────────
class FileItem:
    def __init__(self, path):
        self.path = path; self.name = Path(path).name
        self.status = "waiting"; self.warnings = []; self.report = {}; self.previews = []

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
            ("WB",         "01_wb_preview.png"),
            ("Exposed",    "02_exposed_preview.png"),
            ("Lobe",       "03_hot_mask.png"),
            ("Chart Tile", "cc_detected_tile.jpg"),
            ("Rectified",  "cc_rectified_final.jpg"),
            ("Swatches",   "cc_swatch_comparison.jpg"),
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

        # Toggle button
        self._adv_btn = QPushButton("⚙  Advanced Options  ▼")
        self._adv_btn.setObjectName("adv_toggle")
        self._adv_btn.setCheckable(True); self._adv_btn.setChecked(False)
        self._adv_btn.toggled.connect(self._toggle_advanced)
        self._lay.addWidget(self._adv_btn)

        self._lay.addStretch()

        # Hide advanced initially
        for grp in self._adv_groups:
            grp.setVisible(False)

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
        grp = QGroupBox("Calibration"); f = QFormLayout(grp)

        self.wb_source = QComboBox()
        self.wb_source.addItems(["auto","chart","sphere","dome","kelvin","manual","none"])
        self.wb_source.setToolTip(
            "auto   — chart (if found, conf≥0.2) → sphere → none\n"
            "chart  — search HDRI tiles for ColorChecker\n"
            "sphere — neutralise grey sphere render\n"
            "dome   — grey-world upper hemisphere\n"
            "kelvin — colour temperature (set in Advanced)\n"
            "manual — explicit R,G,B multipliers (set in Advanced)\n"
            "none   — skip WB")

        self.exp_source = QComboBox()
        self.exp_source.addItems(["auto","chart","sphere","metering","none"])
        self.exp_source.setToolTip(
            "auto     — chart if found, else sphere\n"
            "chart    — patch 22 luma → absolute photometric\n"
            "sphere   — render grey sphere, normalise to albedo\n"
            "metering — dome metering (configure in Advanced)\n"
            "none     — skip exposure scaling")

        self.center_hdri = QCheckBox("Centre HDRI on sun")
        self.center_hdri.setChecked(True)
        self.center_hdri.setToolTip("Shift azimuth so the sun sits at the centre column (φ=0)")
        self.input_colorspace = QComboBox()
        self.input_colorspace.addItems(["auto", "acescg", "srgb"])
        self.input_colorspace.setToolTip(
            "Input primaries of the source image.\n"
            "auto   - EXR/HDR defaults to ACEScg, JPG/PNG to sRGB\n"
            "acescg - treat input as linear ACEScg\n"
            "srgb   - treat input as linear sRGB and convert to ACEScg before processing"
        )

        f.addRow("WB source",       self.wb_source)
        f.addRow("Exposure source", self.exp_source)
        f.addRow("Input primaries", self.input_colorspace)
        f.addRow("",                self.center_hdri)
        self._lay.addWidget(grp)

    def _build_advanced_groups(self):
        self._adv_groups: list[QWidget] = []

        # ── WB detail ─────────────────────────────────────────────────────
        grp = QGroupBox("White Balance — Detail"); f = QFormLayout(grp)
        self.wb_kelvin = QDoubleSpinBox()
        self.wb_kelvin.setRange(2000,12000); self.wb_kelvin.setValue(5600); self.wb_kelvin.setSuffix(" K")
        self.wb_kelvin.setToolTip("Used when WB source = kelvin")
        self.wb_rgb = QLineEdit("1.0,1.0,1.0")
        self.wb_rgb.setPlaceholderText("R,G,B  e.g. 1.05,1.0,0.95")
        self.wb_rgb.setToolTip("Used when WB source = manual")
        self.dome_wb_mode = QComboBox()
        self.dome_wb_mode.addItems(["upper_dome","full_dome","hot_exclude"])
        self.dome_wb_mode.setToolTip("Sub-mode for WB source = dome")
        f.addRow("Kelvin",    self.wb_kelvin)
        f.addRow("RGB scale", self.wb_rgb)
        f.addRow("Dome mode", self.dome_wb_mode)
        self._adv_groups.append(grp)

        # ── Exposure detail ───────────────────────────────────────────────
        grp = QGroupBox("Exposure — Detail"); f = QFormLayout(grp)
        self.metering_mode = QComboBox()
        self.metering_mode.addItems(["bottom_dome","whole_scene","swatch","upper_hemi_irradiance"])
        self.metering_mode.setToolTip(
            "Used when exposure source = metering\n"
            "bottom_dome            — median of lower hemisphere\n"
            "whole_scene            — median of full panorama\n"
            "upper_hemi_irradiance  — cosine-weighted E_upper\n"
            "swatch                 — sample a specific pixel")
        self.meter_target = QDoubleSpinBox()
        self.meter_target.setRange(0.001,100.0); self.meter_target.setValue(0.18); self.meter_target.setDecimals(4)
        self.meter_target.setToolTip("0.18 = 18% grey.  1.0 for normalised IBL.")
        self.albedo = QDoubleSpinBox()
        self.albedo.setRange(0.01,1.0); self.albedo.setValue(0.18); self.albedo.setDecimals(4)
        self.albedo.setToolTip("Reflectance of reference grey card / sphere (default 0.18)")
        f.addRow("Metering mode", self.metering_mode)
        f.addRow("Meter target",  self.meter_target)
        f.addRow("Albedo",        self.albedo)
        self._adv_groups.append(grp)

        # ── Sun / Hot Lobe ────────────────────────────────────────────────
        grp = QGroupBox("Sun / Hot Lobe"); f = QFormLayout(grp)
        self.sun_threshold = QDoubleSpinBox()
        self.sun_threshold.setRange(0.01,0.99); self.sun_threshold.setValue(0.1); self.sun_threshold.setDecimals(3)
        self.sun_threshold.setToolTip("Fraction below peak defining the lobe.\n0.1 → lobe = [0.9×peak … peak].  Lower = tighter disc.")
        self.sun_upper_only = QCheckBox("Upper hemisphere only")
        self.lobe_neutralise = QDoubleSpinBox()
        self.lobe_neutralise.setRange(0.0,1.0); self.lobe_neutralise.setValue(1.0); self.lobe_neutralise.setDecimals(2)
        self.lobe_neutralise.setToolTip("0 = keep sun colour  |  1 = fully desaturate to white before boost")
        self.sun_gain_ceiling = QDoubleSpinBox()
        self.sun_gain_ceiling.setRange(10,20000); self.sun_gain_ceiling.setValue(2000)
        self.sun_gain_ceiling.setDecimals(0); self.sun_gain_ceiling.setSuffix("×")
        self.sun_gain_ceiling.setToolTip("Hard cap on per-channel sun gain")
        self.sun_gain_rolloff = QDoubleSpinBox()
        self.sun_gain_rolloff.setRange(10,10000); self.sun_gain_rolloff.setValue(500)
        self.sun_gain_rolloff.setDecimals(0); self.sun_gain_rolloff.setSuffix("×")
        self.sun_gain_rolloff.setToolTip("Gains above this soft-roll to the ceiling (sqrt curve)")
        f.addRow("Sun threshold",   self.sun_threshold)
        f.addRow("",                self.sun_upper_only)
        f.addRow("Lobe neutralise", self.lobe_neutralise)
        f.addRow("Gain ceiling",    self.sun_gain_ceiling)
        f.addRow("Gain rolloff",    self.sun_gain_rolloff)
        self._adv_groups.append(grp)

        # ── Misc ──────────────────────────────────────────────────────────
        grp = QGroupBox("Misc"); f = QFormLayout(grp)
        self.center_elevation = QCheckBox("Centre elevation too")
        self.center_elevation.setToolTip("Also shift vertically so sun sits on horizon")
        self.sphere_res = QSpinBox()
        self.sphere_res.setRange(32,256); self.sphere_res.setValue(96); self.sphere_res.setSingleStep(16)
        self.sphere_res.setToolTip("Validation sphere render resolution in pixels")
        self.ref_sphere = QLineEdit(); self.ref_sphere.setPlaceholderText("Grey ball plate (optional)"); self.ref_sphere.setReadOnly(True)
        btn_sp = QPushButton("…"); btn_sp.setMaximumWidth(26)
        btn_sp.clicked.connect(lambda: self._browse(self.ref_sphere, "Sphere plate (*.jpg *.jpeg *.png)"))
        row_sp = QHBoxLayout(); row_sp.addWidget(self.ref_sphere); row_sp.addWidget(btn_sp)
        f.addRow("",           self.center_elevation)
        f.addRow("Sphere res", self.sphere_res)
        f.addRow("Ref sphere", row_sp)
        self._adv_groups.append(grp)

        # Insert all into layout (before the toggle button gets added)
        for grp in self._adv_groups:
            self._lay.addWidget(grp)

    def _toggle_advanced(self, checked: bool):
        for grp in self._adv_groups:
            grp.setVisible(checked)
        self._adv_btn.setText(
            "⚙  Advanced Options  ▲" if checked else "⚙  Advanced Options  ▼")

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
        cfg.wb_source       = self.wb_source.currentText()
        cfg.exposure_source = self.exp_source.currentText()
        cfg.center_hdri     = self.center_hdri.isChecked()

        # Advanced — WB detail
        cfg.kelvin    = self.wb_kelvin.value()    if cfg.wb_source == "kelvin" else None
        cfg.rgb_scale = self.wb_rgb.text().strip() if cfg.wb_source == "manual" else None
        cfg.dome_wb   = self.dome_wb_mode.currentText() if cfg.wb_source == "dome" else None

        # Advanced — Exposure
        cfg.metering_mode  = self.metering_mode.currentText()
        cfg.meter_target   = self.meter_target.value()
        cfg.albedo         = self.albedo.value()

        # Advanced — Lobe
        cfg.sun_threshold    = self.sun_threshold.value()
        cfg.sun_upper_only   = self.sun_upper_only.isChecked()
        cfg.lobe_neutralise  = self.lobe_neutralise.value()
        cfg.sun_gain_ceiling = self.sun_gain_ceiling.value()
        cfg.sun_gain_rolloff = self.sun_gain_rolloff.value()

        # Advanced — Misc
        cfg.center_elevation = self.center_elevation.isChecked()
        cs = self.input_colorspace.currentText()
        cfg.colorspace = None if cs == "auto" else cs
        cfg.sphere_res = self.sphere_res.value()
        ref = self.ref_sphere.text().strip()
        cfg.ref_sphere = ref if ref else None

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

        self._status = QStatusBar(); self.setStatusBar(self._status)
        self._status.showMessage("Ready  ·  Drop EXR / HDR files to begin")

    # ── File queue ────────────────────────────────────────────────────────
    def _on_files_dropped(self, paths):
        added = 0
        for p in paths:
            if not any(fi.path == p for fi in self._file_items):
                fi = FileItem(p); self._file_items.append(fi)
                it = QListWidgetItem(f"{ICON_WAIT}  {fi.name}"); it.setForeground(QColor(fi.color()))
                self._file_list.addItem(it); added += 1
        self._update_q()
        if self._file_list.count() and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)
        if added: self._status.showMessage(f"Added {added} file(s)")

    def _clear_queue(self):
        if self._running: return
        self._file_items.clear(); self._file_list.clear()
        self._preview.clear(); self._report.clear(); self._update_q()

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
            result = run_validate(fi.path, self._settings.build_config(fi.path))
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
        fi.status = "running"; fi.warnings = []; fi.previews = []
        self._refresh_item(idx); self._file_list.setCurrentRow(idx)
        cfg = self._settings.build_config(fi.path); cfg.validate_only = validate_only
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
    cli = [a for a in sys.argv[1:] if a.lower().endswith((".exr",".hdr",".jpg",".jpeg",".png"))]
    if cli: win._on_files_dropped(cli)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
