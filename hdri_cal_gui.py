"""
hdri_cal_gui.py  —  PySide6 front-end for the HDRI calibration pipeline.

Usage:
    python hdri_cal_gui.py

Requires:
    pip install PySide6
    (plus all hdri_cal.py dependencies)
"""

from __future__ import annotations
import sys, os, json, traceback, threading, time, types
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# ── PySide6 ───────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QGroupBox, QTextEdit, QFileDialog, QProgressBar,
    QScrollArea, QSizePolicy, QTabWidget, QFrame, QToolBar,
    QStatusBar, QMessageBox, QAbstractItemView, QSlider,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QSize, QTimer, QMimeData,
    QRunnable, QThreadPool, Slot,
)
from PySide6.QtGui import (
    QPixmap, QImage, QDragEnterEvent, QDropEvent, QColor,
    QPalette, QFont, QAction, QIcon, QPainter, QBrush, QPen,
)

import numpy as np

# ── Dark VFX palette ──────────────────────────────────────────────────────────
VFX_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a1e;
    color: #d4d4d8;
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 12px;
}
QSplitter::handle { background: #2a2a30; width: 2px; height: 2px; }
QGroupBox {
    border: 1px solid #3a3a42;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: 600;
    color: #a0a0b0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background-color: #2a2a32;
    border: 1px solid #3a3a44;
    border-radius: 3px;
    padding: 5px 14px;
    color: #c8c8d4;
    font-size: 12px;
}
QPushButton:hover { background-color: #32323e; border-color: #5a5a70; }
QPushButton:pressed { background-color: #22222a; }
QPushButton:disabled { color: #505060; border-color: #2a2a32; }
QPushButton#run_btn {
    background-color: #1a472a;
    border-color: #2d6a42;
    color: #7fffb0;
    font-weight: 700;
    font-size: 13px;
    padding: 8px 24px;
}
QPushButton#run_btn:hover { background-color: #1e5530; border-color: #3a8050; }
QPushButton#run_btn:disabled { background-color: #1a2a1e; color: #3a6040; }
QPushButton#validate_btn {
    background-color: #1a2a47;
    border-color: #2d4a6a;
    color: #7fc8ff;
    font-weight: 600;
}
QPushButton#validate_btn:hover { background-color: #1e3255; }
QPushButton#abort_btn {
    background-color: #471a1a;
    border-color: #6a2d2d;
    color: #ff9090;
    font-weight: 600;
}
QPushButton#abort_btn:hover { background-color: #551e1e; }
QComboBox {
    background-color: #22222c;
    border: 1px solid #3a3a44;
    border-radius: 3px;
    padding: 3px 8px;
    color: #c8c8d4;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #22222c;
    border: 1px solid #4a4a58;
    selection-background-color: #2a3a50;
    color: #c8c8d4;
}
QCheckBox { spacing: 6px; color: #b8b8c8; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #4a4a58; border-radius: 2px;
    background: #1e1e28;
}
QCheckBox::indicator:checked { background: #2a6a4a; border-color: #3a8a5a; }
QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1e1e28;
    border: 1px solid #3a3a44;
    border-radius: 3px;
    padding: 3px 6px;
    color: #c8c8d4;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
    border-color: #4a6a9a;
}
QTextEdit {
    background-color: #13131a;
    border: 1px solid #2a2a36;
    border-radius: 3px;
    color: #a8c8a8;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
}
QListWidget {
    background-color: #16161e;
    border: 1px solid #2a2a36;
    border-radius: 3px;
    color: #c0c0d0;
}
QListWidget::item { padding: 4px 8px; border-bottom: 1px solid #1e1e28; }
QListWidget::item:selected { background-color: #1e3050; color: #90c0ff; }
QListWidget::item:hover { background-color: #1a1a28; }
QProgressBar {
    background-color: #1e1e28;
    border: 1px solid #2a2a36;
    border-radius: 2px;
    height: 6px;
    text-align: center;
}
QProgressBar::chunk { background-color: #2a7a4a; border-radius: 2px; }
QTabWidget::pane { border: 1px solid #2a2a36; }
QTabBar::tab {
    background: #1e1e28; border: 1px solid #2a2a36;
    padding: 5px 14px; color: #808090;
    border-bottom: none;
}
QTabBar::tab:selected { background: #1a1a1e; color: #c8c8d4; border-bottom: 1px solid #1a1a1e; }
QScrollBar:vertical {
    background: #16161e; width: 8px; border: none;
}
QScrollBar::handle:vertical { background: #3a3a4a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QLabel#header { font-size: 18px; font-weight: 700; color: #e8e8f0; letter-spacing: -0.5px; }
QLabel#subheader { font-size: 11px; color: #606070; }
QLabel#section { font-size: 11px; font-weight: 600; color: #7a8a9a; text-transform: uppercase; letter-spacing: 0.5px; }
QLabel#status_ok { color: #60c080; font-size: 11px; }
QLabel#status_warn { color: #e0a030; font-size: 11px; }
QLabel#status_err { color: #e05050; font-size: 11px; }
QFrame#divider { background-color: #2a2a36; max-height: 1px; }
QFrame#drop_zone {
    border: 2px dashed #3a3a50;
    border-radius: 8px;
    background-color: #14141c;
}
QFrame#drop_zone[dragover="true"] {
    border-color: #4a7a9a;
    background-color: #141c24;
}
QStatusBar { background-color: #13131a; color: #606070; font-size: 11px; }
QToolBar { background-color: #16161e; border-bottom: 1px solid #2a2a30; spacing: 4px; }
"""

# ── File status icons (unicode) ───────────────────────────────────────────────
ICON_WAIT    = "○"
ICON_RUNNING = "◉"
ICON_OK      = "✓"
ICON_WARN    = "⚠"
ICON_ERROR   = "✕"
ICON_SKIP    = "–"

# ── Pipeline config dataclass (mirrors argparse namespace) ─────────────────────
@dataclass
class PipelineConfig:
    input:                  str = ""
    out:                    str = ""
    debug_dir:              str = "debug_hdri"
    res:                    str = "full"
    colorspace:             Optional[str] = None
    validate_only:          bool = False

    # WB
    colorchecker_in_hdri:   bool = False
    colorchecker:           Optional[str] = None
    sphere_wb:              bool = False
    dome_wb:                Optional[str] = None
    kelvin:                 Optional[float] = None
    rgb_scale:              Optional[str] = None
    wb_swatch:              Optional[str] = None

    # Metering
    metering_mode:          str = "bottom_dome"
    meter_stat:             str = "median"
    meter_target:           float = 0.18
    swatch:                 Optional[str] = None
    swatch_size:            int = 5

    # Solve target
    solve_target:           str = "auto"

    # Hot lobe
    sun_threshold:          float = 0.1
    sun_upper_only:         bool = False
    sun_blur_px:            int = 0

    # Centering
    center_hdri:            bool = True
    center_elevation:       bool = False

    # Gain solve
    sphere_solve:           str = "energy_conservation"
    lobe_neutralise:        float = 1.0
    albedo:                 float = 0.18
    sphere_res:             int = 96
    direct_highlight_target: float = 0.32
    target_peak_ratio:      float = 2.5

    # Reference sphere
    ref_sphere:             Optional[str] = None
    ref_sphere_cx:          int = -1
    ref_sphere_cy:          int = -1
    ref_sphere_r:           int = -1
    ref_sphere_albedo:      float = 0.18


def config_to_namespace(cfg: PipelineConfig):
    """Convert PipelineConfig to argparse-compatible Namespace."""
    ns = types.SimpleNamespace()
    for k, v in cfg.__dict__.items():
        setattr(ns, k.replace("-", "_"), v)
    return ns


# ── Pipeline runner (imports hdri_cal and calls run_pipeline) ──────────────────
def _build_run_func():
    """Import hdri_cal and wire up a callable run_pipeline(cfg, log_cb, progress_cb)."""
    try:
        import hdri_cal as _hc
        return _hc
    except ImportError as e:
        return None


# ── Worker thread ──────────────────────────────────────────────────────────────
class PipelineSignals(QObject):
    log       = Signal(str)       # log line
    warning   = Signal(str)       # warning line
    progress  = Signal(int, int)  # current, total steps
    preview   = Signal(str)       # path to preview image
    done      = Signal(dict)      # result dict
    error     = Signal(str)       # error message


class PipelineWorker(QRunnable):
    def __init__(self, cfg: PipelineConfig, signals: PipelineSignals):
        super().__init__()
        self.cfg     = cfg
        self.signals = signals
        self._abort  = False

    def abort(self):
        self._abort = True

    @Slot()
    def run(self):
        import hdri_cal as hc
        import io, sys

        result = {"warnings": [], "report": {}, "previews": []}
        orig_log  = hc.log
        orig_warn = hc.warn

        def _log(msg):
            self.signals.log.emit(str(msg))
            orig_log(msg)

        def _warn(msg):
            self.signals.warning.emit(str(msg))
            result["warnings"].append(str(msg))
            orig_warn(msg)

        hc.log  = _log
        hc.warn = _warn

        try:
            args = config_to_namespace(self.cfg)
            # Run pipeline - it writes files and returns via side-effects
            hc._run_pipeline(args)

            # Collect preview images
            dd = self.cfg.debug_dir
            for fname in [
                "01_wb_preview.png",
                "02_exposed_preview.png",
                "03_hot_mask.png",
                "09_verify_sphere_final.png",
                "07_corrected_preview.png",
                "colorchecker/cc_swatch_comparison.jpg",
                "colorchecker/cc_erp_swatches.jpg",
            ]:
                p = os.path.join(dd, fname)
                if os.path.exists(p):
                    result["previews"].append(p)
                    self.signals.preview.emit(p)

            # Load report
            rp = os.path.join(dd, "report.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    result["report"] = json.load(f)

            self.signals.done.emit(result)

        except Exception as e:
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            hc.log  = orig_log
            hc.warn = orig_warn


# ── Drop zone widget ────────────────────────────────────────────────────────────
class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("drop_zone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._icon = QLabel("⊕")
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet("font-size: 28px; color: #3a3a52; border: none;")
        self._text = QLabel("Drop EXR / HDR files here  ·  or click to browse")
        self._text.setAlignment(Qt.AlignCenter)
        self._text.setStyleSheet("color: #4a4a60; font-size: 12px; border: none;")
        lay.addWidget(self._icon)
        lay.addWidget(self._text)

        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDRI files", "",
            "HDRI Images (*.exr *.hdr *.jpg *.jpeg *.png);;All Files (*.*)")
        if paths:
            self.files_dropped.emit(paths)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("dragover", "true")
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, e):
        self.setProperty("dragover", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, e: QDropEvent):
        self.setProperty("dragover", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        paths = [u.toLocalFile() for u in e.mimeData().urls()
                 if u.toLocalFile().lower().endswith(
                     (".exr", ".hdr", ".jpg", ".jpeg", ".png"))]
        if paths:
            self.files_dropped.emit(paths)


# ── File queue item ─────────────────────────────────────────────────────────────
class FileItem:
    def __init__(self, path: str):
        self.path     = path
        self.name     = Path(path).name
        self.status   = "waiting"   # waiting | running | ok | warn | error | skip
        self.warnings = []
        self.report   = {}
        self.previews = []

    def status_icon(self) -> str:
        return {
            "waiting": ICON_WAIT, "running": ICON_RUNNING,
            "ok": ICON_OK, "warn": ICON_WARN,
            "error": ICON_ERROR, "skip": ICON_SKIP,
        }.get(self.status, "?")

    def status_color(self) -> str:
        return {
            "waiting": "#606070", "running": "#70a0e0",
            "ok": "#50b070", "warn": "#c09030",
            "error": "#c04040", "skip": "#505060",
        }.get(self.status, "#606070")


# ── Preview panel ───────────────────────────────────────────────────────────────
class PreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._tabs = QTabWidget()
        lay.addWidget(self._tabs)

        self._preview_labels: dict[str, QLabel] = {}
        self._no_preview = QLabel("No preview available")
        self._no_preview.setAlignment(Qt.AlignCenter)
        self._no_preview.setStyleSheet("color: #404050; font-size: 13px;")

        # Tabs for different preview stages
        for tab_name, tab_key in [
            ("WB",       "01_wb_preview.png"),
            ("Exposed",  "02_exposed_preview.png"),
            ("Hot Mask", "03_hot_mask.png"),
            ("Chart",    "colorchecker/cc_erp_swatches.jpg"),
            ("Swatches", "colorchecker/cc_swatch_comparison.jpg"),
            ("Final",    "09_verify_sphere_final.png"),
        ]:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #12121a;")
            lbl.setMinimumHeight(200)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            scroll = QScrollArea()
            scroll.setWidget(lbl)
            scroll.setWidgetResizable(True)
            scroll.setAlignment(Qt.AlignCenter)
            self._tabs.addTab(scroll, tab_name)
            self._preview_labels[tab_key] = lbl

        self.clear()

    def clear(self):
        for lbl in self._preview_labels.values():
            lbl.setText("─")
            lbl.setStyleSheet("background: #12121a; color: #303040; font-size: 30px;")
            lbl.setPixmap(QPixmap())

    def update_preview(self, path: str):
        """Update the matching tab when a preview image is ready."""
        fname = os.path.basename(path)
        # Match by suffix
        for key, lbl in self._preview_labels.items():
            if path.endswith(key) or fname == os.path.basename(key):
                px = QPixmap(path)
                if not px.isNull():
                    lbl.setPixmap(px.scaled(
                        lbl.width() or 600, lbl.height() or 400,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setStyleSheet("background: #12121a;")
                    lbl.setText("")
                break


# ── Report panel ────────────────────────────────────────────────────────────────
class ReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._warn_label = QLabel("Warnings")
        self._warn_label.setObjectName("section")
        lay.addWidget(self._warn_label)

        self._warn_list = QListWidget()
        self._warn_list.setMaximumHeight(120)
        lay.addWidget(self._warn_list)

        self._energy_label = QLabel("Energy Validation")
        self._energy_label.setObjectName("section")
        lay.addWidget(self._energy_label)

        self._energy_grid = QFormLayout()
        self._energy_widget = QWidget()
        self._energy_widget.setLayout(self._energy_grid)
        lay.addWidget(self._energy_widget)

        lay.addStretch()

    def clear(self):
        self._warn_list.clear()
        while self._energy_grid.rowCount():
            self._energy_grid.removeRow(0)

    def update(self, report: dict, warnings: list):
        self.clear()
        for w in warnings:
            item = QListWidgetItem(f"⚠  {w}")
            item.setForeground(QColor("#c09030"))
            self._warn_list.addItem(item)
        if not warnings:
            item = QListWidgetItem("✓  No warnings")
            item.setForeground(QColor("#50b070"))
            self._warn_list.addItem(item)

        def _row(label, val, ok=None):
            lbl = QLabel(str(val))
            if ok is True:    lbl.setObjectName("status_ok")
            elif ok is False: lbl.setObjectName("status_err")
            self._energy_grid.addRow(label, lbl)

        ev = report.get("energy_validation", {})
        if ev:
            _row("E_upper",         f"{ev.get('E_upper', 0):.4f}")
            _row("E_lower",         f"{ev.get('E_lower', 0):.4f}")
            _row("Flat card ↑",     f"{ev.get('pred_flat_card_up', 0):.5f}")
            _row("Sphere mean",     f"{ev.get('pred_sphere_mean', 0):.5f}")
            _row("Rendered sphere", f"{ev.get('rendered_sphere_mean', 0):.5f}")
            err = ev.get("rendered_vs_analytical_err", 1.0)
            _row("Rendered vs analytical", f"{err:.1%}", ok=(err < 0.10))

        # Validate-only report structure
        energy = report.get("energy", {})
        clamping = report.get("clamping", {})
        if energy:
            _row("E_upper",        f"{energy.get('E_upper', 0):.4f}")
            _row("E_lower",        f"{energy.get('E_lower', 0):.4f}")
            ci = energy.get("chroma_imbalance", 0)
            _row("Chroma imbalance", f"{ci:.4f}", ok=(ci < 0.05))
        if clamping:
            likely = clamping.get("likely_clipped", False)
            hr = clamping.get("headroom_ratio", 0)
            cf = clamping.get("clip_fraction", 0)
            _row("Lum max",        f"{clamping.get('lum_max', 0):.2f}")
            _row("Clip fraction",  f"{cf:.4%}", ok=(cf < 0.001))
            _row("Headroom ratio", f"{hr:.3f}", ok=(hr > 1.05))
            _row("Likely clipped", "YES ⚠" if likely else "No", ok=(not likely))


# ── Settings panel ──────────────────────────────────────────────────────────────
class SettingsPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        w = QWidget()
        self.setWidget(w)
        lay = QVBoxLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(8, 8, 8, 8)

        # ── Output ────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output")
        f_out = QFormLayout(grp_out)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Same folder as input  (leave blank)")
        self.out_suffix = QLineEdit("_cal")
        self.out_suffix.setMaximumWidth(80)
        self.debug_dir = QLineEdit("debug_hdri")
        self.res = QComboBox()
        self.res.addItems(["full", "half", "quarter"])

        f_out.addRow("Output suffix", self.out_suffix)
        f_out.addRow("Output dir", self.out_dir)
        f_out.addRow("Debug dir", self.debug_dir)
        f_out.addRow("Resolution", self.res)
        lay.addWidget(grp_out)

        # ── White Balance ─────────────────────────────────────────────────
        grp_wb = QGroupBox("White Balance")
        f_wb = QFormLayout(grp_wb)

        self.wb_mode = QComboBox()
        self.wb_mode.addItems([
            "auto (chart → sphere → none)",
            "colorchecker in HDRI",
            "sphere render",
            "dome grey-world",
            "colour temperature (K)",
            "manual RGB scale",
            "none",
        ])
        self.wb_mode.setToolTip(
            "auto: use chart if found, else sphere render, else no WB.\n"
            "colorchecker in HDRI: sweep tiles for CC24 chart.\n"
            "sphere render: neutralise Lambertian sphere mean.\n"
            "dome grey-world: grey-world assumption on upper sky.\n"
        )

        self.wb_kelvin = QDoubleSpinBox()
        self.wb_kelvin.setRange(2000, 12000)
        self.wb_kelvin.setValue(5600)
        self.wb_kelvin.setSuffix(" K")
        self.wb_kelvin.setEnabled(False)

        self.wb_rgb = QLineEdit("1.0,1.0,1.0")
        self.wb_rgb.setEnabled(False)
        self.wb_rgb.setPlaceholderText("R,G,B  e.g. 1.05,1.0,0.95")

        self.dome_wb_mode = QComboBox()
        self.dome_wb_mode.addItems(["upper_dome", "full_dome", "hot_exclude"])
        self.dome_wb_mode.setEnabled(False)

        f_wb.addRow("WB method", self.wb_mode)
        f_wb.addRow("  Kelvin", self.wb_kelvin)
        f_wb.addRow("  RGB scale", self.wb_rgb)
        f_wb.addRow("  Dome mode", self.dome_wb_mode)

        self.wb_mode.currentIndexChanged.connect(self._wb_mode_changed)
        lay.addWidget(grp_wb)

        # ── Reference files ───────────────────────────────────────────────
        grp_ref = QGroupBox("Reference Files (optional)")
        f_ref = QFormLayout(grp_ref)

        self.ref_chart = QLineEdit()
        self.ref_chart.setPlaceholderText("Drop chart plate .jpg / .png")
        self.ref_chart.setReadOnly(True)
        btn_chart = QPushButton("Browse…")
        btn_chart.clicked.connect(lambda: self._browse_ref(self.ref_chart,
            "Chart plate (*.jpg *.jpeg *.png *.tif)"))
        row_chart = QHBoxLayout()
        row_chart.addWidget(self.ref_chart)
        row_chart.addWidget(btn_chart)

        self.ref_sphere = QLineEdit()
        self.ref_sphere.setPlaceholderText("Drop grey ball plate .jpg / .png")
        self.ref_sphere.setReadOnly(True)
        btn_sphere = QPushButton("Browse…")
        btn_sphere.clicked.connect(lambda: self._browse_ref(self.ref_sphere,
            "Sphere plate (*.jpg *.jpeg *.png *.tif)"))
        row_sphere = QHBoxLayout()
        row_sphere.addWidget(self.ref_sphere)
        row_sphere.addWidget(btn_sphere)

        f_ref.addRow("Chart plate", row_chart)
        f_ref.addRow("Sphere plate", row_sphere)
        lay.addWidget(grp_ref)

        # ── Energy / Exposure ─────────────────────────────────────────────
        grp_exp = QGroupBox("Exposure / Energy")
        f_exp = QFormLayout(grp_exp)

        self.solve_target = QComboBox()
        self.solve_target.addItems(["auto", "patch22", "upper_hemi", "dominant_dir"])
        self.solve_target.setToolTip(
            "auto: patch22 if chart found, meter_target otherwise.\n"
            "patch22: albedo × π / 4 (absolute chart calibration).\n"
            "upper_hemi: normalise to upper hemisphere E_upper.\n"
            "dominant_dir: normalise to hemisphere facing sun.\n"
            "  Best for low-sun / coloured sky scenes."
        )

        self.metering_mode = QComboBox()
        self.metering_mode.addItems([
            "bottom_dome", "whole_scene", "swatch",
            "upper_hemi_irradiance", "dominant_dir_irradiance",
        ])

        self.meter_target = QDoubleSpinBox()
        self.meter_target.setRange(0.001, 100.0)
        self.meter_target.setValue(0.18)
        self.meter_target.setDecimals(4)
        self.meter_target.setToolTip("0.18 = 18% grey.  Use 1.0 for normalised IBL.")

        self.albedo = QDoubleSpinBox()
        self.albedo.setRange(0.01, 1.0)
        self.albedo.setValue(0.18)
        self.albedo.setDecimals(4)

        f_exp.addRow("Solve target", self.solve_target)
        f_exp.addRow("Metering mode", self.metering_mode)
        f_exp.addRow("Meter target", self.meter_target)
        f_exp.addRow("Albedo", self.albedo)
        lay.addWidget(grp_exp)

        # ── Hot lobe ──────────────────────────────────────────────────────
        grp_lobe = QGroupBox("Sun / Hot Lobe")
        f_lobe = QFormLayout(grp_lobe)

        self.sun_threshold = QDoubleSpinBox()
        self.sun_threshold.setRange(0.01, 0.99)
        self.sun_threshold.setValue(0.1)
        self.sun_threshold.setDecimals(3)
        self.sun_threshold.setToolTip(
            "Fraction below peak that defines the lobe boundary.\n"
            "0.1 → lobe spans [0.9×peak, peak].\n"
            "Lower = tighter sun disc.  Higher = includes more halo.")

        self.sun_upper_only = QCheckBox("Upper hemisphere only")
        self.sun_upper_only.setChecked(False)

        self.lobe_neutralise = QDoubleSpinBox()
        self.lobe_neutralise.setRange(0.0, 1.0)
        self.lobe_neutralise.setValue(1.0)
        self.lobe_neutralise.setDecimals(2)
        self.lobe_neutralise.setToolTip(
            "0 = keep sun colour as-is.\n"
            "1 = fully desaturate sun to neutral white before boost.\n"
            "0.8 = allow slight warmth through.")

        self.sphere_solve = QComboBox()
        self.sphere_solve.addItems([
            "energy_conservation", "direct_highlight",
            "iterative_peak_ratio", "none",
        ])

        f_lobe.addRow("Sun threshold", self.sun_threshold)
        f_lobe.addRow("", self.sun_upper_only)
        f_lobe.addRow("Lobe neutralise", self.lobe_neutralise)
        f_lobe.addRow("Gain solve mode", self.sphere_solve)
        lay.addWidget(grp_lobe)

        # ── Advanced ──────────────────────────────────────────────────────
        grp_adv = QGroupBox("Advanced")
        f_adv = QFormLayout(grp_adv)

        self.center_hdri = QCheckBox("Centre HDRI on sun azimuth")
        self.center_hdri.setChecked(True)

        self.colorspace = QComboBox()
        self.colorspace.addItems(["auto", "acescg", "srgb"])

        self.sphere_res = QSpinBox()
        self.sphere_res.setRange(32, 256)
        self.sphere_res.setValue(96)
        self.sphere_res.setSingleStep(16)

        f_adv.addRow("", self.center_hdri)
        f_adv.addRow("Colorspace", self.colorspace)
        f_adv.addRow("Sphere res", self.sphere_res)
        lay.addWidget(grp_adv)

        lay.addStretch()

    def _wb_mode_changed(self, idx):
        mode = self.wb_mode.currentText()
        self.wb_kelvin.setEnabled("temperature" in mode)
        self.wb_rgb.setEnabled("manual" in mode)
        self.dome_wb_mode.setEnabled("dome" in mode)

    def _browse_ref(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            line_edit.setText(path)

    def build_config(self, input_path: str) -> PipelineConfig:
        """Build a PipelineConfig from current widget state."""
        cfg = PipelineConfig()
        cfg.input = input_path

        # Output path
        p = Path(input_path)
        out_dir = self.out_dir.text().strip() or str(p.parent)
        suffix  = self.out_suffix.text().strip() or "_cal"
        cfg.out = str(Path(out_dir) / (p.stem + suffix + ".exr"))

        cfg.debug_dir = self.debug_dir.text().strip() or "debug_hdri"
        # Per-file debug dir to avoid collisions
        cfg.debug_dir = str(Path(out_dir) / (p.stem + "_debug"))

        cfg.res = self.res.currentText()

        cs = self.colorspace.currentText()
        cfg.colorspace = None if cs == "auto" else cs

        # WB
        mode = self.wb_mode.currentText()
        if "colorchecker" in mode:
            cfg.colorchecker_in_hdri = True
        elif "sphere" in mode:
            cfg.sphere_wb = True
        elif "dome" in mode:
            cfg.dome_wb = self.dome_wb_mode.currentText()
        elif "temperature" in mode:
            cfg.kelvin = self.wb_kelvin.value()
        elif "manual" in mode:
            cfg.rgb_scale = self.wb_rgb.text().strip()
        # "auto" and "none" leave defaults

        # Reference files
        ref_chart = self.ref_chart.text().strip()
        if ref_chart:
            cfg.colorchecker = ref_chart

        ref_sphere = self.ref_sphere.text().strip()
        if ref_sphere:
            cfg.ref_sphere = ref_sphere

        # Exposure / Energy
        cfg.solve_target    = self.solve_target.currentText()
        cfg.metering_mode   = self.metering_mode.currentText()
        cfg.meter_target    = self.meter_target.value()
        cfg.albedo          = self.albedo.value()

        # Hot lobe
        cfg.sun_threshold   = self.sun_threshold.value()
        cfg.sun_upper_only  = self.sun_upper_only.isChecked()
        cfg.lobe_neutralise = self.lobe_neutralise.value()
        cfg.sphere_solve    = self.sphere_solve.currentText()

        # Advanced
        cfg.center_hdri     = self.center_hdri.isChecked()
        cfg.sphere_res      = self.sphere_res.value()

        return cfg


# ── Log panel ────────────────────────────────────────────────────────────────────
class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Pipeline Log"))
        btn_clear = QPushButton("Clear")
        btn_clear.setMaximumWidth(60)
        btn_clear.clicked.connect(self.clear)
        hdr.addStretch()
        hdr.addWidget(btn_clear)
        lay.addLayout(hdr)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        lay.addWidget(self._text)

    def append(self, msg: str, color: str = "#a8c8a8"):
        self._text.setTextColor(QColor(color))
        self._text.append(msg)
        self._text.verticalScrollBar().setValue(
            self._text.verticalScrollBar().maximum())

    def append_warn(self, msg: str):
        self.append(f"⚠  {msg}", "#e0a030")

    def append_error(self, msg: str):
        self.append(f"✕  {msg}", "#e05050")

    def clear(self):
        self._text.clear()


# ── Validate-only report ──────────────────────────────────────────────────────
def run_validate(path: str, cfg: PipelineConfig) -> dict:
    """
    Run hdri_cal._run_pipeline in validate_only mode.
    Reads the JSON report and returns a normalised dict.
    """
    try:
        import hdri_cal as hc, types, json as _json
        args = config_to_namespace(cfg)
        args.input         = path
        args.validate_only = True
        args.debug_dir     = cfg.debug_dir
        os.makedirs(args.debug_dir, exist_ok=True)

        hc._run_pipeline(args)

        rp = os.path.join(args.debug_dir, "report.json")
        if os.path.exists(rp):
            with open(rp) as f:
                report = _json.load(f)
        else:
            report = {}

        energy   = report.get("energy", {})
        clamping = report.get("clamping", {})
        h, w_    = report.get("working_resolution", [0, 0])

        return {
            "path":              path,
            "resolution":        f"{w_}×{h}",
            "orientation_ratio": report.get("orientation_energy_ratio", 0),
            "E_upper":           energy.get("E_upper", 0),
            "clip_fraction":     clamping.get("clip_fraction", 0),
            "lum_max":           clamping.get("lum_max", 0),
            "likely_clipped":    clamping.get("likely_clipped", False),
            "headroom_ratio":    clamping.get("headroom_ratio", 0),
            "chroma_imbalance":  energy.get("chroma_imbalance", 0),
            "E_upper_rgb":       energy.get("E_upper_rgb", [0,0,0]),
            "warnings":          report.get("warnings", []),
            "report":            report,
            "ok":                True,
        }
    except Exception as e:
        return {"path": path, "ok": False, "error": str(e), "warnings": [str(e)],
                "report": {}}


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDRI Calibration  ·  VFX Pipeline Tool")
        self.resize(1400, 900)

        self._file_items:   List[FileItem]  = []
        self._current_item: Optional[FileItem] = None
        self._worker:       Optional[PipelineWorker] = None
        self._signals:      Optional[PipelineSignals] = None
        self._pool          = QThreadPool()
        self._pool.setMaxThreadCount(1)   # serial — one file at a time
        self._running       = False
        self._abort_flag    = False

        self._build_ui()
        self._check_hdri_cal()

    # ── UI Construction ───────────────────────────────────────────────────
    def _build_ui(self):
        self.setStyleSheet(VFX_STYLE)

        # Toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self._run_btn = QPushButton("▶  Process")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._on_run)

        self._validate_btn = QPushButton("⚡  Validate Only")
        self._validate_btn.setObjectName("validate_btn")
        self._validate_btn.clicked.connect(self._on_validate)

        self._abort_btn = QPushButton("■  Abort")
        self._abort_btn.setObjectName("abort_btn")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)

        tb.addWidget(self._run_btn)
        tb.addWidget(self._validate_btn)
        tb.addWidget(self._abort_btn)
        tb.addSeparator()

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setMaximumHeight(14)
        self._progress.setVisible(False)
        tb.addWidget(self._progress)

        # Central splitter: left | centre | right
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ── LEFT: file queue ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(230)
        left.setMaximumWidth(340)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)

        hdr_lbl = QLabel("HDRI Cal")
        hdr_lbl.setObjectName("header")
        sub_lbl = QLabel("VFX IBL Calibration Pipeline")
        sub_lbl.setObjectName("subheader")
        left_lay.addWidget(hdr_lbl)
        left_lay.addWidget(sub_lbl)

        div = QFrame()
        div.setObjectName("divider")
        div.setFixedHeight(1)
        left_lay.addWidget(div)

        self._drop_zone = DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        left_lay.addWidget(self._drop_zone)

        queue_hdr = QHBoxLayout()
        queue_hdr.addWidget(QLabel("File Queue"))
        btn_clear_q = QPushButton("✕")
        btn_clear_q.setMaximumWidth(28)
        btn_clear_q.setMaximumHeight(22)
        btn_clear_q.clicked.connect(self._clear_queue)
        queue_hdr.addStretch()
        queue_hdr.addWidget(btn_clear_q)
        left_lay.addLayout(queue_hdr)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        left_lay.addWidget(self._file_list)

        self._queue_status = QLabel("0 files")
        self._queue_status.setObjectName("subheader")
        left_lay.addWidget(self._queue_status)

        splitter.addWidget(left)

        # ── CENTRE: preview + log ──────────────────────────────────────────
        centre = QWidget()
        centre_lay = QVBoxLayout(centre)
        centre_lay.setContentsMargins(4, 4, 4, 4)
        centre_lay.setSpacing(4)

        v_split = QSplitter(Qt.Vertical)

        self._preview = PreviewPanel()
        v_split.addWidget(self._preview)

        bottom_tabs = QTabWidget()
        self._log_panel    = LogPanel()
        self._report_panel = ReportPanel()
        bottom_tabs.addTab(self._log_panel, "Log")
        bottom_tabs.addTab(self._report_panel, "Report")
        v_split.addWidget(bottom_tabs)

        v_split.setSizes([550, 200])
        centre_lay.addWidget(v_split)
        splitter.addWidget(centre)

        # ── RIGHT: settings ────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(360)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        settings_lbl = QLabel("Settings")
        settings_lbl.setObjectName("section")
        right_lay.addWidget(settings_lbl)

        self._settings = SettingsPanel()
        right_lay.addWidget(self._settings)
        splitter.addWidget(right)

        splitter.setSizes([280, 780, 320])

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready  ·  Drop EXR/HDR files to begin")

    # ── File queue management ─────────────────────────────────────────────
    def _on_files_dropped(self, paths: list):
        added = 0
        for p in paths:
            if not any(fi.path == p for fi in self._file_items):
                fi = FileItem(p)
                self._file_items.append(fi)
                item = QListWidgetItem(f"{ICON_WAIT}  {fi.name}")
                item.setForeground(QColor(fi.status_color()))
                self._file_list.addItem(item)
                added += 1
        self._update_queue_status()
        if self._file_list.count() > 0 and self._file_list.currentRow() < 0:
            self._file_list.setCurrentRow(0)
        if added:
            self._status.showMessage(f"Added {added} file(s)  ·  {len(self._file_items)} total")

    def _clear_queue(self):
        if self._running:
            return
        self._file_items.clear()
        self._file_list.clear()
        self._preview.clear()
        self._report_panel.clear()
        self._update_queue_status()

    def _update_queue_status(self):
        n     = len(self._file_items)
        ok    = sum(1 for fi in self._file_items if fi.status == "ok")
        warn  = sum(1 for fi in self._file_items if fi.status == "warn")
        err   = sum(1 for fi in self._file_items if fi.status == "error")
        parts = [f"{n} file{'s' if n!=1 else ''}"]
        if ok:   parts.append(f"{ok} ✓")
        if warn: parts.append(f"{warn} ⚠")
        if err:  parts.append(f"{err} ✕")
        self._queue_status.setText("  ·  ".join(parts))

    def _update_list_item(self, idx: int):
        if idx < 0 or idx >= self._file_list.count():
            return
        fi   = self._file_items[idx]
        item = self._file_list.item(idx)
        item.setText(f"{fi.status_icon()}  {fi.name}")
        item.setForeground(QColor(fi.status_color()))

    def _on_file_selected(self, row: int):
        if row < 0 or row >= len(self._file_items):
            return
        fi = self._file_items[row]
        self._current_item = fi
        self._preview.clear()
        for p in fi.previews:
            self._preview.update_preview(p)
        self._report_panel.update(fi.report, fi.warnings)

    # ── Run / Validate ────────────────────────────────────────────────────
    def _on_run(self):
        if not self._file_items:
            QMessageBox.information(self, "No files", "Add HDRI files to the queue first.")
            return
        if self._running:
            return
        self._run_all(validate_only=False)

    def _on_validate(self):
        if not self._file_items:
            QMessageBox.information(self, "No files", "Add HDRI files to the queue first.")
            return
        if self._running:
            return
        self._run_validate_all()

    def _on_abort(self):
        self._abort_flag = True
        if self._worker:
            self._worker.abort()
        self._status.showMessage("Aborting…")

    def _set_running(self, state: bool):
        self._running = state
        self._run_btn.setEnabled(not state)
        self._validate_btn.setEnabled(not state)
        self._abort_btn.setEnabled(state)
        self._progress.setVisible(state)

    def _run_all(self, validate_only=False):
        """Process all waiting files sequentially."""
        self._abort_flag = False
        self._set_running(True)
        self._log_panel.clear()

        waiting = [i for i, fi in enumerate(self._file_items)
                   if fi.status in ("waiting", "warn", "error")]
        if not waiting:
            self._set_running(False)
            return

        self._progress.setRange(0, len(waiting))
        self._progress.setValue(0)

        self._run_queue = list(waiting)
        self._run_done  = 0
        self._process_next(validate_only)

    def _run_validate_all(self):
        """Fast validate-only pass — no pipeline, just energy checks."""
        self._set_running(True)
        self._log_panel.clear()

        for i, fi in enumerate(self._file_items):
            if self._abort_flag:
                break
            fi.status = "running"
            self._update_list_item(i)
            self._file_list.setCurrentRow(i)
            QApplication.processEvents()

            result = run_validate(fi.path, self._settings.build_config(fi.path))

            fi.warnings = result.get("warnings", [])
            fi.report   = result

            if not result.get("ok", False):
                fi.status = "error"
                self._log_panel.append_error(f"{fi.name}: {result.get('error')}")
            else:
                warns = []
                if result["orientation_ratio"] < 1.2:
                    warns.append(f"Low upper/lower ratio {result['orientation_ratio']:.2f} — may be upside down")
                if result["clip_fraction"] > 0.001:
                    warns.append(f"Clipping: {result['clip_fraction']:.2%} of pixels at max ({result['lum_max']:.1f})")
                if result["chroma_imbalance"] > 0.10:
                    warns.append(f"Upper-hemi chroma imbalance {result['chroma_imbalance']:.2f} — coloured illuminant")

                fi.warnings = result.get("warnings", [])
                fi.report   = result.get("report", {})
                fi.status   = "warn" if fi.warnings else "ok"
                clipped_str = "⚠ CLIPPED" if result.get("likely_clipped") else f"clip={result.get('clip_fraction',0):.3%}"
                self._log_panel.append(
                    f"{fi.name}  {result.get('resolution','?')}  "
                    f"E_upper={result.get('E_upper',0):.3f}  "
                    f"{clipped_str}  "
                    f"chroma={result.get('chroma_imbalance',0):.3f}",
                    "#90c8f0" if not fi.warnings else "#e0c040")
                for w in fi.warnings:
                    self._log_panel.append_warn(f"  {fi.name}: {w}")

            self._update_list_item(i)

        self._update_queue_status()
        self._set_running(False)
        self._status.showMessage("Validation complete")

        # Auto-select first result
        if self._file_items:
            self._file_list.setCurrentRow(0)

    def _process_next(self, validate_only=False):
        if self._abort_flag or not self._run_queue:
            self._set_running(False)
            self._status.showMessage(
                "Aborted" if self._abort_flag else
                f"Done — {self._run_done} file(s) processed")
            self._update_queue_status()
            return

        idx = self._run_queue.pop(0)
        fi  = self._file_items[idx]
        fi.status   = "running"
        fi.warnings = []
        fi.previews = []
        self._update_list_item(idx)
        self._file_list.setCurrentRow(idx)

        cfg = self._settings.build_config(fi.path)
        cfg.validate_only = validate_only

        os.makedirs(cfg.debug_dir, exist_ok=True)

        self._status.showMessage(f"Processing: {fi.name}")
        self._log_panel.append(f"\n{'─'*60}\n▶  {fi.name}", "#7090c0")

        self._signals = PipelineSignals()
        self._signals.log.connect(self._log_panel.append)
        self._signals.warning.connect(self._log_panel.append_warn)
        self._signals.preview.connect(self._on_preview_ready)
        self._signals.done.connect(lambda r, i=idx, fi=fi: self._on_done(r, i, fi, validate_only))
        self._signals.error.connect(lambda e, i=idx, fi=fi: self._on_error(e, i, fi, validate_only))

        self._worker = PipelineWorker(cfg, self._signals)
        self._pool.start(self._worker)

    def _on_preview_ready(self, path: str):
        if self._current_item:
            self._current_item.previews.append(path)
        self._preview.update_preview(path)

    def _on_done(self, result: dict, idx: int, fi: FileItem, validate_only: bool):
        fi.warnings = result.get("warnings", [])
        fi.report   = result.get("report", {})
        fi.status   = "warn" if fi.warnings else "ok"
        self._update_list_item(idx)
        self._report_panel.update(fi.report, fi.warnings)
        self._run_done += 1
        self._progress.setValue(self._progress.value() + 1)
        self._log_panel.append(f"✓  {fi.name}  complete", "#50c080")
        QTimer.singleShot(50, lambda: self._process_next(validate_only))

    def _on_error(self, err: str, idx: int, fi: FileItem, validate_only: bool):
        fi.status = "error"
        fi.warnings = [err[:200]]
        self._update_list_item(idx)
        self._log_panel.append_error(f"{fi.name}:\n{err}")
        self._run_done += 1
        self._progress.setValue(self._progress.value() + 1)
        QTimer.singleShot(50, lambda: self._process_next(validate_only))

    # ── hdri_cal.py check ────────────────────────────────────────────────
    def _check_hdri_cal(self):
        try:
            import hdri_cal
            # Check for the _run_pipeline callable
            if not hasattr(hdri_cal, "_run_pipeline"):
                self._status.showMessage(
                    "⚠  hdri_cal.py is missing _run_pipeline() — "
                    "processing unavailable. Validate-only mode works.")
                self._run_btn.setEnabled(False)
            else:
                self._status.showMessage(
                    "Ready  ·  hdri_cal pipeline loaded  ·  Drop files to begin")
        except ImportError:
            self._status.showMessage(
                "⚠  hdri_cal.py not found — place it alongside hdri_cal_gui.py")
            self._run_btn.setEnabled(False)
            self._validate_btn.setEnabled(False)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HDRI Cal")
    app.setOrganizationName("VFX Pipeline")

    # Accept EXR/HDR files as command-line args
    win = MainWindow()
    win.show()

    cli_files = [a for a in sys.argv[1:]
                 if a.lower().endswith((".exr", ".hdr", ".jpg", ".jpeg", ".png"))]
    if cli_files:
        win._on_files_dropped(cli_files)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
