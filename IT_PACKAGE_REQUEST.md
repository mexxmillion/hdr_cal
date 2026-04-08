# IT Software Installation Request — HDR Calibration Pipeline
# Target: Windows 10 x64, Python 3.10, fully offline workstation

## PREREQUISITE: Python Runtime
- **Python 3.10.x (64-bit Windows installer)** — https://www.python.org/downloads/release/python-31011/
  - File: `python-3.10.11-amd64.exe`
  - Install with "Add to PATH" checked

---

## ALL PYTHON PACKAGES (42 total, Windows 10 x64 wheels for Python 3.10)

Download all `.whl` files from https://pypi.org/project/<package_name>/#files
Platform: **win_amd64**, Python: **cp310**

Install order matters — install in the numbered groups below, each group after the previous.

### Group 1 — Zero-dependency packages (install first)
| # | Package | Version | File | Size |
|---|---------|---------|------|------|
| 1 | numpy | 2.2.6 | `numpy-2.2.6-cp310-cp310-win_amd64.whl` | 12.9 MB |
| 2 | Pillow | 12.2.0 | `pillow-12.2.0-cp310-cp310-win_amd64.whl` | 6.8 MB |
| 3 | typing-extensions | 4.15.0 | `typing_extensions-4.15.0-py3-none-any.whl` | 44 KB |
| 4 | six | 1.17.0 | `six-1.17.0-py2.py3-none-any.whl` | 11 KB |
| 5 | certifi | 2026.2.25 | `certifi-2026.2.25-py3-none-any.whl` | 153 KB |
| 6 | charset-normalizer | 3.4.7 | `charset_normalizer-3.4.7-cp310-cp310-win_amd64.whl` | 155 KB |
| 7 | idna | 3.11 | `idna-3.11-py3-none-any.whl` | 69 KB |
| 8 | urllib3 | 2.6.3 | `urllib3-2.6.3-py3-none-any.whl` | 129 KB |
| 9 | packaging | 26.0 | `packaging-26.0-py3-none-any.whl` | 73 KB |
| 10 | cycler | 0.12.1 | `cycler-0.12.1-py3-none-any.whl` | 8 KB |
| 11 | pyparsing | 3.3.2 | `pyparsing-3.3.2-py3-none-any.whl` | 120 KB |
| 12 | kiwisolver | 1.5.0 | `kiwisolver-1.5.0-cp310-cp310-win_amd64.whl` | 72 KB |
| 13 | mpmath | 1.3.0 | `mpmath-1.3.0-py3-none-any.whl` | 524 KB |
| 14 | click | 8.3.2 | `click-8.3.2-py3-none-any.whl` | 106 KB |
| 15 | psutil | 7.2.2 | `psutil-7.2.2-cp37-abi3-win_amd64.whl` | 135 KB |
| 16 | PyYAML | 6.0.3 | `pyyaml-6.0.3-cp310-cp310-win_amd64.whl` | 155 KB |
| 17 | filelock | 3.25.2 | `filelock-3.25.2-py3-none-any.whl` | 26 KB |
| 18 | fsspec | 2026.3.0 | `fsspec-2026.3.0-py3-none-any.whl` | 198 KB |
| 19 | networkx | 3.4.2 | `networkx-3.4.2-py3-none-any.whl` | 1.6 MB |
| 20 | MarkupSafe | 3.0.3 | `markupsafe-3.0.3-cp310-cp310-win_amd64.whl` | 15 KB |
| 21 | setuptools | 81.0.0 | `setuptools-81.0.0-py3-none-any.whl` | 1.0 MB |

### Group 2 — Packages depending on Group 1
| # | Package | Version | File | Size |
|---|---------|---------|------|------|
| 22 | opencv-python | 4.13.0.92 | `opencv_python-4.13.0.92-cp37-abi3-win_amd64.whl` | 38 MB |
| 23 | scipy | 1.15.3 | `scipy-1.15.3-cp310-cp310-win_amd64.whl` | 39 MB |
| 24 | imageio | 2.37.3 | `imageio-2.37.3-py3-none-any.whl` | 310 KB |
| 25 | contourpy | 1.3.2 | `contourpy-1.3.2-cp310-cp310-win_amd64.whl` | 216 KB |
| 26 | fonttools | 4.62.1 | `fonttools-4.62.1-cp310-cp310-win_amd64.whl` | 1.5 MB |
| 27 | python-dateutil | 2.9.0.post0 | `python_dateutil-2.9.0.post0-py2.py3-none-any.whl` | 225 KB |
| 28 | requests | 2.33.1 | `requests-2.33.1-py3-none-any.whl` | 63 KB |
| 29 | sympy | 1.14.0 | `sympy-1.14.0-py3-none-any.whl` | 6.0 MB |
| 30 | Jinja2 | 3.1.6 | `jinja2-3.1.6-py3-none-any.whl` | 132 KB |
| 31 | shiboken6 | 6.11.0 | `shiboken6-6.11.0-cp310-abi3-win_amd64.whl` | 1.2 MB |

### Group 3 — Packages depending on Group 2
| # | Package | Version | File | Size |
|---|---------|---------|------|------|
| 32 | matplotlib | 3.10.8 | `matplotlib-3.10.8-cp310-cp310-win_amd64.whl` | 7.8 MB |
| 33 | torch | 2.11.0 | `torch-2.11.0-cp310-cp310-win_amd64.whl` | 114.5 MB |
| 34 | colour-science | 0.4.6 | `colour_science-0.4.6-py3-none-any.whl` | 2.4 MB |
| 35 | polars-runtime-32 | 1.39.3 | `polars_runtime_32-1.39.3-cp310-abi3-win_amd64.whl` | 45 MB |
| 36 | PySide6-Essentials | 6.11.0 | `pyside6_essentials-6.11.0-cp310-abi3-win_amd64.whl` | 72 MB |

### Group 4 — Packages depending on Group 3
| # | Package | Version | File | Size |
|---|---------|---------|------|------|
| 37 | torchvision | 0.26.0 | `torchvision-0.26.0-cp310-cp310-win_amd64.whl` | 3.5 MB |
| 38 | ultralytics-thop | 2.0.18 | `ultralytics_thop-2.0.18-py3-none-any.whl` | 28 KB |
| 39 | polars | 1.39.3 | `polars-1.39.3-py3-none-any.whl` | 805 KB |
| 40 | PySide6-Addons | 6.11.0 | `pyside6_addons-6.11.0-cp310-abi3-win_amd64.whl` | 161 MB |

### Group 5 — Top-level packages (install last)
| # | Package | Version | File | Size |
|---|---------|---------|------|------|
| 41 | ultralytics | 8.4.36 | `ultralytics-8.4.36-py3-none-any.whl` | 1.2 MB |
| 42 | colour-checker-detection | 0.2.1 | `colour_checker_detection-0.2.1-py3-none-any.whl` | 32.5 MB |
| 43 | PySide6 | 6.11.0 | `pyside6-6.11.0-cp310-abi3-win_amd64.whl` | 564 KB |

---

## TOTAL DOWNLOAD SIZE: ~552 MB

## INSTALLATION COMMAND (run on the workstation)

Copy all `.whl` files to a folder (e.g. `C:\wheels\`) then run:

```cmd
pip install --no-index --find-links=C:\wheels\ -r requirements.txt
```

Or install one group at a time:
```cmd
pip install --no-index --find-links=C:\wheels\ numpy Pillow typing-extensions six certifi charset-normalizer idna urllib3 packaging cycler pyparsing kiwisolver mpmath click psutil PyYAML filelock fsspec networkx MarkupSafe setuptools
pip install --no-index --find-links=C:\wheels\ opencv-python scipy imageio contourpy fonttools python-dateutil requests sympy Jinja2 shiboken6
pip install --no-index --find-links=C:\wheels\ matplotlib torch colour-science polars-runtime-32 PySide6-Essentials
pip install --no-index --find-links=C:\wheels\ torchvision ultralytics-thop polars PySide6-Addons
pip install --no-index --find-links=C:\wheels\ ultralytics colour-checker-detection PySide6
```

## ALSO NEEDED: YOLOv8 MODEL FILE (offline — cannot auto-download)

Since the workstation has no internet, the model must be copied manually:
- **File:** `colour-checker-detection-l-seg.pt` (~88 MB)
- **Already in repo at:** `models/colour-checker-detection-l-seg.pt`
- Copy it to the workstation alongside the project files

## OPTIONAL: OpenEXR Python bindings (for native .exr I/O)
- Package: `openexr`
- File: check https://pypi.org/project/openexr/#files for cp310-win_amd64 wheel
- **Note:** opencv-python already handles .exr files, so this is optional
