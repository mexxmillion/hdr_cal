# CC24 Detector — Training Plan

Goal: replace the brittle two-pass YOLO+OpenCV path with a model that outputs **4 ordered corners** (TL→TR→BR→BL) directly. No more `minAreaRect` → `_best_rotation` guessing.

Existing infra to reuse:
- ultralytics is already installed in `conda env hdri_cal`
- model files live in `W:/git/hdr_cal/models/`
- integration point: replace `_yolo_detect` ([colorchecker_erp.py:81](colorchecker_erp.py:81)) or `_detect_in_tile` ([colorchecker_erp.py:737](colorchecker_erp.py:737))
- CC24 reference colours in [colorchecker_erp.py:141](colorchecker_erp.py:141) (`CC24_LINEAR_SRGB`) — for synthetic data

---

## Phase 1 — Task & model choice (pick one, then commit)

| Option | What it predicts | Pros | Cons |
|---|---|---|---|
| **A. YOLO-pose, 4 keypoints** | bbox + 4 ordered corners | one model, ordered output, drops the rotation problem | needs keypoint annotations; existing seg datasets must be converted |
| B. YOLO-seg + corner refinement | mask → corner regression head | reuses seg datasets directly | still needs an ordering head; two-stage |
| C. Keypoint-only transformer (DETR-style) | 4 corners + class | best accuracy on perspective | overkill, slower training, smaller community |

**Recommendation: A — ultralytics `YOLOv11-pose` (n or s variant).** Same toolchain as today, smallest disruption. `model = YOLO('yolo11n-pose.pt')` → fine-tune.

Ordering convention (lock this now): **TL → TR → BR → BL with the white patch (#19) in BL corner.** That matches the existing `_order_quad_tl_tr_br_bl` and the manual picker. The model learns absolute orientation, not just shape.

---

## Phase 2 — Datasets

Roboflow Universe has multiple CC24 / Macbeth / X-Rite ColorChecker projects. Candidates to evaluate (need to verify license + quality before pulling):

1. **Search Roboflow for**: `colorchecker`, `color checker`, `macbeth`, `x-rite`, `color calibration card`.
2. **Useful tags**: `keypoints` (rare), `instance segmentation` (common), `bounding box` (most common).
3. **Filter for**: ≥ 500 images, varied lighting, real-world (not just studio), permissive license (CC-BY / public domain / MIT).

Likely outcome: 1–3 segmentation datasets, no keypoint datasets. Plan accordingly — **we'll need to derive keypoints from masks**, which means a one-time conversion script.

**Synthetic supplement (cheap, mandatory):**
- Render the CC24 reference (`CC24_LINEAR_SRGB`) onto random backgrounds with random perspective + lighting.
- 2k–5k synthetic + real Roboflow = better than either alone for keypoint ordering.
- Synthetic gives **guaranteed ordered keypoints**, real gives texture/lighting realism.

Save plan: `data/cc_train/{real,synth}/{train,val,test}/{images,labels}/`.

---

## Phase 3 — Data conversion

Two converters to write:

1. **`roboflow_seg_to_pose.py`** — for each segmentation polygon: fit minAreaRect, pick 4 corners, **order them using the chart's colour content** (find white patch position via colour search → that's BL). Saves YOLO-pose `.txt`:
   ```
   <class> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <vis1> <kpt2_x> <kpt2_y> <vis2> ...
   ```
   Manual review for ~5% of conversions to verify ordering correctness.

2. **`synth_cc24.py`** — already have prior art in [test_synthetic_chart.py](test_synthetic_chart.py). Generate:
   - random HDR backgrounds (or just sampled real EXRs from the user's library)
   - random homography (perspective)
   - exposure variation (±3 EV)
   - chart scale 5%–40% of frame
   - partial occlusion / rotation
   - YOLO-pose labels written directly from the synthesis

---

## Phase 4 — Training

```python
from ultralytics import YOLO
m = YOLO('yolo11s-pose.pt')           # s, not n — keypoints need capacity
m.train(
    data='data/cc_train/cc24.yaml',
    epochs=200,
    imgsz=1024,                        # match the GUI's rectilinear tile size
    batch=16,                          # tune to GPU VRAM
    patience=30,
    optimizer='AdamW',
    lr0=1e-3,
    augment=True,
    mosaic=0.5,                        # less aggressive than default — keypoints suffer from mosaic
    degrees=180,                       # full rotation range
    perspective=0.001,                 # significant perspective augmentation
    fliplr=0.0, flipud=0.0,            # NO flips — would break keypoint ordering
    hsv_h=0.02, hsv_s=0.5, hsv_v=0.5,  # exposure/colour augmentation
)
```

**Critical**: `fliplr=0` and `flipud=0`. Flips break the TL/TR/BR/BL ordering.

Target metrics:
- mAP@0.5 (bbox) ≥ 0.95
- Keypoint OKS@0.5 ≥ 0.90
- Mean corner error ≤ 5 px on 1024 imgsz validation

---

## Phase 5 — Integration

Drop in via a new function in [colorchecker_erp.py](colorchecker_erp.py):

```python
def _yolo_pose_detect(image_u8_rgb: np.ndarray, conf: float = 0.25):
    """Returns [(conf, quad_xyxyxyxy_px)] — quad is already ordered TL,TR,BR,BL."""
    model = _get_yolo_pose_model()         # new loader, separate from segmentation
    res = model(image_bgr, conf=conf, verbose=False, imgsz=1024)
    out = []
    for r in res:
        for box, kpts in zip(r.boxes, r.keypoints):
            quad = kpts.xy[0].cpu().numpy().astype(np.float32)  # (4, 2)
            out.append((float(box.conf), quad))
    return out
```

Then in `_detect_in_tile` (or a new `_detect_in_tile_pose`):
- Call `_yolo_pose_detect` on the tile.
- Skip `_quad_from_mask`, `_best_rotation`, and the chroma-score sanity gate.
- Call `_finalize_detection_from_quad` directly with the ordered quad.

Two-line config switch in `find_colorchecker_in_rect`: prefer pose model if present, fall back to existing path.

---

## Phase 6 — Validation against the real failure cases

Hold-out test set:
- The user's `data/input_new.png` (the screenshot that triggered this rewrite)
- 5+ varied scenes from the user's HDRI library
- Synthetic stress cases: 60°+ perspective, partial occlusion, 0.5 EV underexposure

Acceptance bar: **all 4 corners within 1 swatch-cell of ground truth on every hold-out scene.** Anything less and it's not worth replacing the manual path.

---

## Phase 7 — Decision tree

After Phase 4 training:
- mAP < 0.9 → revisit data (more synth, cleaner labels) before more epochs
- mAP good but corner order wrong on real data → the ordering heuristic in conversion script is broken; fix and retrain
- mAP good, order right, still fails on the user's scenes → integrate as the auto-detect path; manual stays as the override

---

## Open questions (answer before Phase 2)

1. **GPU available?** Where? `W:\conda\envs\hdri_cal` shares VRAM with what?
2. **License constraint** on the trained model? (Affects which Roboflow datasets are usable.)
3. **Keep the existing seg model around** as a secondary, or rip it out once pose works?
4. **HDR robustness** — train on display-mapped LDR (cheap) or on EV-jittered linear (more realistic)? Display-mapped is what YOLO sees in production anyway.
5. **Budget**: how many hours of training + iteration are we willing to spend before falling back to "manual only"?
