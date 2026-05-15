# Drive: Road, Lane, and Edge Segmentation

Drive is a multi-task road-scene perception project for **road/drivable-area segmentation**, **lane segmentation**, and **edge/boundary extraction** on a BDD100K-derived dataset. The project explores two complementary design directions:

1. **YOLO-style supervised segmentation models** for fast multi-task perception.
2. **Knowledge Distillation (KD)** using a stronger SegFormer teacher and a lightweight EfficientNet-B0 student.

The final project objective is not only high segmentation accuracy, but a practical **accuracy–speed–memory trade-off** suitable for future real-time or embedded deployment.

---

## Project Overview

Given an RGB driving image resized to `384 × 640`, the model predicts three dense outputs:

| Output      |       Shape | Meaning                                  |
| ----------- | ----------: | ---------------------------------------- |
| `road_mask` |    `[H, W]` | Drivable road / road-region segmentation |
| `lane_mask` |    `[H, W]` | Lane-marking segmentation                |
| `edge_mask` | `[1, H, W]` | Lane/road boundary or edge extraction    |

The model output tensors are:

| Output logits |          Shape | Description        |
| ------------- | -------------: | ------------------ |
| `road_logits` | `[B, 2, H, W]` | background vs road |
| `lane_logits` | `[B, 2, H, W]` | background vs lane |
| `edge_logits` | `[B, 1, H, W]` | binary edge logit  |

---

## Repository Contents

This repository contains:

```text
Drive/
├── Drive-YOLO-S+ notebook(s)       # Kaggle/IPYNB supervised architecture runs
├── Drive-YOLO-S-Nano notebook(s)   # Kaggle/IPYNB compact model runs
└── Drive-KD/                       # Full standalone KD codebase
```

The `Drive-KD/` directory is the complete standalone training pipeline containing model definitions, dataset loaders, losses, KD losses, metrics, scripts, post-processing utilities, tests, and orchestration scripts.

---

## Model Families

### 1. Drive-YOLO-S+

Drive-YOLO-S+ is the stronger supervised YOLO-style architecture.

Main architectural choices:

* **P1 high-resolution branch** for thin lane and boundary detail.
* **P2–P5 feature pyramid** for multi-scale scene understanding.
* **SPPF** for efficient large-context aggregation.
* **BiFPN-lite** for learnable multi-scale feature fusion.
* Three task heads: road, lane, and edge.

Conceptual data flow:

```text
RGB image
    ↓
YOLO-style CNN backbone
    ↓
P1, P2, P3, P4, P5 feature maps
    ↓
SPPF on deepest feature
    ↓
BiFPN-lite feature fusion
    ↓
Road head / Lane head / Edge head
```

Drive-YOLO-S+ prioritizes accuracy and spatial detail. The P1 branch improves lane and edge localization, but increases memory and compute cost.

---

### 2. Drive-YOLO-S-Nano

Drive-YOLO-S-Nano is the compact supervised variant.

Design goals:

* Reduce model size.
* Reduce latency.
* Preserve acceptable road/lane/edge segmentation quality.
* Explore deployment-oriented architecture scaling.

Compared with Drive-YOLO-S+, the Nano variant uses a reduced capacity design with fewer channels/blocks and a smaller compute footprint.

Conceptually:

```text
RGB image
    ↓
Reduced YOLO-style backbone
    ↓
Lightweight multi-scale neck
    ↓
Compact road/lane/edge heads
```

Nano is not expected to outperform S+ in raw segmentation quality. Its purpose is to explore the small-model regime.

---

### 3. Drive-KD

Drive-KD is the full knowledge-distillation pipeline.

It uses:

| Role    | Model                                       |
| ------- | ------------------------------------------- |
| Teacher | SegFormer-B1 / MiT-B1 encoder               |
| Student | EfficientNet-B0 encoder + SPPF + BiFPN-lite |

Teacher flow:

```text
RGB image
    ↓
SegFormer-B1 / MiT-B1 encoder
    ↓
SegFormer-style decoder
    ↓
Road / lane / edge heads
```

Student flow:

```text
RGB image
    ↓
EfficientNet-B0 pretrained encoder
    ↓
P1, P2, P3, P4, P5 features
    ↓
1×1 channel adapters
    ↓
SPPF on P5
    ↓
BiFPN-lite
    ↓
Road / lane / edge heads
```

Distillation signals:

* **Probability KD**: student matches teacher road/lane/edge soft probability maps.
* **Boundary KD**: student edge head learns from teacher-derived boundary maps.
* **Attention KD**: student spatial attention maps are aligned with teacher attention maps.
* **Supervised loss** remains active using ground-truth masks.

The teacher cache is frozen during student training. Gradients flow only through the student network.

---

## Dataset

The project uses a processed BDD100K-derived dataset containing RGB images and generated/processed masks.

Expected input structure for local Drive-KD training:

```text
D:/Datasets/
├── bdd100k/
│   └── images/
│       └── 100k/
│           ├── train/
│           └── val/
│
└── drive-bdd100k-processed/
    └── drive-stage1-bdd100k-processed/
        ├── manifests/
        │   ├── manifest_train_portable.csv
        │   └── manifest_val_portable.csv
        ├── road_masks/
        ├── lane_masks/
        └── edge_masks/
```

The local config file is:

```text
Drive-KD/configs/dataset/bdd100k_local.yaml
```

Update the dataset paths in that file before running local training.

---

## Loss Functions

The supervised loss is task-specific:

```text
L_supervised = λ_road L_road + λ_lane L_lane + λ_edge L_edge
```

### Road Loss

```text
L_road = CrossEntropy + Dice
```

Reasoning:

* Cross-Entropy gives pixelwise classification pressure.
* Dice encourages region-level overlap.
* Road is a large dense region, so IoU/Dice behavior matters.

### Lane Loss

```text
L_lane = Focal Cross-Entropy + Dice
```

Reasoning:

* Lane pixels are sparse.
* Focal CE downweights easy background pixels.
* Dice helps with thin-structure overlap.

### Edge Loss

```text
L_edge = BCEWithLogits + Dice
```

Reasoning:

* Edge prediction is binary.
* Edge pixels are sparse.
* BCE gives pixelwise binary supervision.
* Dice improves sparse foreground overlap.

### KD Loss

```text
L_total = L_supervised
        + λ_prob L_probability_KD
        + λ_boundary L_boundary_KD
        + λ_attention L_attention_KD
```

Probability KD teaches the student soft teacher probabilities. Boundary KD improves localization of edge-like structures. Attention KD encourages the student to look at the same spatial regions as the teacher.

---

## Gradient Flow Summary

During supervised training:

```text
Losses
  ↓
Road / lane / edge heads
  ↓
Neck / feature fusion
  ↓
Backbone
```

During KD training:

```text
Ground truth masks + teacher cache
        ↓
KD and supervised losses
        ↓
Student heads
        ↓
Student BiFPN-lite
        ↓
Student EfficientNet-B0 encoder
```

The teacher is not updated during KD student training. Teacher outputs are precomputed and loaded from cache.

---

## Metrics

Different outputs require different evaluation metrics.

| Task              | Primary Metric         | Reason                                          |
| ----------------- | ---------------------- | ----------------------------------------------- |
| Road segmentation | Road IoU               | Best for large-region overlap                   |
| Lane segmentation | Lane F1                | Handles sparse lane pixels better than accuracy |
| Edge extraction   | Edge F1                | Better for thin sparse boundary structures      |
| Deployment        | FPS / latency / params | Required for practical model selection          |

Additional metrics include precision, recall, Dice, IoU, pixel accuracy, specificity, parameter count, model size, and FPS.

---

## Metrics Achieved

| Model             | Road IoU ↑ | Lane F1 ↑ | Edge F1 ↑ | Notes                                                          |
| ----------------- | ---------: | --------: | --------: | -------------------------------------------------------------- |
| Drive-YOLO-S+     |   `0.8279` |  `0.7470` |  `0.5841` | Strong supervised model; P100 run reached timeout              |
| Drive-YOLO-S-Nano |   `0.7906` |  `0.6993` |  `0.5417` | Compact variant; Dual T4 run suffered dying-kernel instability |
| Drive-KD          |   `0.8703` |  `0.7877` |  `0.6180` | Best overall score across all three tasks                      |

Drive-KD achieved the strongest overall validation metrics, especially on lane F1 and edge F1, while retaining a lightweight student design.

---

## Final Experiment Notes

### Drive-YOLO-S+

The final S+ Kaggle run on P100 continued until timeout. This suggests the architecture was trainable, but full training exceeded Kaggle runtime limits due to the heavier P1 + SPPF + BiFPN-lite design.

### Drive-YOLO-S-Nano

The Nano run on Dual T4 suffered dying-kernel issues after a few initial epochs. This is likely a notebook/runtime instability issue involving resource pressure rather than a pure model-theory failure.

### Final Direction

The KD design was chosen as the best final direction because it combines:

* Strong teacher guidance.
* Lightweight student inference.
* Improved road/lane/edge metrics.
* A cleaner deployment-oriented architecture.

---

## Drive-KD Setup

### Recommended Python Version

Use:

```text
Python 3.11.9 64-bit
```

### Environment Setup on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

Install PyTorch CUDA wheel according to your GPU/driver. Recommended first attempt:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install repo dependencies:

```powershell
pip install -r requirements.txt
pip install -e .
```

Verify installation:

```powershell
python -c "import drive_kd; print(drive_kd.__version__); print('drive_kd import ok')"
```

---

## Drive-KD Execution

### 1. Run Lightweight Tests

```powershell
pytest -q tests/test_imports.py tests/test_student_forward.py tests/test_losses.py tests/test_kd_losses.py tests/test_metrics.py
```

### 2. Offline Smoke Run

This checks code flow without downloading pretrained weights:

```powershell
python run_smoke.py `
  --offline-model-init `
  --train-samples 8 `
  --val-samples 4 `
  --batch-size 1
```

### 3. Pretrained Smoke Run

This checks pretrained SegFormer/EfficientNet initialization:

```powershell
python run_smoke.py `
  --train-samples 16 `
  --val-samples 4 `
  --batch-size 1
```

### 4. Full Pipeline

```powershell
python run_project.py `
  --steps all
```

### 5. Resume From Later Stages

After teacher training:

```powershell
python run_project.py `
  --steps cache,student,evaluate,benchmark,export
```

Only evaluate/export:

```powershell
python run_project.py `
  --steps evaluate,benchmark,export
```

---

## Manual Script Order

```powershell
python scripts/00_build_absolute_manifests.py `
  --config configs/dataset/bdd100k_local.yaml

python scripts/01_check_dataset.py `
  --config configs/dataset/bdd100k_local.yaml

python scripts/02_train_teacher_segformer_b1.py `
  --config configs/experiments/exp_000_teacher_segformer_b1.yaml

python scripts/03_generate_teacher_cache.py `
  --config configs/experiments/exp_001_teacher_cache.yaml `
  --checkpoint outputs/teacher_segformer_b1/checkpoints/best.pt

python scripts/04_train_student_effb0_kd.py `
  --config configs/experiments/exp_002_student_effb0_full_kd.yaml

python scripts/05_evaluate_student.py `
  --config configs/experiments/exp_002_student_effb0_full_kd.yaml `
  --checkpoint outputs/student_effb0_kd/checkpoints/best.pt

python scripts/06_benchmark_fps.py `
  --config configs/experiments/exp_002_student_effb0_full_kd.yaml `
  --checkpoint outputs/student_effb0_kd/checkpoints/best.pt `
  --batch-size 1 `
  --image-height 384 `
  --image-width 640

python scripts/07_export_student.py `
  --config configs/experiments/exp_002_student_effb0_full_kd.yaml `
  --checkpoint outputs/student_effb0_kd/checkpoints/best.pt `
  --format all
```

---

## Expected Drive-KD Outputs

```text
outputs/teacher_segformer_b1/checkpoints/best.pt
outputs/student_effb0_kd/checkpoints/best.pt
data/cache/teacher_segformer_b1/
data/manifests/train_kd.csv
data/manifests/val_kd.csv
outputs/evaluation/evaluation_summary.json
outputs/evaluation/fps_report.json
outputs/exports/drive_effb0_bifpn_kd.onnx
outputs/exports/drive_effb0_bifpn_kd.torchscript.pt
```

The final Drive-KD model provides the best result among the tested versions and establishes a practical direction for deployable road-scene perception.
