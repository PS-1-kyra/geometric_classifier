# Project PS-1: Geometric Primitive Classifier & Sim-to-Real Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Lead Engineer:** Pranaya Shrestha  
**System Role:** Stage 2 Multimodal Geometric Primitive Classifier & Sim-to-Real Generalization  
**Project:** PS-1 — Class-Agnostic Object Detection via Geometric Primitive Analysis and Domain-Randomized Synthetic Data  

---

## 1. Executive Summary: The 2-Stage Class-Agnostic Paradigm

In conventional automated retail and robotic inventory inspection, detectors are trained to identify fine-grained brand SKUs (*"Coca-Cola 330ml Can"*, *"Lays Barbecue 50g"*). This conventional paradigm suffers from critical limitations:
1. **Extreme SKU Proliferation:** Tens of thousands of dynamic retail products require continuous, unsustainable retraining.
2. **Texture & Artwork Fragility:** Standard RGB detectors overfit to printed packaging graphics and collapse under lighting shifts or surface texture noise.
3. **Severe Data Scarcity:** Hand-annotating thousands of physical product boxes is cost-prohibitive.

**The PS-1 Architecture decomposes perception into two distinct class-agnostic stages:**
- **Stage 1 (Object Proposal Network):** Class-agnostic foreground localization (YOLOv8s-World / SAHI) predicting *"any retail object"* bounding boxes.
- **Stage 2 (Geometric Primitive Classifier - This Repository):** A multimodal shape classifier predicting the fundamental **3D Physical Primitive** (`0_flat`, `1_cylindrical`, `2_cuboid`, `3_irregular`) by fusing self-supervised visual tokens (DINOv2) with differential metric depth calculus (Depth Anything V2).

---

## 2. Multimodal Fusion Architecture: Pipeline C

To eliminate the **Feature Drowning Problem** (where 768-D DINOv2 visual features drown out 35-D geometric features during random decision tree subsampling), Pipeline C enforces mathematical parity via an **Equalized 64-D Latent Projection**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Input Bounding Box Crop (15% Padded)                     │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
            [RGB Image Stream]              [Monocular Metric Depth]
                       │                               │
                       ▼                               ▼
               DINOv2 ViT-S/14                Depth Anything V2
               (CLS + Mean Patch)             (3x3 Mask Erosion)
                       │                               │
                   [ 768-D ]                       [ 35-D ]
            Visual Feature Vector          Geometric Normal Calculus
                       │                               │
                       ▼                               ▼
                 StandardScaler                   RobustScaler
                       │                               │
                 PCA Projection                  PCA Projection
                   (32 Comp)                       (32 Comp)
                       │                               │
                   [ 32-D ]                        [ 32-D ]
                 Visual Latent                  Geometric Latent
                       │                               │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       Equalized 64-D Multimodal Latent
                                  [ 64-D ]
                                       │
                                       ▼
                    Isotonically Calibrated Soft-Voting Ensemble
                    ├── ExtraTreesClassifier (400 estimators)
                    ├── HistGradientBoosting (300 estimators)
                    └── RandomForestClassifier (350 estimators)
                                       │
                                       ▼
                     Physical Solidity & Irregular Guardrails
                                       │
                                       ▼
                     Calibrated Posterior Probabilities:
                  [P(Flat), P(Cylindrical), P(Cuboid), P(Irregular)]
```

---

## 3. Label Taxonomy & Real $\leftrightarrow$ Synthetic Index Mapping

The system categorizes retail objects into 4 physical 3D primitive categories:

| Class Index (Real) | Synthetic Index | Class Name | Physical Characteristics & Examples |
| :---: | :---: | :---: | :--- |
| **0** | **2** | **0_flat** | Negligible depth variance ($\sigma_Z \approx 0$), high $N_z \approx 1.0$. Sachets, blister cards, chocolate packs. |
| **1** | **1** | **1_cylindrical** | Horizontal normal curvature ($\partial Z/\partial x \gg 0$), zero vertical curvature ($\partial Z/\partial y \approx 0$). Cans, bottles, jars. |
| **2** | **0** | **2_cuboid** | Orthogonal planar facets, step-edge depth discontinuities, high solidity. Cereal boxes, tea cartons, medicine boxes. |
| **3** | **3** | **3_irregular** | Multi-modal depth distribution, asymmetric vertical taper ($W_1/W_3 \neq 1.0$). Chip bags, refill pouches, spray bottles. |

### The Universal Bijective Mapping:
```python
REAL_TO_SYNTH_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
SYNTH_TO_REAL_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
```

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA-capable GPU (recommended, CPU inference supported)

### Setup
```bash
git clone https://github.com/your-username/ps1-primitive-geometry-classifier.git
cd ps1-primitive-geometry-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Download Pretrained Models
Place the trained checkpoints into the repository root:
- `real_model.pkl` (Champion Real-Trained Pipeline C Checkpoint)
- `synthetic_model.pkl` (Champion Generalized Synthetic Checkpoint)

*(If using Git LFS, run `git lfs pull` to download binary weights).*

---

## 5. Quickstart Inference: Stage 1 $\to$ Stage 2 Integration

Here is the minimal snippet connecting your Stage 1 detector (YOLO / Mask R-CNN proposals) to the Stage 2 Geometric Classifier:

```python
from PIL import Image
from real_classifier import RealClassifier

# 1. Initialize classifier once
classifier = RealClassifier(model_path="real_model.pkl")

# 2. Feed crop from Stage 1 proposal
pil_crop = Image.open("sample_crop.png")
result = classifier.predict(pil_crop)

# 3. Output prediction & calibrated confidence
print(f"Predicted Class : {result['class']} (Index: {result['class_idx']})")
print(f"Confidence      : {result['confidence'] * 100:.2f}%")
print(f"Probabilities   : {result['probabilities']}")
print(f"Uncertain?      : {result['uncertain']}")
```

### Synthetic Model Inference (Sim-to-Real Benchmark):
```python
from synthetic_classifier import SyntheticClassifier

classifier = SyntheticClassifier(model_path="synthetic_model.pkl")
result = classifier.predict(pil_crop)
print(result)
```

---

## 6. End-to-End Retraining Instructions

### 6.1 Real Pipeline Retraining
1. **Prepare Raw Crops:** Organize original product crops into:
   ```text
   dataset/
   ├── 0_flat/
   ├── 1_cylindrical/
   ├── 2_cuboid/
   └── 3_irregular/
   ```
2. **Generate Zero-Leakage Group Splits:**
   ```bash
   python scripts/generate_splits.py --dataset-dir dataset --splits-dir splits
   ```
3. **Run Target-Balanced Augmentations (10x expansion on Train groups only):**
   ```bash
   python scripts/augment_crops.py --dataset-dir dataset --target-cap 1000 --train-groups splits/train_groups.csv
   ```
4. **Train Champion Model:**
   ```bash
   python train_real.py --cache-dir outputs/cache_v2 --output-model real_model.pkl
   ```

---

### 6.2 Synthetic Pipeline Retraining
1. **Prepare BlenderProc Scenes:** Place synthetic scene renders and COCO annotations in `dataset_synthetic/`.
2. **Train Synthetic Champion:**
   ```bash
   python train_synthetic.py --cache-dir cache --output-model synthetic_model.pkl
   ```

---

## 7. Repository Structure

```text
├── .gitignore                   # Git hygiene configuration
├── requirements.txt             # Pinned project dependencies
├── README.md                    # Master technical documentation
├── Nimesh_guide.md              # Stage 1 integration specification
│
├── real_classifier.py           # Production inference wrapper for Real Champion
├── synthetic_classifier.py      # Production inference wrapper for Synthetic Champion
├── real_model.pkl               # Real-Trained Model Checkpoint (199 MB)
├── synthetic_model.pkl          # Synthetic-Trained Model Checkpoint (142 MB)
│
├── train_real.py                # End-to-end training script for Real Pipeline C
├── train_synthetic.py           # End-to-end training script for Synthetic Champion
│
├── scripts/                     # Preprocessing & Data Engineering Tools
│   ├── __init__.py
│   ├── harvest_crops.py         # Bounding box cropper with 15% contextual padding
│   ├── augment_crops.py         # 16-compound physical/photometric crop augmenter
│   └── generate_splits.py       # Zero-leakage GroupShuffleSplit partition generator
│
└── src/                         # Unified Foundational Engine
    ├── __init__.py              # Package initialization & re-exports
    ├── feature_extractor_v2.py  # 35-D Monocular Depth & Surface Normal Calculus
    ├── feature_extractor.py     # Backward-compatible feature extractor interface
    ├── fusion_engine.py         # Dual-branch 64-D Latent PCA Projection Engine
    ├── pipeline_c.py            # Calibrated Soft-Voting Ensemble Classifier
    ├── generalized_champion.py  # Feature-masked generalized synthetic pipeline
    ├── dataset_engine.py        # Group-aware dataset loader & split generator
    └── synthetic_loader.py      # BlenderProc dataset loader & class bijection
```

---

## 8. Empirical Benchmark Highlights

- **Real-Trained Champion (Pipeline C):**
  - **85.96% – 87.63% Balanced Accuracy** on 815 unseen real physical products (Zero Group Leakage).
  - Per-class recall: Flat = **90.0%**, Cylindrical = **96.7%**, Cuboid = **67.9%**, Irregular = **96.0%**.
- **Synthetic Zero-Shot Sim-to-Real Transfer:**
  - **53.84% Zero-Shot Real Balanced Accuracy** (trained on 100% procedural BlenderProc scenes with 0 real samples).
  - **62.63% Sim-to-Real Transfer Ratio** relative to the real reference baseline.
- **Environmental Robustness (6,035 Perturbed Crops):**
  - While pure RGB detectors suffer a catastrophic **-53.9 pp drop** under texture noise, the 35-D Surface Normal Geometry Engine drops by only **-3.3 pp**, proving surface differential geometry is texture-invariant.

---

## 9. Citation & Contact

If you use this codebase or the 35-D Geometric Primitive Analysis framework in your research, please cite:
```bibtex
@article{shrestha2026ps1,
  title={Class-Agnostic Object Detection via Geometric Primitive Analysis and Domain-Randomized Synthetic Data},
  author={Shrestha, Pranaya and Kayastha, Swoham and Adhikari, Swarnim and Khatiwada, Nimesh},
  journal={Project PS-1 Technical Report},
  year={2026}
}
```
