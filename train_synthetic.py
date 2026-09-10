#!/usr/bin/env python3
"""
========================================================================================
train_synthetic.py — End-to-End Synthetic Pipeline Champion Training Script
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
OVERVIEW & OBJECTIVE (FROM BASIC TO ADVANCED)
----------------------------------------------------------------------------------------
[Basic Concept]:
  In traditional supervised computer vision, training deep neural networks requires
  thousands of manually annotated real-world images, which is expensive and time-consuming.
  In this project, we train a classifier exclusively on SYNTHETIC 3D scenes (generated
  using BlenderProc and CAD models) and test its zero-shot transfer capability on REAL retail
  product crops captured in supermarkets under complex lighting, clutter, and occlusions.

[The Sim-to-Real Domain Gap]:
  When models are trained on pristine synthetic CAD renderings, they often suffer a severe
  drop in performance when evaluated on real-world crops ("Sim-to-Real Gap").
  Why does this happen?
  1. Texture & Lighting Differences: CAD renders have clean synthetic shaders, whereas real
     store images have packaging cellophane specular reflections, motion blur, and ambient gradients.
  2. Depth Distortion: Real retail depth maps (predicted via monocular depth estimation models
     like Depth Anything V2) contain estimation noise, edge bleeding, and curvature softening,
     while synthetic CAD depth is mathematically exact.
  3. Feature Distribution Shift: Specifically, planarity features (e.g., 'planar_surface_ratio',
     'curved_surface_ratio', and 'std_Nz') behave radically differently between synthetic CAD
     facets and real retail packaging.

[The Solution: Generalized Champion Pipeline C]:
  This script implements the four breakthrough pillars discovered during extensive experimentation:
  1. FEATURE MASKING:
     We identify and prune features with high Jensen-Shannon divergence or distribution shift
     between CAD renders and real items:
     ['planar_surface_ratio', 'curved_surface_ratio', 'std_Nz'].
     Dropping these 3 features prevents the decision trees from overfitting to synthetic artifacts.

  2. MODALITY RATIO BALANCING (48-D RGB + 16-D GEOMETRY):
     Standard concatenated features (e.g., 768-D RGB + 35-D Geometry) cause "Feature Drowning",
     where tree splits almost exclusively pick RGB features (768 opportunities vs 35).
     By projecting RGB to 48-D PCA and Geometry to 16-D PCA (a 3:1 ratio), we provide sufficient
     visual capacity for DINOv2 while preserving significant geometric influence.

  3. CALIBRATED SOFT-VOTING ENSEMBLE:
     Combines three diverse, complementary tree-based learners:
     - ExtraTreesClassifier (350 estimators, max_features='sqrt')
     - HistGradientBoostingClassifier (250 iterations, L2 reg=1.0)
     - RandomForestClassifier (300 estimators, max_features='sqrt')
     All calibrated with 5-fold Isotonic Probability Calibration (`CalibratedClassifierCV`).

  4. PRIOR SENSITIVITY RE-WEIGHTING:
     Synthetic object frequency differs from real shelf product distribution.
     We apply empirical prior re-weighting: W = [1.8, 1.0, 1.4, 0.45]
     corresponding to [0: Flat, 1: Cylindrical, 2: Cuboid, 3: Irregular].
     This boosts Flat and Cuboid recall while penalizing false positive pouch (Irregular) alarms.

[Sim-to-Real Benchmark Goal]:
  Transfers from ~7,125 synthetic training crops to 815 zero-shot real retail crops,
  achieving 86.87% Balanced Accuracy and 101.06% Sim-to-Real Transfer Ratio (surpassing
  even real in-domain baseline models!).
========================================================================================
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report

# --------------------------------------------------------------------------------------
# PATH RESOLUTION & ENVIRONMENT SETUP
# --------------------------------------------------------------------------------------
# Resolve the repository root dynamically so imports from `src` work cleanly whether run
# from the repository root, from the scripts directory, or imported as an external tool.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import the core champion architecture and synthetic dataset taxonomy
from src.generalized_champion import (
    GeneralizedChampionPipelineC,
    CLASS_NAMES,
    DROP_FEATURES,
    CLASS_WEIGHTS,
)
from src.synthetic_loader import SYNTH_TO_REAL_MAP, SyntheticPrimitiveDataset

# Reference baseline: In-domain real-trained model achieved 85.96% Balanced Accuracy on the 815 set.
# Transfer Ratio = (Synthetic Zero-Shot Real BAcc / Real Baseline BAcc) * 100%
REAL_BASELINE_REF = 85.96


def train_synthetic_champion(
    X_train_rgb: np.ndarray,
    X_train_geo: np.ndarray,
    y_train: np.ndarray,
    X_eval_rgb: np.ndarray | None = None,
    X_eval_geo: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    output_path: str = "synthetic_model.pkl"
) -> dict:
    """
    Trains the Generalized Champion Sim-to-Real model on synthetic data and saves the artifact.

    Mathematical Workflow:
      1. Feature Masking:
         Extracts geometric feature columns and removes indices corresponding to DROP_FEATURES
         ('planar_surface_ratio', 'curved_surface_ratio', 'std_Nz').
      2. Dual-Branch Standardization & PCA:
         - Visual:   X_rgb -> StandardScaler -> PCA(48-D) -> F_rgb
         - Geometry: X_geo -> RobustScaler  -> PCA(16-D) -> F_geo
         - Concatenation: F_fused = [F_rgb, F_geo] in R^64
      3. Regularized Soft-Voting Ensemble:
         Fits ExtraTrees + HistGradientBoosting + RandomForest with class balancing.
      4. 5-Fold Isotonic Probability Calibration:
         Calibrates raw ensemble probabilities to reflect true posterior confidences.
      5. Optional Zero-Shot Real Evaluation:
         If real validation/test matrices are supplied, evaluates out-of-domain transfer
         against the canonical 815 real retail test crop benchmark.
      6. Checkpoint Serialization:
         Dumps the complete bundle to `output_path` (default: 'synthetic_model.pkl').

    Args:
        X_train_rgb: Precomputed synthetic RGB feature matrix (N_synth, D_rgb).
        X_train_geo: Precomputed synthetic 35-D geometric feature matrix (N_synth, 35).
        y_train:     Ground truth synthetic labels mapped to the Real 4-class space [0..3].
        X_eval_rgb:  Optional real test RGB features for zero-shot transfer evaluation.
        X_eval_geo:  Optional real test Geometry features for zero-shot transfer evaluation.
        y_eval:      Optional real test ground truth labels in [0..3].
        output_path: Target pickle filepath for the trained model bundle.

    Returns:
        bundle: Dictionary containing model metadata, transfer metrics, and fitted pipeline.
    """
    print("=" * 75)
    print("  TRAINING SYNTHETIC GENERALIZED CHAMPION (FEATURE-MASKED + CALIBRATED)")
    print("=" * 75)
    print(f"Synthetic training samples: {len(y_train):5d} (RGB: {X_train_rgb.shape[1]}-D, Geo: {X_train_geo.shape[1]}-D)")
    print(f"Dropped shifted features  : {DROP_FEATURES}")
    print(f"PCA Projections           : 48-D RGB + 16-D Geometry (64-D Balanced Fused Vector)")
    print(f"Class prior re-weighting  : {CLASS_WEIGHTS}\n")

    # Instantiate the Generalized Champion model architecture
    t0 = time.time()
    model = GeneralizedChampionPipelineC(
        n_rgb_comp=48,
        n_geo_comp=16,
        drop_features=DROP_FEATURES,
        class_weights=CLASS_WEIGHTS,
        seed=42
    )

    # Fit the dual-branch preprocessing, ensemble, and calibration
    model.fit(X_train_rgb, X_train_geo, y_train)
    fit_duration = time.time() - t0
    print(f"[OK] Synthetic champion fitted in {fit_duration:.2f}s\n")

    # Evaluate zero-shot transfer on real retail crops if evaluation set is provided
    bacc_real = 0.0
    tr_ratio = 0.0
    if X_eval_rgb is not None and X_eval_geo is not None and y_eval is not None:
        print("=" * 75)
        print(f"  EVALUATING ZERO-SHOT SIM-TO-REAL ON {len(y_eval)} REAL CROPS")
        print("=" * 75)
        preds_real = model.predict(X_eval_rgb, X_eval_geo)
        bacc_real = float(balanced_accuracy_score(y_eval, preds_real))
        acc_real = float(accuracy_score(y_eval, preds_real))
        tr_ratio = float((bacc_real * 100.0 / REAL_BASELINE_REF) * 100.0)

        print(f"  * Real Balanced Accuracy : {bacc_real * 100:.2f}%")
        print(f"  * Real Overall Accuracy  : {acc_real * 100:.2f}%")
        print(f"  * Sim-to-Real Transfer   : {tr_ratio:.2f}% (Ref: {REAL_BASELINE_REF:.2f}%)\n")
        print("Classification Report:")
        print(classification_report(y_eval, preds_real, target_names=CLASS_NAMES, digits=4))

    # Package model bundle with comprehensive reproducibility metadata
    bundle = {
        "model": model,
        "pipeline": model,
        "class_weights": CLASS_WEIGHTS,
        "drop_features": DROP_FEATURES,
        "n_rgb_comp": 48,
        "n_geo_comp": 16,
        "class_names": CLASS_NAMES,
        "bacc_815": round(bacc_real * 100.0, 2),
        "transfer_ratio": round(tr_ratio, 2),
        "approach_name": "Pipeline C Synthetic Generalized Champion (Feature-Masked + Calibrated Prior)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save model bundle to disk
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SAVED CHECKPOINT] -> {output_path} ({file_size_mb:.1f} MB)")
    return bundle


def main():
    """
    Main CLI entrypoint for training the synthetic champion model.
    Parses CLI arguments, locates precomputed feature caches, converts label taxonomies,
    and invokes the training routine.
    """
    parser = argparse.ArgumentParser(
        description="Train Synthetic Generalized Champion Classifier (Zero-Shot Sim-to-Real)."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Synthetic feature cache directory containing .npz feature archives"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="synthetic_model.pkl",
        help="Destination pickle file path for trained champion model"
    )
    args = parser.parse_args()

    # Search for synthetic feature caches across standard project locations
    candidate_cache_dirs = [
        Path(args.cache_dir),
        SCRIPT_DIR / "cache",
        SCRIPT_DIR.parent / "synthetic_pipeline" / "results" / "cache"
    ]
    cache_dir = next((d for d in candidate_cache_dirs if d.exists()), None)

    if cache_dir is None:
        raise FileNotFoundError(
            f"Synthetic cache directory not found. Checked:\n"
            + "\n".join([f"  - {d}" for d in candidate_cache_dirs])
        )

    # Required cache files: batch generated + shelf full scenes
    f_batch = cache_dir / "syn_train_batch_250_dav2_clean_seed42.npz"
    f_shelf = cache_dir / "syn_train_shelf_full_dav2_clean_seed42.npz"

    if not (f_batch.exists() and f_shelf.exists()):
        raise FileNotFoundError(
            f"Missing required synthetic cache files in '{cache_dir}'.\n"
            f"Expected: {f_batch.name} and {f_shelf.name}"
        )

    print(f"[INFO] Loading synthetic caches from {cache_dir} ...")
    d_b = np.load(f_batch)
    d_s = np.load(f_shelf)

    # Stack batches to form the consolidated synthetic training dataset
    X_train_rgb = np.vstack([d_b["X_rgb"], d_s["X_rgb"]])
    X_train_geo = np.vstack([d_b["X_geo"], d_s["X_geo"]])
    y_train_syn = np.concatenate([d_b["y_labels"], d_s["y_labels"]])

    # Convert synthetic taxonomy indices to real standard taxonomy indices:
    # Synthetic generator: 0: cuboid, 1: cylindrical, 2: flat, 3: irregular
    # Real classifier:    0: flat,   1: cylindrical, 2: cuboid, 3: irregular
    y_train = np.array([SYNTH_TO_REAL_MAP[y] for y in y_train_syn], dtype=np.int32)

    # Check for the 815 real evaluation set to measure zero-shot Sim-to-Real transfer
    X_eval_rgb, X_eval_geo, y_eval = None, None, None
    real_cache = SCRIPT_DIR.parent / "retail_geometry_project" / "outputs" / "cache_v2"
    if (real_cache / "train_features_v2.npz").exists() and (real_cache / "test_features_v2.npz").exists():
        d_tr = np.load(real_cache / "train_features_v2.npz")
        d_val = np.load(real_cache / "val_features_v2.npz")
        d_ts = np.load(real_cache / "test_features_v2.npz")

        # The canonical 815 pristine real evaluation benchmark comprises:
        # 1 pristine crop per training product group + all pristine val crops + all pristine test crops
        unique_groups, first_idx = np.unique(d_tr["group_keys"], return_index=True)
        X_eval_rgb = np.vstack([d_tr["X_rgb"][first_idx], d_val["X_rgb"], d_ts["X_rgb"]])
        X_eval_geo = np.vstack([d_tr["X_geo"][first_idx], d_val["X_geo"], d_ts["X_geo"]])
        y_eval = np.concatenate([d_tr["y_labels"][first_idx], d_val["y_labels"], d_ts["y_labels"]]).astype(np.int32)

    # Execute training and export the model bundle
    train_synthetic_champion(
        X_train_rgb, X_train_geo, y_train,
        X_eval_rgb=X_eval_rgb, X_eval_geo=X_eval_geo, y_eval=y_eval,
        output_path=args.output_model
    )


if __name__ == "__main__":
    main()
