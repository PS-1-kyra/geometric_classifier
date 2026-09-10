#!/usr/bin/env python3
"""
=============================================================
train_real.py — End-to-End Real Pipeline C Training Script
=============================================================
Engineer: Pranaya Shrestha
Project:  PS-1 Class-Agnostic Primitive Analysis

This script trains the Real Champion Multimodal Primitive Classifier
from pre-extracted feature caches (or raw images if no cache exists).

Training Pipeline:
  1. RGB Branch: Pretrained DINOv2 ViT-S/14 (768-D) -> StandardScaler -> PCA(32-D).
     DINOv2 provides powerful visual representations learned via self-supervised
     distillation on 142M images. The 768-D features capture both global
     semantics (CLS token) and local texture patterns (mean patch tokens).

  2. Geometry Branch: Depth Anything V2 + 35-D Geometry -> RobustScaler -> PCA(32-D).
     Monocular depth estimation followed by surface normal computation and
     statistical feature extraction. RobustScaler handles outliers from
     noisy depth predictions.

  3. Fused Representation: [F_RGB_32, F_Geo_32] in R^64.
     CRITICAL DESIGN: Both modalities are projected to the SAME dimensionality
     (32-D each). This "equalized projection" ensures that tree-based classifiers
     give equal attention to visual and geometric features. Without this,
     the 768-D DINOv2 features would "drown" the 35-D geometry features
     (trees would almost never split on geometry).

  4. Ensemble: ExtraTrees + HistGradientBoosting + RandomForest
     wrapped in 5-fold CalibratedClassifierCV(method='isotonic').
     Three diverse base learners with complementary inductive biases,
     combined via soft-voting (probability averaging).

  5. Evaluates on 100% PRISTINE unseen product test groups (Zero Data Leakage).
     Test crops come from physical products NEVER seen during training,
     not even in augmented form. This is the gold standard for evaluation.

  6. Saves model bundle directly to real_model.pkl.
     The bundle contains the fitted FusionApproachA_EqualizedProjection
     object, class names, feature names, and evaluation metrics.
=============================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import time      # Timing model training duration
import pickle    # Serialization for saving model checkpoints
import argparse  # CLI argument parsing
from pathlib import Path
import numpy as np   # Numerical arrays for feature matrices
import pandas as pd  # DataFrame operations (not heavily used here)
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix

# ── Path Setup ──────────────────────────────────────────────────────────
# Ensure the repository root is on sys.path so `from src.xxx` imports work
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── Internal Imports ────────────────────────────────────────────────────
from src.dataset_engine import CLASS_NAMES, NUM_CLASSES, load_partitioned_datasets
from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2
from src.fusion_engine import FusionApproachA_EqualizedProjection, evaluate_fusion_model
from src.pipeline_c import PipelineC


def train_from_features(X_rgb_tr, X_geo_tr, y_tr, X_rgb_val, X_geo_val, y_val, X_rgb_te, X_geo_te, y_te, output_path="real_model.pkl"):
    """
    Fits FusionApproachA_EqualizedProjection on pre-extracted features
    and evaluates on validation and test sets.

    This function is the core training logic, separated from data loading
    so it can be called with either cached features or freshly extracted ones.

    Args:
        X_rgb_tr (np.ndarray): Training DINOv2 features, shape (N_train, 768).
        X_geo_tr (np.ndarray): Training geometry features, shape (N_train, 35).
        y_tr (np.ndarray): Training labels, shape (N_train,), values in [0,3].
        X_rgb_val (np.ndarray): Validation DINOv2 features.
        X_geo_val (np.ndarray): Validation geometry features.
        y_val (np.ndarray): Validation labels.
        X_rgb_te (np.ndarray): Test DINOv2 features.
        X_geo_te (np.ndarray): Test geometry features.
        y_te (np.ndarray): Test labels.
        output_path (str): Destination path for the saved .pkl checkpoint.

    Returns:
        dict: Model bundle containing the fitted model, metrics, and metadata.
    """
    print("=" * 75)
    print("  TRAINING REAL CHAMPION: EQUALIZED 64-D PROJECTION PIPELINE C")
    print("=" * 75)
    print(f"Train samples : {len(y_tr):4d} (RGB: {X_rgb_tr.shape[1]}-D, Geo: {X_geo_tr.shape[1]}-D)")
    print(f"Val samples   : {len(y_val):4d} (100% pristine unseen product groups)")
    print(f"Test samples  : {len(y_te):4d} (100% pristine unseen product groups)\n")

    # ── Model Training ──────────────────────────────────────────────────
    # FusionApproachA_EqualizedProjection.fit() internally:
    #   1. Fits StandardScaler on X_rgb_tr, projects via PCA to 32-D
    #   2. Fits RobustScaler on X_geo_tr, projects via PCA to 32-D
    #   3. Concatenates to 64-D fused vector
    #   4. Trains base ensemble (ET + HGB + RF)
    #   5. Wraps in CalibratedClassifierCV for isotonic calibration
    t0 = time.time()
    model = FusionApproachA_EqualizedProjection(n_rgb_comp=32, n_geo_comp=32, seed=42)
    model.fit(X_rgb_tr, X_geo_tr, y_tr)
    fit_duration = time.time() - t0
    print(f"[OK] Model successfully fitted in {fit_duration:.2f}s\n")

    # ── Validation Evaluation ───────────────────────────────────────────
    # Balanced accuracy is used (not regular accuracy) because class
    # distributions may be imbalanced. Balanced accuracy averages
    # per-class recall, giving equal weight to each class.
    val_preds = model.predict(X_rgb_val, X_geo_val)
    val_bacc = balanced_accuracy_score(y_val, val_preds)
    print(f"  * Validation Balanced Accuracy : {val_bacc * 100:.2f}%")

    # ── Test Evaluation ─────────────────────────────────────────────────
    # This is the FINAL evaluation — computed ONCE on pristine test crops
    # from physical products NEVER seen during training.
    test_preds = model.predict(X_rgb_te, X_geo_te)
    test_bacc = balanced_accuracy_score(y_te, test_preds)
    test_acc = accuracy_score(y_te, test_preds)

    print("\n" + "=" * 75)
    print("  PRISTINE TEST SET EVALUATION (ZERO GROUP LEAKAGE)")
    print("=" * 75)
    print(f"  * Test Balanced Accuracy : {test_bacc * 100:.2f}%")
    print(f"  * Test Overall Accuracy  : {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    # classification_report shows per-class precision, recall, F1-score
    print(classification_report(y_te, test_preds, target_names=CLASS_NAMES, digits=4))

    # ── Save Model Bundle ───────────────────────────────────────────────
    # The bundle dict contains everything needed to reconstruct and use
    # the model without access to the original training data.
    bundle = {
        "model": model,                                         # Fitted FusionApproachA_EqualizedProjection
        "champion_name": "Approach A: Equalized 64-D Projection (PCA 32-D + PCA 32-D)",
        "balanced_accuracy": float(test_bacc),                 # Test balanced accuracy
        "class_names": CLASS_NAMES,                            # ["Flat", "Cylindrical", "Cuboid", "Irregular"]
        "geometric_feature_names": GEOMETRIC_FEATURE_NAMES_V2, # Ordered list of 35 feature names
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")        # Training timestamp
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[SAVED CHECKPOINT] -> {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")
    return bundle


def main():
    """
    CLI entry point for real pipeline training.

    Supports two modes:
      1. Cached features (fast): If pre-extracted .npz feature files exist,
         loads them directly and trains the model in seconds.
      2. From scratch (slow): If no cache exists, loads raw images and
         extracts features using DINOv2 + Depth Anything V2 (requires GPU).
    """
    parser = argparse.ArgumentParser(description="Train Real Champion Pipeline C Classifier.")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache_v2",
                        help="Feature cache directory containing pre-extracted .npz files")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                        help="Dataset folder if extracting features from scratch")
    parser.add_argument("--output-model", type=str, default="real_model.pkl",
                        help="Destination pickle file for the trained model")
    args = parser.parse_args()

    # ── Locate Feature Cache ────────────────────────────────────────────
    # Feature caches store pre-extracted DINOv2 + geometry features in
    # compressed NumPy format (.npz). This avoids re-running the expensive
    # neural network inference every time we retrain the model.
    cache_p = Path(args.cache_dir)
    f_tr = cache_p / "train_features_v2.npz"     # Training features
    f_val = cache_p / "val_features_v2.npz"       # Validation features
    f_te = cache_p / "test_features_v2.npz"       # Test features

    # Also check the legacy retail_geometry_project cache location
    # (useful when running from the unified geometric_classifier/ directory)
    if not (f_tr.exists() and f_val.exists() and f_te.exists()):
        alt_cache = SCRIPT_DIR.parent / "retail_geometry_project" / "outputs" / "cache_v2"
        if (alt_cache / "train_features_v2.npz").exists():
            print(f"[INFO] Using verified feature cache from {alt_cache}")
            f_tr = alt_cache / "train_features_v2.npz"
            f_val = alt_cache / "val_features_v2.npz"
            f_te = alt_cache / "test_features_v2.npz"

    # ── Train from Cache or from Scratch ────────────────────────────────
    if f_tr.exists() and f_val.exists() and f_te.exists():
        # Fast path: Load pre-extracted features (seconds)
        print(f"[INFO] Loading pre-extracted 768-D DINOv2 + 35-D Geometry features...")
        d_tr = np.load(f_tr)
        d_val = np.load(f_val)
        d_te = np.load(f_te)

        train_from_features(
            d_tr["X_rgb"], d_tr["X_geo"], d_tr["y_labels"],
            d_val["X_rgb"], d_val["X_geo"], d_val["y_labels"],
            d_te["X_rgb"], d_te["X_geo"], d_te["y_labels"],
            output_path=args.output_model
        )
    else:
        # Slow path: Extract features from raw images (requires GPU, ~30-60 min)
        print(f"[INFO] Feature cache not found. Extracting features directly from '{args.dataset_dir}'...")
        # load_partitioned_datasets returns sample record lists with file paths and labels
        train_records, val_records, test_records = load_partitioned_datasets(dataset_dir=args.dataset_dir)
        # PipelineC handles batched DINOv2 + DepthAnything feature extraction
        pipeline = PipelineC()

        # build_dataset_feature_matrix processes all images and returns
        # (X_rgb, X_geo, reliabilities, y_labels, group_keys) as NumPy arrays
        X_rgb_tr, X_geo_tr, rel_tr, y_tr, _ = pipeline.build_dataset_feature_matrix(train_records)
        X_rgb_val, X_geo_val, rel_val, y_val, _ = pipeline.build_dataset_feature_matrix(val_records)
        X_rgb_te, X_geo_te, rel_te, y_te, _ = pipeline.build_dataset_feature_matrix(test_records)

        train_from_features(
            X_rgb_tr, X_geo_tr, y_tr,
            X_rgb_val, X_geo_val, y_val,
            X_rgb_te, X_geo_te, y_te,
            output_path=args.output_model
        )


if __name__ == "__main__":
    main()
