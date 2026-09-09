#!/usr/bin/env python3
"""
=============================================================
train_real.py — End-to-End Real Pipeline C Training Script
=============================================================
Engineer: Pranaya Shrestha
Project:  PS-1 Class-Agnostic Primitive Analysis

Trains the Real Champion Multimodal Primitive Classifier:
  1. RGB Branch: Pretrained DINOv2 ViT-S/14 (768-D) -> StandardScaler -> PCA(32-D).
  2. Geometry Branch: Depth Anything V2 + 35-D Geometry -> RobustScaler -> PCA(32-D).
  3. Fused Representation: [F_RGB_32, F_Geo_32] in R^64 (Equalized 50% / 50% split).
  4. Ensemble: ExtraTrees + HistGradientBoosting + RandomForest
               wrapped in 5-fold CalibratedClassifierCV(method='isotonic').
  5. Evaluates on 100% PRISTINE unseen product test groups (Zero Data Leakage).
  6. Saves model bundle directly to real_model.pkl.
=============================================================
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix

# Ensure repo root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset_engine import CLASS_NAMES, NUM_CLASSES, load_partitioned_datasets
from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2
from src.fusion_engine import FusionApproachA_EqualizedProjection, evaluate_fusion_model
from src.pipeline_c import PipelineC


def train_from_features(X_rgb_tr, X_geo_tr, y_tr, X_rgb_val, X_geo_val, y_val, X_rgb_te, X_geo_te, y_te, output_path="real_model.pkl"):
    """
    Fits FusionApproachA_EqualizedProjection on train features and evaluates on validation and test sets.
    """
    print("=" * 75)
    print("  TRAINING REAL CHAMPION: EQUALIZED 64-D PROJECTION PIPELINE C")
    print("=" * 75)
    print(f"Train samples : {len(y_tr):4d} (RGB: {X_rgb_tr.shape[1]}-D, Geo: {X_geo_tr.shape[1]}-D)")
    print(f"Val samples   : {len(y_val):4d} (100% pristine unseen product groups)")
    print(f"Test samples  : {len(y_te):4d} (100% pristine unseen product groups)\n")

    t0 = time.time()
    model = FusionApproachA_EqualizedProjection(n_rgb_comp=32, n_geo_comp=32, seed=42)
    model.fit(X_rgb_tr, X_geo_tr, y_tr)
    fit_duration = time.time() - t0
    print(f"[OK] Model successfully fitted in {fit_duration:.2f}s\n")

    # Evaluate on Validation Set
    val_preds = model.predict(X_rgb_val, X_geo_val)
    val_bacc = balanced_accuracy_score(y_val, val_preds)
    print(f"  * Validation Balanced Accuracy : {val_bacc * 100:.2f}%")

    # Evaluate on Test Set
    test_preds = model.predict(X_rgb_te, X_geo_te)
    test_bacc = balanced_accuracy_score(y_te, test_preds)
    test_acc = accuracy_score(y_te, test_preds)

    print("\n" + "=" * 75)
    print("  PRISTINE TEST SET EVALUATION (ZERO GROUP LEAKAGE)")
    print("=" * 75)
    print(f"  * Test Balanced Accuracy : {test_bacc * 100:.2f}%")
    print(f"  * Test Overall Accuracy  : {test_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_te, test_preds, target_names=CLASS_NAMES, digits=4))

    bundle = {
        "model": model,
        "champion_name": "Approach A: Equalized 64-D Projection (PCA 32-D + PCA 32-D)",
        "balanced_accuracy": float(test_bacc),
        "class_names": CLASS_NAMES,
        "geometric_feature_names": GEOMETRIC_FEATURE_NAMES_V2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[SAVED CHECKPOINT] -> {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train Real Champion Pipeline C Classifier.")
    parser.add_argument("--cache-dir", type=str, default="outputs/cache_v2", help="Feature cache directory")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset folder if extracting from scratch")
    parser.add_argument("--output-model", type=str, default="real_model.pkl", help="Destination pickle file")
    args = parser.parse_args()

    cache_p = Path(args.cache_dir)
    f_tr = cache_p / "train_features_v2.npz"
    f_val = cache_p / "val_features_v2.npz"
    f_te = cache_p / "test_features_v2.npz"

    # Also check parent retail_geometry_project caches if local cache is absent
    if not (f_tr.exists() and f_val.exists() and f_te.exists()):
        alt_cache = SCRIPT_DIR.parent / "retail_geometry_project" / "outputs" / "cache_v2"
        if (alt_cache / "train_features_v2.npz").exists():
            print(f"[INFO] Using verified feature cache from {alt_cache}")
            f_tr = alt_cache / "train_features_v2.npz"
            f_val = alt_cache / "val_features_v2.npz"
            f_te = alt_cache / "test_features_v2.npz"

    if f_tr.exists() and f_val.exists() and f_te.exists():
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
        print(f"[INFO] Feature cache not found. Extracting features directly from '{args.dataset_dir}'...")
        train_records, val_records, test_records = load_partitioned_datasets(dataset_dir=args.dataset_dir)
        pipeline = PipelineC()

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
