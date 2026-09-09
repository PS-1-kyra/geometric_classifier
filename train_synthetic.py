#!/usr/bin/env python3
"""
=============================================================
train_synthetic.py — End-to-End Synthetic Pipeline Champion Training Script
=============================================================
Engineer: Pranaya Shrestha
Project:  PS-1 Class-Agnostic Primitive Analysis

Trains the Generalized Sim-to-Real Champion:
  1. Feature Masking: Eliminates CAD-to-real planarity shifted features:
     ['planar_surface_ratio', 'curved_surface_ratio', 'std_Nz'].
  2. Modality Ratio Balancing: 48-D DINOv2 RGB PCA + 16-D Geometry PCA.
  3. Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RF) with 5-fold Isotonic calibration.
  4. Prior Sensitivity Re-weighting: W = [1.8, 1.0, 1.4, 0.45] (Flat, Cylindrical, Cuboid, Irregular).
  5. Saves model bundle directly to synthetic_model.pkl.
=============================================================
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report

# Ensure repo root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.generalized_champion import GeneralizedChampionPipelineC, CLASS_NAMES, DROP_FEATURES, CLASS_WEIGHTS
from src.synthetic_loader import SYNTH_TO_REAL_MAP, SyntheticPrimitiveDataset

REAL_BASELINE_REF = 85.96


def train_synthetic_champion(X_train_rgb, X_train_geo, y_train, X_eval_rgb=None, X_eval_geo=None, y_eval=None, output_path="synthetic_model.pkl"):
    print("=" * 75)
    print("  TRAINING SYNTHETIC GENERALIZED CHAMPION (FEATURE-MASKED + CALIBRATED)")
    print("=" * 75)
    print(f"Synthetic training samples: {len(y_train):5d} (RGB: {X_train_rgb.shape[1]}-D, Geo: {X_train_geo.shape[1]}-D)")
    print(f"Dropped shifted features  : {DROP_FEATURES}")
    print(f"PCA Projections           : 48-D RGB + 16-D Geometry")
    print(f"Class prior re-weighting  : {CLASS_WEIGHTS}\n")

    t0 = time.time()
    model = GeneralizedChampionPipelineC(
        n_rgb_comp=48,
        n_geo_comp=16,
        drop_features=DROP_FEATURES,
        class_weights=CLASS_WEIGHTS,
        seed=42
    )
    model.fit(X_train_rgb, X_train_geo, y_train)
    fit_duration = time.time() - t0
    print(f"[OK] Synthetic champion fitted in {fit_duration:.2f}s\n")

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

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"[SAVED CHECKPOINT] -> {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB)")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train Synthetic Generalized Champion Classifier.")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Synthetic feature cache directory")
    parser.add_argument("--output-model", type=str, default="synthetic_model.pkl", help="Destination pickle file")
    args = parser.parse_args()

    # Search for synthetic feature caches
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

    f_batch = cache_dir / "syn_train_batch_250_dav2_clean_seed42.npz"
    f_shelf = cache_dir / "syn_train_shelf_full_dav2_clean_seed42.npz"

    if not (f_batch.exists() and f_shelf.exists()):
        raise FileNotFoundError(f"Missing required synthetic cache files in '{cache_dir}'.")

    print(f"[INFO] Loading synthetic caches from {cache_dir} ...")
    d_b = np.load(f_batch)
    d_s = np.load(f_shelf)

    X_train_rgb = np.vstack([d_b["X_rgb"], d_s["X_rgb"]])
    X_train_geo = np.vstack([d_b["X_geo"], d_s["X_geo"]])
    y_train_syn = np.concatenate([d_b["y_labels"], d_s["y_labels"]])

    # Convert synthetic index to real index space
    y_train = np.array([SYNTH_TO_REAL_MAP[y] for y in y_train_syn], dtype=np.int32)

    # Load 815 real evaluation set if available
    X_eval_rgb, X_eval_geo, y_eval = None, None, None
    real_cache = SCRIPT_DIR.parent / "retail_geometry_project" / "outputs" / "cache_v2"
    if (real_cache / "train_features_v2.npz").exists() and (real_cache / "test_features_v2.npz").exists():
        d_tr = np.load(real_cache / "train_features_v2.npz")
        d_val = np.load(real_cache / "val_features_v2.npz")
        d_ts = np.load(real_cache / "test_features_v2.npz")

        unique_groups, first_idx = np.unique(d_tr["group_keys"], return_index=True)
        X_eval_rgb = np.vstack([d_tr["X_rgb"][first_idx], d_val["X_rgb"], d_ts["X_rgb"]])
        X_eval_geo = np.vstack([d_tr["X_geo"][first_idx], d_val["X_geo"], d_ts["X_geo"]])
        y_eval = np.concatenate([d_tr["y_labels"][first_idx], d_val["y_labels"], d_ts["y_labels"]]).astype(np.int32)

    train_synthetic_champion(
        X_train_rgb, X_train_geo, y_train,
        X_eval_rgb=X_eval_rgb, X_eval_geo=X_eval_geo, y_eval=y_eval,
        output_path=args.output_model
    )


if __name__ == "__main__":
    main()
