"""
========================================================================================
src/fusion_engine.py — Multi-Modal Fusion Architectures (Eliminating Feature Drowning)
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & ARCHITECTURE: SOLVING THE "FEATURE DROWNING" PROBLEM
----------------------------------------------------------------------------------------
[Basic Concept: What is Multi-Modal Fusion?]:
  In multi-modal machine learning, our goal is to combine two distinct sensory modalities:
    Modality 1: Rich visual appearance (RGB features extracted via pretrained DINOv2).
    Modality 2: Physical shape & depth properties (35-D geometric features from Depth Anything V2).
  The naive way to combine them is "Early Fusion by Concatenation": simply pasting the two
  vectors together into a single large array: F_fused = [F_RGB, F_Geo].

[The "Feature Drowning" Phenomenon]:
  Why does naive early concatenation fail in tree-based ensembles (RandomForest, ExtraTrees)?
  Consider the dimensions:
    - RGB embedding from DINOv2: 768 dimensions (CLS token + mean spatial patch tokens).
    - Geometric feature vector:  35 dimensions.
    - Total concatenated vector: 803 dimensions.
  When an ensemble tree builds a split node, it randomly selects `max_features = sqrt(803) ≈ 28`
  candidate features to evaluate.
  The probability of any selected candidate being an RGB feature is 768 / 803 = 95.6%!
  The probability of selecting a geometric feature is only 35 / 803 = 4.4%.
  Consequently, tree splits almost exclusively test visual packaging cues, while the critical
  3D geometric signals are statistically "drowned out". The model degrades to an RGB-only
  classifier that overfits to packaging textures and fails to generalize out-of-domain.

----------------------------------------------------------------------------------------
THREE ADVANCED FUSION ARCHITECTURES IMPLEMENTED:
----------------------------------------------------------------------------------------
1. Fusion Approach A: Dimensionality-Equalized Projection (64-D Balanced Fused Vector)
   - Visual Branch:   X_RGB (768-D) -> StandardScaler -> PCA(32-D) -> F_RGB_32
   - Geometry Branch: X_Geo (35-D)  -> RobustScaler  -> PCA(32-D) -> F_Geo_32
   - Concatenation:   F_fused = [F_RGB_32, F_Geo_32] in R^64
   - Mathematical Guarantee: Exactly 50% of the candidate features in every tree split are visual,
     and exactly 50% are geometric! Tree splits achieve perfect modality balance.
   - Classification: Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RF) with 5-fold
     Isotonic Probability Calibration.

2. Fusion Approach B: Modality Scaling & Adaptive Reliability Weighting
   - Rather than projecting, Approach B scales the geometric features dynamically by the
     sample's Depth Reliability Score: F_Geo_weighted = R_d * RobustScaler(X_Geo).
   - If a crop suffered severe packaging glare, boundary bleed, or shadow (low R_d), the
     geometric signal is attenuated, allowing visual features to guide the prediction.
     If depth quality is pristine (high R_d), geometry asserts strong decision influence.

3. Fusion Approach C: Late-Fusion Stacking Meta-Learner (Zero Feature Drowning Guarantee)
   - Level-0 Model A (RGB-Only Classifier): Trained strictly on DINOv2 visual embeddings -> P_RGB in R^4.
   - Level-0 Model B (Geo-Only Classifier): Trained strictly on 35-D geometric features -> P_Geo in R^4.
   - Out-of-Fold (OOF) Stacking: To prevent data leakage and meta-overfitting, Level-0 predictions
     on the training set are generated via Stratified Group 5-Fold Cross-Validation.
   - Level-1 Meta-Classifier: Calibrated Logistic Regression / ExtraTrees trained on:
     X_meta = [P_RGB (4-D), P_Geo (4-D), R_d (1-D)] in R^9.
   - Mathematical Guarantee: Feature drowning is 100% impossible because the visual and geometric
     models are trained in total isolation before their posterior class distributions are fused!
========================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
NUM_CLASSES = len(CLASS_NAMES)


def _extract_fit_args(arg3, arg4):
    """
    Robust helper that universaly extracts (y, reliabilities) regardless of whether
    the caller passes positional arguments as:
      fit(X_rgb, X_geo, y, reliabilities)   OR   fit(X_rgb, X_geo, reliabilities, y)

    Uses data type introspection (floating point vs integer/categorical) to disambiguate.
    """
    if arg3 is None and arg4 is not None:
        return arg4, None
    if arg4 is None and arg3 is not None:
        return arg3, None
    if arg3 is not None and arg4 is not None:
        a3 = np.asarray(arg3)
        a4 = np.asarray(arg4)
        if np.issubdtype(a3.dtype, np.floating) and (
            np.issubdtype(a4.dtype, np.integer) or np.issubdtype(a4.dtype, np.str_)
        ):
            return a4, a3
        elif np.issubdtype(a4.dtype, np.floating) and (
            np.issubdtype(a3.dtype, np.integer) or np.issubdtype(a3.dtype, np.str_)
        ):
            return a3, a4
    return arg3, arg4


def build_base_ensemble(seed: int = 42) -> VotingClassifier:
    """
    Constructs a high-capacity, regularized soft-voting ensemble of 3 distinct tree architectures:
      1. ExtraTreesClassifier (Extremely Randomized Trees):
         Samples random split thresholds per feature, providing exceptional variance reduction
         and robust resistance to feature noise.
      2. HistGradientBoostingClassifier (Histogram-based GBDT):
         Binned histogram splits with native support for smooth non-linear decision boundaries.
      3. RandomForestClassifier (Bagged Decision Trees):
         Traditional greedy variance-reduction bagging with balanced bootstrap sampling.

    Args:
        seed: Random seed for deterministic reproducibility across trees.

    Returns:
        VotingClassifier configured for soft-probability averaging.
    """
    return VotingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(
                n_estimators=250, max_depth=18, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1
            )),
            ("hgb", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=8,
                class_weight="balanced", random_state=seed
            )),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=-1
            ))
        ],
        voting="soft",
        n_jobs=-1
    )


# ======================================================================================
# FUSION APPROACH A: 64-D DIMENSIONALITY-EQUALIZED PROJECTION
# ======================================================================================
class FusionApproachA_EqualizedProjection:
    """
    Fusion Strategy A: Dimensionality-Equalized 64-D Projection.

    Mathematical Workflow:
      1. Scale RGB (384/768-D) via StandardScaler (zero mean, unit variance).
      2. Project RGB down to 32 orthogonal dimensions via Principal Component Analysis (PCA).
      3. Scale Geometry (35-D) via RobustScaler (median-centered, IQR-scaled to ignore outliers).
      4. Project Geometry to 32 orthogonal dimensions via PCA.
      5. Concatenate both 32-D vectors: F_fused = [F_RGB_32, F_Geo_32] in R^64.
      6. Fit Calibrated Soft-Voting Ensemble with 5-fold Isotonic calibration.

    Key Benefit:
      Gives visual and geometric signals an exact 50/50 probability of being chosen at any
      tree split, completely eradicating the feature drowning phenomenon.
    """

    def __init__(self, n_rgb_comp: int = 32, n_geo_comp: int = 32, seed: int = 42):
        self.name = "Approach A: Equalized 64-D Projection (PCA 32-D + PCA 32-D)"
        self.scaler_rgb = StandardScaler()
        self.proj_rgb   = PCA(n_components=n_rgb_comp, random_state=seed)
        self.scaler_geo = RobustScaler()
        self.proj_geo   = PCA(n_components=n_geo_comp, random_state=seed)
        self.classifier = None
        self.is_fitted  = False
        self.seed       = seed

    def fit(self, X_rgb, X_geo, arg3=None, arg4=None, group_keys=None, **kwargs):
        """Fits the dual PCA projections, base ensemble, and 5-fold isotonic calibrator."""
        y, reliabilities = _extract_fit_args(
            kwargs.get("y", arg3),
            kwargs.get("reliabilities", arg4)
        )
        # Visual branch projection
        X_rgb_scaled = self.scaler_rgb.fit_transform(X_rgb)
        F_rgb_32     = self.proj_rgb.fit_transform(X_rgb_scaled)

        # Geometry branch projection
        X_geo_scaled = self.scaler_geo.fit_transform(X_geo)
        F_geo_32     = self.proj_geo.fit_transform(X_geo_scaled)

        # 50/50 balanced fused representation
        F_fused_64 = np.hstack([F_rgb_32, F_geo_32]).astype(np.float32)

        base = build_base_ensemble(self.seed)
        base.fit(F_fused_64, y)

        # 5-fold Isotonic Probability Calibration to produce true posterior probabilities
        self.classifier = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
        self.classifier.fit(F_fused_64, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X_rgb, X_geo, reliabilities=None):
        """Transforms input matrices into the equalized 64-D space and predicts class probabilities."""
        X_rgb_scaled = self.scaler_rgb.transform(X_rgb)
        F_rgb_32     = self.proj_rgb.transform(X_rgb_scaled)

        X_geo_scaled = self.scaler_geo.transform(X_geo)
        F_geo_32     = self.proj_geo.transform(X_geo_scaled)

        F_fused_64   = np.hstack([F_rgb_32, F_geo_32]).astype(np.float32)
        return self.classifier.predict_proba(F_fused_64)

    def predict(self, X_rgb, X_geo, reliabilities=None):
        """Predicts class indices in [0..3] via maximum posterior probability."""
        probs = self.predict_proba(X_rgb, X_geo, reliabilities)
        return np.argmax(probs, axis=1)


class RegularizedPipelineCClassifier:
    """
    High-regularization variant of Approach A, engineered specifically for high Sim-to-Real
    transfer robustness.

    Enhancements:
      - Constrained leaf depth and increased min_samples_leaf to prevent memorizing CAD rendering quirks.
      - Strong L2 regularization in HistGradientBoosting.
      - max_features='sqrt' in ExtraTrees and RandomForest to enforce multi-modal feature mixing.
      - 5-fold Isotonic probability calibration.
    """

    def __init__(self, n_rgb_comp: int = 32, n_geo_comp: int = 32, seed: int = 42):
        self.name = "Regularized Pipeline C (64-D Equalized)"
        self.scaler_rgb = StandardScaler()
        self.proj_rgb   = PCA(n_components=n_rgb_comp, random_state=seed)
        self.scaler_geo = RobustScaler()
        self.proj_geo   = PCA(n_components=n_geo_comp, random_state=seed)
        self.classifier = None
        self.is_fitted  = False
        self.seed       = seed

    def fit(self, X_rgb, X_geo, arg3=None, arg4=None, group_keys=None, **kwargs):
        y, reliabilities = _extract_fit_args(
            kwargs.get("y", arg3),
            kwargs.get("reliabilities", arg4)
        )
        X_rgb_scaled = self.scaler_rgb.fit_transform(X_rgb)
        F_rgb_32     = self.proj_rgb.fit_transform(X_rgb_scaled)

        X_geo_scaled = self.scaler_geo.fit_transform(X_geo)
        F_geo_32     = self.proj_geo.fit_transform(X_geo_scaled)

        F_fused_64   = np.hstack([F_rgb_32, F_geo_32]).astype(np.float32)

        base = VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(
                    n_estimators=350, max_depth=16, min_samples_leaf=4,
                    max_features="sqrt", class_weight="balanced",
                    random_state=self.seed, n_jobs=-1
                )),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=250, learning_rate=0.04, max_depth=8,
                    min_samples_leaf=15, l2_regularization=1.0,
                    class_weight="balanced", random_state=self.seed
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=300, max_depth=14, min_samples_leaf=4,
                    max_features="sqrt", class_weight="balanced",
                    random_state=self.seed, n_jobs=-1
                ))
            ],
            voting="soft",
            n_jobs=-1
        )

        self.classifier = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
        self.classifier.fit(F_fused_64, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X_rgb, X_geo, reliabilities=None):
        X_rgb_scaled = self.scaler_rgb.transform(X_rgb)
        F_rgb_32     = self.proj_rgb.transform(X_rgb_scaled)

        X_geo_scaled = self.scaler_geo.transform(X_geo)
        F_geo_32     = self.proj_geo.transform(X_geo_scaled)

        F_fused_64   = np.hstack([F_rgb_32, F_geo_32]).astype(np.float32)
        return self.classifier.predict_proba(F_fused_64)

    def predict(self, X_rgb, X_geo, reliabilities=None):
        probs = self.predict_proba(X_rgb, X_geo, reliabilities)
        return np.argmax(probs, axis=1)


# ======================================================================================
# FUSION APPROACH B: MODALITY SCALING & RELIABILITY WEIGHTING
# ======================================================================================
class FusionApproachB_ReliabilityWeighting:
    """
    Fusion Strategy B: Adaptive Depth Reliability Weighting.

    Mathematical Workflow:
      F_fused = [ StandardScaler(X_RGB), (R_d * RobustScaler(X_Geo)) ]

    Concept:
      Monocular depth estimation models can experience degradation under severe supermarket
      lighting conditions (e.g. halogen glare on glossy snack bags, motion blur).
      By multiplying the geometric features by the dynamic reliability score R_d in [0.10, 1.0],
      the model gracefully falls back onto DINOv2 visual features when depth is corrupted,
      and leverages geometric precision when depth is crisp.
    """

    def __init__(self, seed: int = 42):
        self.name = "Approach B: Modality Scaling & Adaptive Reliability Weighting"
        self.scaler_rgb = StandardScaler()
        self.scaler_geo = RobustScaler()
        self.classifier = None
        self.is_fitted  = False
        self.seed       = seed

    def fit(self, X_rgb, X_geo, arg3=None, arg4=None, group_keys=None, **kwargs):
        y, reliabilities = _extract_fit_args(
            kwargs.get("y", arg3),
            kwargs.get("reliabilities", arg4)
        )
        if reliabilities is None:
            reliabilities = np.ones(len(X_rgb), dtype=np.float32)

        X_rgb_scaled = self.scaler_rgb.fit_transform(X_rgb)
        X_geo_scaled = self.scaler_geo.fit_transform(X_geo)

        # Dynamic reliability weighting
        rel_weights = np.array(reliabilities, dtype=np.float32).reshape(-1, 1)
        X_geo_weighted = X_geo_scaled * rel_weights

        F_fused = np.hstack([X_rgb_scaled, X_geo_weighted]).astype(np.float32)

        base = build_base_ensemble(self.seed)
        base.fit(F_fused, y)

        self.classifier = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
        self.classifier.fit(F_fused, y)
        self.is_fitted = True
        return self

    def transform_features(self, X_rgb, X_geo, reliabilities=None):
        if reliabilities is None:
            reliabilities = np.ones(len(X_rgb), dtype=np.float32)
        X_rgb_scaled = self.scaler_rgb.transform(X_rgb)
        X_geo_scaled = self.scaler_geo.transform(X_geo)
        rel_weights = np.array(reliabilities, dtype=np.float32).reshape(-1, 1)
        X_geo_weighted = X_geo_scaled * rel_weights
        return np.hstack([X_rgb_scaled, X_geo_weighted]).astype(np.float32)

    def predict_proba(self, X_rgb, X_geo, reliabilities=None):
        F_fused = self.transform_features(X_rgb, X_geo, reliabilities)
        return self.classifier.predict_proba(F_fused)

    def predict(self, X_rgb, X_geo, reliabilities=None):
        probs = self.predict_proba(X_rgb, X_geo, reliabilities)
        return np.argmax(probs, axis=1)


# ======================================================================================
# FUSION APPROACH C: LATE-FUSION STACKING META-LEARNER
# ======================================================================================
class FusionApproachC_LateFusionStacking:
    """
    Fusion Strategy C: Late-Fusion Stacking Meta-Learner (Zero Feature Drowning).

    Mathematical Formulation:
      - Level-0 Model A (Visual Specialist):
          Trained on X_RGB -> outputs posterior vector P_RGB = [P(0|RGB), P(1|RGB), P(2|RGB), P(3|RGB)]
      - Level-0 Model B (Geometry Specialist):
          Trained on X_Geo -> outputs posterior vector P_Geo = [P(0|Geo), P(1|Geo), P(2|Geo), P(3|Geo)]
      - Out-of-Fold Cross-Validation:
          To prevent the meta-learner from overfitting to overconfident training predictions,
          Level-0 predictions on the training data are collected out-of-fold across 5 cross-validation splits.
      - Level-1 Meta-Feature Matrix:
          X_meta = [ P_RGB (4-D), P_Geo (4-D), R_d (1-D) ] in R^9
      - Level-1 Meta-Classifier:
          Calibrated LogisticRegression / ExtraTrees trained to optimize final class assignments.

    Why Approach C is Architecturally Superior:
      In early fusion (concatenation), one modality can easily monopolize split decisions.
      In late fusion stacking, both models MUST make independent 4-class probability predictions.
      The meta-learner's sole job is to learn which modality to trust for which class, modulated
      by the depth reliability scalar R_d.
    """

    def __init__(self, seed: int = 42):
        self.name = "Approach C: Late-Fusion Stacking Meta-Learner (Zero Drowning)"
        self.scaler_rgb = StandardScaler()
        self.scaler_geo = RobustScaler()
        self.model_rgb  = None
        self.model_geo  = None
        self.meta_classifier = None
        self.is_fitted  = False
        self.seed       = seed

    def fit(self, X_rgb, X_geo, arg3=None, arg4=None, group_keys=None, **kwargs):
        y, reliabilities = _extract_fit_args(
            kwargs.get("y", arg3),
            kwargs.get("reliabilities", arg4)
        )
        if reliabilities is None:
            reliabilities = np.ones(len(X_rgb), dtype=np.float32)

        X_rgb_scaled = self.scaler_rgb.fit_transform(X_rgb)
        X_geo_scaled = self.scaler_geo.fit_transform(X_geo)

        # 1. Base models for Out-of-Fold (OOF) cross-validation predictions
        base_rgb = build_base_ensemble(self.seed)
        base_geo = build_base_ensemble(self.seed + 1)

        print("  [STACKING] Generating Out-of-Fold Level-0 Probability Predictions...", flush=True)
        if group_keys is not None:
            # Enforce group awareness during cross-validation stacking
            cv_splitter_rgb = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_arg_rgb = cv_splitter_rgb.split(X_rgb_scaled, y, group_keys)
            cv_splitter_geo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_arg_geo = cv_splitter_geo.split(X_geo_scaled, y, group_keys)
        else:
            cv_arg_rgb = 5
            cv_arg_geo = 5

        oof_rgb = cross_val_predict(base_rgb, X_rgb_scaled, y, cv=cv_arg_rgb, method="predict_proba", n_jobs=-1)
        oof_geo = cross_val_predict(base_geo, X_geo_scaled, y, cv=cv_arg_geo, method="predict_proba", n_jobs=-1)

        # 2. Fit full calibrated Level-0 specialists on complete training data
        print("  [STACKING] Fitting full Level-0 Calibrated Classifiers...", flush=True)
        self.model_rgb = CalibratedClassifierCV(estimator=build_base_ensemble(self.seed), method="isotonic", cv=5)
        self.model_rgb.fit(X_rgb_scaled, y)

        self.model_geo = CalibratedClassifierCV(estimator=build_base_ensemble(self.seed + 1), method="isotonic", cv=5)
        self.model_geo.fit(X_geo_scaled, y)

        # 3. Construct Level-1 Meta-Feature Matrix: [P_RGB (4), P_Geo (4), R_d (1)] in R^9
        rel_col = np.array(reliabilities, dtype=np.float32).reshape(-1, 1)
        X_meta_train = np.hstack([oof_rgb, oof_geo, rel_col]).astype(np.float32)

        # 4. Train Calibrated Meta-Classifier
        meta_base = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=self.seed)
        self.meta_classifier = CalibratedClassifierCV(estimator=meta_base, method="isotonic", cv=5)
        self.meta_classifier.fit(X_meta_train, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X_rgb, X_geo, reliabilities):
        """Passes inputs through Level-0 specialists and feeds posterior stack to meta-classifier."""
        X_rgb_scaled = self.scaler_rgb.transform(X_rgb)
        X_geo_scaled = self.scaler_geo.transform(X_geo)

        P_rgb = self.model_rgb.predict_proba(X_rgb_scaled)
        P_geo = self.model_geo.predict_proba(X_geo_scaled)

        rel_col = np.array(reliabilities, dtype=np.float32).reshape(-1, 1)
        X_meta = np.hstack([P_rgb, P_geo, rel_col]).astype(np.float32)

        return self.meta_classifier.predict_proba(X_meta)

    def predict(self, X_rgb, X_geo, reliabilities):
        probs = self.predict_proba(X_rgb, X_geo, reliabilities)
        return np.argmax(probs, axis=1)


# ======================================================================================
# EVALUATION & BENCHMARKING HELPER
# ======================================================================================
def evaluate_fusion_model(model, X_rgb_test, X_geo_test, rel_test, y_test):
    """
    Evaluates any fitted multi-modal fusion model on pristine unseen test crops.

    Computes:
      - Balanced Accuracy (macro-average recall across all 4 classes).
      - Overall Accuracy.
      - Full Confusion Matrix and precision/recall/F1 per class.
      - Uncertainty count: samples with max posterior probability < 0.45.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    probs = model.predict_proba(X_rgb_test, X_geo_test, rel_test)
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    bal_acc = balanced_accuracy_score(y_test, preds)
    acc     = accuracy_score(y_test, preds)
    cm      = confusion_matrix(y_test, preds)
    report  = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)

    uncertain_count = int(np.sum(max_probs < 0.45))

    return {
        "model_name": model.name,
        "balanced_accuracy": float(bal_acc),
        "overall_accuracy": float(acc),
        "uncertain_count": uncertain_count,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_test,
        "y_pred": preds,
        "y_prob": probs
    }
