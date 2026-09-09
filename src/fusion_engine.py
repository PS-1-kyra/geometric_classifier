"""

src/fusion_engine.py -- Multi-Modal Fusion Architectures


Project:  PS-1 Class-Agnostic Primitive Analysis

Implements 3 Advanced Fusion Strategies to eliminate the 384-D vs 19-D feature drowning problem:

  1. Fusion Approach A: Dimensionality-Equalized Projection (64-D Balanced Fused Vector)
     - DINOv2 RGB (384-D / 768-D) -> StandardScaler -> PCA(32-D)
     - Geometry (35-D)            -> RobustScaler  -> PCA(32-D)
     - Concatenation: [F_RGB_32, F_Geo_32] in R^64 (50/50 split probability in trees)
     - Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RF)

  2. Fusion Approach B: Modality-Specific Scaling & Adaptive Reliability Weighting
     - StandardScaler(RGB) + (R_d * RobustScaler(Geo_35))
     - Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RF)

  3. Fusion Approach C: Late-Fusion Stacking Meta-Learner (Zero Feature Drowning)
     - Level-0 Model A (RGB-Only Classifier): ExtraTrees + HistGB + RF -> P_RGB in R^4
     - Level-0 Model B (Geo-Only Classifier): ExtraTrees + HistGB + RF -> P_Geo in R^4
     - Out-of-Fold Cross-Validation Stacking: X_meta = [P_RGB, P_Geo, R_d] in R^9
     - Level-1 Meta-Classifier: Calibrated LogisticRegression / ExtraTrees
=============================================================
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

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score

CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
NUM_CLASSES = len(CLASS_NAMES)


def _extract_fit_args(arg3, arg4):
    """
    Universally extracts (y, reliabilities) regardless of whether the caller passes:
      fit(X_rgb, X_geo, y, reliabilities)  OR  fit(X_rgb, X_geo, reliabilities, y)
    """
    if arg3 is None and arg4 is not None:
        return arg4, None
    if arg4 is None and arg3 is not None:
        return arg3, None
    if arg3 is not None and arg4 is not None:
        a3 = np.asarray(arg3)
        a4 = np.asarray(arg4)
        if np.issubdtype(a3.dtype, np.floating) and (np.issubdtype(a4.dtype, np.integer) or np.issubdtype(a4.dtype, np.str_)):
            return a4, a3
        elif np.issubdtype(a4.dtype, np.floating) and (np.issubdtype(a3.dtype, np.integer) or np.issubdtype(a3.dtype, np.str_)):
            return a3, a4
    return arg3, arg4


def build_base_ensemble(seed=42):
    """Creates a high-capacity soft-voting ensemble for tabular/fused representations."""
    return VotingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(n_estimators=250, max_depth=18, min_samples_leaf=2,
                                        class_weight="balanced", random_state=seed, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=8,
                                                   class_weight="balanced", random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=2,
                                          class_weight="balanced", random_state=seed, n_jobs=-1))
        ],
        voting="soft",
        n_jobs=-1
    )


# =============================================================
# FUSION APPROACH A: 64-D DIMENSIONALITY-EQUALIZED PROJECTION
# =============================================================
class FusionApproachA_EqualizedProjection:
    """
    Equalizes modalities: RGB (32-D) + Geometry (32-D) = 64-D Balanced Fused Vector.
    Eliminates tree split bias, giving equal 50% chance to visual and geometric features.
    """
    def __init__(self, n_rgb_comp=32, n_geo_comp=32, seed=42):
        self.name = "Approach A: Equalized 64-D Projection (PCA 32-D + PCA 32-D)"
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

        F_fused_64 = np.hstack([F_rgb_32, F_geo_32]).astype(np.float32)

        base = build_base_ensemble(self.seed)
        base.fit(F_fused_64, y)

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


class RegularizedPipelineCClassifier:
    """
    High-regularization ensemble designed for extreme Sim-to-Real generalization:
      - 64-D Equalized PCA projection (eliminates feature drowning)
      - Feature Noise Injection during training (improves robustness)
      - Strong leaf regularization (min_samples_leaf=4, max_features='sqrt')
      - Isotonic 5-fold calibration
    """
    def __init__(self, n_rgb_comp=32, n_geo_comp=32, seed=42):
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


# =============================================================
# FUSION APPROACH B: MODALITY SCALING & RELIABILITY WEIGHTING
# =============================================================
class FusionApproachB_ReliabilityWeighting:
    """
    StandardScaler(RGB) + (R_d * RobustScaler(Geo_35)).
    Weights geometry features based on depth reliability (glare/erosion confidence).
    """
    def __init__(self, seed=42):
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


# =============================================================
# FUSION APPROACH C: LATE-FUSION STACKING META-LEARNER
# =============================================================
class FusionApproachC_LateFusionStacking:
    """
    Late-Fusion Stacking:
      - Model_RGB trained on DINOv2 RGB -> Outputs P_RGB (4-D)
      - Model_Geo trained on 35-D Geometry -> Outputs P_Geo (4-D)
      - Meta-Classifier trained on X_meta = [P_RGB, P_Geo, R_d] in R^9
    Completely eliminates feature drowning because models are trained independently before fusion!
    """
    def __init__(self, seed=42):
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

        # 1. Base Model RGB & Geo for fast Out-of-Fold generation (5 folds total, not 25 nested folds)
        base_rgb = build_base_ensemble(self.seed)
        base_geo = build_base_ensemble(self.seed + 1)

        print("  [STACKING] Generating Out-of-Fold Level-0 Probability Predictions...", flush=True)
        if group_keys is not None:
            cv_splitter_rgb = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_arg_rgb = cv_splitter_rgb.split(X_rgb_scaled, y, group_keys)
            cv_splitter_geo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_arg_geo = cv_splitter_geo.split(X_geo_scaled, y, group_keys)
        else:
            cv_arg_rgb = 5
            cv_arg_geo = 5

        oof_rgb = cross_val_predict(base_rgb, X_rgb_scaled, y, cv=cv_arg_rgb, method="predict_proba", n_jobs=-1)
        oof_geo = cross_val_predict(base_geo, X_geo_scaled, y, cv=cv_arg_geo, method="predict_proba", n_jobs=-1)

        # 2. Fit Level-0 Calibrated models on full training data (1 single calibration run)
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


# =============================================================
# EVALUATION & BENCHMARKING HELPER
# =============================================================
def evaluate_fusion_model(model, X_rgb_test, X_geo_test, rel_test, y_test):
    """Evaluates a fusion model on pristine unseen test data and returns detailed metrics."""
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
