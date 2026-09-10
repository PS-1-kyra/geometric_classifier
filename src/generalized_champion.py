"""

src/generalized_champion.py -- Generalized Champion Pipeline C Model Definition


Project: PS-1 Class-Agnostic Geometric Primitive Analysis

This module defines the GeneralizedChampionPipelineC model architecture,
which is the CHAMPION model for sim-to-real transfer learning. It was designed
to train on synthetic BlenderProc-generated data and generalize to real-world
retail shelf imagery without fine-tuning.

Architecture Overview:
  1. Feature Masking: Drop 3 domain-shifted features that behave differently
     in synthetic (perfect CAD) vs. real (noisy depth) data
  2. Dual-Branch PCA Projection:
     - RGB Branch:  768-D DINOv2 → StandardScaler → PCA(48-D)
     - Geo Branch:  32-D filtered → RobustScaler → PCA(16-D)
     - Fused: [48-D RGB, 16-D Geo] = 64-D balanced vector
  3. Calibrated Soft-Voting Ensemble:
     - ExtraTrees (150 trees, max_depth=12)
     - HistGradientBoosting (150 iters, max_depth=6)
     - RandomForest (150 trees, max_depth=12)
     - Wrapped in CalibratedClassifierCV(method='isotonic', cv=5)
  4. Calibrated Class Prior Re-weighting: W = [1.8, 1.0, 1.4, 0.45]

Design Rationale:
  - 48-D RGB (vs 32-D in real pipeline): DINOv2 visual features transfer well
    across domains, so we preserve more visual information.
  - 16-D Geo (vs 32-D in real pipeline): Geometric features have more domain
    gap, so we compress them more aggressively to reduce overfitting to
    synthetic geometry artifacts.
  - Feature masking removes features where synthetic-to-real domain shift is
    most severe (planar/curved surface ratios and Z-normal variance).

"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import numpy as np  # Core numerical computing

# scikit-learn components for the ML pipeline:
from sklearn.decomposition import PCA              # Principal Component Analysis for dimensionality reduction
from sklearn.preprocessing import StandardScaler, RobustScaler  # Feature scaling
# StandardScaler: Removes mean, scales to unit variance. Best for normally-distributed features.
# RobustScaler: Uses median and IQR instead of mean/std. Resistant to outliers in geometric features.

from sklearn.ensemble import (
    ExtraTreesClassifier,              # Extremely Randomized Trees — faster than RF, more randomization
    HistGradientBoostingClassifier,    # Histogram-based Gradient Boosting — fast, handles class imbalance
    RandomForestClassifier,            # Random Forest — bagging ensemble of decision trees
    VotingClassifier                   # Soft-voting ensemble: averages probability outputs
)
from sklearn.calibration import CalibratedClassifierCV  # Isotonic/Platt calibration for reliable probabilities

# Import the ordered list of all 35 geometric feature names.
# This is needed to identify which features to drop by name → index mapping.
from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2

# =============================================================================
# CONSTANTS
# =============================================================================
# Canonical class label ordering used throughout the project.
# Index 0 = Flat, 1 = Cylindrical, 2 = Cuboid, 3 = Irregular
CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]

# Features to DROP before PCA projection.
# These 3 features exhibit the strongest domain shift between synthetic and real:
#   planar_surface_ratio: In synthetic data, perfect CAD surfaces are truly planar.
#     In real data, monocular depth estimation introduces curvature noise on flat surfaces.
#   curved_surface_ratio: Complementary to planar_surface_ratio (same domain gap).
#   std_Nz: Standard deviation of Z-component of surface normals. Synthetic depth
#     maps are noiseless, giving very low std_Nz for flat surfaces. Real depth
#     estimation adds noise that inflates std_Nz unpredictably.
DROP_FEATURES = ["planar_surface_ratio", "curved_surface_ratio", "std_Nz"]

# Per-class prior re-weighting vector.
# Applied AFTER the calibrated ensemble produces raw probabilities.
# Purpose: Compensate for systematic mismatch between synthetic training
# distribution and real-world retail shelf class frequencies.
#   Flat (1.8):       Significantly up-weighted. Flat objects (boxes, trays) are
#                     common in real shelves but under-represented in synthetic data.
#   Cylindrical (1.0): No adjustment needed — well-balanced across domains.
#   Cuboid (1.4):     Moderately up-weighted. Second most common real-world class.
#   Irregular (0.45):  Heavily down-weighted. The model over-predicts "Irregular"
#                     because depth noise in real images mimics irregular geometry.
CLASS_WEIGHTS = [1.8, 1.0, 1.4, 0.45]


# =============================================================================
# MODEL CLASS
# =============================================================================
class GeneralizedChampionPipelineC:
    """
    Generalized Sim-to-Real Champion Classifier.

    This is a complete, self-contained ML pipeline that goes from raw features
    (768-D DINOv2 + 35-D Geometry) to calibrated 4-class probability predictions.

    The pipeline is designed for SERIALIZATION via pickle — all components
    (scalers, PCA projectors, ensemble classifier) are stored as instance
    attributes and saved/loaded together in a single .pkl file.

    Typical lifecycle:
      1. fit(X_rgb, X_geo, y)      — Train on synthetic data
      2. pickle.dump(model, f)      — Save to .pkl
      3. model = pickle.load(f)     — Load for inference
      4. model.predict(X_rgb, X_geo) — Classify real-world crops
    """

    def __init__(self, n_rgb_comp=48, n_geo_comp=16, drop_features=None, class_weights=None, seed=42):
        """
        Initialize the dual-branch projection and ensemble parameters.

        All components are created here but NOT fitted — fitting happens in fit().

        Args:
            n_rgb_comp (int): Number of PCA components for DINOv2 features.
                Default 48 retains ~95% variance of 768-D DINOv2 embeddings.
            n_geo_comp (int): Number of PCA components for geometric features.
                Default 16 for the 32 remaining features after dropping 3.
            drop_features (list[str]): Feature names to mask out.
                Default: ["planar_surface_ratio", "curved_surface_ratio", "std_Nz"]
            class_weights (list[float]): Per-class prior re-weighting factors.
                Applied after ensemble prediction. Default: [1.8, 1.0, 1.4, 0.45]
            seed (int): Random seed for reproducibility across all components.
        """
        self.n_rgb_comp = n_rgb_comp
        self.n_geo_comp = n_geo_comp
        self.drop_features = drop_features or DROP_FEATURES
        self.class_weights = class_weights or CLASS_WEIGHTS
        self.seed = seed

        # ── RGB Branch Components ────────────────────────────────────────
        # StandardScaler: Normalizes each of the 768 DINOv2 dimensions to
        # zero mean and unit variance. This is important because PCA is
        # sensitive to feature scales — without scaling, high-variance
        # dimensions would dominate the principal components.
        self.scaler_rgb = StandardScaler()
        # PCA: Reduces 768-D → 48-D by finding the directions of maximum
        # variance in the scaled feature space. These 48 components capture
        # the most discriminative visual patterns learned by DINOv2.
        self.pca_rgb = PCA(n_components=n_rgb_comp, random_state=seed)

        # ── Geometry Branch Components ───────────────────────────────────
        # RobustScaler: Uses median and interquartile range (IQR) instead of
        # mean and standard deviation. This makes it resistant to outliers,
        # which are common in geometric features extracted from noisy depth maps.
        self.scaler_geo = RobustScaler()
        # PCA: Reduces 32-D filtered geometry → 16-D compressed representation.
        self.pca_geo = PCA(n_components=n_geo_comp, random_state=seed)

        # ── Classifier ───────────────────────────────────────────────────
        self.clf = None              # Set during fit() — CalibratedClassifierCV instance
        self.keep_geo_indices = []   # Cached column indices after feature masking

    def _filter_geo(self, X_geo):
        """
        Removes domain-shifted features from the 35-D geometry vector.

        Implementation detail: On the first call, this method computes which
        column indices to keep by matching feature names against DROP_FEATURES.
        The result is cached in self.keep_geo_indices for subsequent calls.

        Args:
            X_geo (np.ndarray): Raw geometry features, shape (N, 35).

        Returns:
            np.ndarray: Filtered features, shape (N, 32).
                35 original - 3 dropped = 32 retained features.
        """
        # Lazy initialization: compute keep indices on first call
        if not self.keep_geo_indices:
            # Get feature names for the columns we have
            all_names = GEOMETRIC_FEATURE_NAMES_V2[:X_geo.shape[1]]
            # Keep only features NOT in the drop list
            self.keep_geo_indices = [i for i, name in enumerate(all_names) if name not in self.drop_features]
        # Advanced NumPy indexing: select only the kept columns
        return X_geo[:, self.keep_geo_indices]

    def fit(self, X_rgb, X_geo, y):
        """
        Fits the complete pipeline on training data.

        Training sequence:
          1. RGB Branch: fit StandardScaler → fit PCA → project to 48-D
          2. Geo Branch: filter features → fit RobustScaler → fit PCA → project to 16-D
          3. Concatenate to 64-D fused vector
          4. Train calibrated soft-voting ensemble on fused features

        The ensemble consists of 3 diverse base classifiers:
          - ExtraTrees: High randomization, fast training, good with many features
          - HistGradientBoosting: Sequential boosting, captures feature interactions
          - RandomForest: Bagging ensemble, reduces variance
        All use class_weight="balanced" to handle class imbalance internally.
        The ensemble is wrapped in CalibratedClassifierCV with isotonic calibration
        to ensure probability outputs are well-calibrated (not just accurate).

        Args:
            X_rgb (np.ndarray): DINOv2 feature matrix, shape (N_train, 768).
            X_geo (np.ndarray): Geometric feature matrix, shape (N_train, 35).
            y (np.ndarray): Class labels, shape (N_train,), values in [0, 3].

        Returns:
            self: For method chaining (e.g., model.fit(X, y).predict(X_test))
        """
        # ── Step 1: RGB Branch ───────────────────────────────────────────
        X_rgb_sc = self.scaler_rgb.fit_transform(X_rgb)  # Fit scaler on training data and transform
        Z_rgb = self.pca_rgb.fit_transform(X_rgb_sc)      # Fit PCA and project: (N, 768) → (N, 48)

        # ── Step 2: Geometry Branch ──────────────────────────────────────
        X_geo_filt = self._filter_geo(X_geo)              # Drop shifted features: (N, 35) → (N, 32)
        X_geo_sc = self.scaler_geo.fit_transform(X_geo_filt)  # Scale with RobustScaler
        # Dynamically adjust PCA components if fewer features than requested
        # (safety check: can't have more components than features)
        actual_comp = min(self.n_geo_comp, X_geo_filt.shape[1])
        self.pca_geo = PCA(n_components=actual_comp, random_state=self.seed)
        Z_geo = self.pca_geo.fit_transform(X_geo_sc)      # Project: (N, 32) → (N, 16)

        # ── Step 3: Fusion ───────────────────────────────────────────────
        # Horizontal concatenation: [48-D RGB | 16-D Geo] = 64-D fused vector
        Z_fused = np.hstack([Z_rgb, Z_geo])

        # ── Step 4: Calibrated Soft-Voting Ensemble ──────────────────────
        # Build 3 diverse base classifiers with complementary inductive biases:
        rf = RandomForestClassifier(
            n_estimators=150,           # Number of trees in the forest
            max_depth=12,               # Maximum tree depth (prevents overfitting)
            min_samples_leaf=4,         # Minimum samples per leaf (regularization)
            max_features="sqrt",        # Consider sqrt(64) ≈ 8 features per split
            class_weight="balanced",    # Auto-adjusts weights inversely proportional to class frequencies
            random_state=self.seed,
            n_jobs=-1                   # Use all CPU cores for parallel training
        )
        et = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1
        )
        hgb = HistGradientBoostingClassifier(
            max_iter=150,               # Number of boosting iterations
            max_depth=6,                # Shallower trees for gradient boosting (standard practice)
            min_samples_leaf=15,        # More conservative leaf size for boosting
            class_weight="balanced",
            random_state=self.seed
        )

        # Soft-voting ensemble: averages the predicted probability distributions
        # from all 3 base classifiers. This is better than hard voting because
        # it preserves probability information and produces smoother decisions.
        base_ensemble = VotingClassifier(
            estimators=[("et", et), ("hgb", hgb), ("rf", rf)],
            voting="soft"  # Average probabilities (vs "hard" = majority vote)
        )

        # Isotonic calibration: Fits a non-parametric monotonic function that
        # maps raw ensemble probabilities to calibrated probabilities.
        # 5-fold CV ensures the calibration is learned on held-out data.
        # After calibration, a predicted probability of 0.7 truly means ~70%
        # of samples with that score belong to the predicted class.
        self.clf = CalibratedClassifierCV(estimator=base_ensemble, cv=5, method="isotonic")
        self.clf.fit(Z_fused, y)
        return self

    def transform(self, X_rgb, X_geo):
        """
        Transforms raw features through the FITTED dual-branch pipeline.

        Uses the scalers and PCA projectors fitted during fit() to transform
        new (test/inference) data. Does NOT refit — applies the learned
        transformations from training data.

        Args:
            X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
            X_geo (np.ndarray): Geometric features, shape (N, 35).

        Returns:
            np.ndarray: Fused feature vector, shape (N, 64).
        """
        # Transform (not fit_transform!) — uses learned parameters
        X_rgb_sc = self.scaler_rgb.transform(X_rgb)
        Z_rgb = self.pca_rgb.transform(X_rgb_sc)

        X_geo_filt = self._filter_geo(X_geo)
        X_geo_sc = self.scaler_geo.transform(X_geo_filt)
        Z_geo = self.pca_geo.transform(X_geo_sc)
        return np.hstack([Z_rgb, Z_geo])

    def predict_proba(self, X_rgb, X_geo, class_weights=None):
        """
        Returns calibrated class probabilities with prior re-weighting.

        The re-weighting step adjusts the ensemble's raw probability output
        to compensate for class distribution mismatch between synthetic
        training data and real-world test distributions.

        Mathematical formulation:
          P_raw = CalibratedEnsemble(PCA(Scale(X_rgb)), PCA(Scale(filter(X_geo))))
          P_weighted = P_raw * W                    (element-wise multiplication)
          P_final = P_weighted / sum(P_weighted)     (re-normalization to sum=1)

        Args:
            X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
            X_geo (np.ndarray): Geometric features, shape (N, 35).
            class_weights (list[float], optional): Override default class weights.

        Returns:
            np.ndarray: Calibrated, re-weighted probability matrix, shape (N, 4).
                Each row sums to ~1.0.
        """
        Z = self.transform(X_rgb, X_geo)
        probs = self.clf.predict_proba(Z)

        # Apply per-class prior re-weighting
        w = class_weights if class_weights is not None else self.class_weights
        if w is not None:
            w_arr = np.array(w, dtype=np.float32)
            probs = probs * w_arr  # Broadcast: (N, 4) * (4,) → (N, 4)
            # Re-normalize rows to sum to 1.0
            # Adding 1e-7 epsilon prevents division-by-zero if all probs are 0
            probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-7)
        return probs

    def predict(self, X_rgb, X_geo, class_weights=None):
        """
        Returns hard class predictions (argmax of re-weighted probabilities).

        Args:
            X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
            X_geo (np.ndarray): Geometric features, shape (N, 35).
            class_weights (list[float], optional): Override default class weights.

        Returns:
            np.ndarray: Predicted class indices, shape (N,), values in {0, 1, 2, 3}.
        """
        probs = self.predict_proba(X_rgb, X_geo, class_weights=class_weights)
        return np.argmax(probs, axis=1)
