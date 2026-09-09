"""

src/generalized_champion.py -- Generalized Champion Pipeline C Model Definition


Project: PS-1 Class-Agnostic Geometric Primitive Analysis

Defines the GeneralizedChampionPipelineC model architecture:
  - 48-D DINOv2 RGB Projection (StandardScaler + PCA)
  - 16-D Geometry Projection (RobustScaler + PCA with Top-3 CAD shifted features dropped)
  - Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RandomForest with 5-fold isotonic CV)
  - Calibrated Class Prior Re-weighting: W = [1.8, 1.0, 1.4, 0.45]

"""

import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2

CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
DROP_FEATURES = ["planar_surface_ratio", "curved_surface_ratio", "std_Nz"]
CLASS_WEIGHTS = [1.8, 1.0, 1.4, 0.45]


class GeneralizedChampionPipelineC:
    def __init__(self, n_rgb_comp=48, n_geo_comp=16, drop_features=None, class_weights=None, seed=42):
        self.n_rgb_comp = n_rgb_comp
        self.n_geo_comp = n_geo_comp
        self.drop_features = drop_features or DROP_FEATURES
        self.class_weights = class_weights or CLASS_WEIGHTS
        self.seed = seed

        self.scaler_rgb = StandardScaler()
        self.pca_rgb = PCA(n_components=n_rgb_comp, random_state=seed)

        self.scaler_geo = RobustScaler()
        self.pca_geo = PCA(n_components=n_geo_comp, random_state=seed)

        self.clf = None
        self.keep_geo_indices = []

    def _filter_geo(self, X_geo):
        if not self.keep_geo_indices:
            all_names = GEOMETRIC_FEATURE_NAMES_V2[:X_geo.shape[1]]
            self.keep_geo_indices = [i for i, name in enumerate(all_names) if name not in self.drop_features]
        return X_geo[:, self.keep_geo_indices]

    def fit(self, X_rgb, X_geo, y):
        # 1. RGB branch
        X_rgb_sc = self.scaler_rgb.fit_transform(X_rgb)
        Z_rgb = self.pca_rgb.fit_transform(X_rgb_sc)

        # 2. Geometry branch (filtered)
        X_geo_filt = self._filter_geo(X_geo)
        X_geo_sc = self.scaler_geo.fit_transform(X_geo_filt)
        actual_comp = min(self.n_geo_comp, X_geo_filt.shape[1])
        self.pca_geo = PCA(n_components=actual_comp, random_state=self.seed)
        Z_geo = self.pca_geo.fit_transform(X_geo_sc)

        Z_fused = np.hstack([Z_rgb, Z_geo])

        # 3. Soft-Voting Calibrated Ensemble
        rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=4,
                                    max_features="sqrt", class_weight="balanced", random_state=self.seed, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=150, max_depth=12, min_samples_leaf=4,
                                   max_features="sqrt", class_weight="balanced", random_state=self.seed, n_jobs=-1)
        hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=6, min_samples_leaf=15,
                                             class_weight="balanced", random_state=self.seed)

        base_ensemble = VotingClassifier(
            estimators=[("et", et), ("hgb", hgb), ("rf", rf)],
            voting="soft"
        )

        self.clf = CalibratedClassifierCV(estimator=base_ensemble, cv=5, method="isotonic")
        self.clf.fit(Z_fused, y)
        return self

    def transform(self, X_rgb, X_geo):
        X_rgb_sc = self.scaler_rgb.transform(X_rgb)
        Z_rgb = self.pca_rgb.transform(X_rgb_sc)

        X_geo_filt = self._filter_geo(X_geo)
        X_geo_sc = self.scaler_geo.transform(X_geo_filt)
        Z_geo = self.pca_geo.transform(X_geo_sc)
        return np.hstack([Z_rgb, Z_geo])

    def predict_proba(self, X_rgb, X_geo, class_weights=None):
        Z = self.transform(X_rgb, X_geo)
        probs = self.clf.predict_proba(Z)
        w = class_weights if class_weights is not None else self.class_weights
        if w is not None:
            w_arr = np.array(w, dtype=np.float32)
            probs = probs * w_arr
            probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-7)
        return probs

    def predict(self, X_rgb, X_geo, class_weights=None):
        probs = self.predict_proba(X_rgb, X_geo, class_weights=class_weights)
        return np.argmax(probs, axis=1)
