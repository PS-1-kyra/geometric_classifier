"""

synthetic_classifier.py -- Reusable Synthetic-Trained Classifier


Project: PS-1 Class-Agnostic Geometric Primitive Analysis

Reusable inference interface for Pranaya's Synthetic-Trained Classifier
(Feature-Masked + Calibrated Prior Re-weighting).

Architecture:
  - Visual Backbone: Pretrained DINOv2 ViT-S/14 (768-D: 384-D CLS + 384-D Mean Patch)
  - Geometric Backbone: Depth Anything V2 + 35-D Geometry Extractor V2
  - Feature Masking: Eliminates uninformative depth artifacts (planar_surface_ratio, etc.)
  - Classifier: Calibrated Soft-Voting Ensemble (ExtraTrees + HistGB + RF) with Prior Re-weighting
  - Uncertainty Threshold: Confidence < 0.40 -> uncertain

Usage:
  from synthetic_classifier import SyntheticClassifier, CLASS_NAMES

  classifier = SyntheticClassifier()
  result = classifier.predict(rgb_crop, depth_crop=None, mask=None)
  print(result["class"], result["confidence"], result["probabilities"])

"""

import os
import sys
import pickle
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Setup Paths relative to this file
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOTS = [
    FILE_DIR,
    FILE_DIR.parent,
    FILE_DIR / "src",
    FILE_DIR.parent / "retail_geometry_project",
    FILE_DIR.parent / "synthetic_pipeline",
]
for p in PROJECT_ROOTS:
    p_str = str(p)
    if p.is_dir() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Import Geometry Extractor and Model Definition
from src.feature_extractor_v2 import extract_geometry_features_v2, GeometricFeatureExtractorV2, GEOMETRIC_FEATURE_NAMES_V2
try:
    from src.generalized_champion import GeneralizedChampionPipelineC, CLASS_NAMES, CLASS_WEIGHTS, DROP_FEATURES
except ImportError:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler

    CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
    CLASS_WEIGHTS = [1.8, 1.0, 1.4, 0.45]
    DROP_FEATURES = ["planar_surface_ratio", "curved_surface_ratio", "std_Nz"]

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

    if "src.generalized_champion" not in sys.modules:
        mod = types.ModuleType("src.generalized_champion")
        mod.GeneralizedChampionPipelineC = GeneralizedChampionPipelineC
        mod.CLASS_NAMES = CLASS_NAMES
        mod.CLASS_WEIGHTS = CLASS_WEIGHTS
        mod.DROP_FEATURES = DROP_FEATURES
        if "src" not in sys.modules:
            sys.modules["src"] = types.ModuleType("src")
        sys.modules["src.generalized_champion"] = mod

# Default Model Candidate Paths
DEFAULT_MODEL_CANDIDATES = [
    FILE_DIR / "synthetic_model.pkl",
    FILE_DIR / "Pranaya_to_Nimesh" / "synthetic_model.pkl",
    FILE_DIR.parent / "Pranaya_to_Nimesh" / "synthetic_model.pkl",
    FILE_DIR / "synthetic_pipeline" / "results" / "pipeline_c_synthetic_champion.pkl",
    FILE_DIR.parent / "synthetic_pipeline" / "results" / "pipeline_c_synthetic_champion.pkl",
    FILE_DIR / "pipeline_c_synthetic_champion.pkl",
    FILE_DIR / "results" / "pipeline_c_synthetic_champion.pkl",
]

UNCERTAINTY_THRESHOLD = 0.40  # Confidence < 40% flagged as uncertain


class ModelUnpickler(pickle.Unpickler):
    """Custom unpickler that safely remaps legacy module paths to local src."""
    def find_class(self, module, name):
        remap = {
            "retail_geometry_project.src": "src",
            "synthetic_pipeline.src": "src",
            "retail_geometry_project": "src",
            "synthetic_pipeline": "src",
        }
        for old_prefix, new_prefix in remap.items():
            if module.startswith(old_prefix):
                module = module.replace(old_prefix, new_prefix)
        if module in ("fusion_engine", "pipeline_c", "generalized_champion", "feature_extractor", "feature_extractor_v2"):
            module = f"src.{module}"
        return super().find_class(module, name)


def safe_pickle_load(file_or_path):
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            return ModelUnpickler(f).load()
    return ModelUnpickler(file_or_path).load()


class SyntheticClassifier:
    """
    Pranaya's Synthetic-Trained Primitive Classifier.

    Accepts 15% context-padded RGB crops (and optional depth/masks), extracts
    DINOv2 ViT-S/14 768-D and 35-D Geometry Extractor V2 features, and outputs
    calibrated 4-class probabilities for Flat, Cylindrical, Cuboid, and Irregular.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initializes backbones (DINOv2 + Depth Anything V2) and loads the trained checkpoint once.

        Args:
            model_path (str or Path, optional): Custom path to synthetic_model.pkl.
            device (torch.device or str, optional): Device to run PyTorch inference on ('cuda' or 'cpu').
        """
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SyntheticClassifier] Initializing on device: {self.device}")


        # 1. Resolve & Load Model Bundle
        self.model_path = self._resolve_model_path(model_path)
        print(f"[SyntheticClassifier] Loading model checkpoint from: {self.model_path.name}")
        self.bundle = safe_pickle_load(self.model_path)

        self.model = self.bundle["model"]
        self.class_weights = self.bundle.get("class_weights", [1.8, 1.0, 1.4, 0.45])
        self.class_names = self.bundle.get("class_names", CLASS_NAMES)

        # 2. Load DINOv2 ViT-S/14 Backbone
        print("[SyntheticClassifier] Loading DINOv2 ViT-S/14 visual backbone...")
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.dinov2.eval()

        self.dino_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Load 35-D Geometric Feature Extractor
        print("[SyntheticClassifier] Loading Depth Anything V2 + 35-D Geometry Extractor...")
        self.geo_extractor = GeometricFeatureExtractorV2(device=self.device)
        print("[SyntheticClassifier] Ready for inference.\n")

    def _resolve_model_path(self, model_path):
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"[SyntheticClassifier] Provided model_path not found: {model_path}")

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"[SyntheticClassifier] No synthetic model checkpoint found in default locations:\n"
            + "\n".join([f"  - {c}" for c in DEFAULT_MODEL_CANDIDATES])
        )

    def extract_dino_features(self, rgb_crop):
        """Extracts 768-D representation (384-D CLS + 384-D Mean Spatial Patch Tokens)."""
        pil_crop = Image.fromarray(rgb_crop)
        t_crop = self.dino_transform(pil_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                features_dict = self.dinov2.forward_features(t_crop)
                cls_token = features_dict["x_norm_clstoken"].squeeze(0).cpu().numpy()
                patch_tokens = features_dict["x_norm_patchtokens"].squeeze(0).cpu().numpy()
                mean_patch = np.mean(patch_tokens, axis=0)
                dino_768 = np.concatenate([cls_token, mean_patch]).astype(np.float32)
            except Exception:
                cls_out = self.dinov2(t_crop).squeeze(0).cpu().numpy()
                dino_768 = np.concatenate([cls_out, cls_out]).astype(np.float32)

        return dino_768

    def extract_geometry_features(self, rgb_crop, depth_crop=None, mask=None):
        """Extracts 35-D Geometric features and depth reliability."""
        geo_dict, geo_vec, rel, clean_depth, eroded_mask = self.geo_extractor.extract_features_and_reliability(
            rgb_crop, raw_depth=depth_crop, raw_mask=mask
        )
        return geo_vec, rel, clean_depth, eroded_mask, geo_dict

    def predict(self, rgb_crop, depth_crop=None, mask=None):
        """
        Runs full multimodal inference on a single 15% context-padded object crop.

        Args:
            rgb_crop (np.ndarray): RGB crop of the detected object (H, W, 3), uint8.
            depth_crop (np.ndarray, optional): Metric depth crop (H, W), float32. If None, auto-predicted.
            mask (np.ndarray, optional): Binary segmentation mask (H, W), uint8 or bool.

        Returns:
            dict: Structured prediction dictionary:
                {
                    "class": "Cuboid",
                    "class_idx": 2,
                    "confidence": 0.87,
                    "probabilities": {
                        "Flat": 0.02,
                        "Cylindrical": 0.05,
                        "Cuboid": 0.87,
                        "Irregular": 0.06
                    },
                    "uncertain": False
                }
        """
        if isinstance(rgb_crop, Image.Image):
            rgb_crop = np.array(rgb_crop.convert("RGB"))
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)

        if rgb_crop is None or rgb_crop.size == 0 or rgb_crop.shape[0] < 5 or rgb_crop.shape[1] < 5:
            raise ValueError("[SyntheticClassifier] Invalid or empty rgb_crop passed to predict().")

        # 1. Feature Extraction
        dino_768 = self.extract_dino_features(rgb_crop)
        geo_35, rel, clean_depth, eroded_mask, geo_dict = self.extract_geometry_features(rgb_crop, depth_crop, mask)

        # 2. Model Prediction
        probs = self.model.predict_proba(
            dino_768.reshape(1, -1),
            geo_35.reshape(1, -1),
            class_weights=self.class_weights
        )[0]

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        pred_label = self.class_names[class_idx]
        is_uncertain = bool(confidence < UNCERTAINTY_THRESHOLD)

        prob_dict = {
            self.class_names[k]: round(float(probs[k]), 4)
            for k in range(len(self.class_names))
        }

        return {
            "class": pred_label,
            "class_idx": class_idx,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "uncertain": is_uncertain
        }



# Standalone Unit Test

if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING SYNTHETIC CLASSIFIER INFERENCE INTERFACE")
    print("=" * 70)

    # Instantiate classifier
    clf = SyntheticClassifier()

    # Create dummy synthetic crop (simulate a product crop)
    dummy_crop = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)

    # Test prediction
    result = clf.predict(dummy_crop)

    print("\nPrediction Result:")
    print(f"  * Class        : {result['class']} (index {result['class_idx']})")
    print(f"  * Confidence   : {result['confidence']*100:.2f}%")
    print(f"  * Probabilities: {result['probabilities']}")
    print(f"  * Uncertain    : {result['uncertain']}")

    # If sample crop exists, test on it
    candidate_test_imgs = [
        FILE_DIR / "sample_crop.png",
        FILE_DIR.parent / "retail_geometry_project" / "dataset" / "0_flat" / "crop_0018.png",
        FILE_DIR / "retail_geometry_project" / "dataset" / "0_flat" / "crop_0018.png"
    ]
    test_img_path = next((p for p in candidate_test_imgs if p.exists()), None)
    if test_img_path:
        img_bgr = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        real_res = clf.predict(img_rgb)
        print(f"\nReal Crop Test ({test_img_path.name}):")
        print(f"  * Predicted    : {real_res['class']} (Conf: {real_res['confidence']*100:.2f}%)")
        print(f"  * Probabilities: {real_res['probabilities']}")
        print(f"  * Uncertain    : {real_res['uncertain']}")

    print("\n[OK] SyntheticClassifier test passed successfully!")
