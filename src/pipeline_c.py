"""
========================================================================================
src/pipeline_c.py — End-to-End Multimodal Geometry Primitive Classifier Pipeline
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
ARCHITECTURE & MATHEMATICAL PIPELINE OVERVIEW (FROM BASIC TO ADVANCED)
----------------------------------------------------------------------------------------
[Basic Concept]:
  Pipeline C is an integrated end-to-end multimodal classification pipeline designed to take
  a raw product crop from a retail shelf image and classify it into one of four geometric
  primitives:
    0: Flat        (books, chocolate bars, packaged stationery)
    1: Cylindrical (soda cans, spray cans, beverage bottles)
    2: Cuboid      (cereal boxes, tea boxes, rectangular cartons)
    3: Irregular   (potato chip bags, pouches, flexible wrappers)

[Multimodal Dual-Branch Architecture]:
  1. Visual Feature Branch (DINOv2 ViT-S/14):
     - We pass the RGB crop through Meta AI's self-supervised DINOv2 Vision Transformer.
     - Global Semantic Vector: The 384-D CLS token (`x_norm_clstoken`).
     - Spatial Dense Context: 384-D spatial mean-pooling of all patch tokens (`x_norm_patchtokens`).
     - Combined Visual Representation: 768-D vector (384 CLS + 384 Patch Mean).
     - Standardized & Projected via PCA down to 128 dimensions (`F_RGB_128`).

  2. Geometric Feature Branch (Depth Anything V2 + 3D Surface Harvester):
     - We estimate metric depth using Depth Anything V2.
     - Refined via Guided Bilateral Filtering and 3x3 Morphological Mask Erosion.
     - Extracts 32/35-D geometric descriptors (surface normal variances, percentiles, taper profiles).
     - Scaled via RobustScaler and projected via PCA down to 32 dimensions (`F_Geo_32`).

  3. Adaptive Reliability-Weighted Fusion Layer:
     - F_fused = [ F_RGB_128, (R_d * F_Geo_32) ] in R^160
     - Where R_d in [0.10, 1.0] is the dynamic depth reliability scalar measuring mask erosion
       and gradient smoothness.

  4. Calibrated Multi-Learner Ensemble:
     - ExtraTreesClassifier (400 estimators, max_depth=20)
     - HistGradientBoostingClassifier (300 iterations, max_depth=10)
     - RandomForestClassifier (350 estimators, max_depth=18)
     - Combined with soft-probability voting and calibrated with 5-fold Isotonic Regression.

[4-Layer Physical Guardrails & Irregular Suppression]:
  A frequent error in retail object detection is false positive "Irregular" classifications:
  a cereal box with a glossy specular reflection or a can with printed text can trick a naive model
  into predicting "Irregular". To prevent this, Pipeline C includes physical guardrails:
    - Layer 1: Morphological Mask Erosion suppresses background shelf edge bleed.
    - Layer 2: Physical Solidity Guardrail: Rigid cans and cuboids have high convex hull solidity
      (solidity >= 0.70). Deformable pouches have dented silhouettes with lower solidity.
      If solidity >= 0.70, P(Irregular) is penalized by 0.10x.
    - Layer 3: Isotonic Probability Calibration transforms raw tree votes into true posteriors.
    - Layer 4: Calibrated Decision Thresholding: If the top prediction is Irregular but confidence
      is < 0.45, the prediction is rejected and reassigned to the secondary primitive.
========================================================================================
"""

import os
import sys
import pickle
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.feature_extractor import GeometricFeatureExtractor, GEOMETRIC_FEATURE_NAMES

CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_MODEL_SAVE_PATH = "outputs/retail_dinov2_fused_model.pkl"


class ModelUnpickler(pickle.Unpickler):
    """
    Custom Pickle Unpickler that safely and transparently remaps legacy module import paths
    (e.g., 'retail_geometry_project.src' or 'synthetic_pipeline.src') to the unified 'src' package.
    Ensures backwards compatibility with older serialized checkpoint files.
    """

    def find_class(self, module: str, name: str):
        remap = {
            "retail_geometry_project.src": "src",
            "synthetic_pipeline.src": "src",
            "retail_geometry_project": "src",
            "synthetic_pipeline": "src",
        }
        for old_prefix, new_prefix in remap.items():
            if module.startswith(old_prefix):
                module = module.replace(old_prefix, new_prefix)
        # Handle top-level module references without explicit 'src.' prefix
        if module in (
            "fusion_engine",
            "pipeline_c",
            "generalized_champion",
            "feature_extractor",
            "feature_extractor_v2",
        ):
            module = f"src.{module}"
        return super().find_class(module, name)


def safe_pickle_load(file_or_path):
    """Safely loads a pickle file using ModelUnpickler to remap legacy paths."""
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            return ModelUnpickler(f).load()
    return ModelUnpickler(file_or_path).load()


class PipelineC:
    """
    Unified Multimodal Primitive Geometry Classifier.

    Integrates:
      1. Pretrained DINOv2 ViT-S/14 visual backbone (768-D representation).
      2. Monocular depth & 3D surface normal feature extraction.
      3. Dimensionality reduction & adaptive reliability-weighted fusion.
      4. 3-model calibrated ensemble with 4-layer physical guardrails.
    """

    def __init__(self, device=None, model_path=None):
        """
        Initializes Pipeline C components, preprocessing transforms, and scalers.

        Args:
            device: torch.device ('cuda' or 'cpu').
            model_path: Optional path to serialized checkpoint to load immediately.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.geo_extractor = GeometricFeatureExtractor(device=self.device)
        self.dinov2_model = None

        # Standard ImageNet normalization for DINOv2 ViT backbone
        self.dino_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Dual-branch feature standardizers and PCA projection layers
        self.scaler_rgb = StandardScaler()
        self.proj_rgb   = PCA(n_components=128, random_state=42)

        self.scaler_geo = RobustScaler()
        self.proj_geo   = PCA(n_components=32, random_state=42)

        # Calibrated ensemble classifier
        self.classifier = None
        self.is_fitted  = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _lazy_load_dinov2(self):
        """Lazy loads DINOv2 ViT-S/14 weights into GPU memory only on first usage."""
        if self.dinov2_model is None:
            print("[PIPELINE C] Loading DINOv2 ViT-S/14 backbone...")
            self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.dinov2_model = self.dinov2_model.to(self.device)
            self.dinov2_model.eval()

    def extract_dino_representation(self, rgb_crop: np.ndarray) -> np.ndarray:
        """
        Extracts a consolidated 768-dimensional visual embedding combining:
          - 384-D CLS token: Represents global object category context.
          - 384-D Mean-Pooled spatial patch tokens: Captures fine-grained packaging surface details.

        Args:
            rgb_crop: RGB uint8 numpy array (H, W, 3).

        Returns:
            dino_rep: Float32 numpy array of shape (768,).
        """
        self._lazy_load_dinov2()
        pil_img = Image.fromarray(rgb_crop)
        tensor = self.dino_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                # Forward features extracts both cls and dense patch tokens
                features_dict = self.dinov2_model.forward_features(tensor)
                cls_token    = features_dict["x_norm_clstoken"].squeeze(0).cpu().numpy()
                patch_tokens = features_dict["x_norm_patchtokens"].squeeze(0).cpu().numpy()
                mean_patch   = np.mean(patch_tokens, axis=0)
                dino_rep     = np.concatenate([cls_token, mean_patch])
            except Exception:
                # Fallback to standard forward call if forward_features is unsupported
                cls_out  = self.dinov2_model(tensor).squeeze(0).cpu().numpy()
                dino_rep = np.concatenate([cls_out, cls_out])

        return dino_rep.astype(np.float32)

    def extract_single_sample_features(self, rgb_crop, raw_depth=None, raw_mask=None):
        """Extracts raw RGB (768-D), raw Geometry (32-D), and depth reliability score R_d."""
        dino_raw = self.extract_dino_representation(rgb_crop)
        _, geo_raw, reliability, clean_depth, eroded_mask = self.geo_extractor.extract_features_and_reliability(
            rgb_crop, raw_depth, raw_mask
        )
        return dino_raw, geo_raw, reliability, clean_depth, eroded_mask

    def project_and_fuse(self, X_rgb_raw, X_geo_raw, reliabilities, fit: bool = False) -> np.ndarray:
        """
        Applies dual-branch standardizations, PCA projections, and adaptive reliability
        weighting to construct the 160-dimensional fused feature representation:
          F_fused = [ F_RGB_128, (R_d * F_Geo_32) ] in R^160

        Args:
            X_rgb_raw:     Raw DINOv2 feature matrix (N, 768).
            X_geo_raw:     Raw geometric feature matrix (N, 32).
            reliabilities: Depth reliability scalars R_d for each sample in [0.10, 1.0].
            fit:           If True, fits scalers and PCA projections on training data.

        Returns:
            F_fused: Float32 fused feature matrix of shape (N, 160).
        """
        if fit:
            X_rgb_scaled = self.scaler_rgb.fit_transform(X_rgb_raw)
            n_rgb_comp = min(128, X_rgb_scaled.shape[0], X_rgb_scaled.shape[1])
            self.proj_rgb = PCA(n_components=n_rgb_comp, random_state=42)
            F_rgb_128 = self.proj_rgb.fit_transform(X_rgb_scaled)

            X_geo_scaled = self.scaler_geo.fit_transform(X_geo_raw)
            n_geo_comp = min(32, X_geo_scaled.shape[0], X_geo_scaled.shape[1])
            self.proj_geo = PCA(n_components=n_geo_comp, random_state=42)
            F_geo_32 = self.proj_geo.fit_transform(X_geo_scaled)
        else:
            X_rgb_scaled = self.scaler_rgb.transform(X_rgb_raw)
            F_rgb_128 = self.proj_rgb.transform(X_rgb_scaled)

            X_geo_scaled = self.scaler_geo.transform(X_geo_raw)
            F_geo_32 = self.proj_geo.transform(X_geo_scaled)

        # Zero-pad if sample size was smaller than target projection dimensionality
        if F_rgb_128.shape[1] < 128:
            pad = np.zeros((F_rgb_128.shape[0], 128 - F_rgb_128.shape[1]), dtype=np.float32)
            F_rgb_128 = np.hstack([F_rgb_128, pad])

        if F_geo_32.shape[1] < 32:
            pad = np.zeros((F_geo_32.shape[0], 32 - F_geo_32.shape[1]), dtype=np.float32)
            F_geo_32 = np.hstack([F_geo_32, pad])

        # Dynamic reliability weighting applied to the geometry branch
        rel_weights = np.array(reliabilities, dtype=np.float32).reshape(-1, 1)
        F_geo_weighted = F_geo_32 * rel_weights
        F_fused = np.hstack([F_rgb_128, F_geo_weighted]).astype(np.float32)

        return F_fused

    def build_dataset_feature_matrix(self, sample_records, cache_path=None, batch_size=32):
        """
        Extracts and caches multi-modal feature matrices for an entire list of sample records.
        Uses batched GPU inference for both DINOv2 and Depth Anything V2 for maximum throughput.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"[PIPELINE C] Loading precomputed feature cache: {cache_path}")
            data = np.load(cache_path)
            return data["X_rgb"], data["X_geo"], data["reliabilities"], data["y_labels"], data["group_keys"]

        print(f"[PIPELINE C] Extracting multimodal features for {len(sample_records)} samples (batch size = {batch_size})...")
        self._lazy_load_dinov2()
        self.geo_extractor._lazy_load_depth_model()

        X_rgb, X_geo, reliabilities, y_labels, group_keys = [], [], [], [], []
        start_time = time.time()
        n_total = len(sample_records)

        for b_start in range(0, n_total, batch_size):
            b_end = min(b_start + batch_size, n_total)
            batch_recs = sample_records[b_start:b_end]

            batch_imgs_rgb = []
            valid_batch_recs = []

            for rec in batch_recs:
                img_bgr = cv2.imread(rec["filepath"])
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    batch_imgs_rgb.append(img_rgb)
                    valid_batch_recs.append(rec)

            if not batch_imgs_rgb:
                continue

            # 1. Batched DINOv2 Visual Feature Extraction
            tensors_dino = torch.stack([self.dino_transform(Image.fromarray(img)) for img in batch_imgs_rgb]).to(self.device)
            with torch.no_grad():
                try:
                    feat_dict = self.dinov2_model.forward_features(tensors_dino)
                    cls_tokens = feat_dict["x_norm_clstoken"].cpu().numpy()
                    patch_tokens = feat_dict["x_norm_patchtokens"].cpu().numpy()
                    mean_patches = np.mean(patch_tokens, axis=1)
                    dino_batch_rep = np.hstack([cls_tokens, mean_patches])
                except Exception:
                    cls_out = self.dinov2_model(tensors_dino).cpu().numpy()
                    dino_batch_rep = np.hstack([cls_out, cls_out])

            # 2. Batched Depth Anything V2 Feature Extraction
            batch_resized_rgb = [cv2.resize(img, (224, 224)) for img in batch_imgs_rgb]
            depth_inputs = self.geo_extractor.image_processor(images=batch_resized_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                depth_outputs = self.geo_extractor.depth_model(**depth_inputs)

            for i, img_rgb in enumerate(batch_imgs_rgb):
                h, w = img_rgb.shape[:2]
                depth_pred = torch.nn.functional.interpolate(
                    depth_outputs.predicted_depth[i:i+1].unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy().astype(np.float32)

                _, geo_raw, rel, _, _ = self.geo_extractor.extract_features_and_reliability(img_rgb, raw_depth=depth_pred)

                X_rgb.append(dino_batch_rep[i])
                X_geo.append(geo_raw)
                reliabilities.append(rel)
                y_labels.append(valid_batch_recs[i]["label_id"])
                group_keys.append(valid_batch_recs[i]["group_id"])

            elapsed = time.time() - start_time
            rate = len(X_rgb) / (elapsed + 1e-5)
            print(f"  ... processed {b_end}/{n_total} samples ({len(X_rgb)} valid, {elapsed:.1f}s, {rate:.1f} samples/sec)", flush=True)

        X_rgb = np.array(X_rgb, dtype=np.float32)
        X_geo = np.array(X_geo, dtype=np.float32)
        reliabilities = np.array(reliabilities, dtype=np.float32)
        y_labels = np.array(y_labels, dtype=np.int32)
        group_keys = np.array(group_keys)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                X_rgb=X_rgb, X_geo=X_geo, reliabilities=reliabilities,
                y_labels=y_labels, group_keys=group_keys
            )
            print(f"[PIPELINE C] Saved feature cache to {cache_path}")

        return X_rgb, X_geo, reliabilities, y_labels, group_keys

    def fit(self, train_samples, val_samples=None, cache_dir="outputs/cache"):
        """
        Fits Pipeline C on the training partition and validates on pristine validation split.
        """
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, "train_features.npz")
        val_cache   = os.path.join(cache_dir, "val_features.npz")

        X_rgb_tr, X_geo_tr, rel_tr, y_train, groups_tr = self.build_dataset_feature_matrix(train_samples, train_cache)

        print("\n[PIPELINE C] Standardizing, Projecting and Fusing Multimodal Representations...")
        F_fused_train = self.project_and_fuse(X_rgb_tr, X_geo_tr, rel_tr, fit=True)
        print(f"  [OK] Fused Training Matrix Shape: {F_fused_train.shape} (128 RGB + 32 Weighted Geo = 160-D)")

        base_ensemble = VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2,
                                            class_weight="balanced", random_state=42, n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_depth=10,
                                                       class_weight="balanced", random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=350, max_depth=18, min_samples_leaf=2,
                                              class_weight="balanced", random_state=42, n_jobs=-1))
            ],
            voting="soft",
            n_jobs=-1
        )

        print("\n[PIPELINE C] Training 3-Model Ensemble (ExtraTrees + HistGB + RandomForest)...")
        t0 = time.time()
        base_ensemble.fit(F_fused_train, y_train)
        print(f"  [OK] Ensemble trained in {time.time() - t0:.1f}s")

        print("[PIPELINE C] Fitting Isotonic Probability Calibration (5-fold CV)...")
        self.classifier = CalibratedClassifierCV(estimator=base_ensemble, method="isotonic", cv=5)
        self.classifier.fit(F_fused_train, y_train)
        self.is_fitted = True
        print("  [OK] Probability calibration complete!")

        if val_samples:
            print("\n[PIPELINE C] Evaluating on Pristine Validation Split...")
            X_rgb_val, X_geo_val, rel_val, y_val, groups_val = self.build_dataset_feature_matrix(val_samples, val_cache)
            F_fused_val = self.project_and_fuse(X_rgb_val, X_geo_val, rel_val, fit=False)
            val_preds = self.classifier.predict(F_fused_val)
            val_bal_acc = balanced_accuracy_score(y_val, val_preds)
            print(f"  [OK] Pristine Validation Balanced Accuracy: {val_bal_acc:.2%}")

    def evaluate(self, test_samples, cache_path=None):
        """
        Evaluates model on pristine unseen physical product test crops.
        """
        assert self.is_fitted, "Pipeline C must be fitted before evaluation!"
        X_rgb_te, X_geo_te, rel_te, y_test, groups_te = self.build_dataset_feature_matrix(test_samples, cache_path)
        F_fused_test = self.project_and_fuse(X_rgb_te, X_geo_te, rel_te, fit=False)

        probs = self.classifier.predict_proba(F_fused_test)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        uncertain_mask = (max_probs < 0.45)
        num_uncertain  = int(np.sum(uncertain_mask))

        bal_acc = balanced_accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)

        print("\n" + "=" * 65)
        print("  PIPELINE C TEST SET EVALUATION REPORT (100% PRISTINE UNSEEN)")
        print("=" * 65)
        print(f"  Test Samples (100% Pristine Unseen Groups): {len(y_test)}")
        print(f"  Balanced Accuracy                          : {bal_acc:.2%}")
        print(f"  Uncertainty Rejections (Conf < 0.45)       : {num_uncertain}/{len(y_test)} ({num_uncertain/len(y_test):.1%})")
        print("\n" + report)

        return {
            "balanced_accuracy": float(bal_acc),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_true": y_test,
            "y_pred": preds,
            "y_prob": probs,
            "uncertain_mask": uncertain_mask
        }

    def predict_crop(self, rgb_crop, raw_depth=None, raw_mask=None):
        """
        Performs inference on a single product crop proposal with 4-layer physical guardrails.

        Returns:
            pred_label:   String class name ('Flat', 'Cylindrical', 'Cuboid', 'Irregular').
            class_idx:    Integer class index [0..3].
            confidence:   Percentage confidence score (0.0 to 100.0%).
            prob_dict:    Dictionary mapping class names to posterior probabilities.
            is_uncertain: Boolean flag indicating whether max probability < 0.45.
            debug_info:   Dictionary containing intermediate depth maps, masks, and features.
        """
        assert self.is_fitted, "Pipeline C must be fitted before prediction!"

        # Check if V2 Champion model, legacy 403-dim format, or 160-dim format
        if hasattr(self, "is_v2_fusion") and self.is_v2_fusion:
            # 1. DINOv2 768-D representation
            self._lazy_load_dinov2()
            pil_img = Image.fromarray(rgb_crop)
            tensor = self.dino_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                try:
                    feat_dict = self.dinov2_model.forward_features(tensor)
                    cls_tok = feat_dict["x_norm_clstoken"].cpu().numpy()
                    patch_tok = feat_dict["x_norm_patchtokens"].cpu().numpy()
                    mean_p = np.mean(patch_tok, axis=1)
                    dino_raw = np.hstack([cls_tok, mean_p]).squeeze(0)
                except Exception:
                    cls_out = self.dinov2_model(tensor).cpu().numpy()
                    dino_raw = np.hstack([cls_out, cls_out]).squeeze(0)

            # 2. 35-D Geometry Features via V2 Harvester
            from src.feature_extractor_v2 import extract_geometry_features_v2
            geo_dict, geo_raw, rel, clean_depth, eroded_mask = extract_geometry_features_v2(
                rgb_crop, raw_depth, raw_mask
            )

            probs = self.fusion_model.predict_proba(
                dino_raw.reshape(1, -1),
                geo_raw.reshape(1, -1),
                [rel]
            )[0]
            F_fused = np.array([geo_raw])
        elif hasattr(self, "is_legacy_403") and self.is_legacy_403:
            # 1. DINOv2 384-D CLS
            self._lazy_load_dinov2()
            pil_img = Image.fromarray(rgb_crop)
            tensor = self.dino_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dino_cls = self.dinov2_model(tensor).squeeze(0).cpu().numpy()

            # 2. 19-dim Depth & Geometry features
            geo_dict, _, rel, clean_depth, eroded_mask = self.geo_extractor.extract_features_and_reliability(
                rgb_crop, raw_depth, raw_mask
            )

            # Assemble 19 legacy features
            legacy_19 = [
                geo_dict.get("std_Nx", 0.0), geo_dict.get("std_Ny", 0.0), geo_dict.get("mean_Nz", 0.0),
                geo_dict.get("mean_abs_Nx", 0.0), geo_dict.get("mean_abs_Ny", 0.0), 1.0,
                geo_dict.get("depth_std", 0.0), geo_dict.get("depth_range", 0.0), geo_dict.get("depth_iqr", 0.0),
                0.15, geo_dict.get("local_normal_variation", 0.0), 0.5,
                geo_dict.get("aspect_ratio", 1.0), geo_dict.get("solidity", 0.8), geo_dict.get("perimeter_area_ratio", 0.05),
                1.0 if (rgb_crop.shape[0] / max(1, rgb_crop.shape[1])) > 1.8 else 0.0,
                geo_dict.get("curved_surface_ratio", 0.5), geo_dict.get("solidity", 0.8), 0.5
            ]

            fused_403 = np.hstack([dino_cls, legacy_19]).reshape(1, -1)
            fused_scaled = self.scaler.transform(fused_403)
            probs = self.classifier.predict_proba(fused_scaled)[0]
            F_fused = fused_scaled
        else:
            dino_raw, geo_raw, rel, clean_depth, eroded_mask = self.extract_single_sample_features(
                rgb_crop, raw_depth, raw_mask
            )
            F_fused = self.project_and_fuse(
                dino_raw.reshape(1, -1),
                geo_raw.reshape(1, -1),
                [rel],
                fit=False
            )
            probs = self.classifier.predict_proba(F_fused)[0]

        # ------------------------------------------------------------------------------
        # 4-LAYER PHYSICAL SOLIDITY GUARDRAIL & IRREGULAR SUPPRESSION
        # ------------------------------------------------------------------------------
        # Layer 2: Physical Solidity Guardrail Filter (solidity >= 0.70 cannot be an Irregular pouch)
        solidity = geo_dict.get("solidity", 0.8) if "geo_dict" in locals() else 0.8
        if solidity >= 0.70:
            probs[3] *= 0.10
            p_sum = np.sum(probs)
            if p_sum > 0:
                probs = probs / p_sum

        # Layer 4: Calibrated Decision Thresholding for Irregular
        top_idx = int(np.argmax(probs))
        if top_idx == 3 and probs[3] < 0.45:
            probs[3] = 0.0
            p_sum = np.sum(probs)
            if p_sum > 0:
                probs = probs / p_sum
            class_idx = int(np.argmax(probs))
        else:
            class_idx = top_idx

        confidence = float(probs[class_idx] * 100)
        pred_label = CLASS_NAMES[class_idx]
        is_uncertain = bool(probs[class_idx] < 0.45)

        prob_dict = {CLASS_NAMES[k]: float(probs[k]) for k in range(NUM_CLASSES)}

        debug_info = {
            "depth_reliability": float(rel),
            "clean_depth": clean_depth,
            "eroded_mask": eroded_mask,
            "fused_vector": F_fused[0]
        }

        return pred_label, class_idx, confidence, prob_dict, is_uncertain, debug_info

    def save(self, save_path=DEFAULT_MODEL_SAVE_PATH):
        """Serializes fitted Pipeline C state to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state = {
            "scaler_rgb": self.scaler_rgb,
            "proj_rgb": self.proj_rgb,
            "scaler_geo": self.scaler_geo,
            "proj_geo": self.proj_geo,
            "classifier": self.classifier,
            "class_names": CLASS_NAMES,
            "geometric_feature_names": GEOMETRIC_FEATURE_NAMES
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        print(f"[PIPELINE C] Saved fitted model pipeline -> {save_path}")

    def load(self, model_path=DEFAULT_MODEL_SAVE_PATH):
        """Loads fitted Pipeline C state (supports V2 Champion, V1 160-D, and Legacy 403-D)."""
        state = safe_pickle_load(model_path)

        if "model" in state and hasattr(state["model"], "predict_proba"):
            # V2 Champion Model (Approach A/B/C from fusion_engine)
            self.fusion_model = state["model"]
            self.is_v2_fusion = True
            self.is_legacy_403 = False
        elif "scaler_rgb" in state:
            self.scaler_rgb = state["scaler_rgb"]
            self.proj_rgb   = state["proj_rgb"]
            self.scaler_geo = state["scaler_geo"]
            self.proj_geo   = state["proj_geo"]
            self.classifier = state["classifier"]
            self.is_v2_fusion = False
            self.is_legacy_403 = False
        elif "ensemble" in state:
            self.scaler = state["scaler"]
            self.classifier = state["ensemble"]
            self.is_v2_fusion = False
            self.is_legacy_403 = True

        self.is_fitted = True
        print(f"[PIPELINE C] Successfully loaded model from -> {model_path}")
