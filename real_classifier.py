"""

real_classifier.py -- Reusable Real-Trained Classifier


Project: PS-1 Class-Agnostic Geometric Primitive Analysis

Reusable inference interface for Pranaya's Real-Trained Classifier
(Dimensionality-Equalized 64-D Projection + Calibrated Soft-Voting Ensemble).

Architecture:
  - Visual Backbone: Pretrained DINOv2 ViT-S/14 (768-D -> 128-D / 32-D PCA Projection)
  - Geometric Backbone: Depth Anything V2 + 35-D Geometry Extractor V2 (-> 32-D PCA Projection)
  - Fusion Strategy: Equalized 64-D / 160-D representation (50/50 visual vs geometric split)
  - Classifier: Calibrated Ensemble (ExtraTrees + HistGradientBoosting + RandomForest)
  - Guardrails: Physical Solidity Guardrail + Irregular Decision Thresholding
  - Uncertainty Threshold: Confidence < 0.45 -> uncertain

Usage:
  from real_classifier import RealClassifier, CLASS_NAMES

  classifier = RealClassifier()
  result = classifier.predict(rgb_crop, depth_crop=None, mask=local_mask)
  print(result["class"], result["confidence"], result["probabilities"])

"""

import os
import sys
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
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

# Import Pipeline C and classes
from src.pipeline_c import PipelineC, CLASS_NAMES, NUM_CLASSES
from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2
from src.fusion_engine import FusionApproachA_EqualizedProjection

# Default Model Candidate Paths
DEFAULT_MODEL_CANDIDATES = [
    FILE_DIR / "real_model.pkl",
    FILE_DIR / "geometric_classifier" / "real_model.pkl",
    FILE_DIR.parent / "geometric_classifier" / "real_model.pkl",
    FILE_DIR / "retail_geometry_project" / "outputs" / "retail_dinov2_fused_model.pkl",
    FILE_DIR.parent / "retail_geometry_project" / "outputs" / "retail_dinov2_fused_model.pkl",
    FILE_DIR / "outputs" / "retail_dinov2_fused_model.pkl",
    FILE_DIR / "retail_dinov2_fused_model.pkl",
]

UNCERTAINTY_THRESHOLD = 0.45  # Confidence < 45% flagged as uncertain


class RealClassifier:
    """
    Pranaya's Real-Trained Primitive Classifier.

    Accepts 15% context-padded RGB crops (and optional depth/masks), extracts
    DINOv2 ViT-S/14 embeddings and 35-D Geometry features, projects into an
    equalized representation, and outputs calibrated 4-class probabilities for
    Flat, Cylindrical, Cuboid, and Irregular.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initializes backbones and loads the real-trained Pipeline C checkpoint once.

        Args:
            model_path (str or Path, optional): Custom path to real_model.pkl.
            device (torch.device or str, optional): Device to run PyTorch inference on ('cuda' or 'cpu').
        """
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RealClassifier] Initializing on device: {self.device}")

        # Resolve Model Path
        self.model_path = self._resolve_model_path(model_path)
        print(f"[RealClassifier] Loading model checkpoint from: {self.model_path.name}")

        # Initialize underlying Pipeline C engine
        self.pipeline_c = PipelineC(device=self.device, model_path=str(self.model_path))
        self.class_names = CLASS_NAMES
        print("[RealClassifier] Ready for inference.\n")

    def _resolve_model_path(self, model_path):
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"[RealClassifier] Provided model_path not found: {model_path}")

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"[RealClassifier] No real model checkpoint found in default locations:\n"
            + "\n".join([f"  - {c}" for c in DEFAULT_MODEL_CANDIDATES])
        )

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
                    "confidence": 0.91,
                    "probabilities": {
                        "Flat": 0.01,
                        "Cylindrical": 0.03,
                        "Cuboid": 0.91,
                        "Irregular": 0.05
                    },
                    "uncertain": False,
                    "debug": {
                        "depth_reliability": 0.85,
                        "clean_depth": np.ndarray,
                        "eroded_mask": np.ndarray,
                        "fused_vector": np.ndarray
                    }
                }
        """
        if isinstance(rgb_crop, Image.Image):
            rgb_crop = np.array(rgb_crop.convert("RGB"))
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)

        if rgb_crop is None or rgb_crop.size == 0 or rgb_crop.shape[0] < 5 or rgb_crop.shape[1] < 5:
            raise ValueError("[RealClassifier] Invalid or empty rgb_crop passed to predict().")

        # Run inference via Pipeline C's predict_crop
        pred_label, class_idx, conf_pct, prob_dict, is_uncertain, debug_info = self.pipeline_c.predict_crop(
            rgb_crop, raw_depth=depth_crop, raw_mask=mask
        )

        # Normalize confidence to 0.0 - 1.0 scale
        confidence_val = float(conf_pct / 100.0) if conf_pct > 1.0 else float(conf_pct)
        is_uncertain = bool(confidence_val < UNCERTAINTY_THRESHOLD)

        # Format probability dictionary to clean rounded floats (0.0 to 1.0)
        formatted_probs = {
            k: round(float(v), 4) for k, v in prob_dict.items()
        }

        return {
            "class": pred_label,
            "class_idx": class_idx,
            "confidence": round(confidence_val, 4),
            "probabilities": formatted_probs,
            "uncertain": is_uncertain,
            "debug": {
                "depth_reliability": round(float(debug_info.get("depth_reliability", 0.0)), 4),
                "clean_depth": debug_info.get("clean_depth"),
                "eroded_mask": debug_info.get("eroded_mask"),
                "fused_vector": debug_info.get("fused_vector")
            }
        }



# Standalone Unit Test

if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING REAL CLASSIFIER INFERENCE INTERFACE")
    print("=" * 70)

    # Instantiate classifier
    clf = RealClassifier()

    # Create dummy crop (simulate a product crop)
    dummy_crop = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)

    # Test prediction
    result = clf.predict(dummy_crop)

    print("\nDummy Crop Prediction:")
    print(f"  * Class        : {result['class']} (index {result['class_idx']})")
    print(f"  * Confidence   : {result['confidence']*100:.2f}%")
    print(f"  * Probabilities: {result['probabilities']}")
    print(f"  * Uncertain    : {result['uncertain']}")
    print(f"  * Reliability  : {result['debug']['depth_reliability']}")

    # If real sample crop exists, test on it
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
        print(f"  * Reliability  : {real_res['debug']['depth_reliability']}")

    print("\n[OK] RealClassifier test passed successfully!")
