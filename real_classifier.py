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

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os       # OS-level path operations (checking file existence, etc.)
import sys      # System-level operations (modifying Python's module search path)
import pickle   # Serialization library for loading trained model checkpoints (.pkl files)
from pathlib import Path  # Object-oriented filesystem path handling (cross-platform)

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import cv2          # OpenCV — image loading (BGR format), color conversion, resizing
import numpy as np  # NumPy — core numerical computing (arrays, linear algebra)
import torch        # PyTorch — deep learning framework (GPU acceleration, DINOv2 backbone)
from PIL import Image  # Pillow — Python Imaging Library for PIL.Image format conversion

# =============================================================================
# DYNAMIC PATH CONFIGURATION
# =============================================================================
# Resolve the absolute path to this file's parent directory.
# This allows the script to locate sibling modules (src/) regardless of
# where it is invoked from (e.g., project root vs. scripts/ vs. notebooks/).
FILE_DIR = Path(__file__).resolve().parent

# Register multiple candidate directories on sys.path so that Python can
# discover and import the `src` package and its submodules.
# This supports running from:
#   - The geometric_classifier/ folder itself
#   - A parent directory (e.g., "kyra proj/")
#   - Legacy paths (retail_geometry_project/, synthetic_pipeline/)
PROJECT_ROOTS = [
    FILE_DIR,                                     # Current directory (geometric_classifier/)
    FILE_DIR.parent,                              # Parent directory (kyra proj/)
    FILE_DIR / "src",                             # Direct src/ access
    FILE_DIR.parent / "retail_geometry_project",  # Legacy real pipeline root
    FILE_DIR.parent / "synthetic_pipeline",       # Legacy synthetic pipeline root
]
for p in PROJECT_ROOTS:
    p_str = str(p)
    # Only add existing directories that aren't already in sys.path
    # to avoid duplicate entries and import confusion
    if p.is_dir() and p_str not in sys.path:
        sys.path.insert(0, p_str)  # insert(0, ...) gives these paths highest priority

# =============================================================================
# INTERNAL MODULE IMPORTS
# =============================================================================
# PipelineC: The core multimodal inference engine that handles:
#   - DINOv2 visual feature extraction (768-D)
#   - Depth Anything V2 geometric feature extraction (35-D)
#   - Feature fusion (equalized projection) and ensemble classification
# CLASS_NAMES: Canonical ordered label list ["Flat", "Cylindrical", "Cuboid", "Irregular"]
# NUM_CLASSES: Integer count of classes (4)
from src.pipeline_c import PipelineC, CLASS_NAMES, NUM_CLASSES

# GEOMETRIC_FEATURE_NAMES_V2: Ordered list of all 35 geometric feature names
# used by the V2 feature extractor (surface normals, depth stats, silhouette, etc.)
from src.feature_extractor_v2 import GEOMETRIC_FEATURE_NAMES_V2

# FusionApproachA_EqualizedProjection: The champion fusion model class.
# This is the class stored inside real_model.pkl — needed for pickle deserialization
# to reconstruct the model object.
from src.fusion_engine import FusionApproachA_EqualizedProjection

# =============================================================================
# DEFAULT MODEL SEARCH PATHS
# =============================================================================
# When no explicit model_path is provided, the classifier will search these
# candidate locations in order and use the first one that exists.
# This makes the classifier portable across different directory structures.
DEFAULT_MODEL_CANDIDATES = [
    FILE_DIR / "real_model.pkl",                                                  # Same directory as this script
    FILE_DIR / "geometric_classifier" / "real_model.pkl",                         # Nested geometric_classifier/
    FILE_DIR.parent / "geometric_classifier" / "real_model.pkl",                  # Sibling geometric_classifier/
    FILE_DIR / "retail_geometry_project" / "outputs" / "retail_dinov2_fused_model.pkl",   # Legacy real pipeline output
    FILE_DIR.parent / "retail_geometry_project" / "outputs" / "retail_dinov2_fused_model.pkl",  # Parent legacy path
    FILE_DIR / "outputs" / "retail_dinov2_fused_model.pkl",                       # Direct outputs/ folder
    FILE_DIR / "retail_dinov2_fused_model.pkl",                                   # Flat file in current dir
]

# =============================================================================
# CONFIDENCE THRESHOLD
# =============================================================================
# If the classifier's maximum class probability falls below this threshold,
# the prediction is flagged as "uncertain". This acts as a safety guardrail
# to prevent the system from making confident but wrong predictions on
# ambiguous or out-of-distribution inputs.
# Value rationale: 0.45 (45%) means the model must assign at least 45%
# probability to its top class, otherwise the prediction is unreliable.
UNCERTAINTY_THRESHOLD = 0.45


class RealClassifier:
    """
    Pranaya's Real-Trained Primitive Classifier.

    This class wraps the full multimodal inference pipeline (Pipeline C) into
    a simple, single-method interface. It handles:
      1. Model discovery and loading (with multi-path fallback)
      2. Input validation and format conversion (NumPy, PIL)
      3. Feature extraction (DINOv2 768-D + 35-D Geometry)
      4. Dimensionality-equalized fusion (64-D or 160-D fused vector)
      5. Calibrated ensemble classification (ExtraTrees + HistGB + RF)
      6. Post-processing: confidence normalization, uncertainty flagging

    Accepts 15% context-padded RGB crops (and optional depth/masks), extracts
    DINOv2 ViT-S/14 embeddings and 35-D Geometry features, projects into an
    equalized representation, and outputs calibrated 4-class probabilities for
    Flat, Cylindrical, Cuboid, and Irregular.

    Lifecycle:
      1. __init__() — loads model checkpoint + initializes neural backbones (one-time cost)
      2. predict()  — runs inference on a single crop (fast, can be called repeatedly)
    """

    def __init__(self, model_path=None, device=None):
        """
        Initializes backbones and loads the real-trained Pipeline C checkpoint once.

        This constructor performs three expensive operations that are cached for
        all subsequent predict() calls:
          1. Resolves the model .pkl file from candidate paths
          2. Loads the pre-trained Pipeline C engine (which internally loads
             DINOv2 ViT-S/14 and Depth Anything V2 neural networks)
          3. Stores the canonical class name mapping

        Args:
            model_path (str or Path, optional): Custom path to real_model.pkl.
                If None, searches DEFAULT_MODEL_CANDIDATES in order.
            device (torch.device or str, optional): Device to run PyTorch inference
                on ('cuda' or 'cpu'). If None, auto-detects CUDA availability.
                GPU is strongly recommended for production throughput.
        """
        # Auto-detect GPU availability if no device is specified.
        # CUDA (NVIDIA GPU) provides ~10-50x speedup for DINOv2 and Depth Anything V2.
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RealClassifier] Initializing on device: {self.device}")

        # Step 1: Resolve which .pkl file to load (supports multiple fallback paths)
        self.model_path = self._resolve_model_path(model_path)
        print(f"[RealClassifier] Loading model checkpoint from: {self.model_path.name}")

        # Step 2: Initialize the underlying Pipeline C inference engine.
        # PipelineC.__init__ will:
        #   - Load DINOv2 ViT-S/14 (pretrained Vision Transformer, ~21M params)
        #   - Load Depth Anything V2 (monocular depth estimator)
        #   - Load the trained scikit-learn ensemble from the .pkl checkpoint
        #   - Set up StandardScaler, PCA projectors, and CalibratedClassifierCV
        self.pipeline_c = PipelineC(device=self.device, model_path=str(self.model_path))

        # Store the canonical class names for output formatting
        self.class_names = CLASS_NAMES  # ["Flat", "Cylindrical", "Cuboid", "Irregular"]
        print("[RealClassifier] Ready for inference.\n")

    def _resolve_model_path(self, model_path):
        """
        Resolves the model checkpoint file path with multi-location fallback.

        Search strategy:
          1. If user provides an explicit path and it exists, use it immediately.
          2. Otherwise, iterate through DEFAULT_MODEL_CANDIDATES and return the
             first path that exists on disk.
          3. If no candidate exists, raise FileNotFoundError with all searched paths.

        Args:
            model_path: User-specified path (str/Path) or None for auto-discovery.

        Returns:
            Path: Resolved absolute path to the model .pkl file.

        Raises:
            FileNotFoundError: If no valid model checkpoint is found anywhere.
        """
        # Case 1: User provided an explicit path
        if model_path:
            p = Path(model_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"[RealClassifier] Provided model_path not found: {model_path}")

        # Case 2: Auto-discover from candidate locations
        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate

        # Case 3: Nothing found — provide helpful error listing all searched paths
        raise FileNotFoundError(
            f"[RealClassifier] No real model checkpoint found in default locations:\n"
            + "\n".join([f"  - {c}" for c in DEFAULT_MODEL_CANDIDATES])
        )

    def predict(self, rgb_crop, depth_crop=None, mask=None):
        """
        Runs full multimodal inference on a single 15% context-padded object crop.

        Processing Pipeline (executed in order):
          1. Input Validation & Format Conversion
             - Accepts both np.ndarray (H,W,3 uint8 RGB) and PIL.Image
             - Validates minimum dimensions (5x5 pixels)
          2. DINOv2 Visual Feature Extraction (768-D)
             - CLS token (384-D) + Mean-pooled spatial patches (384-D)
          3. Depth Anything V2 Geometric Feature Extraction (35-D)
             - Monocular depth estimation → surface normals, curvatures,
               depth statistics, silhouette descriptors
          4. Dimensionality-Equalized Fusion
             - RGB (768-D → 32-D via PCA) + Geo (35-D → 32-D via PCA) = 64-D
             - This prevents the 768-D DINOv2 features from "drowning" the
               35-D geometric features in tree-based classifiers
          5. Calibrated Ensemble Classification
             - ExtraTrees + HistGradientBoosting + RandomForest
             - Isotonic calibration ensures output probabilities are reliable
          6. Post-Processing
             - Physical Solidity Guardrail: suppresses "Irregular" for solid objects
             - Irregular Decision Thresholding: requires >45% confidence for Irregular
             - Confidence normalization to [0.0, 1.0] scale
             - Uncertainty flagging if max probability < UNCERTAINTY_THRESHOLD

        Args:
            rgb_crop (np.ndarray or PIL.Image): RGB crop of the detected object.
                Expected shape: (H, W, 3), dtype: uint8, color space: RGB.
                Should include ~15% context padding around the object bounding box.
            depth_crop (np.ndarray, optional): Pre-computed metric depth crop (H, W),
                dtype: float32. If None, Depth Anything V2 will auto-predict depth
                from the RGB crop (monocular depth estimation).
            mask (np.ndarray or PIL.Image, optional): Binary segmentation mask (H, W),
                dtype: uint8 or bool. If None, a conservative center mask (90% area)
                is auto-generated.

        Returns:
            dict: Structured prediction dictionary with the following keys:
                {
                    "class": str,           # Human-readable class name (e.g., "Cuboid")
                    "class_idx": int,        # Integer class index [0-3]
                    "confidence": float,     # Calibrated confidence in [0.0, 1.0]
                    "probabilities": dict,   # Per-class probability distribution
                    "uncertain": bool,       # True if confidence < 0.45
                    "debug": dict            # Internal diagnostics for inspection
                }

                Example:
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

        Raises:
            ValueError: If rgb_crop is None, empty, or smaller than 5x5 pixels.
        """
        # ── Input Format Conversion ──────────────────────────────────────
        # Accept PIL.Image inputs by converting to NumPy RGB arrays.
        # This makes the API more flexible (works with both OpenCV and Pillow).
        if isinstance(rgb_crop, Image.Image):
            rgb_crop = np.array(rgb_crop.convert("RGB"))
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)

        # ── Input Validation ─────────────────────────────────────────────
        # Guard against degenerate inputs that would crash feature extraction.
        # Minimum 5x5 pixels needed for meaningful surface normal computation.
        if rgb_crop is None or rgb_crop.size == 0 or rgb_crop.shape[0] < 5 or rgb_crop.shape[1] < 5:
            raise ValueError("[RealClassifier] Invalid or empty rgb_crop passed to predict().")

        # ── Core Inference via Pipeline C ────────────────────────────────
        # predict_crop() executes the full pipeline:
        #   DINOv2 extraction → Depth estimation → Geometry extraction →
        #   PCA projection → Fusion → Ensemble prediction → Guardrails
        # Returns: (label_str, class_idx, confidence_pct, prob_dict, uncertain_bool, debug_dict)
        pred_label, class_idx, conf_pct, prob_dict, is_uncertain, debug_info = self.pipeline_c.predict_crop(
            rgb_crop, raw_depth=depth_crop, raw_mask=mask
        )

        # ── Confidence Normalization ─────────────────────────────────────
        # Pipeline C returns confidence as percentage (0-100), but our API
        # contract is [0.0, 1.0]. Handle both cases defensively.
        confidence_val = float(conf_pct / 100.0) if conf_pct > 1.0 else float(conf_pct)

        # ── Uncertainty Detection ────────────────────────────────────────
        # Flag predictions where the model isn't confident enough.
        # This is critical for production safety — uncertain predictions
        # should trigger human review or fallback logic.
        is_uncertain = bool(confidence_val < UNCERTAINTY_THRESHOLD)

        # ── Probability Formatting ───────────────────────────────────────
        # Round probabilities to 4 decimal places for clean JSON serialization
        # and ensure all values are native Python floats (not NumPy types).
        formatted_probs = {
            k: round(float(v), 4) for k, v in prob_dict.items()
        }

        # ── Assemble Structured Output ───────────────────────────────────
        return {
            "class": pred_label,                    # e.g., "Cuboid"
            "class_idx": class_idx,                 # e.g., 2
            "confidence": round(confidence_val, 4), # e.g., 0.9134
            "probabilities": formatted_probs,       # Full 4-class distribution
            "uncertain": is_uncertain,              # True if low confidence
            "debug": {
                # Depth reliability score Rd ∈ [0.1, 1.0] — indicates how trustworthy
                # the monocular depth estimate is (lower = more noise/glare)
                "depth_reliability": round(float(debug_info.get("depth_reliability", 0.0)), 4),
                # Cleaned depth map after guided bilateral filtering
                "clean_depth": debug_info.get("clean_depth"),
                # Binary mask after 3x3 morphological erosion (boundary bleed removal)
                "eroded_mask": debug_info.get("eroded_mask"),
                # The final fused feature vector that was fed to the ensemble classifier
                "fused_vector": debug_info.get("fused_vector")
            }
        }



# =============================================================================
# STANDALONE UNIT TEST
# =============================================================================
# This block runs only when executing `python real_classifier.py` directly.
# It verifies that the classifier can:
#   1. Load the model checkpoint without errors
#   2. Process a dummy random crop through the full pipeline
#   3. Optionally test on a real sample crop if one exists on disk

if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING REAL CLASSIFIER INFERENCE INTERFACE")
    print("=" * 70)

    # Instantiate classifier (triggers model loading + backbone initialization)
    clf = RealClassifier()

    # Create a dummy random crop to verify the pipeline runs end-to-end.
    # Shape: (120, 100, 3) = Height 120px, Width 100px, 3 RGB channels.
    # Values: random uint8 [0, 255] — this won't produce meaningful predictions
    # but validates that all shapes, types, and operations work correctly.
    dummy_crop = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)

    # Run prediction on the dummy crop
    result = clf.predict(dummy_crop)

    # Print results in a structured format
    print("\nDummy Crop Prediction:")
    print(f"  * Class        : {result['class']} (index {result['class_idx']})")
    print(f"  * Confidence   : {result['confidence']*100:.2f}%")
    print(f"  * Probabilities: {result['probabilities']}")
    print(f"  * Uncertain    : {result['uncertain']}")
    print(f"  * Reliability  : {result['debug']['depth_reliability']}")

    # If a real sample crop exists on disk, test on it for more meaningful results.
    # This searches several candidate locations where real dataset crops might be.
    candidate_test_imgs = [
        FILE_DIR / "sample_crop.png",
        FILE_DIR.parent / "retail_geometry_project" / "dataset" / "0_flat" / "crop_0018.png",
        FILE_DIR / "retail_geometry_project" / "dataset" / "0_flat" / "crop_0018.png"
    ]
    # Use the first candidate that exists on disk (or None if none found)
    test_img_path = next((p for p in candidate_test_imgs if p.exists()), None)
    if test_img_path:
        # OpenCV loads in BGR format, so convert to RGB for our pipeline
        img_bgr = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        real_res = clf.predict(img_rgb)
        print(f"\nReal Crop Test ({test_img_path.name}):")
        print(f"  * Predicted    : {real_res['class']} (Conf: {real_res['confidence']*100:.2f}%)")
        print(f"  * Probabilities: {real_res['probabilities']}")
        print(f"  * Uncertain    : {real_res['uncertain']}")
        print(f"  * Reliability  : {real_res['debug']['depth_reliability']}")

    print("\n[OK] RealClassifier test passed successfully!")
