"""

synthetic_classifier.py -- Reusable Synthetic-Trained Classifier


Project: PS-1 Class-Agnostic Geometric Primitive Analysis

Reusable inference interface for Pranaya's Synthetic-Trained Classifier
(Feature-Masked + Calibrated Prior Re-weighting).

This classifier was trained entirely on BlenderProc-generated synthetic data
and is designed to generalize to real-world retail shelf imagery via
sim-to-real transfer learning techniques.

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

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os       # OS-level file/directory operations
import sys      # System-level operations (sys.path manipulation, module injection)
import pickle   # Python object serialization (for loading .pkl model checkpoints)
import types    # Dynamic module creation (used for fallback class registration)
from pathlib import Path  # Object-oriented filesystem paths (cross-platform)

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import cv2                    # OpenCV — image I/O and color conversion (BGR↔RGB)
import numpy as np            # NumPy — numerical arrays, linear algebra, statistics
import torch                  # PyTorch — deep learning framework, GPU acceleration
import torchvision.transforms as T  # Torchvision — image preprocessing transforms
                                    # (Resize, ToTensor, Normalize for DINOv2 input)
from PIL import Image         # Pillow — PIL.Image format for DINOv2's transform pipeline

# =============================================================================
# DYNAMIC PATH CONFIGURATION
# =============================================================================
# Resolve the absolute path to this file's parent directory.
# Path(__file__).resolve() gives the full filesystem path, .parent strips the filename.
FILE_DIR = Path(__file__).resolve().parent

# Register multiple candidate directories on sys.path so that `from src.xxx import yyy`
# works regardless of the working directory or invocation context.
# sys.path.insert(0, ...) gives these paths highest priority during import resolution.
PROJECT_ROOTS = [
    FILE_DIR,                                     # geometric_classifier/ itself
    FILE_DIR.parent,                              # Parent directory (kyra proj/)
    FILE_DIR / "src",                             # Direct src/ subpackage
    FILE_DIR.parent / "retail_geometry_project",  # Legacy real pipeline root
    FILE_DIR.parent / "synthetic_pipeline",       # Legacy synthetic pipeline root
]
for p in PROJECT_ROOTS:
    p_str = str(p)
    if p.is_dir() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# =============================================================================
# INTERNAL MODULE IMPORTS — Feature Extraction
# =============================================================================
# extract_geometry_features_v2: Convenience function that creates a global
#   GeometricFeatureExtractorV2 singleton and calls extract_features_and_reliability()
# GeometricFeatureExtractorV2: Class that wraps Depth Anything V2 + 35-D feature harvesting
# GEOMETRIC_FEATURE_NAMES_V2: Ordered list of all 35 geometric feature names
from src.feature_extractor_v2 import extract_geometry_features_v2, GeometricFeatureExtractorV2, GEOMETRIC_FEATURE_NAMES_V2

# =============================================================================
# INTERNAL MODULE IMPORTS — Model Definition (with Fallback)
# =============================================================================
# Try to import the GeneralizedChampionPipelineC class from the src package.
# This class definition is REQUIRED for pickle deserialization — Python's pickle
# needs the class definition in scope to reconstruct the serialized model object.
try:
    from src.generalized_champion import GeneralizedChampionPipelineC, CLASS_NAMES, CLASS_WEIGHTS, DROP_FEATURES
except ImportError:
    # ─── FALLBACK: Inline Class Definition ───────────────────────────
    # If the src.generalized_champion module can't be imported (e.g., missing
    # scikit-learn dependencies or broken paths), we define the essential class
    # inline so that pickle deserialization can still succeed.
    # This is a defensive measure for deployment environments where the full
    # src/ package might not be properly installed.

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler

    # Canonical class label ordering (matches both real and synthetic pipelines)
    CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]

    # Prior sensitivity re-weighting vector W = [1.8, 1.0, 1.4, 0.45]
    # These weights compensate for class prior imbalances between synthetic
    # training data and real-world retail shelf distributions:
    #   Flat (1.8):       Up-weighted — under-represented in synthetic data
    #   Cylindrical (1.0): Baseline — well-represented in both domains
    #   Cuboid (1.4):     Up-weighted — moderate synthetic under-representation
    #   Irregular (0.45): Down-weighted — over-predicted due to depth noise
    CLASS_WEIGHTS = [1.8, 1.0, 1.4, 0.45]

    # Features to drop during inference because they exhibit strong domain shift
    # between synthetic (perfect CAD geometry) and real (noisy depth estimation):
    #   planar_surface_ratio: Synthetic objects have perfect planar surfaces
    #   curved_surface_ratio: Complementary to planar — same domain gap
    #   std_Nz: Z-normal variance is artificially low in synthetic depth maps
    DROP_FEATURES = ["planar_surface_ratio", "curved_surface_ratio", "std_Nz"]

    class GeneralizedChampionPipelineC:
        """
        Fallback inline definition of the Generalized Champion model.

        This class implements the same interface as src.generalized_champion but
        is defined here as a safety net for pickle deserialization. The model
        architecture uses:
          1. Feature masking (drop CAD-shifted features)
          2. Dual-branch PCA projection (48-D RGB + 16-D Geometry)
          3. Calibrated class prior re-weighting

        This is NOT used for training — only for inference after loading a
        pre-trained model from a .pkl checkpoint.
        """
        def __init__(self, n_rgb_comp=48, n_geo_comp=16, drop_features=None, class_weights=None, seed=42):
            """
            Initialize dual-branch projection parameters.

            Args:
                n_rgb_comp (int): Number of PCA components for DINOv2 RGB features.
                    48-D was found optimal for sim-to-real transfer (vs 32-D or 64-D).
                n_geo_comp (int): Number of PCA components for geometric features.
                    16-D after dropping 3 shifted features from the original 35-D.
                drop_features (list): Feature names to mask out before PCA.
                class_weights (list): Per-class prior re-weighting factors.
                seed (int): Random seed for reproducibility.
            """
            self.n_rgb_comp = n_rgb_comp
            self.n_geo_comp = n_geo_comp
            self.drop_features = drop_features or DROP_FEATURES
            self.class_weights = class_weights or CLASS_WEIGHTS
            self.seed = seed

            # RGB branch: StandardScaler (zero-mean, unit-variance) → PCA projection
            self.scaler_rgb = StandardScaler()
            self.pca_rgb = PCA(n_components=n_rgb_comp, random_state=seed)

            # Geometry branch: RobustScaler (median/IQR-based, outlier-resistant) → PCA
            # RobustScaler is chosen over StandardScaler because geometric features
            # can have extreme outliers from depth estimation failures.
            self.scaler_geo = RobustScaler()
            self.pca_geo = PCA(n_components=n_geo_comp, random_state=seed)

            self.clf = None              # The trained ensemble classifier (set during fit())
            self.keep_geo_indices = []   # Column indices to keep after feature masking

        def _filter_geo(self, X_geo):
            """
            Removes domain-shifted features from the 35-D geometry vector.

            On first call, computes which column indices to keep by comparing
            feature names against DROP_FEATURES. Caches the result for speed.

            Args:
                X_geo (np.ndarray): Raw geometry features, shape (N, 35).

            Returns:
                np.ndarray: Filtered geometry features, shape (N, 32).
            """
            if not self.keep_geo_indices:
                all_names = GEOMETRIC_FEATURE_NAMES_V2[:X_geo.shape[1]]
                self.keep_geo_indices = [i for i, name in enumerate(all_names) if name not in self.drop_features]
            return X_geo[:, self.keep_geo_indices]

        def transform(self, X_rgb, X_geo):
            """
            Transforms raw features through the fitted dual-branch pipeline.

            Pipeline: Scale → PCA project → Concatenate
              RGB: (N, 768) → StandardScaler → PCA → (N, 48)
              Geo: (N, 35) → filter → (N, 32) → RobustScaler → PCA → (N, 16)
              Fused: (N, 48+16) = (N, 64)

            Args:
                X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
                X_geo (np.ndarray): Geometric features, shape (N, 35).

            Returns:
                np.ndarray: Fused feature vector, shape (N, 64).
            """
            # RGB branch
            X_rgb_sc = self.scaler_rgb.transform(X_rgb)
            Z_rgb = self.pca_rgb.transform(X_rgb_sc)
            # Geometry branch (with feature masking)
            X_geo_filt = self._filter_geo(X_geo)
            X_geo_sc = self.scaler_geo.transform(X_geo_filt)
            Z_geo = self.pca_geo.transform(X_geo_sc)
            # Concatenate for the final fused representation
            return np.hstack([Z_rgb, Z_geo])

        def predict_proba(self, X_rgb, X_geo, class_weights=None):
            """
            Returns calibrated class probabilities with optional prior re-weighting.

            The re-weighting step adjusts raw ensemble probabilities to compensate
            for class distribution mismatch between synthetic training data and
            real-world test distributions. After weighting, probabilities are
            re-normalized to sum to 1.0.

            Args:
                X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
                X_geo (np.ndarray): Geometric features, shape (N, 35).
                class_weights (list, optional): Per-class weights. Uses self.class_weights if None.

            Returns:
                np.ndarray: Re-weighted probability matrix, shape (N, 4).
            """
            Z = self.transform(X_rgb, X_geo)
            probs = self.clf.predict_proba(Z)
            # Apply prior re-weighting
            w = class_weights if class_weights is not None else self.class_weights
            if w is not None:
                w_arr = np.array(w, dtype=np.float32)
                probs = probs * w_arr  # Element-wise multiply each class probability by its weight
                # Re-normalize so rows sum to 1.0 (epsilon prevents division by zero)
                probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-7)
            return probs

        def predict(self, X_rgb, X_geo, class_weights=None):
            """
            Returns hard class predictions (argmax of re-weighted probabilities).

            Args:
                X_rgb (np.ndarray): DINOv2 features, shape (N, 768).
                X_geo (np.ndarray): Geometric features, shape (N, 35).
                class_weights (list, optional): Per-class weights.

            Returns:
                np.ndarray: Predicted class indices, shape (N,), values in [0, 3].
            """
            probs = self.predict_proba(X_rgb, X_geo, class_weights=class_weights)
            return np.argmax(probs, axis=1)

    # ─── Dynamic Module Registration ────────────────────────────────────
    # When pickle deserializes synthetic_model.pkl, it looks up the class via
    # `sys.modules["src.generalized_champion"].GeneralizedChampionPipelineC`.
    # If the real src.generalized_champion module failed to import, we inject
    # our fallback class into sys.modules so pickle can find it.
    if "src.generalized_champion" not in sys.modules:
        mod = types.ModuleType("src.generalized_champion")
        mod.GeneralizedChampionPipelineC = GeneralizedChampionPipelineC
        mod.CLASS_NAMES = CLASS_NAMES
        mod.CLASS_WEIGHTS = CLASS_WEIGHTS
        mod.DROP_FEATURES = DROP_FEATURES
        if "src" not in sys.modules:
            sys.modules["src"] = types.ModuleType("src")
        sys.modules["src.generalized_champion"] = mod

# =============================================================================
# DEFAULT MODEL SEARCH PATHS
# =============================================================================
# Ordered list of candidate paths to search when no explicit model_path is given.
# The classifier uses the first path that exists on disk.
DEFAULT_MODEL_CANDIDATES = [
    FILE_DIR / "synthetic_model.pkl",                                                        # Same directory
    FILE_DIR / "geometric_classifier" / "synthetic_model.pkl",                               # Nested
    FILE_DIR.parent / "geometric_classifier" / "synthetic_model.pkl",                        # Sibling
    FILE_DIR / "synthetic_pipeline" / "results" / "pipeline_c_synthetic_champion.pkl",       # Legacy synthetic
    FILE_DIR.parent / "synthetic_pipeline" / "results" / "pipeline_c_synthetic_champion.pkl",# Parent legacy
    FILE_DIR / "pipeline_c_synthetic_champion.pkl",                                          # Flat
    FILE_DIR / "results" / "pipeline_c_synthetic_champion.pkl",                              # Results subdir
]

# =============================================================================
# CONFIDENCE THRESHOLD
# =============================================================================
# Lower threshold than the real classifier (0.40 vs 0.45) because synthetic-trained
# models inherently have lower confidence on real-world data due to domain gap.
# Setting this too high would flag too many predictions as uncertain.
UNCERTAINTY_THRESHOLD = 0.40


# =============================================================================
# CUSTOM PICKLE UNPICKLER FOR LEGACY MODEL COMPATIBILITY
# =============================================================================
class ModelUnpickler(pickle.Unpickler):
    """
    Custom unpickler that safely remaps legacy module paths to local src.

    Background: Python's pickle serializes the full module path of each class
    (e.g., "synthetic_pipeline.src.generalized_champion.GeneralizedChampionPipelineC").
    When the directory structure changes (e.g., from synthetic_pipeline/ to
    geometric_classifier/), pickle can't find the class and crashes.

    Solution: Override find_class() to intercept module lookups and redirect
    legacy paths to the new unified `src` package.

    Remap table:
      "retail_geometry_project.src.X"  →  "src.X"   (real pipeline legacy)
      "synthetic_pipeline.src.X"       →  "src.X"   (synthetic pipeline legacy)
      "retail_geometry_project.X"      →  "src.X"   (alternative legacy format)
      "synthetic_pipeline.X"           →  "src.X"   (alternative legacy format)
      "fusion_engine"                  →  "src.fusion_engine"    (bare module name)
      "pipeline_c"                     →  "src.pipeline_c"       (bare module name)
      etc.
    """
    def find_class(self, module, name):
        """
        Intercepts pickle's class lookup and remaps legacy module paths.

        Args:
            module (str): The module path stored in the pickle file.
            name (str): The class/function name within that module.

        Returns:
            The resolved class object from the remapped module.
        """
        # Map old package prefixes to the new unified "src" package
        remap = {
            "retail_geometry_project.src": "src",
            "synthetic_pipeline.src": "src",
            "retail_geometry_project": "src",
            "synthetic_pipeline": "src",
        }
        for old_prefix, new_prefix in remap.items():
            if module.startswith(old_prefix):
                module = module.replace(old_prefix, new_prefix)

        # Handle bare module names (no package prefix) by adding "src."
        if module in ("fusion_engine", "pipeline_c", "generalized_champion", "feature_extractor", "feature_extractor_v2"):
            module = f"src.{module}"

        # Delegate to the standard unpickler with the remapped module path
        return super().find_class(module, name)


def safe_pickle_load(file_or_path):
    """
    Safely loads a pickle file using the custom ModelUnpickler.

    Accepts either a file path (str/Path) or an already-opened file object.
    Always uses ModelUnpickler to handle legacy module path remapping.

    Args:
        file_or_path: Path to pickle file (str/Path) or open file handle.

    Returns:
        The deserialized Python object (typically a dict containing the model).
    """
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as f:
            return ModelUnpickler(f).load()
    return ModelUnpickler(file_or_path).load()


# =============================================================================
# SYNTHETIC CLASSIFIER CLASS
# =============================================================================
class SyntheticClassifier:
    """
    Pranaya's Synthetic-Trained Primitive Classifier.

    This classifier was trained on BlenderProc-generated synthetic shelf scenes
    and is designed to generalize to real retail imagery without fine-tuning
    (zero-shot sim-to-real transfer).

    Key differences from RealClassifier:
      1. Feature Masking: Drops 3 CAD-shifted features (planar_surface_ratio,
         curved_surface_ratio, std_Nz) that have systematic domain gap
      2. PCA Dimensions: 48-D RGB + 16-D Geo (vs 32+32 in real pipeline)
         - More RGB dimensions because DINOv2 features transfer well
         - Fewer Geo dimensions because depth features have more domain gap
      3. Prior Re-weighting: Adjusts class priors to match real-world distribution
      4. Self-contained: Extracts DINOv2 features directly (doesn't use Pipeline C)
         because the synthetic model bundle stores the GeneralizedChampionPipelineC
         object which handles its own projection and classification

    Accepts 15% context-padded RGB crops (and optional depth/masks), extracts
    DINOv2 ViT-S/14 768-D and 35-D Geometry Extractor V2 features, and outputs
    calibrated 4-class probabilities for Flat, Cylindrical, Cuboid, and Irregular.
    """

    def __init__(self, model_path=None, device=None):
        """
        Initializes backbones (DINOv2 + Depth Anything V2) and loads the trained checkpoint once.

        Initialization sequence:
          1. Resolve and load the .pkl model bundle (contains GeneralizedChampionPipelineC
             object + class weights + metadata)
          2. Load DINOv2 ViT-S/14 backbone for 768-D visual feature extraction
          3. Load Depth Anything V2 for monocular depth estimation + 35-D geometry

        Args:
            model_path (str or Path, optional): Custom path to synthetic_model.pkl.
                If None, searches DEFAULT_MODEL_CANDIDATES in order.
            device (torch.device or str, optional): Device to run PyTorch inference on
                ('cuda' or 'cpu'). GPU strongly recommended for DINOv2 + DepthAnything.
        """
        # Auto-detect GPU (CUDA) availability for neural network inference
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SyntheticClassifier] Initializing on device: {self.device}")


        # ── Step 1: Resolve & Load Model Bundle ─────────────────────────
        # The .pkl bundle is a Python dict containing:
        #   "model": GeneralizedChampionPipelineC (fitted sklearn model)
        #   "class_weights": [1.8, 1.0, 1.4, 0.45]
        #   "class_names": ["Flat", "Cylindrical", "Cuboid", "Irregular"]
        #   "bacc_815": float (balanced accuracy on 815 real test crops)
        #   "transfer_ratio": float (sim-to-real transfer ratio %)
        self.model_path = self._resolve_model_path(model_path)
        print(f"[SyntheticClassifier] Loading model checkpoint from: {self.model_path.name}")
        self.bundle = safe_pickle_load(self.model_path)

        # Extract the core model and its configuration from the bundle
        self.model = self.bundle["model"]  # GeneralizedChampionPipelineC instance
        self.class_weights = self.bundle.get("class_weights", [1.8, 1.0, 1.4, 0.45])
        self.class_names = self.bundle.get("class_names", CLASS_NAMES)

        # ── Step 2: Load DINOv2 ViT-S/14 Visual Backbone ────────────────
        # DINOv2 (Distillation with No Labels v2) is a self-supervised Vision
        # Transformer trained by Meta/FAIR on 142M images.
        # ViT-S/14 = Small variant with 14×14 patch size (~21M parameters).
        # It produces a 384-D CLS token and 256 spatial patch tokens (each 384-D).
        # We concatenate CLS + mean(patches) = 768-D total representation.
        print("[SyntheticClassifier] Loading DINOv2 ViT-S/14 visual backbone...")
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.dinov2.eval()  # Set to evaluation mode (disables dropout, batch norm in eval mode)

        # Define the image preprocessing transform required by DINOv2:
        #   1. Resize to 224×224 (ViT's expected input resolution)
        #   2. Convert PIL Image to PyTorch tensor (HWC uint8 → CHW float32 [0,1])
        #   3. Normalize with ImageNet mean/std (the statistics DINOv2 was trained with)
        self.dino_transform = T.Compose([
            T.Resize((224, 224)),                                          # Bilinear resize
            T.ToTensor(),                                                  # [0,255] → [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],                       # ImageNet channel means
                        std=[0.229, 0.224, 0.225]),                        # ImageNet channel stds
        ])

        # ── Step 3: Load 35-D Geometric Feature Extractor ───────────────
        # GeometricFeatureExtractorV2 wraps:
        #   - Depth Anything V2 Small: Monocular depth estimation network
        #   - Surface normal computation (Sobel gradients on depth map)
        #   - 35-D feature harvesting (depth stats, normals, silhouette, etc.)
        print("[SyntheticClassifier] Loading Depth Anything V2 + 35-D Geometry Extractor...")
        self.geo_extractor = GeometricFeatureExtractorV2(device=self.device)
        print("[SyntheticClassifier] Ready for inference.\n")

    def _resolve_model_path(self, model_path):
        """
        Resolves the model checkpoint path with multi-location fallback.

        Searches DEFAULT_MODEL_CANDIDATES in order and returns the first match.
        Raises FileNotFoundError with all searched paths if nothing is found.

        Args:
            model_path: User-specified path or None for auto-discovery.

        Returns:
            Path: Absolute path to the .pkl model file.
        """
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
        """
        Extracts 768-D DINOv2 visual representation from an RGB crop.

        The representation combines two complementary signals:
          - CLS token (384-D): Global image-level summary, captures overall
            object appearance, category-level semantics
          - Mean Spatial Patch Tokens (384-D): Average of all 256 patch tokens,
            captures fine-grained texture, edge patterns, and local details

        Concatenating both gives a richer 768-D representation than either alone.

        Fallback: If forward_features() fails (older DINOv2 versions), falls back
        to standard forward() which only returns the CLS token (duplicated to 768-D).

        Args:
            rgb_crop (np.ndarray): RGB image crop, shape (H, W, 3), dtype uint8.

        Returns:
            np.ndarray: 768-D feature vector, dtype float32.
        """
        # Convert NumPy array to PIL Image (required by torchvision transforms)
        pil_crop = Image.fromarray(rgb_crop)
        # Apply preprocessing: resize → tensor → normalize, then add batch dim
        t_crop = self.dino_transform(pil_crop).unsqueeze(0).to(self.device)

        # Run forward pass with no gradient computation (inference only — saves memory)
        with torch.no_grad():
            try:
                # forward_features() returns a dict with separate CLS and patch tokens
                features_dict = self.dinov2.forward_features(t_crop)
                cls_token = features_dict["x_norm_clstoken"].squeeze(0).cpu().numpy()    # (384,)
                patch_tokens = features_dict["x_norm_patchtokens"].squeeze(0).cpu().numpy()  # (256, 384)
                mean_patch = np.mean(patch_tokens, axis=0)  # Average over spatial patches → (384,)
                dino_768 = np.concatenate([cls_token, mean_patch]).astype(np.float32)  # (768,)
            except Exception:
                # Fallback: use standard forward() which returns CLS-only (384-D)
                # Duplicate it to maintain 768-D dimensionality for downstream compatibility
                cls_out = self.dinov2(t_crop).squeeze(0).cpu().numpy()
                dino_768 = np.concatenate([cls_out, cls_out]).astype(np.float32)

        return dino_768

    def extract_geometry_features(self, rgb_crop, depth_crop=None, mask=None):
        """
        Extracts 35-D geometric features and depth reliability score.

        Processing pipeline:
          1. Monocular depth estimation via Depth Anything V2 (if no depth provided)
          2. Guided bilateral filtering (edge-preserving depth denoising)
          3. 3x3 morphological mask erosion (removes boundary artifacts)
          4. Object-normalized depth map computation
          5. 35-D feature extraction:
             - Surface normal statistics (9 features)
             - Depth distribution & percentiles (10 features)
             - Spatial asymmetry & taper profiles (5 features)
             - Silhouette & contour descriptors (8 features)
             - Composite interaction ratios (3 features)
          6. Depth reliability scoring Rd ∈ [0.1, 1.0]

        Args:
            rgb_crop (np.ndarray): RGB crop, shape (H, W, 3), uint8.
            depth_crop (np.ndarray, optional): Pre-computed depth map (H, W), float32.
            mask (np.ndarray, optional): Binary object mask (H, W).

        Returns:
            tuple: (geo_vec, reliability, clean_depth, eroded_mask, geo_dict)
                - geo_vec: 35-D feature vector (np.ndarray, float32)
                - reliability: Depth reliability score Rd ∈ [0.1, 1.0]
                - clean_depth: Filtered depth map after guided bilateral filtering
                - eroded_mask: Binary mask after 3x3 erosion
                - geo_dict: Dictionary mapping feature names to values
        """
        geo_dict, geo_vec, rel, clean_depth, eroded_mask = self.geo_extractor.extract_features_and_reliability(
            rgb_crop, raw_depth=depth_crop, raw_mask=mask
        )
        return geo_vec, rel, clean_depth, eroded_mask, geo_dict

    def predict(self, rgb_crop, depth_crop=None, mask=None):
        """
        Runs full multimodal inference on a single 15% context-padded object crop.

        Processing Pipeline:
          1. Input validation and format conversion (NumPy/PIL)
          2. DINOv2 768-D visual feature extraction
          3. 35-D geometric feature extraction (with auto depth estimation)
          4. Feature masking (drop CAD-shifted features)
          5. Dual-branch PCA projection (48-D RGB + 16-D Geo)
          6. Calibrated ensemble prediction with prior re-weighting
          7. Uncertainty flagging (confidence < 40%)

        Note: Unlike RealClassifier, this does NOT apply the physical solidity
        guardrail or irregular suppression, because those heuristics were
        calibrated on real-world data distributions.

        Args:
            rgb_crop (np.ndarray or PIL.Image): RGB crop of the detected object.
                Shape: (H, W, 3), dtype: uint8. Should include ~15% context padding.
            depth_crop (np.ndarray, optional): Metric depth crop (H, W), float32.
                If None, auto-predicted via Depth Anything V2.
            mask (np.ndarray or PIL.Image, optional): Binary segmentation mask (H, W).
                If None, a conservative center mask is auto-generated.

        Returns:
            dict: Structured prediction dictionary:
                {
                    "class": str,          # e.g., "Cuboid"
                    "class_idx": int,       # e.g., 2
                    "confidence": float,    # Calibrated confidence in [0.0, 1.0]
                    "probabilities": dict,  # Per-class probability distribution
                    "uncertain": bool       # True if confidence < 0.40
                }

        Raises:
            ValueError: If rgb_crop is None, empty, or smaller than 5x5 pixels.
        """
        # ── Input Format Conversion ──────────────────────────────────────
        if isinstance(rgb_crop, Image.Image):
            rgb_crop = np.array(rgb_crop.convert("RGB"))
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.array(mask)

        # ── Input Validation ─────────────────────────────────────────────
        if rgb_crop is None or rgb_crop.size == 0 or rgb_crop.shape[0] < 5 or rgb_crop.shape[1] < 5:
            raise ValueError("[SyntheticClassifier] Invalid or empty rgb_crop passed to predict().")

        # ── Step 1: Feature Extraction ───────────────────────────────────
        # Extract 768-D DINOv2 visual features
        dino_768 = self.extract_dino_features(rgb_crop)
        # Extract 35-D geometric features + depth reliability score
        geo_35, rel, clean_depth, eroded_mask, geo_dict = self.extract_geometry_features(rgb_crop, depth_crop, mask)

        # ── Step 2: Model Prediction ─────────────────────────────────────
        # The GeneralizedChampionPipelineC.predict_proba() internally:
        #   a) Filters out CAD-shifted features (planar_surface_ratio, etc.)
        #   b) Scales features (StandardScaler for RGB, RobustScaler for Geo)
        #   c) Projects via PCA (48-D RGB + 16-D Geo = 64-D fused)
        #   d) Runs the calibrated ensemble (ExtraTrees + HistGB + RF)
        #   e) Applies class prior re-weighting (W = [1.8, 1.0, 1.4, 0.45])
        #   f) Re-normalizes probabilities to sum to 1.0
        probs = self.model.predict_proba(
            dino_768.reshape(1, -1),    # (1, 768) — single sample batch
            geo_35.reshape(1, -1),      # (1, 35) — single sample batch
            class_weights=self.class_weights
        )[0]  # [0] to extract single sample from batch dimension

        # ── Step 3: Post-Processing ──────────────────────────────────────
        # Determine the predicted class (highest probability)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        pred_label = self.class_names[class_idx]
        # Flag as uncertain if below threshold
        is_uncertain = bool(confidence < UNCERTAINTY_THRESHOLD)

        # Format probability dictionary with clean rounded floats
        prob_dict = {
            self.class_names[k]: round(float(probs[k]), 4)
            for k in range(len(self.class_names))
        }

        # ── Assemble Structured Output ───────────────────────────────────
        return {
            "class": pred_label,                     # Human-readable label
            "class_idx": class_idx,                  # Integer index [0-3]
            "confidence": round(confidence, 4),      # Max probability
            "probabilities": prob_dict,              # Full 4-class distribution
            "uncertain": is_uncertain                # Uncertainty flag
        }



# =============================================================================
# STANDALONE UNIT TEST
# =============================================================================
# Verifies that the synthetic classifier can load, extract features, and predict.

if __name__ == "__main__":
    print("=" * 70)
    print("  TESTING SYNTHETIC CLASSIFIER INFERENCE INTERFACE")
    print("=" * 70)

    # Instantiate classifier (triggers model + backbone loading)
    clf = SyntheticClassifier()

    # Create a dummy random crop for smoke testing.
    # Random pixels won't produce meaningful predictions but validate the pipeline.
    dummy_crop = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)

    # Run prediction
    result = clf.predict(dummy_crop)

    # Display results
    print("\nPrediction Result:")
    print(f"  * Class        : {result['class']} (index {result['class_idx']})")
    print(f"  * Confidence   : {result['confidence']*100:.2f}%")
    print(f"  * Probabilities: {result['probabilities']}")
    print(f"  * Uncertain    : {result['uncertain']}")

    # Test on a real sample crop if one exists on disk
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
