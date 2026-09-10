"""
src/__init__.py — Shared Core Framework for PS-1 Primitive Geometry Classification.

This file makes the `src/` directory a Python package and provides a unified
public API by re-exporting all major components. Importing from `src` directly
(e.g., `from src import PipelineC`) is equivalent to importing from the
individual submodules.

Package Architecture:
  src/
  ├── __init__.py                 ← You are here (public API re-exports)
  ├── feature_extractor.py        ← V1 backward-compatible wrapper
  ├── feature_extractor_v2.py     ← 35-D Geometry & Profile Harvester (primary)
  ├── fusion_engine.py            ← 3 Multi-Modal Fusion Architectures (A, B, C)
  ├── generalized_champion.py     ← Synthetic Champion Model Definition
  ├── pipeline_c.py               ← Full Multimodal Inference Pipeline
  ├── dataset_engine.py           ← Group-Aware Dataset Splitting (zero leakage)
  └── synthetic_loader.py         ← BlenderProc Synthetic Data Loader
"""

# =============================================================================
# FEATURE EXTRACTION MODULES
# =============================================================================
# GeometricFeatureExtractorV2: Primary 35-D feature extractor using Depth Anything V2.
#   Extracts surface normals, depth statistics, silhouette descriptors, and composite ratios.
# extract_geometry_features_v2: Convenience function using a global singleton extractor.
# GEOMETRIC_FEATURE_NAMES_V2: Ordered list of all 35 feature names (used for DataFrame columns).
from src.feature_extractor_v2 import (
    GeometricFeatureExtractorV2,
    extract_geometry_features_v2,
    GEOMETRIC_FEATURE_NAMES_V2,
)

# GeometricFeatureExtractor: Backward-compatible alias for V2 extractor.
#   Exists so that legacy pickle files referencing "src.feature_extractor.GeometricFeatureExtractor"
#   can still be deserialized without errors.
# GEOMETRIC_FEATURE_NAMES: Alias for GEOMETRIC_FEATURE_NAMES_V2.
from src.feature_extractor import (
    GeometricFeatureExtractor,
    GEOMETRIC_FEATURE_NAMES,
)

# =============================================================================
# FUSION ARCHITECTURES (3 Approaches to Multimodal Feature Combination)
# =============================================================================
# FusionApproachA_EqualizedProjection: ★ CHAMPION ★
#   64-D balanced fusion (32-D RGB PCA + 32-D Geo PCA). Eliminates tree split bias
#   by giving equal dimensionality to both modalities.
# FusionApproachB_ReliabilityWeighting:
#   Scales geometry features by depth reliability score Rd. Better when depth
#   quality varies significantly across samples.
# FusionApproachC_LateFusionStacking:
#   Trains separate RGB and Geo classifiers, then stacks their probability
#   outputs with a meta-learner. Zero feature drowning by design.
# evaluate_fusion_model: Utility function for computing balanced accuracy,
#   confusion matrices, and classification reports on test data.
# REAL_CLASS_NAMES: Alias to avoid name collision with pipeline_c.CLASS_NAMES.
from src.fusion_engine import (
    FusionApproachA_EqualizedProjection,
    FusionApproachB_ReliabilityWeighting,
    FusionApproachC_LateFusionStacking,
    evaluate_fusion_model,
    CLASS_NAMES as REAL_CLASS_NAMES,
)

# =============================================================================
# PIPELINE C — Full Multimodal Inference Engine
# =============================================================================
# PipelineC: End-to-end inference pipeline that handles DINOv2 extraction,
#   depth estimation, geometry features, fusion, and classification in one call.
# CLASS_NAMES: Canonical class ordering ["Flat", "Cylindrical", "Cuboid", "Irregular"].
# NUM_CLASSES: Integer count of classes (4).
from src.pipeline_c import (
    PipelineC,
    CLASS_NAMES,
    NUM_CLASSES,
)

# =============================================================================
# GENERALIZED CHAMPION — Synthetic-Trained Model Architecture
# =============================================================================
# GeneralizedChampionPipelineC: The model class used for sim-to-real transfer.
#   Uses feature masking + calibrated prior re-weighting for domain adaptation.
# DROP_FEATURES: List of features to mask out due to synthetic→real domain shift.
# CLASS_WEIGHTS: Prior re-weighting vector [1.8, 1.0, 1.4, 0.45] for Flat/Cyl/Cub/Irr.
from src.generalized_champion import (
    GeneralizedChampionPipelineC,
    DROP_FEATURES,
    CLASS_WEIGHTS,
)

# =============================================================================
# REAL DATASET ENGINE — Group-Aware Splitting
# =============================================================================
# discover_product_groups: Scans dataset folders and groups crops by physical product.
# generate_and_save_splits: Creates stratified 70/15/15 train/val/test splits
#   at the product-group level (prevents data leakage from augmented variants).
# load_partitioned_datasets: Loads samples respecting split assignments.
#   Train gets originals + augmented; Val/Test get pristine originals ONLY.
# LABEL_MAPPING: Maps folder names to class indices (e.g., "0_flat" → 0).
from src.dataset_engine import (
    discover_product_groups,
    generate_and_save_splits,
    load_partitioned_datasets,
    LABEL_MAPPING,
)

# =============================================================================
# SYNTHETIC DATASET ENGINE — BlenderProc Data Loader
# =============================================================================
# SyntheticPrimitiveDataset: PyTorch Dataset for loading per-object crops from
#   BlenderProc-generated synthetic scenes with COCO annotations.
# GEOMETRY_CLASSES: Synthetic class index mapping (different ordering than real).
# REAL_TO_SYNTH_MAP: Bijection from real class indices to synthetic indices.
#   {0: 2, 1: 1, 2: 0, 3: 3} — Flat↔Cuboid are swapped, others match.
# SYNTH_TO_REAL_MAP: Inverse bijection (synthetic → real).
from src.synthetic_loader import (
    SyntheticPrimitiveDataset,
    GEOMETRY_CLASSES,
    REAL_TO_SYNTH_MAP,
    SYNTH_TO_REAL_MAP,
)

# =============================================================================
# PUBLIC API — Controls what `from src import *` exports
# =============================================================================
__all__ = [
    # Feature Extraction
    "GeometricFeatureExtractorV2",
    "GeometricFeatureExtractor",
    "extract_geometry_features_v2",
    "GEOMETRIC_FEATURE_NAMES_V2",
    "GEOMETRIC_FEATURE_NAMES",
    # Fusion & Architectures
    "FusionApproachA_EqualizedProjection",
    "FusionApproachB_ReliabilityWeighting",
    "FusionApproachC_LateFusionStacking",
    "evaluate_fusion_model",
    "PipelineC",
    "GeneralizedChampionPipelineC",
    "DROP_FEATURES",
    "CLASS_WEIGHTS",
    "CLASS_NAMES",
    "REAL_CLASS_NAMES",
    "NUM_CLASSES",
    # Real Dataset Engine
    "discover_product_groups",
    "generate_and_save_splits",
    "load_partitioned_datasets",
    "LABEL_MAPPING",
    # Synthetic Dataset Engine
    "SyntheticPrimitiveDataset",
    "GEOMETRY_CLASSES",
    "REAL_TO_SYNTH_MAP",
    "SYNTH_TO_REAL_MAP",
]
