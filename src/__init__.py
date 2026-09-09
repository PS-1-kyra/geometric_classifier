"""
Shared Core Framework for Project PS-1 Primitive Geometry Classification.

Provides unified feature extraction, multimodal fusion, dataset loading,
and model architectures for both Real and Synthetic pipelines.
"""

from src.feature_extractor_v2 import (
    GeometricFeatureExtractorV2,
    extract_geometry_features_v2,
    GEOMETRIC_FEATURE_NAMES_V2,
)

from src.feature_extractor import (
    GeometricFeatureExtractor,
    GEOMETRIC_FEATURE_NAMES,
)

from src.fusion_engine import (
    FusionApproachA_EqualizedProjection,
    FusionApproachB_ReliabilityWeighting,
    FusionApproachC_LateFusionStacking,
    evaluate_fusion_model,
    CLASS_NAMES as REAL_CLASS_NAMES,
)

from src.pipeline_c import (
    PipelineC,
    CLASS_NAMES,
    NUM_CLASSES,
)

from src.generalized_champion import (
    GeneralizedChampionPipelineC,
    DROP_FEATURES,
    CLASS_WEIGHTS,
)

from src.dataset_engine import (
    discover_product_groups,
    generate_and_save_splits,
    load_partitioned_datasets,
    LABEL_MAPPING,
)

from src.synthetic_loader import (
    SyntheticPrimitiveDataset,
    GEOMETRY_CLASSES,
    REAL_TO_SYNTH_MAP,
    SYNTH_TO_REAL_MAP,
)

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
