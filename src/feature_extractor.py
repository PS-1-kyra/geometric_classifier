"""
src/feature_extractor.py -- Backward-compatible wrapper & re-export of GeometricFeatureExtractor.

Provides seamless compatibility for modules/pickles importing from src.feature_extractor.
"""

from src.feature_extractor_v2 import (
    GeometricFeatureExtractorV2 as GeometricFeatureExtractor,
    GeometricFeatureExtractorV2,
    GEOMETRIC_FEATURE_NAMES_V2 as GEOMETRIC_FEATURE_NAMES,
    GEOMETRIC_FEATURE_NAMES_V2,
    extract_geometry_features_v2 as extract_geometry_features,
    extract_geometry_features_v2,
    LOCAL_DEPTH_PATH,
    HF_DEPTH_ID,
)

__all__ = [
    "GeometricFeatureExtractor",
    "GeometricFeatureExtractorV2",
    "GEOMETRIC_FEATURE_NAMES",
    "GEOMETRIC_FEATURE_NAMES_V2",
    "extract_geometry_features",
    "extract_geometry_features_v2",
    "LOCAL_DEPTH_PATH",
    "HF_DEPTH_ID",
]
