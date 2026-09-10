"""
src/feature_extractor.py -- Backward-Compatible Wrapper & Re-export of GeometricFeatureExtractor.

This module exists SOLELY for backward compatibility with legacy pickle files
and older code that imports from `src.feature_extractor` instead of the newer
`src.feature_extractor_v2`.

When Python's pickle deserializes a model checkpoint (.pkl file) that was saved
when the class was defined in `src.feature_extractor`, it looks for the class
at that exact module path. By re-exporting the V2 class under the V1 name,
we ensure old pickles can still be loaded without errors.

All actual implementation lives in src/feature_extractor_v2.py.

Re-export Mapping:
  GeometricFeatureExtractor     → GeometricFeatureExtractorV2     (class alias)
  GEOMETRIC_FEATURE_NAMES       → GEOMETRIC_FEATURE_NAMES_V2     (constant alias)
  extract_geometry_features     → extract_geometry_features_v2   (function alias)
  LOCAL_DEPTH_PATH, HF_DEPTH_ID → passed through directly
"""

# Import everything from V2 and create backward-compatible aliases.
# The `as` keyword creates a new name pointing to the same object,
# so `GeometricFeatureExtractor` and `GeometricFeatureExtractorV2` are
# both references to the exact same class — no wrapping or overhead.
from src.feature_extractor_v2 import (
    GeometricFeatureExtractorV2 as GeometricFeatureExtractor,   # V1 name → V2 class
    GeometricFeatureExtractorV2,                                 # Also export under V2 name
    GEOMETRIC_FEATURE_NAMES_V2 as GEOMETRIC_FEATURE_NAMES,     # V1 name → V2 constant
    GEOMETRIC_FEATURE_NAMES_V2,                                  # Also export V2 name
    extract_geometry_features_v2 as extract_geometry_features,  # V1 name → V2 function
    extract_geometry_features_v2,                                # Also export V2 name
    LOCAL_DEPTH_PATH,   # Filesystem path to locally cached Depth Anything V2 model weights
    HF_DEPTH_ID,        # HuggingFace model ID for downloading Depth Anything V2
)

# __all__ controls what `from src.feature_extractor import *` exports.
# Both V1 (aliased) and V2 (original) names are exported for maximum compatibility.
__all__ = [
    "GeometricFeatureExtractor",      # Legacy alias → GeometricFeatureExtractorV2
    "GeometricFeatureExtractorV2",    # Current class
    "GEOMETRIC_FEATURE_NAMES",        # Legacy alias → GEOMETRIC_FEATURE_NAMES_V2
    "GEOMETRIC_FEATURE_NAMES_V2",     # Current constant
    "extract_geometry_features",      # Legacy alias → extract_geometry_features_v2
    "extract_geometry_features_v2",   # Current function
    "LOCAL_DEPTH_PATH",               # Local model cache path
    "HF_DEPTH_ID",                    # HuggingFace model identifier
]
