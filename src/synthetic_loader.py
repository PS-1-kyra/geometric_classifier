"""
========================================================================================
src/synthetic_loader.py — PyTorch Dataset Loader for PS-1 Synthetic Geometry Data
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
OVERVIEW & OBJECTIVE (FROM BASIC TO ADVANCED)
----------------------------------------------------------------------------------------
[Basic Concept]:
  In standard computer vision, a PyTorch Dataset yields images and class labels one by one.
  Here, our synthetic training data is created using BlenderProc, a photorealistic rendering
  engine built on Blender. Each render contains an entire retail shelf or tabletop populated
  with multiple 3D CAD objects, along with COCO-formatted 2D bounding boxes and camera metadata.
  This module consumes those full synthetic scenes and yields individual object crops, paired
  with geometric class labels and camera elevation angles.

[The Geometric Taxonomy & Bijection Problem]:
  A classic pitfall in multi-pipeline machine learning is index mismatch between data sources:
    - Real Standard Taxonomy:
        0: Flat, 1: Cylindrical, 2: Cuboid, 3: Irregular
    - BlenderProc Synthetic Taxonomy:
        0: Cuboid, 1: Cylindrical, 2: Flat, 3: Irregular

  Notice that classes 0 and 2 are swapped between real and synthetic conventions!
  To solve this, we define a formal mathematical bijection (one-to-one and onto mapping):
    REAL_TO_SYNTH_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
    SYNTH_TO_REAL_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
  This bijection guarantees seamless bidirectional translation between real retail models
  and synthetic training data without silent label corruption.

[Advanced Feature: Scene-Level Subset Sampling]:
  Why sample at the scene level instead of the crop level?
  In a synthetic scene, all objects share identical lighting conditions, camera elevation,
  and rendering artifacts. If we randomly sampled individual crops, crops from the same
  scene would be scattered across train and validation splits, causing synthetic scene leakage.
  Therefore, the parameter `subset_ratio` shuffles and selects complete SCENES deterministically,
  then gathers all annotations within those selected scenes.

[Camera Elevation Guardrails]:
  Real retail shelf cameras operate within an angle of 30° to 70° relative to the floor.
  If synthetic data were generated from extreme top-down (90°) or ground-level (0°) views,
  the geometric cues (e.g., elliptical rims of cans, perspective foreshortening of boxes)
  would mislead the classifier. This loader strictly verifies that every scene's camera
  elevation falls within [30.0°, 70.0°].
========================================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------------------
# CONSTANTS & TAXONOMY SPECIFICATIONS
# --------------------------------------------------------------------------------------
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")

# Valid camera elevation bounds representing realistic retail shelf perspectives
CAMERA_ELEVATION_RANGE: tuple[float, float] = (30.0, 70.0)

# Geometry taxonomy internal to the BlenderProc synthetic generator
GEOMETRY_CLASSES: dict[str, int] = {
    "cuboid":       0,
    "cylindrical":  1,
    "flat":         2,
    "irregular":    3,
}

# Bijective mappings between Real standard index space and Synthetic generator index space
# Real:      0=flat,   1=cylindrical, 2=cuboid, 3=irregular
# Synthetic: 0=cuboid, 1=cylindrical, 2=flat,   3=irregular
REAL_TO_SYNTH_MAP: dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}
SYNTH_TO_REAL_MAP: dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}


class SyntheticPrimitiveDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset yielding individual per-object crops from PS-1 synthetic scenes.

    Each index in `__getitem__` corresponds to one object annotation (one product crop),
    extracted from its parent scene image via its COCO bounding box coordinates.

    Key Features:
      1. On-the-fly precision cropping with boundary clamping.
      2. Scene-level subset sampling (preserving scene integrity).
      3. Automatic COCO annotation loading and validation.
      4. Camera elevation verification against realistic retail store geometry.

    Args:
        root_dir:           Root directory of the synthetic dataset (must contain
                            'images/', 'annotations/annotations_coco.json', 'splits/').
        split:              Dataset partition to load: 'train', 'val', or 'test'.
        transform:          Optional torchvision transform pipeline applied to the PIL crop.
        subset_ratio:       Fraction of scenes to keep in (0.0, 1.0] for ablation studies.
        seed:               Deterministic RNG seed for scene-level subset sampling.
        validate_elevation: If True, asserts camera elevation is strictly within [30°, 70°].
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Callable | None = None,
        subset_ratio: float = 1.0,
        seed: int = 42,
        validate_elevation: bool = True,
    ) -> None:
        super().__init__()

        # Input validation checks
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}; got {split!r}")
        if not (0.0 < float(subset_ratio) <= 1.0):
            raise ValueError(f"subset_ratio must be in (0.0, 1.0]; got {subset_ratio}")

        self.root: Path = Path(root_dir)
        self.split: str = split
        self.transform: Callable | None = transform
        self.subset_ratio: float = float(subset_ratio)
        self.seed: int = int(seed)
        self.validate_elevation: bool = validate_elevation

        # Execution sequence: load annotations -> resolve split IDs -> validate scenes -> sample
        self._load_coco()
        self._load_split_ids()
        self._load_scene_metadata_and_validate()
        self._apply_subset_sampling()

    def _load_coco(self) -> None:
        """
        Parses COCO-formatted JSON annotations containing image metadata, categories,
        and bounding box annotations for all objects in the synthetic dataset.
        """
        coco_path = self.root / "annotations" / "annotations_coco.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO annotations file at: {coco_path}")

        with open(coco_path, "r") as f:
            coco = json.load(f)

        # Index images and categories for O(1) dictionary lookups
        self._images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        self._category_name_by_id: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco["categories"]
        }
        self._annotations_all: list[dict] = list(coco["annotations"])

    def _load_split_ids(self) -> None:
        """
        Reads the split file (`splits/{split}.txt`) containing scene file stems and maps
        them to their corresponding integer `image_id`s in the COCO manifest.
        """
        split_file = self.root / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        # Clean whitespace and empty lines from split file
        stems = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        stem_to_id: dict[str, int] = {}
        for img in self._images_by_id.values():
            stem = Path(img["file_name"]).stem
            stem_to_id[stem] = img["id"]

        # Resolve image IDs belonging to this split
        resolved: set[int] = {stem_to_id[s] for s in stems if s in stem_to_id}
        if not resolved:
            raise ValueError(
                f"No image_ids resolved from splits/{self.split}.txt — "
                f"verify that stem values match image file_names."
            )
        self._split_image_ids: set[int] = resolved

    def _load_scene_metadata_and_validate(self) -> None:
        """
        Loads per-scene JSON metadata and validates that the camera elevation angle
        falls within the realistic physical range [30°, 70°].
        """
        self._scene_meta_by_id: dict[int, dict] = {}
        lo, hi = CAMERA_ELEVATION_RANGE

        for img_id in sorted(self._split_image_ids):
            img = self._images_by_id[img_id]
            stem = Path(img["file_name"]).stem
            meta_path = self.root / "metadata" / "per_scene" / f"{stem}_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing per-scene metadata: {meta_path}")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Camera elevation angle check
            camera = meta.get("camera", {})
            elev = camera.get("camera_elevation")
            if self.validate_elevation:
                if elev is None:
                    raise ValueError(f"Scene {stem}: 'camera.camera_elevation' field is missing.")
                elev = float(elev)
                if not (lo <= elev <= hi):
                    raise ValueError(
                        f"Scene {stem}: camera_elevation={elev}° is outside realistic bounds [{lo}°, {hi}°]."
                    )
            self._scene_meta_by_id[img_id] = meta

    def _apply_subset_sampling(self) -> None:
        """
        Executes scene-level subset sampling deterministically using NumPy's Generator.
        Keeps all object annotations belonging to the retained subset of scenes.
        """
        rng = np.random.default_rng(seed=self.seed)
        scene_ids = np.array(sorted(self._split_image_ids), dtype=np.int64)
        rng.shuffle(scene_ids)

        # Calculate number of scenes to keep based on subset_ratio
        n_keep = max(1, int(round(len(scene_ids) * self.subset_ratio)))
        kept = np.sort(scene_ids[:n_keep])
        self._kept_scene_ids: set[int] = {int(x) for x in kept}

        # Filter annotations belonging strictly to kept scenes
        anns = [a for a in self._annotations_all if a["image_id"] in self._kept_scene_ids]
        # Sort annotations stably by (image_id, annotation_id)
        anns.sort(key=lambda a: (a["image_id"], a["id"]))
        self._annotations: list[dict] = anns

    def __len__(self) -> int:
        """Returns the total number of object crop annotations available in this split."""
        return len(self._annotations)

    def __getitem__(self, idx: int) -> tuple[Any, int, dict[str, Any]]:
        """
        Fetches and crops an individual object instance from its parent scene.

        Workflow:
          1. Resolves annotation by index.
          2. Loads the full scene RGB image via PIL.
          3. Extracts the bounding box [x, y, w, h] and clamps coordinates to image boundaries.
          4. Crops the sub-region and applies user-specified transforms (e.g. DINOv2 normalization).
          5. Returns (image_crop, label_idx, metadata_dict).

        Returns:
            image:     PIL.Image RGB crop (or torch.Tensor if transform is applied).
            label_idx: Integer class label in [0..3] (BlenderProc synthetic index space).
            metadata:  Dictionary containing annotation_id, image_id, bbox, and camera_elevation.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        ann = self._annotations[idx]
        img_id = ann["image_id"]
        img_info = self._images_by_id[img_id]
        scene = self._scene_meta_by_id[img_id]

        img_path = self.root / img_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Scene image file not found: {img_path}")

        # Load full scene image
        full = Image.open(str(img_path)).convert("RGB")
        W_img, H_img = full.size

        # Extract COCO bounding box [x_min, y_min, width, height]
        x, y, w, h = ann["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        # Precision coordinate clamping to avoid out-of-bounds indexing
        x1 = max(0, min(W_img - 1, x1))
        y1 = max(0, min(H_img - 1, y1))
        x2 = max(x1 + 1, min(W_img, x2))
        y2 = max(y1 + 1, min(H_img, y2))

        # Crop the object region
        crop = full.crop((x1, y1, x2, y2))

        # Resolve category name and map to synthetic class index
        geom_name = ann.get("geometry_label") or self._category_name_by_id.get(ann["category_id"], "")
        label_idx = GEOMETRY_CLASSES.get(geom_name)
        if label_idx is None:
            raise ValueError(f"Unknown geometry_label={geom_name!r} on annotation {ann['id']}.")

        # Apply optional image transformations
        image: Any = crop
        if self.transform is not None:
            image = self.transform(image)

        # Build detailed metadata dictionary
        metadata: dict[str, Any] = {
            "annotation_id":    int(ann["id"]),
            "image_id":         int(img_id),
            "geometry_label":   geom_name,
            "bbox":             [float(x), float(y), float(w), float(h)],
            "camera_elevation": float(scene["camera"]["camera_elevation"]),
        }

        return image, int(label_idx), metadata

    @property
    def classes(self) -> list[str]:
        """Returns the list of geometry class names ordered by synthetic class index."""
        return sorted(GEOMETRY_CLASSES, key=GEOMETRY_CLASSES.get)

    @property
    def num_classes(self) -> int:
        """Returns the number of geometric primitive classes (4)."""
        return len(GEOMETRY_CLASSES)

    @property
    def n_scenes(self) -> int:
        """Returns the number of unique synthetic scenes currently retained."""
        return len(self._kept_scene_ids)

    @classmethod
    def from_n_scenes(
        cls,
        root_dir: str,
        split: str,
        n_scenes: int,
        transform=None,
        seed: int = 42,
    ) -> "SyntheticPrimitiveDataset":
        """
        Convenience factory constructor to initialize a dataset with an exact number of scenes.

        Args:
            root_dir:  Dataset root folder.
            split:     Split name ('train', 'val', 'test').
            n_scenes:  Exact number of scenes to sample.
            transform: Optional image transform.
            seed:      RNG seed for deterministic sampling.
        """
        full = cls(root_dir, split, transform=transform, subset_ratio=1.0, seed=seed)
        total = len(full._split_image_ids)
        if n_scenes > total:
            raise ValueError(f"Requested n_scenes={n_scenes} exceeds total available ({total}) in '{split}'.")
        ratio = n_scenes / total
        return cls(root_dir, split, transform=transform, subset_ratio=ratio, seed=seed)

    @staticmethod
    def available_scenes(root_dir: str, split: str) -> int:
        """Returns the total number of scenes listed in the partition text file."""
        split_file = Path(root_dir) / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        return sum(1 for line in split_file.read_text().splitlines() if line.strip())
