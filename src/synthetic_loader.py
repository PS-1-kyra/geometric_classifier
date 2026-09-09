"""
SyntheticPrimitiveDataset — PyTorch Dataset loader for PS-1 synthetic geometry data.

Consumes BlenderProc-generated synthetic scenes, returning per-annotation object crops
for downstream primitive geometry classifier training.

Geometry Taxonomy & Bijection:
    Real Standard Index:      0: flat, 1: cylindrical, 2: cuboid, 3: irregular
    Synthetic Generator Index: 0: cuboid, 1: cylindrical, 2: flat, 3: irregular

    REAL_TO_SYNTH_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
    SYNTH_TO_REAL_MAP = {0: 2, 1: 1, 2: 0, 3: 3}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
CAMERA_ELEVATION_RANGE: tuple[float, float] = (30.0, 70.0)

# Geometry taxonomy in BlenderProc generator
GEOMETRY_CLASSES: dict[str, int] = {
    "cuboid":       0,
    "cylindrical":  1,
    "flat":         2,
    "irregular":    3,
}

# Real <-> Synthetic Class Index Bijections
REAL_TO_SYNTH_MAP: dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}
SYNTH_TO_REAL_MAP: dict[int, int] = {0: 2, 1: 1, 2: 0, 3: 3}


class SyntheticPrimitiveDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset yielding per-object crops from a PS-1 synthetic dataset.

    Each item corresponds to one annotation (one object instance), NOT one scene.
    subset_ratio operates at the scene level (e.g. N scenes).

    Args:
        root_dir:      Path to dataset root (contains images/, annotations/, etc.)
        split:         'train', 'val', or 'test'
        transform:     Optional callable applied to the cropped image before returning
        subset_ratio:  Fraction of scenes to keep, in (0.0, 1.0]. Deterministic given seed.
        seed:          RNG seed for the scene-level subset sampling step.
        validate_elevation: If True, asserts camera elevation is within [30, 70] degrees.

    Returns from __getitem__:
        (image, label_idx, metadata)
            image     — PIL.Image RGB crop (or whatever `transform` returns)
            label_idx — int in [0, 3] mapping to GEOMETRY_CLASSES
            metadata  — dict with keys: annotation_id, image_id, geometry_label,
                        bbox, camera_elevation
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

        self._load_coco()
        self._load_split_ids()
        self._load_scene_metadata_and_validate()
        self._apply_subset_sampling()

    def _load_coco(self) -> None:
        coco_path = self.root / "annotations" / "annotations_coco.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO annotations: {coco_path}")

        with open(coco_path, "r") as f:
            coco = json.load(f)

        self._images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        self._category_name_by_id: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco["categories"]
        }
        self._annotations_all: list[dict] = list(coco["annotations"])

    def _load_split_ids(self) -> None:
        split_file = self.root / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        stems = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        stem_to_id: dict[str, int] = {}
        for img in self._images_by_id.values():
            stem = Path(img["file_name"]).stem
            stem_to_id[stem] = img["id"]

        resolved: set[int] = {stem_to_id[s] for s in stems if s in stem_to_id}
        if not resolved:
            raise ValueError(
                f"No image_ids resolved from splits/{self.split}.txt — "
                f"check that stem values match image file_names."
            )
        self._split_image_ids: set[int] = resolved

    def _load_scene_metadata_and_validate(self) -> None:
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

            camera = meta.get("camera", {})
            elev = camera.get("camera_elevation")
            if self.validate_elevation:
                if elev is None:
                    raise ValueError(f"Scene {stem}: 'camera.camera_elevation' missing.")
                elev = float(elev)
                if not (lo <= elev <= hi):
                    raise ValueError(
                        f"Scene {stem}: camera_elevation={elev}° outside [{lo}°, {hi}°]."
                    )
            self._scene_meta_by_id[img_id] = meta

    def _apply_subset_sampling(self) -> None:
        rng = np.random.default_rng(seed=self.seed)
        scene_ids = np.array(sorted(self._split_image_ids), dtype=np.int64)
        rng.shuffle(scene_ids)

        n_keep = max(1, int(round(len(scene_ids) * self.subset_ratio)))
        kept = np.sort(scene_ids[:n_keep])
        self._kept_scene_ids: set[int] = {int(x) for x in kept}

        anns = [a for a in self._annotations_all if a["image_id"] in self._kept_scene_ids]
        anns.sort(key=lambda a: (a["image_id"], a["id"]))
        self._annotations: list[dict] = anns

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> tuple[Any, int, dict[str, Any]]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range (len={len(self)})")

        ann = self._annotations[idx]
        img_id = ann["image_id"]
        img_info = self._images_by_id[img_id]
        scene = self._scene_meta_by_id[img_id]

        img_path = self.root / img_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        full = Image.open(str(img_path)).convert("RGB")
        W_img, H_img = full.size

        x, y, w, h = ann["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        x1 = max(0, min(W_img - 1, x1))
        y1 = max(0, min(H_img - 1, y1))
        x2 = max(x1 + 1, min(W_img, x2))
        y2 = max(y1 + 1, min(H_img, y2))
        crop = full.crop((x1, y1, x2, y2))

        geom_name = ann.get("geometry_label") or self._category_name_by_id.get(ann["category_id"], "")
        label_idx = GEOMETRY_CLASSES.get(geom_name)
        if label_idx is None:
            raise ValueError(f"Unknown geometry_label={geom_name!r} on annotation {ann['id']}.")

        image: Any = crop
        if self.transform is not None:
            image = self.transform(image)

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
        return sorted(GEOMETRY_CLASSES, key=GEOMETRY_CLASSES.get)

    @property
    def num_classes(self) -> int:
        return len(GEOMETRY_CLASSES)

    @property
    def n_scenes(self) -> int:
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
        full = cls(root_dir, split, transform=transform, subset_ratio=1.0, seed=seed)
        total = len(full._split_image_ids)
        if n_scenes > total:
            raise ValueError(f"Requested n_scenes={n_scenes} exceeds total {total} in '{split}'.")
        ratio = n_scenes / total
        return cls(root_dir, split, transform=transform, subset_ratio=ratio, seed=seed)

    @staticmethod
    def available_scenes(root_dir: str, split: str) -> int:
        split_file = Path(root_dir) / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        return sum(1 for line in split_file.read_text().splitlines() if line.strip())
