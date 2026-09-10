#!/usr/bin/env python3
"""
========================================================================================
scripts/augment_crops.py — Physical & Photometric Crop Augmenter
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & ARCHITECTURE: DOMAIN-SPECIFIC RETAIL AUGMENTATION
----------------------------------------------------------------------------------------
[Basic Concept: Why Standard Augmentations are Insufficient]:
  Standard computer vision augmentations (simple horizontal flips, random crops, slight rotations)
  fail to prepare a retail model for the extreme photometric and physical conditions encountered
  inside real grocery stores and supermarkets.
  In a physical supermarket aisle:
    - Overhead halogen and LED spotlights create blinding specular glare on cellophane wrappers
      and curved aluminum cans.
    - Shelf price strips, shelf lips, and wire dividers partially occlude the bottom 5% to 15%
      of product packaging.
    - Upper shelves cast strong top-down gradient shadow patterns onto lower products.
    - Non-rigid bags (chips, pouches, pasta) undergo physical elastic deformations and wrinkles.

[The 16 Compound Transformation Pipeline]:
  This script implements 16 distinct domain-specific physical and photometric transformations:
    1. Specular Glare Injection (`inject_specular_glare`):
       Synthesizes bright elliptical highlights with Gaussian falloff, simulating halogen reflections.
    2. Bottom Shelf Lip Occlusion (`inject_shelf_occlusion`):
       Occludes the lower horizontal band (5% to 15% height) with shelf divider gray values.
    3. Vision Transformer Patch Cutout (`apply_patch_cutout`):
       Randomly zeros out a rectangular patch (15% to 30% width/height), preventing DINOv2
       from overfitting to specific brand logos and forcing reliance on global geometric shapes.
    4. Overhead Shelf Lighting Gradient (`apply_shelf_shadow`):
       Applies a vertical linear attenuation gradient from 1.0 (top) down to 0.40 (bottom).
    5. Elastic Deformations (`apply_elastic_deformation`):
       Generates a randomized 2D Gaussian displacement field (alpha=15, sigma=3) using `cv2.remap`
       to simulate non-rigid pouch squishing, wrinkles, and pouch dents.
    6. Photometric Color Jitter & Contrast Shift:
       HSV saturation and value perturbations simulating fluorescent vs warm retail store lighting.
    7. Perspective Distortions & Rotations:
       Simulates skewed shopper viewing angles and tilted shelf displays.

[Strict Zero-Data-Leakage Guarantee]:
  If a splits file (`splits/train_groups.csv`) is provided, this augmenter strictly reads the
  allowed training product group IDs and augments ONLY training items up to the specified
  `target_cap` per class (e.g., 1,000 samples).
  Validation and test splits are completely ignored and remain 100% pristine original crops.
========================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Ensure repo root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_engine import LABEL_MAPPING, CLASS_NAMES


def inject_specular_glare(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates supermarket halogen spotlight or LED glare on cellophane and cans.

    Mathematical Workflow:
      1. Generates 1 to 2 random ellipse centers within the middle 60% of the image.
      2. Draws solid white ellipses with random major/minor axes and rotation angles.
      3. Applies a heavy 15x15 Gaussian blur to create soft light dispersion falloff.
      4. Blends with the original image using alpha=0.75, beta=0.25 (additive lighting).

    Args:
        img_bgr: BGR uint8 image of shape (H, W, 3).

    Returns:
        img_glare: Augmented BGR image exhibiting realistic specular flare.
    """
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    num_spots = np.random.randint(1, 3)
    for _ in range(num_spots):
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        axes = (
            np.random.randint(max(4, w // 10), max(8, w // 4)),
            np.random.randint(max(2, h // 20), max(5, h // 8))
        )
        angle = np.random.randint(0, 180)
        cv2.ellipse(res, (cx, cy), axes, angle, 0, 360, (255, 255, 255), -1)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    return cv2.addWeighted(img_bgr, 0.75, res, 0.25, 0)


def inject_shelf_occlusion(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates bottom shelf lip occlusion.
    Overwrites the bottom 5% to 15% of the product crop with dark gray shelf plastic (BGR: 40, 40, 40).
    """
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    occ_h = int(h * np.random.uniform(0.05, 0.15))
    res[h - occ_h:h, :] = (40, 40, 40)
    return res


def apply_patch_cutout(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates partial object occlusion via rectangular patch cutout.
    Forces Vision Transformer attention mechanisms to leverage distributed geometric cues
    rather than memorizing localized packaging brand logos.
    """
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    cw = max(4, int(w * np.random.uniform(0.15, 0.30)))
    ch = max(4, int(h * np.random.uniform(0.15, 0.30)))
    x1 = np.random.randint(0, max(1, w - cw))
    y1 = np.random.randint(0, max(1, h - ch))
    res[y1:y1 + ch, x1:x1 + cw] = 0
    return res


def apply_shelf_shadow(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates overhead supermarket shelf lighting gradients.
    Applies a vertical linear attenuation gradient darkening the lower section of the crop.
    """
    h, w = img_bgr.shape[:2]
    dim = np.random.uniform(0.40, 0.70)
    grad = np.linspace(1.0, dim, h).reshape(h, 1, 1)
    return np.clip(img_bgr.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def apply_elastic_deformation(img_bgr: np.ndarray, alpha: float = 15, sigma: float = 3) -> np.ndarray:
    """
    Applies elastic mesh deformation simulating non-rigid pouches, crumpled wrappers, and squished bags.

    Mathematical Formulation:
      Generates two random displacement fields:
        dx = GaussianBlur(Uniform(-1, 1) * alpha, sigma)
        dy = GaussianBlur(Uniform(-1, 1) * alpha, sigma)
      Uses `cv2.remap` with border reflection to warp pixels along the smooth displacement field.
    """
    h, w = img_bgr.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def generate_compound_variant(img_bgr: np.ndarray, variant_id: int) -> np.ndarray:
    """
    Generates 1 of 16 distinct compound store variants combining physical and photometric perturbations.

    Variant Matrix:
      0: Horizontal flip (mirror symmetry)
      1: Rotation (+10 deg) + Brightness boost (+15)
      2: Specular glare + Perspective tilt
      3: Shelf lip occlusion + Overhead shelf shadow
      4: Elastic deformation + Horizontal flip
      5: Patch cutout + Gaussian blur (defocus blur)
      6: HSV saturation & lighting jitter
      7: Specular glare + Elastic deformation
      8: Rotation (-12 deg) + Shelf lip occlusion
      9: Glare + Cutout + Shadow
      10: Horizontal flip + Shelf shadow
      11: Rotation (+6 deg)
      12: Pure overhead shelf shadow
      13: Pure patch cutout
      14: Pure specular glare
      15: Contrast attenuation (-10% contrast, -10 brightness)
    """
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    v = variant_id % 16

    if v == 0:
        return cv2.flip(res, 1)
    elif v == 1:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
        res = cv2.warpAffine(res, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return cv2.convertScaleAbs(res, alpha=1.2, beta=15)
    elif v == 2:
        res = inject_specular_glare(res)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        m = int(w * 0.06)
        pts2 = np.float32([[m, m], [w - m, 0], [w, h - m], [0, h]])
        return cv2.warpPerspective(res, cv2.getPerspectiveTransform(pts1, pts2), (w, h), borderMode=cv2.BORDER_REFLECT)
    elif v == 3:
        res = inject_shelf_occlusion(res)
        return apply_shelf_shadow(res)
    elif v == 4:
        res = apply_elastic_deformation(res)
        return cv2.flip(res, 1)
    elif v == 5:
        res = apply_patch_cutout(res)
        return cv2.GaussianBlur(res, (3, 3), 0)
    elif v == 6:
        hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.7, 1.3)
        hsv[:, :, 2] *= np.random.uniform(0.85, 1.15)
        res = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return res
    elif v == 7:
        res = inject_specular_glare(res)
        return apply_elastic_deformation(res)
    elif v == 8:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -12, 1.0)
        res = cv2.warpAffine(res, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return inject_shelf_occlusion(res)
    elif v == 9:
        res = inject_specular_glare(res)
        res = apply_patch_cutout(res)
        return apply_shelf_shadow(res)
    elif v == 10:
        res = cv2.flip(res, 1)
        res = apply_shelf_shadow(res)
        return res
    elif v == 11:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 6, 1.0)
        return cv2.warpAffine(res, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif v == 12:
        return apply_shelf_shadow(res)
    elif v == 13:
        return apply_patch_cutout(res)
    elif v == 14:
        return inject_specular_glare(res)
    else:
        return cv2.convertScaleAbs(res, alpha=0.9, beta=-10)


def augment_dataset(
    dataset_dir: str = "dataset",
    target_cap: int = 1000,
    train_groups_csv: str | None = None
):
    """
    Augments dataset crops up to `target_cap` per class.

    Zero-Leakage Enforcement:
      If `train_groups_csv` is supplied, only crops whose file stems match the
      `group_id`s in `train_groups.csv` are eligible for augmentation.
      Crops belonging to validation and test groups are strictly untouched.
    """
    allowed_groups = None
    if train_groups_csv and os.path.exists(train_groups_csv):
        df_train = pd.read_csv(train_groups_csv)
        allowed_groups = set(df_train["group_id"].astype(str))
        print(f"[AUGMENTER] Enforcing Zero-Leakage: Augmenting {len(allowed_groups)} training groups only.")

    for folder_name in LABEL_MAPPING.keys():
        folder_path = Path(dataset_dir) / folder_name
        if not folder_path.exists():
            continue

        orig_files = sorted([
            f for f in folder_path.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and "_aug_" not in f.stem
        ])

        # Filter strictly by allowed training groups
        if allowed_groups is not None:
            orig_files = [f for f in orig_files if f.stem in allowed_groups]

        num_orig = len(orig_files)
        if num_orig == 0:
            continue

        needed = max(0, target_cap - num_orig)
        variants_per_crop = max(1, int(np.ceil(needed / num_orig)))

        print(f"Class {folder_name:14s}: {num_orig:4d} originals -> Generating ~{variants_per_crop} variants/crop (Target: {target_cap})")

        created = 0
        for f in orig_files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            for v_idx in range(variants_per_crop):
                if num_orig + created >= target_cap:
                    break
                aug_img = generate_compound_variant(img, v_idx)
                aug_name = f"{f.stem}_aug_{v_idx+1}.png"
                cv2.imwrite(str(folder_path / aug_name), aug_img)
                created += 1

        print(f"  -> Successfully generated {created} variants for {folder_name}")


def main():
    """CLI entrypoint for dataset augmentation."""
    parser = argparse.ArgumentParser(description="Target-balanced physical crop augmenter (Zero Data Leakage).")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--target-cap", type=int, default=1000, help="Target balanced sample count per class")
    parser.add_argument("--train-groups", type=str, default="splits/train_groups.csv", help="Path to train_groups.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("  PS-1 PHYSICAL & PHOTOMETRIC DATA AUGMENTER")
    print("=" * 70)
    print(f"Dataset directory : {args.dataset_dir}")
    print(f"Target cap / class: {args.target_cap}")
    print(f"Train groups CSV  : {args.train_groups}\n")

    augment_dataset(dataset_dir=args.dataset_dir, target_cap=args.target_cap, train_groups_csv=args.train_groups)
    print("\n[OK] Data augmentation complete!")


if __name__ == "__main__":
    main()
