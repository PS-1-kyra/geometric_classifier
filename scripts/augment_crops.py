#!/usr/bin/env python3
"""
scripts/augment_crops.py — Physical & Photometric Crop Augmenter

Applies 16 compound physical and store-condition transformations:
  - Vision Transformer patch cutout
  - Specular glare simulation (supermarket halogens)
  - Shelf lip occlusion & overhead shelf shadow
  - Elastic deformation & perspective distortion
  - Color jitter & orientation flips

Zero-Leakage Guarantee:
  When a splits CSV is provided, ONLY training groups are augmented.
  Validation and test splits remain 100% pristine.
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


def inject_specular_glare(img_bgr):
    """Simulates supermarket halogen / LED glare on cellophane and cans."""
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    num_spots = np.random.randint(1, 3)
    for _ in range(num_spots):
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        axes = (np.random.randint(max(4, w // 10), max(8, w // 4)),
                np.random.randint(max(2, h // 20), max(5, h // 8)))
        angle = np.random.randint(0, 180)
        cv2.ellipse(res, (cx, cy), axes, angle, 0, 360, (255, 255, 255), -1)
    res = cv2.GaussianBlur(res, (15, 15), 0)
    return cv2.addWeighted(img_bgr, 0.75, res, 0.25, 0)


def inject_shelf_occlusion(img_bgr):
    """Simulates bottom shelf lip occlusion (5% to 15% height)."""
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    occ_h = int(h * np.random.uniform(0.05, 0.15))
    res[h - occ_h:h, :] = (40, 40, 40)
    return res


def apply_patch_cutout(img_bgr):
    """ViT patch cutout simulating partial occlusions."""
    h, w = img_bgr.shape[:2]
    res = img_bgr.copy()
    cw = max(4, int(w * np.random.uniform(0.15, 0.30)))
    ch = max(4, int(h * np.random.uniform(0.15, 0.30)))
    x1 = np.random.randint(0, max(1, w - cw))
    y1 = np.random.randint(0, max(1, h - ch))
    res[y1:y1 + ch, x1:x1 + cw] = 0
    return res


def apply_shelf_shadow(img_bgr):
    """Overhead store lighting gradient shadow."""
    h, w = img_bgr.shape[:2]
    dim = np.random.uniform(0.40, 0.70)
    grad = np.linspace(1.0, dim, h).reshape(h, 1, 1)
    return np.clip(img_bgr.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def apply_elastic_deformation(img_bgr, alpha=15, sigma=3):
    """Elastic warping simulating non-rigid pouches/bags."""
    h, w = img_bgr.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def generate_compound_variant(img_bgr, variant_id):
    """Generates 1 of 16 distinct compound store variants."""
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


def augment_dataset(dataset_dir="dataset", target_cap=1000, train_groups_csv=None):
    """
    Augments dataset folder crops up to target_cap per class.
    If train_groups_csv is provided, ONLY crops belonging to training groups are augmented.
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
