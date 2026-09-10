"""
========================================================================================
src/dataset_engine.py -- Group-Aware Dataset Engine (Zero Data Leakage Architecture)
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & ARCHITECTURE: PREVENTING DATA LEAKAGE IN RETAIL COMPUTER VISION
----------------------------------------------------------------------------------------
[Basic Concept: What is Data Leakage?]:
  In naive machine learning pipelines, a dataset is often split randomly using `train_test_split`.
  In retail computer vision, however, a single physical product (SKU) might be photographed
  multiple times from different angles, or subjected to various data augmentations (glare,
  crops, flips, color shifts).
  If different crops of the SAME physical bottle or cereal box end up in both the training set
  and the test set, the classifier can achieve artificially high test accuracy (e.g. 99%)
  by simply memorizing the specific brand logo or vibrant packaging colors, rather than
  learning the underlying 3D geometric shape (cylindrical vs cuboid). This fatal flaw is known
  as "Group Leakage" or "Identity Leakage".

[Advanced Principle: Group-Aware Stratified Splitting]:
  To guarantee that our primitive geometry classifier genuinely learns class-agnostic shape
  features, we enforce strict physical product isolation:
  1. Product Group Discovery:
     Every crop file name contains a root base identifier (`group_id`). Augmented variants
     bear a suffix like `_aug_1.png`, `_aug_2.png`.
     The engine strips all augmentation suffixes to identify the root physical SKU group.
  2. Stratified GroupShuffleSplit:
     We perform a two-stage stratified partition on the UNIQUE PRODUCT GROUPS (not on images):
       - Stage 1: 70% of physical product groups -> TRAIN split
       - Stage 2: 15% of physical product groups -> VALIDATION split
       - Stage 3: 15% of physical product groups -> TEST split
     Both splits maintain the exact class distribution across the 4 geometric categories.
  3. Formal Leakage Verification:
     The engine mathematically asserts that:
       Intersection(Train_Groups, Val_Groups) == Empty
       Intersection(Train_Groups, Test_Groups) == Empty
       Intersection(Val_Groups, Test_Groups) == Empty
     If even a single product group leaks across partitions, execution halts immediately with
     an AssertionError.

[The "Pristine Evaluation" Standard]:
  A common mistake in ML benchmarks is evaluating models on synthetically augmented test data.
  This engine enforces the "Pristine Evaluation" rule:
    - TRAIN SET: Contains pristine original crops PLUS target-balanced physical augmentations
      (specular glare, patch cutouts, shelf lip shadows) to build deep invariant representations.
    - VAL & TEST SETS: Contain 100% PRISTINE ORIGINAL CROPS ONLY. No augmented variants are ever
      admitted into validation or test partitions.
========================================================================================
"""

import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------
# GLOBAL DIRECTORY DEFAULTS & TAXONOMY CONSTANTS
# --------------------------------------------------------------------------------------
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_SPLITS_DIR  = "splits"

# Canonical 4-class retail primitive geometry mapping
LABEL_MAPPING = {
    "0_flat": 0,          # Flat objects (chocolate bars, books, thin packaged goods)
    "1_cylindrical": 1,   # Cylindrical objects (beverage cans, bottles, deodorant sprays)
    "2_cuboid": 2,        # Cuboid objects (cereal boxes, tea cartons, rectangular boxes)
    "3_irregular": 3      # Irregular / Deformable objects (pouches, bags of chips, shrink-wrapped)
}

CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
NUM_CLASSES = len(CLASS_NAMES)


def discover_product_groups(dataset_dir: str = DEFAULT_DATASET_DIR):
    """
    Scans the dataset directory structure, identifies pristine parent crops, and groups
    all augmented child crops back to their parent physical product group.

    Algorithm:
      1. Iterates through the standard class subfolders ("0_flat", "1_cylindrical", etc.).
      2. For each image file, strips the `_aug_X` suffix to extract the root `group_id`.
      3. Categorizes records into:
         - `original_records`: Crops without the `_aug_` tag (pristine physical items).
         - `all_records`: Consolidated pool of pristine and augmented crops.
         - `groups_dict`: Mapping from `group_id` to its metadata, class label, and variant list.

    Args:
        dataset_dir: Path to the root directory containing the class folders.

    Returns:
        groups_dict (dict): Dictionary mapping group_id -> group metadata dict.
        original_records (list[dict]): List of records for pristine original crops only.
        all_records (list[dict]): List of all records (pristine + augmented).
    """
    groups_dict = {}      # group_id -> dictionary of group attributes & variant files
    original_records = [] # pristine original crops only (used for pristine val/test)
    all_records = []      # all records (used for populating training and tracking)

    for folder_name, label_id in LABEL_MAPPING.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        # Collect and sort all valid image files
        crop_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        for fname in crop_files:
            full_path = os.path.join(folder_path, fname)
            base_name, ext = os.path.splitext(fname)

            # Determine group_id by identifying whether this is an augmented variant
            # Format convention: {original_stem}_aug_{variant_id}.png
            if "_aug_" in base_name:
                group_id = base_name.split("_aug_")[0]
                is_aug = True
            else:
                group_id = base_name
                is_aug = False

            record = {
                "filename": fname,
                "filepath": full_path,
                "folder": folder_name,
                "label_id": label_id,
                "class_name": CLASS_NAMES[label_id],
                "group_id": group_id,
                "is_augmented": is_aug
            }

            # Register new product group in the dictionary
            if group_id not in groups_dict:
                groups_dict[group_id] = {
                    "group_id": group_id,
                    "folder": folder_name,
                    "label_id": label_id,
                    "class_name": CLASS_NAMES[label_id],
                    "original_file": fname if not is_aug else None,
                    "variants": []
                }

            # Update pristine file path or append to augmented variant list
            if not is_aug:
                groups_dict[group_id]["original_file"] = fname
                original_records.append(record)
            else:
                groups_dict[group_id]["variants"].append(fname)

            all_records.append(record)

    return groups_dict, original_records, all_records


def generate_and_save_splits(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    seed: int = 42
):
    """
    Executes a Stratified 70/15/15 Group Partition across physical product groups
    and writes permanent split tables (train_groups.csv, val_groups.csv, test_groups.csv).

    Why Stratified Group Split?
      - Standard random split splits samples: images of the same box leak into train and test.
      - Standard GroupKFold splits groups, but can produce severe class imbalance in smaller classes.
      - Stratified Group Split ensures that every split has the exact 70/15/15 proportion of
        Flat, Cylindrical, Cuboid, and Irregular items, while ensuring that 100% of crops from
        any single product group reside exclusively within ONE partition.

    Mathematical Invariant Verified:
      Set(Train_Groups) ∩ Set(Val_Groups) = ∅
      Set(Train_Groups) ∩ Set(Test_Groups) = ∅
      Set(Val_Groups) ∩ Set(Test_Groups) = ∅

    Args:
        dataset_dir: Root dataset folder containing the class subfolders.
        splits_dir:  Target folder where CSV manifest tables will be saved.
        seed:        RNG seed for deterministic, perfectly reproducible splits.

    Returns:
        df_train (pd.DataFrame): Manifest of product groups assigned to train.
        df_val (pd.DataFrame):   Manifest of product groups assigned to validation.
        df_test (pd.DataFrame):  Manifest of product groups assigned to test.
        groups_dict (dict):      Mapping of all discovered product groups.
        all_records (list):      Consolidated list of all image records.
    """
    os.makedirs(splits_dir, exist_ok=True)
    groups_dict, original_records, all_records = discover_product_groups(dataset_dir)

    unique_groups = list(groups_dict.keys())
    if not unique_groups:
        raise FileNotFoundError(f"[SPLIT ENGINE] No valid product groups found in '{dataset_dir}'.")

    # Extract class labels for each unique physical product group for stratification
    group_labels  = [groups_dict[g]["label_id"] for g in unique_groups]
    group_classes = [groups_dict[g]["class_name"] for g in unique_groups]

    print(f"[SPLIT ENGINE] Found {len(unique_groups)} unique physical product groups across {NUM_CLASSES} classes.")
    for c_id, c_name in enumerate(CLASS_NAMES):
        count = sum(1 for l in group_labels if l == c_id)
        print(f"  - Class {c_id} ({c_name:12s}): {count:3d} physical product groups")

    # Step 1: 70% Train, 30% Temporary (Validation + Test) with Stratification
    train_groups, temp_groups, train_y, temp_y = train_test_split(
        unique_groups, group_labels,
        test_size=0.30,
        random_state=seed,
        stratify=group_labels
    )

    # Step 2: Split Temporary 50/50 -> 15% Validation, 15% Test with Stratification
    val_groups, test_groups, val_y, test_y = train_test_split(
        temp_groups, temp_y,
        test_size=0.50,
        random_state=seed,
        stratify=temp_y
    )

    # Mathematical Anti-Leakage Assertion Check
    train_set, val_set, test_set = set(train_groups), set(val_groups), set(test_groups)
    assert len(train_set & val_set) == 0, "CRITICAL LEAKAGE: Overlapping groups between Train and Val!"
    assert len(train_set & test_set) == 0, "CRITICAL LEAKAGE: Overlapping groups between Train and Test!"
    assert len(val_set & test_set) == 0, "CRITICAL LEAKAGE: Overlapping groups between Val and Test!"
    print("[SPLIT ENGINE] Verification Passed: Exactly 0 overlapping groups across all splits.")

    def save_group_csv(groups, filename, split_name):
        """Helper to serialize partition metadata to CSV."""
        rows = []
        for g in groups:
            info = groups_dict[g]
            rows.append({
                "group_id": g,
                "folder": info["folder"],
                "label_id": info["label_id"],
                "class_name": info["class_name"],
                "original_file": info["original_file"],
                "num_augmented_variants": len(info["variants"]),
                "split": split_name
            })
        df = pd.DataFrame(rows)
        save_path = os.path.join(splits_dir, filename)
        df.to_csv(save_path, index=False)
        print(f"  [OK] Saved {split_name:5s} split ({len(df):3d} groups) -> {save_path}")
        return df

    df_train = save_group_csv(train_groups, "train_groups.csv", "train")
    df_val   = save_group_csv(val_groups,   "val_groups.csv",   "val")
    df_test  = save_group_csv(test_groups,  "test_groups.csv",  "test")

    return df_train, df_val, df_test, groups_dict, all_records


def load_partitioned_datasets(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    splits_dir: str = DEFAULT_SPLITS_DIR
):
    """
    Loads dataset samples strictly respecting the Pristine Validation/Test evaluation rule:
      - Train set: Includes pristine originals + all augmented variants for Train groups.
      - Val set:   Includes PRISTINE original crops ONLY for Val groups (no augmentations).
      - Test set:  Includes PRISTINE original crops ONLY for Test groups (no augmentations).

    This guarantees that the reported validation and test benchmark metrics reflect true
    real-world performance on uncorrupted product photography, preventing inflated scores.

    Args:
        dataset_dir: Root dataset folder path.
        splits_dir:  Directory containing train_groups.csv, val_groups.csv, test_groups.csv.

    Returns:
        train_samples (list[dict]): Training records (original + augmented).
        val_samples (list[dict]):   Validation records (pristine originals only).
        test_samples (list[dict]):  Test records (pristine originals only).
    """
    train_csv = os.path.join(splits_dir, "train_groups.csv")
    val_csv   = os.path.join(splits_dir, "val_groups.csv")
    test_csv  = os.path.join(splits_dir, "test_groups.csv")

    # If splits do not exist yet, generate them automatically
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        generate_and_save_splits(dataset_dir=dataset_dir, splits_dir=splits_dir)

    train_groups = set(pd.read_csv(train_csv)["group_id"].astype(str))
    val_groups   = set(pd.read_csv(val_csv)["group_id"].astype(str))
    test_groups  = set(pd.read_csv(test_csv)["group_id"].astype(str))

    groups_dict, original_records, all_records = discover_product_groups(dataset_dir)

    train_samples = []
    val_samples   = []
    test_samples  = []

    for r in all_records:
        g = r["group_id"]
        if g in train_groups:
            # Training partition: Admit both pristine originals and augmented variants
            train_samples.append(r)
        elif g in val_groups:
            # Validation partition: Admit PRISTINE originals ONLY
            if not r["is_augmented"]:
                val_samples.append(r)
        elif g in test_groups:
            # Test partition: Admit PRISTINE originals ONLY
            if not r["is_augmented"]:
                test_samples.append(r)

    print(f"\n[PARTITION SUMMARY - PRISTINE NON-LEAKED DATASET]")
    print(f"  Train samples (with balanced augmentations) : {len(train_samples):5d} ({len(train_groups)} product groups)")
    print(f"  Val samples   (100% pristine originals only): {len(val_samples):5d} ({len(val_groups)} product groups)")
    print(f"  Test samples  (100% pristine originals only): {len(test_samples):5d} ({len(test_groups)} product groups)")

    return train_samples, val_samples, test_samples
