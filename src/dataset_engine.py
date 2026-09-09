"""
=============================================================
src/dataset_engine.py -- Group-Aware Dataset Engine (Zero Data Leakage)
=============================================================
Engineer: Pranaya Shrestha
Project:  PS-1 Class-Agnostic Primitive Analysis

Responsibilities:
  1. Identifies all physical product groups across 4 classes.
  2. Executes GroupShuffleSplit BEFORE data augmentation:
       - TRAIN SET = 70% physical product groups
       - VAL SET   = 15% physical product groups
       - TEST SET  = 15% physical product groups
  3. Saves permanent splits to:
       - splits/train_groups.csv
       - splits/val_groups.csv
       - splits/test_groups.csv
  4. Enforces PRISTINE Validation and Test sets:
       - Train set receives target-balanced compound augmentations.
       - Val & Test sets contain 100% pristine original crops ONLY.
=============================================================
"""

import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET_DIR = "dataset"
DEFAULT_SPLITS_DIR  = "splits"

LABEL_MAPPING = {
    "0_flat": 0,
    "1_cylindrical": 1,
    "2_cuboid": 2,
    "3_irregular": 3
}
CLASS_NAMES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]
NUM_CLASSES = len(CLASS_NAMES)


def discover_product_groups(dataset_dir=DEFAULT_DATASET_DIR):
    """
    Scans dataset folders, identifies original physical product crops,
    and maps all augmented variants back to their parent group_id.
    """
    groups_dict = {}      # group_id -> list of file records
    original_records = [] # pristine original records only
    all_records = []      # all records (original + augmented)

    for folder_name, label_id in LABEL_MAPPING.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        crop_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        for fname in crop_files:
            full_path = os.path.join(folder_path, fname)
            base_name, ext = os.path.splitext(fname)

            # Determine group_id (strip _aug_X suffix if present)
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

            if group_id not in groups_dict:
                groups_dict[group_id] = {
                    "group_id": group_id,
                    "folder": folder_name,
                    "label_id": label_id,
                    "class_name": CLASS_NAMES[label_id],
                    "original_file": fname if not is_aug else None,
                    "variants": []
                }

            if not is_aug:
                groups_dict[group_id]["original_file"] = fname
                original_records.append(record)
            else:
                groups_dict[group_id]["variants"].append(fname)

            all_records.append(record)

    return groups_dict, original_records, all_records


def generate_and_save_splits(dataset_dir=DEFAULT_DATASET_DIR, splits_dir=DEFAULT_SPLITS_DIR, seed=42):
    """
    Executes a stratified 70/15/15 Group Split across physical product groups
    and writes train_groups.csv, val_groups.csv, test_groups.csv.
    """
    os.makedirs(splits_dir, exist_ok=True)
    groups_dict, original_records, all_records = discover_product_groups(dataset_dir)

    unique_groups = list(groups_dict.keys())
    if not unique_groups:
        raise FileNotFoundError(f"[SPLIT ENGINE] No valid product groups found in '{dataset_dir}'.")

    group_labels  = [groups_dict[g]["label_id"] for g in unique_groups]
    group_classes = [groups_dict[g]["class_name"] for g in unique_groups]

    print(f"[SPLIT ENGINE] Found {len(unique_groups)} unique physical product groups across {NUM_CLASSES} classes.")
    for c_id, c_name in enumerate(CLASS_NAMES):
        count = sum(1 for l in group_labels if l == c_id)
        print(f"  - Class {c_id} ({c_name:12s}): {count:3d} physical product groups")

    # Step 1: 70% Train, 30% Temp (Val + Test)
    train_groups, temp_groups, train_y, temp_y = train_test_split(
        unique_groups, group_labels,
        test_size=0.30,
        random_state=seed,
        stratify=group_labels
    )

    # Step 2: Split Temp 50/50 -> 15% Val, 15% Test
    val_groups, test_groups, val_y, test_y = train_test_split(
        temp_groups, temp_y,
        test_size=0.50,
        random_state=seed,
        stratify=temp_y
    )

    # Anti-leakage verification
    train_set, val_set, test_set = set(train_groups), set(val_groups), set(test_groups)
    assert len(train_set & val_set) == 0, "LEAKAGE: Train and Val overlap!"
    assert len(train_set & test_set) == 0, "LEAKAGE: Train and Test overlap!"
    assert len(val_set & test_set) == 0, "LEAKAGE: Val and Test overlap!"
    print("[SPLIT ENGINE] Verification Passed: 0 overlapping groups across splits.")

    def save_group_csv(groups, filename, split_name):
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


def load_partitioned_datasets(dataset_dir=DEFAULT_DATASET_DIR, splits_dir=DEFAULT_SPLITS_DIR):
    """
    Loads samples strictly respecting pristine test/val rules:
      - Train set: includes original + augmented samples for train groups.
      - Val set:   includes PRISTINE original samples ONLY for val groups.
      - Test set:  includes PRISTINE original samples ONLY for test groups.
    """
    train_csv = os.path.join(splits_dir, "train_groups.csv")
    val_csv   = os.path.join(splits_dir, "val_groups.csv")
    test_csv  = os.path.join(splits_dir, "test_groups.csv")

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
            train_samples.append(r)
        elif g in val_groups:
            if not r["is_augmented"]:
                val_samples.append(r)
        elif g in test_groups:
            if not r["is_augmented"]:
                test_samples.append(r)

    print(f"\n[PARTITION SUMMARY - PRISTINE NON-LEAKED DATASET]")
    print(f"  Train samples (with balanced augmentations) : {len(train_samples):5d} ({len(train_groups)} product groups)")
    print(f"  Val samples   (100% pristine originals only): {len(val_samples):5d} ({len(val_groups)} product groups)")
    print(f"  Test samples  (100% pristine originals only): {len(test_samples):5d} ({len(test_groups)} product groups)")

    return train_samples, val_samples, test_samples
