#!/usr/bin/env python3
"""
========================================================================================
scripts/generate_splits.py — Zero-Data-Leakage GroupShuffleSplit Manifest Generator
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & PURPOSE: REPRODUCIBLE ZERO-LEAKAGE SPLIT GENERATION
----------------------------------------------------------------------------------------
[Basic Concept]:
  Machine learning models evaluated on leaked test data give a dangerous illusion of success.
  In retail, multiple crops of the same physical product must never be allowed to cross split
  boundaries.
  This CLI utility executes the group-aware, stratified 70/15/15 dataset partition using
  `src.dataset_engine.generate_and_save_splits`, generating permanent CSV manifests:
    - `splits/train_groups.csv`: 70% of physical product groups (receives balanced augmentations)
    - `splits/val_groups.csv`:   15% of physical product groups (100% pristine original crops only)
    - `splits/test_groups.csv`:  15% of physical product groups (100% pristine original crops only)

[Guarantees]:
  1. Identity Isolation: No two crops belonging to the same physical SKU will ever appear
     in different splits.
  2. Class Stratification: The natural class proportions of Flat, Cylindrical, Cuboid, and
     Irregular items are preserved identically across train, val, and test partitions.
  3. Determinism: Controlled via the `--seed` flag (default: 42) for exact mathematical
     reproducibility across runs and research environments.
========================================================================================
"""

import sys
import argparse
from pathlib import Path

# Ensure repository root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_engine import (
    generate_and_save_splits,
    DEFAULT_DATASET_DIR,
    DEFAULT_SPLITS_DIR,
)


def main():
    """CLI entrypoint for generating group-aware dataset splits."""
    parser = argparse.ArgumentParser(
        description="Generate zero-leakage physical product group splits (70/15/15)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(REPO_ROOT / DEFAULT_DATASET_DIR),
        help="Path to labeled dataset directory (containing 0_flat, 1_cylindrical, etc.)"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=str(REPO_ROOT / DEFAULT_SPLITS_DIR),
        help="Output directory to write train_groups.csv, val_groups.csv, test_groups.csv"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic, perfectly reproducible partition tables"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PS-1 ZERO-LEAKAGE GROUP SPLIT GENERATOR")
    print("=" * 70)
    print(f"Dataset directory : {args.dataset_dir}")
    print(f"Splits directory  : {args.splits_dir}")
    print(f"RNG Seed          : {args.seed}\n")

    generate_and_save_splits(
        dataset_dir=args.dataset_dir,
        splits_dir=args.splits_dir,
        seed=args.seed
    )
    print("\n[OK] Partition tables successfully generated with zero group leakage!")


if __name__ == "__main__":
    main()
