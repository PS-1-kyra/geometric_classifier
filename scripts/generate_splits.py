#!/usr/bin/env python3
"""
scripts/generate_splits.py — Zero-Leakage GroupShuffleSplit Generator

Executes a group-aware split (70% Train, 15% Val, 15% Test) strictly by physical
product base IDs (`group_id`), ensuring zero data leakage across splits.
"""

import sys
import argparse
from pathlib import Path

# Ensure repo root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_engine import generate_and_save_splits, DEFAULT_DATASET_DIR, DEFAULT_SPLITS_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate zero-leakage physical product group splits.")
    parser.add_argument("--dataset-dir", type=str, default=str(REPO_ROOT / DEFAULT_DATASET_DIR),
                        help="Path to labeled dataset directory (containing 0_flat, 1_cylindrical, etc.)")
    parser.add_argument("--splits-dir", type=str, default=str(REPO_ROOT / DEFAULT_SPLITS_DIR),
                        help="Output directory to write train_groups.csv, val_groups.csv, test_groups.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print("  PS-1 ZERO-LEAKAGE GROUP SPLIT GENERATOR")
    print("=" * 70)
    print(f"Dataset directory : {args.dataset_dir}")
    print(f"Splits directory  : {args.splits_dir}")
    print(f"RNG Seed          : {args.seed}\n")

    generate_and_save_splits(dataset_dir=args.dataset_dir, splits_dir=args.splits_dir, seed=args.seed)
    print("\n[OK] Partition tables successfully generated with zero group leakage!")


if __name__ == "__main__":
    main()
