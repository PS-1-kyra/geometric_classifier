#!/usr/bin/env python3
"""
========================================================================================
scripts/harvest_crops.py — Context-Padded Object Crop Harvester
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & MOTIVATION: THE 15% CONTEXTUAL PADDING PRINCIPLE
----------------------------------------------------------------------------------------
[Basic Concept: Why Not Just Crop the Bounding Box?]:
  When an object detector (such as Mask R-CNN or YOLO) identifies a product on a shelf,
  it returns a tight bounding box [x_min, y_min, x_max, y_max] that tightly hugs the perimeter
  of the item.
  In standard 2D image classification, cropping tightly to this box is common practice.
  However, in 3D geometric primitive classification, tight cropping causes severe failure:
    1. Boundary Bleed & Depth Clipping:
       Monocular depth foundation models (like Depth Anything V2) rely on spatial contextual
       gradients to estimate relative depth. When a product is cropped tightly, the outer
       edges of the object coincide with the image boundary. The depth network cannot see
       where the product ends and the shelf begins, leading to severe edge distortion and
       curvature flattening.
    2. Contour & Silhouette Truncation:
       If a bottle's cap or a can's curved rim is clipped by even 1 pixel, morphological contour
       algorithms (convex hull, Hu moments, eccentricity) produce false geometric statistics.

[The Solution: 15% Contextual Margin Padding]:
  This script expands every proposal bounding box outward by 15% of its width and height:
    x1_padded = max(0, x1 - 0.15 * width)
    y1_padded = max(0, y1 - 0.15 * height)
    x2_padded = min(image_w, x2 + 0.15 * width)
    y2_padded = min(image_h, y2 + 0.15 * height)

  This provides Depth Anything V2 with surrounding shelf background context, allowing it to
  sharply separate the object's foreground 3D surface from the background plane.
========================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Ensure repo root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_padding(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    pad_pct: float = 0.15
) -> tuple[int, int, int, int]:
    """
    Expands a bounding box by `pad_pct` (default 15%) along each boundary, clamped to image dimensions.

    Mathematical Formulation:
      bw = x2 - x1,  bh = y2 - y1
      pad_x = int(bw * pad_pct),  pad_y = int(bh * pad_pct)
      x1_pad = max(0, x1 - pad_x)
      y1_pad = max(0, y1 - pad_y)
      x2_pad = min(img_w, x2 + pad_x)
      y2_pad = min(img_h, y2 + pad_y)

    Args:
        x1, y1, x2, y2: Raw bounding box coordinates.
        img_w, img_h:   Full shelf image dimensions.
        pad_pct:        Fractional expansion factor (default: 0.15).

    Returns:
        Padded and boundary-clamped coordinates (px1, py1, px2, py2).
    """
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_pct)
    py = int(bh * pad_pct)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def harvest_from_boxes(
    image_bgr: np.ndarray,
    boxes: list | np.ndarray,
    output_dir: str,
    stem: str = "img",
    pad_pct: float = 0.15,
    min_size: int = 32
) -> list[str]:
    """
    Crops objects from a shelf image using bounding boxes with contextual padding and saves them.

    Args:
        image_bgr:  Full scene image as uint8 numpy array (H, W, 3).
        boxes:      List or array of bounding boxes [x1, y1, x2, y2, ...].
        output_dir: Destination folder path for saved crops.
        stem:       Prefix stem for generated crop filenames.
        pad_pct:    Contextual padding fraction.
        min_size:   Minimum width and height in pixels to filter out tiny noise detections.

    Returns:
        saved_crops: List of filepaths to successfully saved crop images.
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]
    saved_crops = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        px1, py1, px2, py2 = add_padding(x1, y1, x2, y2, w, h, pad_pct=pad_pct)

        crop_w = px2 - px1
        crop_h = py2 - py1
        # Guardrail against tiny noise detections
        if crop_w < min_size or crop_h < min_size:
            continue

        crop = image_bgr[py1:py2, px1:px2]
        crop_filename = f"{stem}_crop_{idx:04d}.png"
        save_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(save_path, crop)
        saved_crops.append(save_path)

    return saved_crops


def harvest_image_proposals(
    image_path: str | Path,
    output_dir: str,
    device: str = "cpu",
    score_thresh: float = 0.5,
    pad_pct: float = 0.15
) -> list[str]:
    """
    Automated foreground proposal harvesting using a pretrained torchvision Mask R-CNN detector.

    Args:
        image_path:   Path to the shelf scene image.
        output_dir:   Directory to store harvested crops.
        device:       Execution device ('cuda' or 'cpu').
        score_thresh: Minimum objectness confidence threshold for candidate detections.
        pad_pct:      Contextual padding fraction (default: 0.15).

    Returns:
        List of saved crop image filepaths.
    """
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Warning: unable to load image from {image_path}")
        return []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torchvision.transforms.functional.to_tensor(img_rgb).to(device)

    # Load pretrained Mask R-CNN backbone with ResNet-50 FPN
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()

    with torch.no_grad():
        preds = model([tensor])[0]

    # Filter detected bounding boxes by score threshold
    scores = preds["scores"].cpu().numpy()
    boxes = preds["boxes"].cpu().numpy()
    valid_boxes = boxes[scores >= score_thresh]

    stem = Path(image_path).stem
    return harvest_from_boxes(img_bgr, valid_boxes, output_dir, stem=stem, pad_pct=pad_pct)


def main():
    """CLI entrypoint for crop harvesting."""
    parser = argparse.ArgumentParser(description="Harvest 15% context-padded object crops from images.")
    parser.add_argument("--images-dir", type=str, default="raw_images", help="Folder containing raw shelf images")
    parser.add_argument("--output-dir", type=str, default="dataset/harvested_crops", help="Output crops folder")
    parser.add_argument("--pad-pct", type=float, default=0.15, help="Contextual padding margin (default 0.15 = 15%)")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Detection confidence score threshold")
    parser.add_argument("--device", type=str, default="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("  PS-1 CONTEXT-PADDED CROP HARVESTER (15% PADDING)")
    print("=" * 70)
    print(f"Images directory : {args.images_dir}")
    print(f"Output directory : {args.output_dir}")
    print(f"Padding margin   : {args.pad_pct * 100:.1f}%\n")

    images_path = Path(args.images_dir)
    if not images_path.exists():
        print(f"Images folder '{args.images_dir}' does not exist. Creating folder.")
        images_path.mkdir(parents=True, exist_ok=True)
        return

    img_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not img_files:
        print(f"No image files found in '{args.images_dir}'. Place input scenes there and re-run.")
        return

    total_crops = 0
    for img_p in img_files:
        print(f"Processing scene: {img_p.name} ...")
        crops = harvest_image_proposals(
            img_p, args.output_dir,
            device=args.device,
            score_thresh=args.score_thresh,
            pad_pct=args.pad_pct
        )
        print(f"  -> Extracted {len(crops)} crops")
        total_crops += len(crops)

    print(f"\n[OK] Completed! Total crops harvested: {total_crops} in '{args.output_dir}'")


if __name__ == "__main__":
    main()
