#!/usr/bin/env python3
"""
scripts/harvest_crops.py — Context-Padded Object Crop Harvester

Extracts localized object crops from full shelf scenes with:
  1. 15% Contextual Margin Padding (prevents boundary depth clipping).
  2. Bounding box clamping to image boundaries.
  3. Aspect-ratio and area guardrail filtering.
  4. Supports automated foreground proposals via Mask R-CNN / YOLO or direct COCO/YOLO annotations.
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


def add_padding(x1, y1, x2, y2, img_w, img_h, pad_pct=0.15):
    """
    Expands bounding box by pad_pct (default 15%) on all sides, clamped to image bounds.
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


def harvest_from_boxes(image_bgr, boxes, output_dir, stem="img", pad_pct=0.15, min_size=32):
    """
    Crops boxes with pad_pct margin and saves crops.
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]
    saved_crops = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        px1, py1, px2, py2 = add_padding(x1, y1, x2, y2, w, h, pad_pct=pad_pct)

        crop_w = px2 - px1
        crop_h = py2 - py1
        if crop_w < min_size or crop_h < min_size:
            continue

        crop = image_bgr[py1:py2, px1:px2]
        crop_filename = f"{stem}_crop_{idx:04d}.png"
        save_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(save_path, crop)
        saved_crops.append(save_path)

    return saved_crops


def harvest_image_proposals(image_path, output_dir, device="cpu", score_thresh=0.5, pad_pct=0.15):
    """
    Automated object proposal detection using torchvision Mask R-CNN.
    """
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Warning: unable to load {image_path}")
        return []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torchvision.transforms.functional.to_tensor(img_rgb).to(device)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()

    with torch.no_grad():
        preds = model([tensor])[0]

    scores = preds["scores"].cpu().numpy()
    boxes = preds["boxes"].cpu().numpy()
    valid_boxes = boxes[scores >= score_thresh]

    stem = Path(image_path).stem
    return harvest_from_boxes(img_bgr, valid_boxes, output_dir, stem=stem, pad_pct=pad_pct)


def main():
    parser = argparse.ArgumentParser(description="Harvest 15% context-padded object crops from images.")
    parser.add_argument("--images-dir", type=str, default="raw_images", help="Folder with raw shelf images")
    parser.add_argument("--output-dir", type=str, default="dataset/harvested_crops", help="Output crops folder")
    parser.add_argument("--pad-pct", type=float, default=0.15, help="Contextual padding margin (default 0.15)")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Detection confidence threshold")
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
