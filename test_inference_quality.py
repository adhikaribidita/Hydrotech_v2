"""
test_inference_quality.py
=========================
Benchmarks OLD (single-pass) vs NEW (TTA + multi-scale v2) post-processing
pipelines on validation images.

Run with:  python test_inference_quality.py

Prints per image:
  - Component count  (old vs new)
  - Flood coverage % (old vs new)
  - Delta coverage

Saves 5-panel PNG comparisons to: test_results/quality_comparison/
"""

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ── Import improved post-processing from backend ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
from main import clean_mask, clean_mask_v2, _run_tta_inference

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = r"c:\Users\pinak\Desktop\hydrotech\HydroTech-AI-Flood-Detection"
MODEL_PATH = os.path.join(PROJECT, "best_model.pth")
VAL_IMG_DIR = os.path.join(PROJECT, "dataset", "val_images")
OUT_DIR = os.path.join(PROJECT, "test_results", "quality_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE  = 256
THRESHOLD = 0.5
DEVICE    = torch.device("cpu")

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model ...")
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3, classes=1, activation=None,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded.\n")


# ── Metrics helpers ───────────────────────────────────────────────────────────
def count_isolated_pixels(mask_bin):
    """Count foreground pixels that are completely isolated (8-connectivity)."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    return sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] == 1)


def count_components(mask_bin):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    return max(0, num_labels - 1)   # subtract background


# ── Run test ─────────────────────────────────────────────────────────────────
val_images = [f for f in sorted(os.listdir(VAL_IMG_DIR))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))][:8]

if not val_images:
    print("No validation images found in", VAL_IMG_DIR)
    sys.exit(1)

total_comp_old = 0
total_comp_new = 0

HDR = f"{'Image':<30} {'Comp_old':>10} {'Comp_new':>10} {'Flood%_old':>11} {'Flood%_new':>11} {'Delta':>8}"
print(HDR)
print("-" * len(HDR))


def add_label(img_in, label):
    """Add a text label banner above an image."""
    vis = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR) if img_in.ndim == 2 else img_in.copy()
    pad = np.zeros((28, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(pad, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    return np.vstack([pad, vis])


for fname in val_images:
    img_path = os.path.join(VAL_IMG_DIR, fname)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue

    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

    # ── OLD pipeline: single inference + legacy clean_mask ───────────────────
    with torch.no_grad():
        prob_single = torch.sigmoid(model(tensor)).squeeze().numpy()
    old_mask  = clean_mask(prob_single, THRESHOLD)
    comp_old  = count_components(old_mask)
    flood_old = round(float(old_mask.mean() * 100), 2)

    # ── NEW pipeline: TTA ensemble + multi-scale v2 ───────────────────────────
    prob_tta = _run_tta_inference(model, tensor)
    new_mask = clean_mask_v2(
        prob_tta,
        image_bgr_256=resized,
        threshold=THRESHOLD,
        use_crf=False,          # keep CRF off for reproducibility in test
    )
    comp_new  = count_components(new_mask)
    flood_new = round(float(new_mask.mean() * 100), 2)

    total_comp_old += comp_old
    total_comp_new += comp_new

    delta = flood_new - flood_old
    print(f"{fname:<30} {comp_old:>10} {comp_new:>10} {flood_old:>10.2f}% "
          f"{flood_new:>10.2f}% {delta:>+7.2f}%")

    # ── Save 5-panel comparison ───────────────────────────────────────────────
    old_vis  = (old_mask * 255).astype(np.uint8)
    new_vis  = (new_mask * 255).astype(np.uint8)
    heat_old = cv2.applyColorMap((prob_single * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_new = cv2.applyColorMap((prob_tta    * 255).astype(np.uint8), cv2.COLORMAP_JET)

    row = np.hstack([
        add_label(resized,  "Original"),
        add_label(heat_old, f"Heat single"),
        add_label(old_vis,  f"OLD ({comp_old}c,{flood_old}%)"),
        add_label(heat_new, f"Heat TTA"),
        add_label(new_vis,  f"NEW ({comp_new}c,{flood_new}%)"),
    ])
    cv2.imwrite(os.path.join(OUT_DIR, f"cmp_{fname}"), row)

# ── Summary ──────────────────────────────────────────────────────────────────
print("-" * len(HDR))
print(f"\n{'TOTAL':<30} {total_comp_old:>10} {total_comp_new:>10}")
print(f"\nComponent count  OLD pipeline : {total_comp_old}")
print(f"Component count  NEW pipeline : {total_comp_new}")
print(f"\n5-panel comparison images saved to: {OUT_DIR}")
