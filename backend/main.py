import os
from pathlib import Path
from datetime import datetime
from io import BytesIO
import base64

# Force CPU-only execution on Windows when CUDA-enabled PyTorch wheel is present
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors

import segmentation_models_pytorch as smp

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    Path("/app/best_model.pth"),                                    # Docker / Render
    BASE_DIR / "best_model.pth",                                    # local: backend/best_model.pth
    BASE_DIR.parent / "best_model.pth",                             # legacy: project root
    BASE_DIR.parent / "saved_model" / "unetpp_flood_full.pth",
    BASE_DIR.parent / "saved_model" / "unetpp_flood_weights.pth",
]
MODEL_PATH = next((p for p in MODEL_PATHS if p.exists()), MODEL_PATHS[1])
IMG_SIZE = 256
THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: DenseCRF edge refinement.
# Install with: pip install pydensecrf
# If not installed, CRF is silently skipped.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False


def _apply_crf(image_bgr: np.ndarray, prob_map: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """
    Refine a binary probability map with DenseCRF using the original image as
    the pairwise (appearance) feature.  Returns a refined float32 probability map.

    Falls back silently if pydensecrf is not installed.
    """
    if not _HAS_CRF:
        return prob_map

    h, w = prob_map.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Build 2-class softmax: [background_prob, flood_prob]
    flood_prob  = np.clip(prob_map, 1e-6, 1.0 - 1e-6).astype(np.float32)
    bg_prob     = 1.0 - flood_prob
    softmax     = np.stack([bg_prob, flood_prob], axis=0)          # (2, H, W)

    unary = unary_from_softmax(softmax)                            # (2, H*W)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)

    # Appearance kernel — bilateral: colour + position (snaps to image edges)
    d.addPairwiseBilateral(
        sxy=(10, 10),        # spatial std
        srgb=(13, 13, 13),   # colour std — tight so it snaps precisely
        rgbim=np.ascontiguousarray(image_rgb),
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    # Smoothness kernel — Gaussian: position only
    d.addPairwiseGaussian(
        sxy=(3, 3), compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(n_iter)
    refined = np.array(Q).reshape(2, h, w)[1]   # class-1 (flood) probability
    return refined.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING V2: multi-scale morphological pipeline.
# ─────────────────────────────────────────────────────────────────────────────
def clean_mask_v2(
    prob_map: np.ndarray,
    image_bgr_256: np.ndarray | None = None,
    threshold: float = THRESHOLD,
    use_crf: bool = True,
) -> np.ndarray:
    """
    Convert a raw sigmoid probability map to a clean binary flood mask.

    Pipeline:
      1. Gaussian smoothing on raw probability  → reduces 256-grid aliasing
      2. Optional DenseCRF edge refinement      → snaps boundaries to image edges
      3. Hard thresholding
      4. Fine morphological opening             → kills isolated salt pixels
      5. Coarse morphological closing           → fills interior pepper gaps
      6. Connected-component filtering          → drops stray micro-blobs
      7. Hole filling via contour drawing       → closes enclosed lakes
      8. Boundary smoothing (dilate+erode)      → removes staircase jaggies

    Returns:
        uint8 array of shape (H, W) with values 0 or 1.
    """
    h, w = prob_map.shape[:2]
    prob = prob_map.astype(np.float32)

    # ── 1. Gaussian smoothing (σ=1.0 → subtle anti-aliasing) ─────────────────
    prob = cv2.GaussianBlur(prob, (5, 5), sigmaX=1.0)

    # ── 2. Optional DenseCRF refinement ──────────────────────────────────────
    if use_crf and image_bgr_256 is not None:
        prob = _apply_crf(image_bgr_256, prob)

    # ── 3. Threshold ─────────────────────────────────────────────────────────
    mask = (prob > threshold).astype(np.uint8)

    # ── 4. Fine morphological opening (removes isolated salt pixels) ──────────
    # Kernel ≈ 0.8% of shortest dimension (min 3, always odd)
    k_fine = max(3, int(min(h, w) * 0.008))
    if k_fine % 2 == 0:
        k_fine += 1
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fine, k_fine))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_open)

    # ── 5. Coarse morphological closing (fills interior pepper gaps) ──────────
    # Kernel ≈ 4% of shortest dimension (min 7, always odd)
    k_coarse = max(7, int(min(h, w) * 0.04))
    if k_coarse % 2 == 0:
        k_coarse += 1
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_coarse, k_coarse))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_close)

    # ── 6. Connected-component filtering (remove stray micro-blobs) ───────────
    # Threshold: ≥ 0.10% of image area (stricter than previous 0.15%)
    min_area = max(20, int(h * w * 0.001))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    # ── 7. Hole filling (enclosed background regions inside flood) ─────────────
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    # ── 8. Contour smoothing (remove staircase jaggies at boundary) ───────────
    # Dilate then erode with a small round kernel to smooth the contour line
    k_smooth = 3
    kern_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_smooth, k_smooth))
    smoothed = cv2.morphologyEx(filled, cv2.MORPH_DILATE, kern_smooth)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_ERODE,  kern_smooth)

    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Legacy clean_mask (kept for backward compatibility with test scripts)
# ─────────────────────────────────────────────────────────────────────────────
def clean_mask(prob_map: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Backward-compatible wrapper — calls clean_mask_v2 without CRF.
    """
    return clean_mask_v2(prob_map, image_bgr_256=None, threshold=threshold, use_crf=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA)
# Averages predictions from 4 orientations for more robust output.
# ─────────────────────────────────────────────────────────────────────────────
def _run_tta_inference(model, tensor: torch.Tensor) -> np.ndarray:
    """
    Run model at 4 geometric augmentations and average the probability maps.
    Augmentations: original, horizontal flip, vertical flip, both flips.
    Each prediction is un-flipped before averaging so maps align correctly.
    """
    preds: list[np.ndarray] = []

    augments = [
        {},
        {"flip_h": True},
        {"flip_v": True},
        {"flip_h": True, "flip_v": True},
    ]

    for aug in augments:
        t = tensor.clone()
        if aug.get("flip_h"):
            t = t.flip(3)   # flip width (W) dimension
        if aug.get("flip_v"):
            t = t.flip(2)   # flip height (H) dimension

        with torch.inference_mode():
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()

        # Undo flips on the output probability map to realign
        if aug.get("flip_v"):
            prob = np.flipud(prob)
        if aug.get("flip_h"):
            prob = np.fliplr(prob)

        preds.append(prob.astype(np.float32))

    return np.mean(preds, axis=0)


# Use cuda.device_count() > 0 (not just is_available()) to handle the case
# where the CUDA runtime DLL is present but no physical GPU is accessible.
DEVICE = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")

app = FastAPI(title="HydroTech AI Flood Detection API")

# Configure CORS to accept requests from any origin (public inference API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_model():
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    model = create_model()
    # Always load weights to CPU first, then move to target device.
    # This safely handles checkpoints saved on GPU when running CPU-only.
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model


def fallback_inference(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_bin = (mask > 127).astype(np.uint8) * 255

    heatmap = cv2.applyColorMap(mask_bin, cv2.COLORMAP_JET)
    overlay = image_bgr.copy()
    # Deep blue flood overlay (satellite convention)
    overlay[mask_bin == 255] = np.clip(
        overlay[mask_bin == 255].astype(np.float32) * 0.30
        + np.array([210, 70, 0], dtype=np.float32) * 0.70,
        0, 255,
    ).astype(np.uint8)

    flood_percent = round(float(mask_bin.mean() / 255 * 100), 2)
    if flood_percent < 5:
        status = "LOW RISK"
    elif flood_percent < 30:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    return overlay, mask_bin, heatmap, status, f"{flood_percent}%"


model = None
model_load_error = None
try:
    model = load_model()
except Exception as exc:
    model_load_error = str(exc)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH.name),
        "error": model_load_error,
        "fallback_inference": model is None,
        "crf_available": _HAS_CRF,
        "tta_enabled": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: cap image longest-side to MAX_PREVIEW_PX before base64 encoding.
# Prevents the /predict response and /report payload from ballooning to
# tens of MB for large input images (e.g. 4000×1000 px).
# ─────────────────────────────────────────────────────────────────────────────
MAX_PREVIEW_PX = 1024

def _cap_image_size(img_bgr: np.ndarray, max_px: int = MAX_PREVIEW_PX) -> np.ndarray:
    """Proportionally resize img_bgr so the longest side ≤ max_px."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return img_bgr
    scale = max_px / longest
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    body = await file.read()
    pil_img = Image.open(BytesIO(body)).convert("RGB")
    orig_w, orig_h = pil_img.size

    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Run U-Net++ prediction, or fallback to adaptive contours
    if model is not None:
        overlay, mask, heatmap, status, coverage = run_inference(image_bgr)
    else:
        overlay, mask, heatmap, status, coverage = run_inference_fallback(image_bgr)

    if overlay is None:
        raise HTTPException(status_code=500, detail="Inference failed")

    # Cap images to MAX_PREVIEW_PX before encoding to keep payload size manageable.
    # Large originals (e.g. 4000×1000) produce ~47 MB of base64 across 3 images;
    # capping at 1024px longest-side brings this down to under 2 MB.
    overlay_sm  = _cap_image_size(overlay)
    mask_sm     = _cap_image_size(mask)
    heatmap_sm  = _cap_image_size(heatmap)

    _, overlay_png = cv2.imencode(".png", overlay_sm)
    _, mask_png    = cv2.imencode(".png", mask_sm)
    _, heatmap_png = cv2.imencode(".png", heatmap_sm)

    response = {
        "status": status,
        "coverage": coverage,
        "overlay": base64.b64encode(overlay_png.tobytes()).decode("utf-8"),
        "mask": base64.b64encode(mask_png.tobytes()).decode("utf-8"),
        "heatmap": base64.b64encode(heatmap_png.tobytes()).decode("utf-8"),
        "original_width": orig_w,
        "original_height": orig_h,
    }
    return JSONResponse(response)


@app.post("/report")
async def generate_report(payload: dict):
    required = ["status", "coverage", "overlay", "mask", "heatmap", "original_width", "original_height"]
    if not all(key in payload for key in required):
        raise HTTPException(status_code=400, detail="Missing report fields")

    try:
        output = BytesIO()
        pdf = canvas.Canvas(output, pagesize=letter)
        width, height = letter # 612 x 792 points

        # --- STYLE DEFINITIONS ---
        navy_dark = colors.HexColor("#001525")
        cyan_glow = colors.HexColor("#00b4d8")
        slate_text = colors.HexColor("#2f3e46")
        grey_light = colors.HexColor("#f8f9fa")
        red_alarm = colors.HexColor("#e63946")
        green_safe = colors.HexColor("#2a9d8f")
        yellow_warn = colors.HexColor("#f4a261")

        # --- HEADER SECTION ---
        # Top banner background block
        pdf.setFillColor(navy_dark)
        pdf.rect(0, height - 90, width, 90, stroke=0, fill=1)

        # Cyan decorative stripe
        pdf.setFillColor(cyan_glow)
        pdf.rect(0, height - 94, width, 4, stroke=0, fill=1)

        # Logo Vector Graphic
        pdf.setStrokeColor(cyan_glow)
        pdf.setLineWidth(1.5)
        pdf.circle(48, height - 45, 18, stroke=1, fill=0)
        pdf.setStrokeColor(colors.white)
        pdf.circle(48, height - 45, 12, stroke=1, fill=0)
        pdf.setStrokeColor(cyan_glow)
        pdf.line(30, height - 45, 66, height - 45)
        pdf.line(48, height - 63, 48, height - 27)

        # Main Titles
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(85, height - 42, "HYDROTECH ANALYTICAL PLATFORM")
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#90e0ef"))
        pdf.drawString(85, height - 58, "SATELLITE SPECTRAL IMAGERY & DEEP FLOOD SEGMENTATION REPORT")

        # --- CASE METADATA BOX ---
        pdf.setFillColor(grey_light)
        pdf.setStrokeColor(colors.HexColor("#e9ecef"))
        pdf.setLineWidth(1)
        pdf.roundRect(40, height - 200, width - 80, 90, 8, stroke=1, fill=1)

        # Metadata contents
        pdf.setFillColor(slate_text)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(55, height - 130, "SESSION METADATA:")

        pdf.setFont("Helvetica", 9)
        pdf.drawString(55, height - 150, f"Analysis Time:  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        pdf.drawString(55, height - 165, f"Source Size:      {payload['original_width']} x {payload['original_height']} pixels")
        pdf.drawString(55, height - 180, "Inference Mode:  U-Net++ (EfficientNet-B3) + TTA + Multi-Scale Post-Processing")

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(340, height - 130, "THREAT RATINGS:")

        # Risk level status styling
        status = payload['status'].upper()
        if "LOW" in status:
            pdf.setFillColor(green_safe)
        elif "MODERATE" in status:
            pdf.setFillColor(yellow_warn)
        else:
            pdf.setFillColor(red_alarm)

        pdf.drawString(340, height - 150, f"RISK LEVEL:     {status}")
        pdf.setFillColor(navy_dark)
        pdf.drawString(340, height - 168, f"FLOOD COVER:  {payload['coverage']}")

        # --- SPATIAL IMAGERY SECTION (GRID) ---
        pdf.setFillColor(navy_dark)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, height - 225, "SPATIAL WATER SEGMENTATION VISUALIZATIONS")
        pdf.setStrokeColor(colors.HexColor("#dee2e6"))
        pdf.setLineWidth(0.5)
        pdf.line(40, height - 230, width - 40, height - 230)

        # Helper: decode base64 → PIL RGB → ReportLab ImageReader.
        # Using mask=None (not mask='auto') because OpenCV-encoded PNGs have no
        # alpha channel; mask='auto' triggers a slow alpha-detection scan.
        def draw_image(b64_data, x, y, w, h):
            raw = base64.b64decode(b64_data)
            img = Image.open(BytesIO(raw)).convert("RGB")
            img_reader = ImageReader(img)
            # Bounding frame
            pdf.setStrokeColor(colors.HexColor("#ced4da"))
            pdf.setLineWidth(1)
            pdf.rect(x - 2, y - 2, w + 4, h + 4, stroke=1, fill=0)
            pdf.drawImage(img_reader, x, y, width=w, height=h, mask=None)

        # Draw Overlay and Binary Mask side-by-side
        draw_image(payload["overlay"], 50, height - 480, 230, 230)
        pdf.setFillColor(slate_text)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(50, height - 495, "FIGURE 01: CLASSIFIED FLOOD OVERLAY (BLUE VECTORS)")

        draw_image(payload["mask"], 330, height - 480, 230, 230)
        pdf.drawString(330, height - 495, "FIGURE 02: BINARY CLASSIFICATION WATER MASK")

        # Draw Probability Heatmap bottom-left
        draw_image(payload["heatmap"], 50, height - 740, 230, 230)
        pdf.drawString(50, height - 755, "FIGURE 03: SIGMOID PROBABILITY DENSITY HEATMAP")

        # --- TECHNICAL ADVISORY SUMMARY (BOTTOM-RIGHT) ---
        pdf.setFillColor(grey_light)
        pdf.roundRect(330, height - 740, 230, 230, 6, stroke=1, fill=1)

        pdf.setFillColor(navy_dark)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(345, height - 540, "TECHNICAL ASSESSMENT BRIEF")
        pdf.setStrokeColor(colors.HexColor("#e9ecef"))
        pdf.line(345, height - 545, 545, height - 545)

        pdf.setFont("Helvetica", 7.5)
        summary_text = [
            "This automated report classifies surface water boundaries",
            "extracted from multispectral satellite imagery.",
            "",
            "The AI model uses a nested U-Net++ topology with",
            "dense skip paths and an EfficientNet-B3 encoder.",
            "Predictions are refined via Test-Time Augmentation",
            "(4-orientation ensemble) and a multi-scale post-",
            "processing pipeline with optional DenseCRF.",
            "",
            "ACTION ADVISORY:",
            "• Low Risk: Routine monitoring of hydrological basins.",
            "• Moderate Risk: Deploy remote telemetry gauges.",
            "• Critical Risk: Alert localized civic response centers."
        ]

        text_y = height - 565
        for line in summary_text:
            if "ACTION ADVISORY" in line:
                pdf.setFont("Helvetica-Bold", 8)
                pdf.setFillColor(colors.HexColor("#495057"))
            elif "•" in line:
                pdf.setFont("Helvetica", 7.5)
                pdf.setFillColor(slate_text)
            pdf.drawString(345, text_y, line)
            text_y -= 12

        # --- FOOTER SECTION ---
        pdf.setStrokeColor(colors.HexColor("#e9ecef"))
        pdf.setLineWidth(1)
        pdf.line(40, 45, width - 40, 45)
        pdf.setFont("Helvetica-Oblique", 7.5)
        pdf.setFillColor(colors.HexColor("#adb5bd"))
        pdf.drawString(40, 30, "CONFIDENTIAL REPORT - GENERATED AUTOMATICALLY BY HYDROTECH PLATFORM ENGINE")
        pdf.drawRightString(width - 40, 30, "PAGE 1 OF 1")

        pdf.showPage()
        pdf.save()
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=hydrotech_report_{datetime.now().strftime('%Y%m%d')}.pdf"},
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")


def run_inference(image_bgr):
    """
    Full inference pipeline with:
      - Test-Time Augmentation (4-orientation ensemble)
      - Multi-scale morphological post-processing (clean_mask_v2)
      - Optional DenseCRF edge refinement
      - Improved blue-teal flood overlay (satellite convention)
    """
    if model is None:
        return fallback_inference(image_bgr)

    try:
        orig_h, orig_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(DEVICE)
        )

        # ── TTA: average 4-orientation predictions ───────────────────────────
        prob = _run_tta_inference(model, tensor)   # (256, 256) float32

        # ── Multi-scale post-processing + optional CRF ───────────────────────
        clean = clean_mask_v2(
            prob,
            image_bgr_256=resized,   # pass resized image for CRF colour guidance
            threshold=THRESHOLD,
            use_crf=_HAS_CRF,
        )                              # (256, 256) uint8 0/1
        mask = (clean * 255).astype(np.uint8)   # (256, 256) uint8 0/255

        # Resize back to original resolution
        mask_full = cv2.resize(
            mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # Heatmap from TTA-averaged probability (smooth gradient, not binary)
        heatmap_256 = cv2.applyColorMap(
            (prob * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.resize(heatmap_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # ── Flood overlay: deep blue (satellite convention) ──────────────────
        # 30% original + 70% deep blue [BGR: 210, 70, 0]
        overlay = image_bgr.copy()
        flood_pixels = mask_full == 255
        overlay[flood_pixels] = np.clip(
            overlay[flood_pixels].astype(np.float32) * 0.30
            + np.array([210, 70, 0], dtype=np.float32) * 0.70,
            0, 255,
        ).astype(np.uint8)

        flood_percent = round(float(clean.mean() * 100), 2)
        if flood_percent < 5:
            status = "LOW RISK"
        elif flood_percent < 30:
            status = "MODERATE RISK"
        else:
            status = "HIGH RISK"

        return overlay, mask_full, heatmap, status, f"{flood_percent}%"
    except Exception:
        return fallback_inference(image_bgr)



def run_inference_fallback(image_bgr):
    try:
        resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Simple adaptive threshold to mock flooded areas
        thresh_val = int(np.percentile(gray, 60))
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        heatmap = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        overlay = resized.copy()
        
        # Deep blue overlay (satellite convention)
        overlay[mask == 255] = np.clip(
            overlay[mask == 255].astype(np.float32) * 0.30
            + np.array([210, 70, 0], dtype=np.float32) * 0.70,
            0, 255
        ).astype(np.uint8)

        flood_percent = round(float(mask.mean() / 255 * 100), 2)
        if flood_percent < 5:
            status = "LOW RISK"
        elif flood_percent < 30:
            status = "MODERATE RISK"
        else:
            status = "HIGH RISK"

        return overlay, mask, heatmap, status, f"{flood_percent}%"
    except Exception:
        return None, None, None, "ERROR", "0%"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
