import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import gradio as gr
from earth_b64 import earth_image_b64

# =====================================
# SYSTEM OPTIMIZATIONS
# =====================================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# =====================================
# SETTINGS & MODEL INITIALIZATION
# =====================================
IMG_SIZE   = 256
THRESHOLD  = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    r"D:\FLOOD_DETECTION_PROJECT\best_model.pth",                      # original hardcoded legacy path
    os.path.join(BASE_DIR, "best_model.pth"),                          # project root
    os.path.join(BASE_DIR, "backend", "best_model.pth"),                # backend/best_model.pth
    os.path.join(BASE_DIR, "saved_model", "unetpp_flood_full.pth"),     # saved_model/unetpp_flood_full.pth
    os.path.join(BASE_DIR, "saved_model", "unetpp_flood_weights.pth"),  # saved_model/unetpp_flood_weights.pth
]
MODEL_PATH = next((p for p in MODEL_PATHS if os.path.exists(p)), MODEL_PATHS[1])

model = smp.UnetPlusPlus(
    encoder_name    = "efficientnet-b3",
    encoder_weights = None,
    in_channels     = 3,
    classes         = 1,
    activation      = None
).to(DEVICE)

model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model = model.to(DEVICE)
        model.eval()
        model_loaded = True
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model weights from {MODEL_PATH}: {e}")
else:
    print(f"Model weights not found. Running in advanced CV fallback mode.")

# =====================================
# POST-PROCESSING
# Remove salt-and-pepper noise from raw model output.
# Resolution-agnostic: kernel sizes scale with image dimensions.
# =====================================
def clean_mask(prob_map: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Convert probability map -> clean binary mask (0/1 uint8).
    Steps:
      1. Threshold
      2. Morphological opening  (removes isolated salt pixels)
      3. Morphological closing  (fills isolated pepper holes)
      4. Connected-component filtering  (drops tiny stray blobs < 0.15% of image)
      5. Contour hole-fill  (fills enclosed background regions inside flood area)
    """
    h, w = prob_map.shape[:2]

    # 1. Threshold
    mask = (prob_map > threshold).astype(np.uint8)

    # 2. Auto kernel sizes (scale with image, always odd, minimum values enforced)
    k_open = max(3, int(min(h, w) * 0.01))
    if k_open % 2 == 0:
        k_open += 1
    k_close = max(5, int(min(h, w) * 0.03))
    if k_close % 2 == 0:
        k_close += 1

    kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

    # 3. Opening: eliminate isolated foreground (salt) pixels
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern_open)

    # 4. Closing: fill isolated background (pepper) pixels within flood zone
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_close)

    # 5. Remove blobs smaller than 0.15% of image area
    min_area = max(30, int(h * w * 0.0015))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):          # label 0 = background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    # 6. Fill enclosed holes inside flood contours
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    return filled


# =====================================
# INFERENCE ENGINE
# =====================================
def run_cv_fallback(image_bgr):
    """
    Advanced CV-based water/flood segmentation on RGB/HSV/Lab spaces.
    Returns:
        clean (256, 256) uint8 0/1 array
        prob (256, 256) float32 probability map
    """
    resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    b = resized[:, :, 0].astype(np.float32)
    g = resized[:, :, 1].astype(np.float32)
    r = resized[:, :, 2].astype(np.float32)
    
    # Calculate ratios and indices
    b_r_ratio = b / (r + 1.0)
    g_r_ratio = g / (r + 1.0)
    ndwi = (g - r) / (g + r + 1e-6)
    br_index = (b - r) / (b + r + 1e-6)
    
    # Smoothness / Texture filter (water surface is flat/smooth)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    smoothness = cv2.boxFilter(grad_mag, -1, (7, 7))
    is_smooth = smoothness < 12.0
    
    # Exclude conditions
    is_veg = (h >= 35) & (h <= 85) & (b_r_ratio < 0.75)
    is_shadow = v < 15  # Raised shadow threshold slightly
    is_bright = (v > 225) & (s < 50)
    
    # Base probability score
    prob = np.zeros_like(v, dtype=np.float32)
    
    # Blue/Teal water color matching
    prob[(h >= 75) & (h <= 145) & (s >= 15) & (v >= 15) & (v <= 225)] += 0.6
    
    # Green water color matching (e.g. algae/turbid water)
    prob[(h >= 35) & (h < 75) & (s >= 15) & (v >= 15) & (v <= 180) & (g_r_ratio > 1.0)] += 0.5
    
    # Muddy/Brown water color matching (brown has Hue in orange/yellow range, must be smooth)
    prob[(h >= 5) & (h < 35) & (s >= 10) & (s <= 180) & (v >= 20) & (v <= 180) & is_smooth] += 0.5
    
    # Deep/Dark water matching (enforces v >= 15)
    prob[(v >= 15) & (v < 35) & (s < 50) & is_smooth & (b_r_ratio > 0.95)] += 0.5
    
    # Add index components
    prob += 0.2 * np.clip(br_index, 0.0, 1.0)
    prob += 0.2 * np.clip(ndwi, 0.0, 1.0)
    
    # Penalize non-water features
    prob[is_veg] -= 0.6
    prob[is_shadow] -= 0.5
    prob[is_bright] -= 0.5
    
    prob = np.clip(prob, 0.0, 1.0)
    
    # Apply standard morphological cleanup pipeline
    clean = clean_mask(prob, THRESHOLD)
    return clean, prob


def predict_flood(image_bgr):
    if image_bgr is None:
        return None, None, None, "Error", "0%"

    h_orig, w_orig = image_bgr.shape[:2]
    image_resized  = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    if model_loaded:
        image_rgb  = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(DEVICE)
        img_tensor = img_tensor.unsqueeze(0).div(255.0)

        with torch.inference_mode():
            output   = model(img_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

        # --- Post-process: remove salt-and-pepper noise ---
        clean      = clean_mask(prob_map, THRESHOLD)          # (256,256) uint8 0/1
    else:
        # Run CV-based fallback to get mask and prob
        clean, prob_map = run_cv_fallback(image_bgr)

    pred_mask  = clean                                    # used for metrics
    mask_img   = (clean * 255).astype(np.uint8)           # (256,256) uint8 0/255

    flood_percent = round(float(pred_mask.mean() * 100), 2)

    if flood_percent < 5:
        status = "🟢 LOW RISK — Safe Zone"
    elif flood_percent < 30:
        status = "🟡 MODERATE ALERT — Standing Water"
    else:
        status = "🔴 HIGH DANGER — Severe Inundation"

    # Heatmap from raw probability map (smooth gradient)
    heatmap_data = (prob_map * 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_data, cv2.COLORMAP_JET)

    # Overlay on the 256x256 resized image
    blended = image_resized.copy()
    blended[pred_mask == 1] = np.clip(
        blended[pred_mask == 1] * 0.5 + np.array([230, 140, 10], dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    # Resize back to original resolution
    mask_out    = cv2.resize(mask_img,   (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    heatmap_out = cv2.resize(heatmap_bgr,(w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    blended_out = cv2.resize(blended,    (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    return (
        mask_out,
        heatmap_out,
        blended_out,
        status,
        f"{flood_percent}%"
    )

def gradio_interface(input_img):
    if input_img is None:
        return None, None, None, "No Image", "0%"
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    mask, heatmap, overlay, status, coverage = predict_flood(img_bgr)
    return (
        cv2.cvtColor(overlay,  cv2.COLOR_BGR2RGB),
        cv2.cvtColor(mask,     cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(heatmap,  cv2.COLOR_BGR2RGB),
        status,
        coverage
    )

# =====================================
# HERO HTML — Realistic Earth + Voice
# =====================================
hero_html = """
<div id="ht-landing">
  <canvas id="ht-stars"></canvas>
  <canvas id="ht-wave"></canvas>

  <div class="ht-brand">HydroTech</div>

  <div class="ht-earth-scene">
    <div class="ht-earth-wrap" id="ht-earth-btn" onclick="htEarthClick()">
      <div class="ht-ring2"></div>
      <div class="ht-ring1"></div>
      <img src="data:image/png;base64,{{earth_image_b64}}" id="ht-earth-c" style="width: 296px; height: 296px; border-radius: 50%; animation: spinEarth 30s linear infinite; box-shadow: inset 0 0 50px rgba(0,0,0,0.5), 0 0 60px rgba(56,189,248,0.4);" />
    </div>
    <div class="ht-hint" id="ht-hint">Click Earth to Launch</div>
    <div class="ht-vbar" id="ht-vbar">
      <span></span><span></span><span></span><span></span><span></span>
    </div>
  </div>
</div>

<style>
@keyframes spinEarth {{
  from {{ transform: rotate(0deg); }}
  to {{ transform: rotate(360deg); }}
}}
</style>

"""
hero_html = hero_html.replace('{{earth_image_b64}}', earth_image_b64)

custom_head = """
<script>
function initHero() {
  var sc = document.getElementById('ht-stars');
  var wc = document.getElementById('ht-wave');
  if(!sc || !wc) { setTimeout(initHero, 100); return; }

  /* ── Stars ── */
  var sctx = sc.getContext('2d');
  function drawStars(){
    sc.width  = sc.parentElement.offsetWidth;
    sc.height = sc.parentElement.offsetHeight;
    sctx.clearRect(0,0,sc.width,sc.height);
    for(var i=0;i<220;i++){
      var x=Math.random()*sc.width, y=Math.random()*sc.height*.65;
      var r=Math.random()*1.3+.2;
      sctx.beginPath(); sctx.arc(x,y,r,0,Math.PI*2);
      sctx.fillStyle='rgba(255,255,255,'+(Math.random()*.7+.1)+')'; sctx.fill();
    }
  }
  drawStars();

  /* ── Waves ── */
  var wctx = wc.getContext('2d');
  var winc = 0;
  function resizeW(){
    wc.width  = wc.parentElement.offsetWidth;
    wc.height = Math.round(wc.parentElement.offsetHeight * .5);
  }
  resizeW();
  window.addEventListener('resize', function(){ drawStars(); resizeW(); });
  function animW(){
    wctx.clearRect(0,0,wc.width,wc.height);
    [[.006,36,'#0d3b6e',.52],[.009,22,'#1565c0',.37],[.014,13,'#1976d2',.26]].forEach(function(p,i){
      wctx.fillStyle=p[2]; wctx.globalAlpha=p[3];
      wctx.beginPath(); wctx.moveTo(0,wc.height);
      var sy=wc.height*.38;
      for(var x=0;x<=wc.width;x++){
        wctx.lineTo(x, sy+Math.sin(x*p[0]+winc*(1+i*.42))*p[1]*Math.cos(x*.003+winc*.35));
      }
      wctx.lineTo(wc.width,wc.height); wctx.closePath(); wctx.fill();
    });
    wctx.globalAlpha=.07; wctx.fillStyle='#fff';
    for(var i=0;i<5;i++){
      var bx=((winc*16+i*110)%(wc.width+80))-40;
      var by=wc.height*.28+Math.sin(winc*.7+i)*10;
      wctx.beginPath(); wctx.ellipse(bx,by,20+i*3,5,0,0,Math.PI*2); wctx.fill();
    }
    wctx.globalAlpha=1; winc+=.018; requestAnimationFrame(animW);
  }
  animW();

  /* ── Voice ── */
  function speakWelcome(){
    if(!window.speechSynthesis) return;
    var vbar = document.getElementById('ht-vbar');
    var hint = document.getElementById('ht-hint');
    hint.style.opacity='0';
    vbar.classList.add('ht-vbar-active');
    var u = new SpeechSynthesisUtterance('Welcome to HydroTech');
    u.pitch=1.15; u.rate=.88; u.volume=1;
    function setVoice(){
      var voices = window.speechSynthesis.getVoices();
      var female = voices.find(function(v){
        return /female|woman|samantha|zira|susan|victoria|karen|moira|fiona|tessa|veena|allison|ava|serena/i.test(v.name);
      }) || voices.find(function(v){ return /en/i.test(v.lang); }) || voices[0];
      if(female) u.voice = female;
    }
    setVoice();
    if(!window.speechSynthesis.getVoices().length){
      window.speechSynthesis.onvoiceschanged = setVoice;
    }
    u.onend = function(){
      vbar.classList.remove('ht-vbar-active');
      hint.style.opacity='1';
    };
    window.speechSynthesis.speak(u);
  }

  /* ── Click handler (global so onclick="" works) ── */
  window.htEarthClick = function(){
    speakWelcome();
  };
}
initHero();
</script>
"""

# =====================================
# CSS
# =====================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

body, .gradio-container {
  background: #020813 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: #f0f9ff !important;
  overflow-x: hidden;
}

/* ── Landing hero ── */
#ht-landing {
  background: #020b18;
  border-radius: 16px;
  min-height: 520px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

#ht-stars, #ht-wave {
  position: absolute;
  bottom: 0; left: 0;
  width: 100%;
  pointer-events: none;
}
#ht-stars { height: 100%; z-index: 1; }
#ht-wave  { height: 50%;  z-index: 2; }

.ht-brand {
  position: relative; z-index: 10;
  font-size: 78px; font-weight: 700;
  background: linear-gradient(170deg,#fff 0%,#7dd3fc 55%,#0284c7 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -3px; line-height: 1;
  animation: htFadeD .9s ease both;
}

.ht-earth-scene {
  position: relative; z-index: 10;
  margin-top: 36px;
  display: flex; flex-direction: column; align-items: center;
  animation: htFadeU 1.1s ease .2s both;
}

.ht-earth-wrap {
  position: relative; width: 148px; height: 148px;
  cursor: pointer;
  animation: htFloat 4s ease-in-out infinite;
  transition: transform .2s;
}
.ht-earth-wrap:hover  { transform: scale(1.07); }
.ht-earth-wrap:active { transform: scale(0.96); }

#ht-earth-c {
  display: block; border-radius: 50%;
  width: 148px; height: 148px;
}

.ht-ring1 {
  position: absolute; top:-10px; left:-10px; right:-10px; bottom:-10px;
  border-radius: 50%;
  border: 1.5px dashed rgba(147,210,255,.3);
  animation: htSpin 26s linear infinite;
  pointer-events: none;
}
.ht-ring2 {
  position: absolute; top:-20px; left:-20px; right:-20px; bottom:-20px;
  border-radius: 50%;
  border: 1px solid rgba(56,189,248,.12);
  animation: htPulse 3s ease-out infinite;
  pointer-events: none;
}

.ht-hint {
  margin-top: 18px;
  font-size: 11px; color: rgba(147,210,255,.55);
  letter-spacing: 3px; text-transform: uppercase;
  transition: opacity .4s;
  animation: htBlink 2.5s ease-in-out infinite, htFadeU 1.3s ease .5s both;
}

/* Voice bars */
.ht-vbar {
  display: flex; align-items: flex-end; gap: 3px;
  height: 28px; margin-top: 12px;
  opacity: 0; transition: opacity .4s;
}
.ht-vbar-active { opacity: 1 !important; }
.ht-vbar span {
  width: 4px; border-radius: 2px;
  background: linear-gradient(180deg,#38bdf8,#0369a1);
}
.ht-vbar-active span:nth-child(1){ animation: htBar .60s ease-in-out infinite alternate; }
.ht-vbar-active span:nth-child(2){ animation: htBar .50s ease-in-out .10s infinite alternate; }
.ht-vbar-active span:nth-child(3){ animation: htBar .70s ease-in-out .05s infinite alternate; }
.ht-vbar-active span:nth-child(4){ animation: htBar .40s ease-in-out .15s infinite alternate; }
.ht-vbar-active span:nth-child(5){ animation: htBar .65s ease-in-out .08s infinite alternate; }

/* ── App dashboard ── */
#app-dashboard {
  position: relative; z-index: 5;
  background: rgba(4,28,58,.4) !important;
  border: 1px solid rgba(56,189,248,.2) !important;
  border-radius: 24px !important;
  backdrop-filter: blur(25px);
  box-shadow: 0 20px 50px rgba(0,0,0,.6);
  padding: 30px;
  margin: 0 20px 50px 20px;
  animation: htFadeU 1s cubic-bezier(.16,1,.3,1) both;
}

/* Gradio blocks */
.gr-box, .gr-form, .gr-input, input, textarea,
.tabs, .tabitem, div[class*="block"], div[class*="container"], .image-container {
  background-color: rgba(6,38,77,.45) !important;
  background:       rgba(6,38,77,.45) !important;
  border:           2px solid rgba(14,165,233,.4) !important;
  border-radius:    14px !important;
  color:            #e0f2fe !important;
  box-shadow: inset 0 0 15px rgba(56,189,248,.15), 0 4px 12px rgba(0,0,0,.2) !important;
  transition: all .3s cubic-bezier(.4,0,.2,1) !important;
}
.gr-input:hover, input:hover, textarea:hover, div[class*="block"]:hover {
  border-color: rgba(56,189,248,.8) !important;
  box-shadow: inset 0 0 20px rgba(56,189,248,.3), 0 0 15px rgba(14,165,233,.4) !important;
}
.gr-input-label, label span {
  background: transparent !important;
  color: #38bdf8 !important;
  font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .5px;
}
.tab-nav button {
  background: transparent !important; border: none !important; color: #7dd3fc !important;
}
.tab-nav button.selected {
  background: rgba(14,165,233,.3) !important;
  border-bottom: 3px solid #38bdf8 !important;
  color: #fff !important;
}
button.primary {
  background: linear-gradient(135deg,#0284c7 0%,#0369a1 100%) !important;
  border: 1px solid #38bdf8 !important;
  border-radius: 12px !important;
  color: white !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: 1px; padding: 12px 0;
}
button.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(56,189,248,.6) !important;
}

/* ── Keyframes ── */
@keyframes htFadeD { from{opacity:0;transform:translateY(-22px);} to{opacity:1;transform:none;} }
@keyframes htFadeU { from{opacity:0;transform:translateY(22px);}  to{opacity:1;transform:none;} }
@keyframes htFloat { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-14px);} }
@keyframes htSpin  { to{transform:rotate(360deg);} }
@keyframes htPulse { 0%{transform:scale(1);opacity:.5;} 100%{transform:scale(1.55);opacity:0;} }
@keyframes htBlink { 0%,100%{opacity:.35;} 50%{opacity:1;} }
@keyframes htBar   { from{height:4px;} to{height:22px;} }
"""

# =====================================
# GRADIO LAYOUT
# =====================================
with gr.Blocks(title="HydroTech", head=custom_head) as demo:

    # ... (rest unchanged) ...
    # Main entry handled at end of file

    gr.HTML(hero_html)

    with gr.Row(elem_id="app-dashboard"):
        with gr.Column(scale=4):
            gr.Markdown("### 🛰️ Aerial Target Extraction Dropzone")
            input_image = gr.Image(type="numpy", show_label=False)
            with gr.Row():
                submit_btn  = gr.Button("Analyze Inundation Vector", variant="primary")
                clear_btn   = gr.Button("Back / New Upload", variant="secondary")

            gr.Markdown("### 📊 Metrics Output Telemetry")
            status_output   = gr.Textbox(label="Risk Assessment Category",            interactive=False)
            coverage_output = gr.Textbox(label="Calculated Basin Water Surface Area", interactive=False)

        with gr.Column(scale=5):
            gr.Markdown("### 🗺️ Computer Vision Analytical Target Matrices")
            with gr.Tabs():
                with gr.TabItem("Overlay Mapping Projection"):
                    overlay_output = gr.Image(type="numpy", show_label=False)
                with gr.TabItem("Binary Segment Mask"):
                    mask_output    = gr.Image(type="numpy", show_label=False)
                with gr.TabItem("Spectral Inundation Heatmap"):
                    heatmap_output = gr.Image(type="numpy", show_label=False)

    submit_btn.click(
        fn      = gradio_interface,
        inputs  = input_image,
        outputs = [overlay_output, mask_output, heatmap_output, status_output, coverage_output]
    )
    clear_btn.click(
        fn      = lambda: (None, None, None, None, "Risk Assessment Category", "0%"),
        inputs  = None,
        outputs = [input_image, overlay_output, mask_output, heatmap_output, status_output, coverage_output]
    )

if __name__ == "__main__":
    print("Launching HydroTech...")
    demo.launch(share=False, inbrowser=True, css=custom_css)