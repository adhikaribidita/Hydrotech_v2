# HydroTech — AI-Powered Environmental Inundation & Flood Analytics

HydroTech is a cinematic, enterprise-grade flood detection platform that leverages satellite imagery and deep learning to perform real-time water inundation segmentation. 

* **Live Demo (Frontend):** [https://hydrotech-v2.vercel.app/](https://hydrotech-v2.vercel.app/)
* **Inference API (Backend):** [https://hydrotech-api.onrender.com/health](https://hydrotech-api.onrender.com/health)

---

## 🌟 Key Features

1. **Cinematic Landing Page:** Includes an interactive, high-fidelity 3D Earth visualization rendered in real-time with WebGL using React Three Fiber, customizable cloud layers, realistic shaders, and post-processing glows.
2. **AI-Driven Segmentation:** Utilizes a nested U-Net++ topology with an EfficientNet-B3 encoder to perform semantic segmentation of flood regions from spectral satellite imagery.
3. **Robust Fallback Engine:** If PyTorch model weights (`best_model.pth`) are not present, the backend dynamically falls back to an adaptive-thresholding contour analysis pipeline, ensuring 100% uptime.
4. **Advanced Post-Processing (V2):**
   - *Anti-Aliasing:* Gaussian smoothing to reduce grid aliasing.
   - *Edge Refinement:* Optional DenseCRF (Conditional Random Field) boundary snapping.
   - *Noise Clean-up:* Morphological opening (fine) and closing (coarse).
   - *Clustering:* Connected-component filtering to exclude stray micro-blobs.
   - *Refinement:* Hole filling and contour smoothing.
5. **Multi-Scale Test-Time Augmentation (TTA):** Ensembles predictions from 4 geometric orientations (flips) to maximize boundary prediction robustness.
6. **Automated Technical Advisory Reports:** Generates vector-graphic PDF briefs (including stats, risk ratings, overlays, and heatmaps) on the fly via ReportLab, ready for civic response dispatch.

---

## 📂 Project Architecture

```
HydroTech-AI-Flood-Detection/
├── backend/
│   ├── main.py                  # FastAPI service (predict, report generation, health check)
│   ├── requirements.txt         # CPU-optimized Python requirements for Render
│   └── Dockerfile               # Slim Docker setup for cloud deployments
├── frontend/
│   ├── src/
│   │   ├── components/          # 3D Earth, Dropzone, and UI widgets
│   │   ├── pages/               # Landing and Dashboard layouts
│   │   ├── store/               # Zustand global state manager
│   │   └── styles/              # Global Tailwind CSS definitions
│   ├── package.json             # React/Vite dependencies
│   ├── vercel.json              # Client routing rewrites for SPA routing
│   └── vite.config.ts           # Bundler and dev-server configuration
└── scripts/                     # Helper utilities for validation
```

---

## 💻 Local Setup & Development

### Backend (Python 3.10+)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install the packages (headless OpenCV for servers, and optional CPU-only PyTorch wheels for speed):
   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
4. Start the development server:
   ```bash
   python main.py
   ```
   *The API will run on `http://127.0.0.1:8000`. Swagger docs are available at `http://127.0.0.1:8000/docs`.*

### Frontend (Node.js 20+)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the frontend dependencies (with legacy peer deps handling):
   ```bash
   npm install --legacy-peer-deps
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   *The client will run on `http://localhost:3000`.*

---

## 🚀 Cloud Deployment

### Backend (Render)
Deployed via a Docker container on Render to guarantee system-level OpenCV dependency mapping.
- **Root Directory:** `backend`
- **Runtime:** `Docker`
- **Instance Type:** Free (512MB RAM)
- **Model Weight Injection (Optional):** Define `MODEL_URL` as a Build Argument with a direct download link to HuggingFace or S3 to run the full U-Net++ model. If empty, the app runs the adaptive-threshold fallback.

### Frontend (Vercel)
Static SPA deployment configured with Vite presets.
- **Root Directory:** `frontend`
- **Framework Preset:** Vite
- **Install Command:** `npm install --legacy-peer-deps`
- **Environment Variables:** `VITE_API_URL` set to the live Render Backend Web Service URL.
- **Client Routing:** Handled via the rewrite rules in `frontend/vercel.json` redirecting all routes to `index.html`.

---

## 📄 License
This project is for analytical and educational research.
