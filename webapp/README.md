# VehicleVision — CNN Vehicle Classification Web App

A Flask web application for classifying vehicle types from images using four trained CNN architectures. Built for the Pattern Recognition course project by Tania Girish.

**Live demo:** [Deploy to Render](#deploy-to-render)

---

## Features

| Page | Route | Description |
|------|-------|-------------|
| Classifier | `/classifier` | Upload image → select model → classify or compare all 4 |
| Results | `/results` | Detailed prediction + per-class probability bars |
| Comparison | `/comparison` | Benchmark charts, radar, performance table, training figures |
| Models | `/models` | Architecture details for all 4 models + augmentation info |
| About | `/about` | Dataset, methodology, research questions, references |

**Two modes:**
- **Demo mode** (default, no GPU required) — realistic simulated predictions seeded from image content, consistent across requests
- **Live mode** — actual TensorFlow inference using trained `.keras` model files

---

## Local Setup

### 1. Clone / navigate to this directory

```bash
cd webapp/
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

**Demo mode only (fast, no GPU needed):**
```bash
pip install -r requirements.txt
```

**Live inference (requires ~5 min install + model files):**
```bash
# Uncomment the tensorflow-cpu line in requirements.txt first
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for demo mode)
```

### 5. Add model files (live mode only)

Copy trained `.keras` files into `models/`:

```bash
cp /path/to/outputs/models/custom_cnn_best.keras    models/
cp /path/to/outputs/models/mobilenetv2_best.keras   models/
cp /path/to/outputs/models/efficientnetb0_best.keras models/
cp /path/to/outputs/models/resnet50_best.keras       models/
```

Then set `DEMO_MODE=false` in `.env`.

### 6. Run

```bash
python app.py
```

Open **http://localhost:5000**

---

## Deploy to Render

### Quick deploy (demo mode — no model files needed)

1. **Create a new GitHub repo** and push the `webapp/` directory contents as the repo root:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/vehiclevision.git
   git push -u origin main
   ```

2. Go to [render.com](https://render.com) → **New → Web Service**

3. Connect your GitHub repo

4. Render auto-detects `render.yaml`. Review settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 2 --threads 2 --bind 0.0.0.0:$PORT --timeout 120`
   - **Environment:** `DEMO_MODE=true` (already set in render.yaml)

5. Click **Create Web Service** — deploys in ~3 minutes.

### Deploy with live model inference

1. **Uncomment** `tensorflow-cpu==2.15.1` in `requirements.txt` (build takes ~8 min on Render free tier)

2. In Render dashboard → your service → **Environment** → add:
   ```
   DEMO_MODE = false
   ```

3. Add a **Persistent Disk** (Render dashboard → Disks):
   - Mount path: `/opt/models`
   - Size: 2 GB

4. Upload model files to the disk via Render Shell:
   ```bash
   # In Render Shell
   ls /opt/models/
   ```
   Then use `scp` or a startup script to copy models to `/opt/models/`.

5. Update `MODEL_DIR` in `app.py` to match the disk path:
   ```python
   MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
   ```

> **Note:** ResNet50 (172 MB) may exceed Render free tier RAM (512 MB) when combined with TensorFlow. Use EfficientNetB0 (30 MB) or MobileNetV2 (25 MB) for production on free tier.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | (generated) | Flask session secret — set a random string in production |
| `DEMO_MODE` | `true` | `true` = simulated predictions, `false` = live TF inference |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode (never use in production) |
| `PORT` | `5000` | Server port (set automatically by Render) |
| `MODEL_DIR` | `./models` | Path to directory containing `.keras` model files |

---

## Project Structure

```
webapp/
├── app.py                  # Flask app, routes, prediction logic
├── requirements.txt
├── render.yaml             # Render deployment config
├── .env.example
├── .gitignore
├── README.md
├── models/
│   └── .gitkeep            # Place *.keras files here for live mode
├── uploads/
│   └── .gitkeep            # Runtime upload storage (auto-cleaned in production)
├── static/
│   ├── css/style.css       # Full custom CSS — dark slate/cyan theme
│   ├── js/app.js           # Shared JS
│   └── figures/            # Training output figures (PNG)
└── templates/
    ├── base.html
    ├── classifier.html     # Main upload + classify UI
    ├── results.html        # Single + compare results
    ├── comparison.html     # Chart.js benchmark dashboard
    ├── models.html         # Architecture details
    └── about.html          # Project info + dataset + methodology
```

---

## Vehicle Classes

| Icon | Class | Notes |
|------|-------|-------|
| 🛺 | Auto Rickshaws | Three-wheeled motorised rickshaws |
| 🚲 | Bikes | Pedal bicycles |
| 🚗 | Cars | Passenger cars |
| 🏍️ | Motorcycles | Two-wheeled motor vehicles |
| ✈️ | Planes | Aircraft |
| 🚢 | Ships | Watercraft |
| 🚂 | Trains | Railway vehicles |

---

## Model Performance

| Model | Accuracy | F1 | Params | Inference |
|-------|----------|-----|--------|-----------|
| Custom CNN | 89.99% | 0.900 | 1.44M | 5.55 ms |
| MobileNetV2 | 99.17% | 0.992 | 2.59M | 5.12 ms |
| ResNet50 | 98.93% | 0.989 | 24.77M | 6.95 ms |
| **EfficientNetB0** | **99.28%** | **0.993** | **4.38M** | **5.33 ms** |

---

## Tech Stack

- **Backend:** Python 3.10, Flask 3.0, Gunicorn
- **ML:** TensorFlow 2.13 / Keras (live mode)
- **Frontend:** Vanilla JS, Chart.js 4.4, Inter + JetBrains Mono (Google Fonts)
- **Deployment:** Render Web Service

---

## License

MIT — dataset is CC BY-SA 4.0 (Kaggle, Mohamed Maher).
