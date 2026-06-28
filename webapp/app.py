"""
Vehicle Type Classification — Flask Web Application
Pattern Recognition Course · Tania Girish
"""

import os
import json
import time
import uuid
import hashlib
import random
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_DIR     = BASE_DIR / "models"
FIGURES_DIR   = BASE_DIR / "static" / "figures"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vehicle-clf-dev-secret-2025")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_FOLDER.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() in ("1", "true", "yes")
IMG_SIZE  = (224, 224)
CLASS_NAMES = [
    "Auto Rickshaws", "Bikes", "Cars",
    "Motorcycles", "Planes", "Ships", "Trains"
]
CLASS_ICONS = {
    "Auto Rickshaws": "🛺",
    "Bikes":          "🚲",
    "Cars":           "🚗",
    "Motorcycles":    "🏍️",
    "Planes":         "✈️",
    "Ships":          "🚢",
    "Trains":         "🚂",
}

MODEL_META = {
    "custom_cnn": {
        "label":       "Custom CNN",
        "file":        "custom_cnn_best.keras",
        "params":      "1.44M",
        "accuracy":    89.99,
        "f1":          0.900,
        "auc":         0.993,
        "infer_ms":    5.55,
        "color":       "#22d3ee",
        "description": "4× Conv Blocks (32→256) trained from scratch",
        "arch_summary": [
            "Input 224×224×3",
            "Conv2D(32) → BN → ReLU → MaxPool",
            "Conv2D(64) → BN → ReLU → MaxPool",
            "Conv2D(128) → BN → ReLU → MaxPool",
            "Conv2D(256) → BN → ReLU → MaxPool",
            "GlobalAveragePooling2D",
            "Dense(512) + Dropout(0.5)",
            "Dense(256) + Dropout(0.3)",
            "Softmax(7)",
        ],
    },
    "mobilenetv2": {
        "label":       "MobileNetV2",
        "file":        "mobilenetv2_best.keras",
        "params":      "2.59M",
        "accuracy":    99.17,
        "f1":          0.9916,
        "auc":         1.000,
        "infer_ms":    5.12,
        "color":       "#a78bfa",
        "description": "ImageNet pre-trained · inverted residuals · linear bottlenecks",
        "arch_summary": [
            "Input 224×224×3",
            "MobileNetV2 backbone (frozen → fine-tuned)",
            "GlobalAveragePooling2D",
            "Dense(256) + Dropout(0.3)",
            "Softmax(7)",
        ],
    },
    "resnet50": {
        "label":       "ResNet50",
        "file":        "resnet50_best.keras",
        "params":      "24.77M",
        "accuracy":    98.93,
        "f1":          0.9892,
        "auc":         1.000,
        "infer_ms":    6.95,
        "color":       "#fb923c",
        "description": "50-layer residual network · ImageNet pre-trained · skip connections",
        "arch_summary": [
            "Input 224×224×3",
            "ResNet50 backbone (frozen → fine-tuned last 30)",
            "GlobalAveragePooling2D",
            "Dense(256) + Dropout(0.3)",
            "Softmax(7)",
        ],
    },
    "efficientnetb0": {
        "label":       "EfficientNetB0",
        "file":        "efficientnetb0_best.keras",
        "params":      "4.38M",
        "accuracy":    99.28,
        "f1":          0.9928,
        "auc":         1.000,
        "infer_ms":    5.33,
        "color":       "#4ade80",
        "description": "Compound scaling · ImageNet pre-trained · best overall",
        "arch_summary": [
            "Input 224×224×3",
            "EfficientNetB0 backbone (frozen → fine-tuned last 20)",
            "GlobalAveragePooling2D",
            "Dense(256) + Dropout(0.3)",
            "Softmax(7)",
        ],
    },
}

COMPARISON_DATA = {
    "Model":         ["Custom CNN", "MobileNetV2", "ResNet50", "EfficientNetB0"],
    "Accuracy":      [89.99,        99.17,          98.93,      99.28],
    "Precision":     [0.901,        0.9917,         0.9895,     0.9929],
    "Recall":        [0.8998,       0.9916,         0.9893,     0.9928],
    "F1 (Macro)":    [0.900,        0.9916,         0.9892,     0.9928],
    "AUC":           [0.993,        1.000,          1.000,      1.000],
    "Params (M)":    [1.44,         2.59,           24.77,      4.38],
    "Infer (ms)":    [5.55,         5.12,           6.95,       5.33],
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Model loading ──────────────────────────────────────────────────────────────
_loaded_models: dict = {}

def model_available(model_key: str) -> bool:
    path = MODEL_DIR / MODEL_META[model_key]["file"]
    return path.exists()

def load_model_cached(model_key: str):
    """Lazy-load a Keras model; cache in _loaded_models."""
    if model_key in _loaded_models:
        return _loaded_models[model_key]
    model_path = MODEL_DIR / MODEL_META[model_key]["file"]
    if not model_path.exists():
        return None
    try:
        import tensorflow as tf
        log.info(f"Loading model: {model_key} from {model_path}")
        m = tf.keras.models.load_model(str(model_path))
        _loaded_models[model_key] = m
        return m
    except Exception as e:
        log.error(f"Failed to load {model_key}: {e}")
        return None

# ── Prediction helpers ─────────────────────────────────────────────────────────
def preprocess_image(img_path: str):
    """Load and preprocess image to (1, 224, 224, 3)."""
    import tensorflow as tf
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

def real_predict(model_key: str, img_path: str) -> dict:
    """Run actual TF inference."""
    model = load_model_cached(model_key)
    if model is None:
        return demo_predict(model_key, img_path)
    t0 = time.perf_counter()
    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    top_class = max(probs, key=probs.get)
    return {
        "model_key":     model_key,
        "model_label":   MODEL_META[model_key]["label"],
        "top_class":     top_class,
        "icon":          CLASS_ICONS[top_class],
        "confidence":    round(probs[top_class] * 100, 2),
        "probabilities": {k: round(v * 100, 3) for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        "infer_ms":      round(elapsed_ms, 1),
        "demo":          False,
    }

def demo_predict(model_key: str, img_path: str) -> dict:
    """
    Deterministic demo predictions seeded from image content hash.
    Distributions reflect each model's real-world accuracy characteristics.
    """
    # Seed from image bytes for consistency across requests
    with open(img_path, "rb") as f:
        img_hash = int(hashlib.md5(f.read()).hexdigest(), 16)
    rng = random.Random(img_hash + hash(model_key))

    # Pick a "winner" class probabilistically
    winner_idx = rng.randint(0, len(CLASS_NAMES) - 1)
    winner = CLASS_NAMES[winner_idx]

    # Model accuracy shapes how confident the prediction is
    acc = MODEL_META[model_key]["accuracy"] / 100
    winner_prob = rng.uniform(acc * 0.85, min(acc + 0.04, 0.999))

    remaining = 1.0 - winner_prob
    others = [rng.random() for _ in range(len(CLASS_NAMES) - 1)]
    s = sum(others)
    others = [o / s * remaining for o in others]

    probs = {}
    oi = 0
    for name in CLASS_NAMES:
        if name == winner:
            probs[name] = winner_prob
        else:
            probs[name] = others[oi]; oi += 1

    # Simulated inference time (realistic jitter)
    base_ms = MODEL_META[model_key]["infer_ms"]
    infer_ms = round(base_ms + rng.uniform(-1.5, 1.5), 1)

    return {
        "model_key":     model_key,
        "model_label":   MODEL_META[model_key]["label"],
        "top_class":     winner,
        "icon":          CLASS_ICONS[winner],
        "confidence":    round(winner_prob * 100, 2),
        "probabilities": {k: round(v * 100, 3) for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        "infer_ms":      infer_ms,
        "demo":          True,
    }

def predict(model_key: str, img_path: str) -> dict:
    if DEMO_MODE or not model_available(model_key):
        return demo_predict(model_key, img_path)
    return real_predict(model_key, img_path)

# ── File helpers ───────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_upload(file) -> Path:
    ext = file.filename.rsplit(".", 1)[1].lower()
    name = f"{uuid.uuid4().hex}.{ext}"
    dest = UPLOAD_FOLDER / name
    file.save(str(dest))
    return dest

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("classifier"))

@app.route("/classifier")
def classifier():
    return render_template("classifier.html", models=MODEL_META, demo=DEMO_MODE)

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, or WebP."}), 400

    model_key = request.form.get("model", "efficientnetb0")
    if model_key not in MODEL_META:
        return jsonify({"error": "Unknown model"}), 400
    compare_all = request.form.get("compare_all") == "true"

    try:
        img_path = save_upload(file)
    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500

    try:
        if compare_all:
            results = {k: predict(k, str(img_path)) for k in MODEL_META}
            session["last_results"]      = results
            session["last_model_key"]    = model_key
            session["last_img_filename"] = img_path.name
            return jsonify({"compare": True, "results": results, "img": img_path.name})
        else:
            result = predict(model_key, str(img_path))
            session["last_result"]       = result
            session["last_img_filename"] = img_path.name
            return jsonify({"compare": False, "result": result, "img": img_path.name})
    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route("/results")
def results():
    result   = session.get("last_result")
    results  = session.get("last_results")
    img_name = session.get("last_img_filename")
    if not result and not results:
        return redirect(url_for("classifier"))
    return render_template(
        "results.html",
        result=result,
        results=results,
        img_name=img_name,
        models=MODEL_META,
        class_icons=CLASS_ICONS,
        demo=DEMO_MODE,
    )

@app.route("/comparison")
def comparison():
    return render_template(
        "comparison.html",
        models=MODEL_META,
        data=COMPARISON_DATA,
        demo=DEMO_MODE,
    )

@app.route("/models")
def models_page():
    return render_template("models.html", models=MODEL_META, demo=DEMO_MODE)

@app.route("/about")
def about():
    return render_template("about.html", demo=DEMO_MODE)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(COMPARISON_DATA)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "demo": DEMO_MODE})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
