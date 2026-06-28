"""
Vehicle Type Classification — Flask Web Application
Pattern Recognition Course · Tania Girish
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path

import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename
from PIL import Image

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_DIR     = BASE_DIR / "models"
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vehicle-clf-dev-secret-2025")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Auto Rickshaws", "Bikes", "Cars",
               "Motorcycles", "Planes", "Ships", "Trains"]
CLASS_ICONS = {
    "Auto Rickshaws": "🛺", "Bikes": "🚲", "Cars": "🚗",
    "Motorcycles": "🏍️", "Planes": "✈️", "Ships": "🚢", "Trains": "🚂",
}

MODEL_META = {
    "custom_cnn": {
        "label": "Custom CNN", "file": "custom_cnn.tflite",
        "params": "1.44M", "accuracy": 89.99, "f1": 0.900, "auc": 0.993,
        "infer_ms": 5.55, "color": "#22d3ee",
        "description": "4× Conv Blocks (32→256) trained from scratch",
        "preprocess": "normalize",   # divide by 255, already done
        "arch_summary": [
            "Input 224×224×3", "Conv2D(32) → BN → ReLU → MaxPool",
            "Conv2D(64) → BN → ReLU → MaxPool", "Conv2D(128) → BN → ReLU → MaxPool",
            "Conv2D(256) → BN → ReLU → MaxPool", "GlobalAveragePooling2D",
            "Dense(512) + Dropout(0.5)", "Dense(256) + Dropout(0.3)", "Softmax(7)",
        ],
    },
    "mobilenetv2": {
        "label": "MobileNetV2", "file": "mobilenetv2.tflite",
        "params": "2.59M", "accuracy": 99.17, "f1": 0.9916, "auc": 1.000,
        "infer_ms": 5.12, "color": "#a78bfa",
        "description": "ImageNet pre-trained · inverted residuals · linear bottlenecks",
        "preprocess": "mobilenet",   # included in tflite graph (Lambda converted OK)
        "arch_summary": [
            "Input 224×224×3", "MobileNetV2 backbone (frozen → fine-tuned)",
            "GlobalAveragePooling2D", "Dense(256) + Dropout(0.3)", "Softmax(7)",
        ],
    },
    "resnet50": {
        "label": "ResNet50", "file": "resnet50.tflite",
        "params": "24.77M", "accuracy": 98.93, "f1": 0.9892, "auc": 1.000,
        "infer_ms": 6.95, "color": "#fb923c",
        "description": "50-layer residual network · ImageNet pre-trained · skip connections",
        "preprocess": "resnet50",    # Lambda stripped — must apply in app
        "arch_summary": [
            "Input 224×224×3", "ResNet50 backbone (frozen → fine-tuned last 30)",
            "GlobalAveragePooling2D", "Dense(512) + Dropout(0.5)",
            "Dense(256) + Dropout(0.3)", "Softmax(7)",
        ],
    },
    "efficientnetb0": {
        "label": "EfficientNetB0", "file": "efficientnetb0.tflite",
        "params": "4.38M", "accuracy": 99.28, "f1": 0.9928, "auc": 1.000,
        "infer_ms": 5.33, "color": "#4ade80",
        "description": "Compound scaling · ImageNet pre-trained · best overall",
        "preprocess": "efficientnet", # Lambda stripped — must apply in app
        "arch_summary": [
            "Input 224×224×3", "EfficientNetB0 backbone (frozen → fine-tuned last 20)",
            "GlobalAveragePooling2D", "Dense(256) + Dropout(0.3)", "Softmax(7)",
        ],
    },
}

COMPARISON_DATA = {
    "Model":      ["Custom CNN", "MobileNetV2", "ResNet50", "EfficientNetB0"],
    "Accuracy":   [89.99, 99.17, 98.93, 99.28],
    "Precision":  [0.901, 0.9917, 0.9895, 0.9929],
    "Recall":     [0.8998, 0.9916, 0.9893, 0.9928],
    "F1 (Macro)": [0.900, 0.9916, 0.9892, 0.9928],
    "AUC":        [0.993, 1.000, 1.000, 1.000],
    "Params (M)": [1.44, 2.59, 24.77, 4.38],
    "Infer (ms)": [5.55, 5.12, 6.95, 5.33],
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── TFLite interpreter cache ──────────────────────────────────────────────────
_interpreters: dict = {}

def get_interpreter(model_key: str):
    """Load and cache a TFLite interpreter."""
    if model_key in _interpreters:
        return _interpreters[model_key]
    model_path = MODEL_DIR / MODEL_META[model_key]["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    log.info(f"Loading TFLite: {model_key}")
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    _interpreters[model_key] = interp
    return interp

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(img_path: str, preprocess_mode: str) -> np.ndarray:
    """
    Load image and apply the preprocessing that was inside the Lambda layer
    (now stripped from the TFLite graph, so we apply it here).
    """
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)  # shape (224,224,3), values 0-255

    if preprocess_mode == "normalize":
        # Custom CNN: trained with pixels /255
        arr = arr / 255.0

    elif preprocess_mode == "mobilenet":
        # MobileNetV2: Lambda was converted into TFLite graph, so the graph
        # expects raw [0,255] pixels — the graph handles scaling internally
        # (The Lambda converts [0,1] → [-1,1] after the /255 pipeline, but
        #  since the Lambda is embedded in the graph, just feed raw pixels.)
        arr = arr / 255.0  # the embedded Lambda handles the rest

    elif preprocess_mode == "efficientnet":
        # Lambda stripped. EfficientNetB0's preprocess_input: [0,255] → [-1,1]
        # The notebook pipeline normalized /255 first, then Lambda scaled:
        # x*255 → preprocess_input → [-1,1]
        # Net effect on raw pixels: preprocess_input(raw_pixels) = raw/127.5 - 1
        arr = arr / 127.5 - 1.0

    elif preprocess_mode == "resnet50":
        # Lambda stripped. ResNet50: scale255 then resnet preprocess_input.
        # scale255 reverses /255 normalization. resnet preprocess does:
        # convert RGB→BGR, subtract ImageNet means [103.939, 116.779, 123.68]
        arr = arr[:, :, ::-1]  # RGB → BGR
        arr[:, :, 0] -= 103.939
        arr[:, :, 1] -= 116.779
        arr[:, :, 2] -= 123.68

    return np.expand_dims(arr, 0).astype(np.float32)  # (1,224,224,3)

# ── TFLite inference ──────────────────────────────────────────────────────────
def tflite_predict(model_key: str, img_path: str) -> dict:
    meta = MODEL_META[model_key]
    interp = get_interpreter(model_key)

    inp_details  = interp.get_input_details()
    out_details  = interp.get_output_details()

    x = preprocess_image(img_path, meta["preprocess"])

    t0 = time.perf_counter()
    interp.set_tensor(inp_details[0]["index"], x)
    interp.invoke()
    preds = interp.get_tensor(out_details[0]["index"])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    top_class = max(probs, key=probs.get)

    return {
        "model_key":     model_key,
        "model_label":   meta["label"],
        "top_class":     top_class,
        "icon":          CLASS_ICONS[top_class],
        "confidence":    round(probs[top_class] * 100, 2),
        "probabilities": {k: round(v * 100, 3)
                          for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        "infer_ms":      round(elapsed_ms, 1),
        "demo":          False,
    }

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
    return render_template("classifier.html", models=MODEL_META, demo=False)

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, or WebP."}), 400

    model_key  = request.form.get("model", "efficientnetb0")
    compare_all = request.form.get("compare_all") == "true"

    if model_key not in MODEL_META:
        return jsonify({"error": "Unknown model"}), 400

    try:
        img_path = save_upload(file)
    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500

    try:
        if compare_all:
            results = {}
            for k in MODEL_META:
                try:
                    results[k] = tflite_predict(k, str(img_path))
                except Exception as e:
                    log.error(f"Error predicting {k}: {e}")
                    results[k] = {"error": str(e), "model_key": k, "model_label": MODEL_META[k]["label"]}
            session["last_results"]      = results
            session["last_model_key"]    = model_key
            session["last_img_filename"] = img_path.name
            return jsonify({"compare": True, "results": results, "img": img_path.name})
        else:
            result = tflite_predict(model_key, str(img_path))
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
    return render_template("results.html", result=result, results=results,
        img_name=img_name, models=MODEL_META, class_icons=CLASS_ICONS, demo=False)

@app.route("/comparison")
def comparison():
    return render_template("comparison.html", models=MODEL_META, data=COMPARISON_DATA, demo=False)

@app.route("/models")
def models_page():
    return render_template("models.html", models=MODEL_META, demo=False)

@app.route("/about")
def about():
    return render_template("about.html", demo=False)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(COMPARISON_DATA)

@app.route("/health")
def health():
    available = {k: (MODEL_DIR / v["file"]).exists() for k, v in MODEL_META.items()}
    return jsonify({"status": "ok", "models": available})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,
            debug=os.environ.get("FLASK_DEBUG","false").lower()=="true")
