# 🚗 Vehicle Type Classification System Using CNN

> A CNN-based decision-support system for automated vehicle type classification from traffic images.  
> **Pattern Recognition**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Overview

This project develops and benchmarks **four deep learning architectures** for automated vehicle type classification from traffic and parking images:

| Model | Type | Parameters | Expected Accuracy |
|-------|------|-----------|-------------------|
| Custom CNN | Baseline | ~7.2M | 78–84% |
| MobileNetV2 | Transfer Learning | ~3.4M | 88–92% |
| ResNet50 | Transfer Learning | ~25.6M | 91–95% |
| EfficientNetB0 | Transfer Learning | ~5.3M | 90–94% |

### Vehicle Classes (7)
🚲 Bicycle · 🚌 Bus · 🚗 Car · 🏍️ Motorcycle · 🛻 Pickup Truck · 🚛 Truck · 🚐 Van

---

## 🗃️ Dataset

**Vehicle Image Classification Dataset**
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification)
- **Total Images:** 5,600 high-resolution images
- **Classes:** 7 vehicle types
- **License:** CC BY-SA 4.0 (academic use permitted)
- **Split:** 70% train / 15% validation / 15% test (stratified)

**Download before running the notebook:**
```bash
kaggle datasets download -d mohamedmaher5/vehicle-classification
```
Or download manually from the Kaggle link above.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip 23+
- NVIDIA GPU recommended (CUDA 11.8+) or Kaggle Notebook (free GPU)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/vehicle-classification-cnn.git
cd vehicle-classification-cnn
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Kaggle Notebook

The notebook is at `notebooks/vehicle_classification_notebook.py`.

**Option A — Kaggle (Recommended, free GPU):**
1. Upload the `.py` file to a new Kaggle Notebook
2. Add the dataset: Kaggle → Datasets → `mohamedmaher5/vehicle-classification`
3. Enable GPU accelerator (Settings → Accelerator → GPU P100)
4. Run all cells sequentially

**Option B — Local (requires GPU for reasonable speed):**
```bash
# Ensure dataset is downloaded to DATASET_PATH in the config
python notebooks/vehicle_classification_notebook.py
```

After training, copy model files to `webapp/models/` and figures to `webapp/static/figures/`.

---

## 🔬 Methodology

1. **Preprocessing:** Resize to 224×224, normalize to [0,1], ImageNet standardization for TL models
2. **Augmentation:** Flip, rotation, zoom, brightness, contrast, saturation, hue
3. **Custom CNN:** 4× [Conv2D → BatchNorm → ReLU → MaxPool] → GAP → Dense head
4. **Transfer Learning:** Two-phase — feature extraction (frozen base) then fine-tuning
5. **Evaluation:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
6. **Explainability:** Grad-CAM (Selvaraju et al. 2017) on last convolutional layer
7. **Error Analysis:** Per-class error rates, top misclassification pairs

---

## 📈 Key Results (Expected)

| Model | Accuracy | F1 (Macro) | Params (M) | Time (ms) |
|-------|----------|-----------|-----------|-----------|
| Custom CNN | ~81% | ~0.80 | 7.2 | ~12 |
| MobileNetV2 | ~90% | ~0.89 | 3.4 | ~18 |
| ResNet50 | ~93% | ~0.93 | 25.6 | ~28 |
| EfficientNetB0 | ~92% | ~0.91 | 5.3 | ~22 |

---


- Dataset: Mohamed Maher (Kaggle)
- Reference architecture: He et al. (ResNet), Sandler et al. (MobileNetV2), Tan & Le (EfficientNet)
- Explainability: Selvaraju et al. (Grad-CAM, ICCV 2017)
- Framework: TensorFlow/Keras team at Google

## Web Application

A full-stack Flask web application is available in [`webapp/`](webapp/).  
See [`webapp/README.md`](webapp/README.md) for local setup and Render deployment instructions.
