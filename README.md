# 🚗 Vehicle Type Classification System Using CNN

> A CNN-based decision-support system for automated vehicle type classification from traffic images.  
> **Pattern Recognition Course — University Project**

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

## 📂 Repository Structure

```
Vehicle-Type-Classification-System/
├── notebooks/
│   └── vehicle_classification_notebook.py   # Complete Kaggle notebook (18 sections)
│
├── webapp/                                   # Flask web application
│   ├── app.py                                # Main Flask application
│   ├── requirements.txt                      # Web app dependencies
│   ├── models/                               # Trained model files (.keras, .h5)
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   ├── uploads/                          # Uploaded images (runtime)
│   │   ├── gradcam/                          # Grad-CAM outputs (runtime)
│   │   └── figures/                          # Training figures
│   └── templates/
│       ├── base.html
│       ├── index.html                        # Home page
│       ├── upload.html                       # Image upload
│       ├── prediction.html                   # Results page
│       ├── gradcam.html                      # Explainability
│       ├── dashboard.html                    # Analytics
│       ├── comparison.html                   # Model comparison
│       └── about.html                        # Project info
│
├── reports/
│   └── Vehicle_Classification_Proposal.docx # Research proposal
│
├── dataset/                                  # (Empty — download from Kaggle)
├── outputs/                                  # Training outputs
├── requirements.txt                          # Full environment
├── .gitignore
└── LICENSE
```

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

## 🌐 Running the Web Application

### 1. (Optional) Copy trained models
```bash
cp /kaggle/working/models/*.keras webapp/models/
cp /kaggle/working/models/model_config.json webapp/models/
cp /kaggle/working/outputs/figures/*.png webapp/static/figures/
```

### 2. Install web app dependencies
```bash
cd webapp
pip install -r requirements.txt
```

### 3. Launch the application
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

The app runs in **demo mode** (simulated predictions) if trained models are not found.

---

## 📊 Web Application Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Project overview, objectives, dataset info |
| Upload | `/upload` | Image upload with model selection |
| Prediction | `/prediction` | Results + top-5 probabilities |
| Grad-CAM | `/gradcam` | Heatmap overlay + attention explanation |
| Dashboard | `/dashboard` | Metrics, charts, training curves |
| Comparison | `/comparison` | 4-model side-by-side comparison |
| About | `/about` | Background, methodology, research questions |

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
| **ResNet50** | **~93%** | **~0.93** | 25.6 | ~28 |
| EfficientNetB0 | ~92% | ~0.91 | 5.3 | ~22 |

---

## 📖 Key Reference

> Khan, S. U., Hussain, A., Ali, S., & Bhutta, M. N. M. (2024). *Deep Learning-Based Vehicle Type and Color Classification to Support Safe Autonomous Driving.* Applied Sciences, 14(4), 1600. https://doi.org/10.3390/app14041600

---

## 🔮 Future Work

- Object detection with YOLO/Faster R-CNN for multi-vehicle scenes
- Video-based temporal modeling (LSTM, 3D-CNN)
- Edge deployment via TensorFlow Lite quantization
- Adversarial robustness evaluation
- Night/weather condition generalization

---

## ⚠️ Ethical Considerations

This system is a **decision-support tool** and should NOT be used as a fully autonomous decision-maker in safety-critical contexts. Human oversight remains essential. The dataset may not represent all global traffic conditions, vehicle types, or environmental variations. Deployment in new environments requires re-evaluation and validation.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
Dataset: CC BY-SA 4.0 — download separately from Kaggle.

---

## 🙏 Acknowledgements

- Dataset: Mohamed Maher (Kaggle)
- Reference architecture: He et al. (ResNet), Sandler et al. (MobileNetV2), Tan & Le (EfficientNet)
- Explainability: Selvaraju et al. (Grad-CAM, ICCV 2017)
- Framework: TensorFlow/Keras team at Google
