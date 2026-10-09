# 🔬 End-to-End Skin Cancer Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![CI](https://github.com/vishalvemula1/skin-cancer/actions/workflows/ci.yml/badge.svg)](https://github.com/vishalvemula1/skin-cancer/actions)
[![Deep Learning](https://img.shields.io/badge/Domain-Computer%20Vision-red)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 FastAPI Backend & Skin Cancer Detection API

The project now ships with a **production-ready FastAPI backend** for real-time skin lesion classification.

### Quickstart (API)

```bash
# 1. Create & activate venv
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install torch (CPU or GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU-only
# pip install torch torchvision   # GPU

# 3. Install API dependencies
pip install -r requirements-api.txt

# 4. (Optional) put your trained weights at models/model_weights.pth
#    The API will start even without weights (predictions will be random)

# 5. Run the server
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser-friendly drag-and-drop UI |
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Upload an image → get prediction |
| `GET` | `/docs` | Interactive Swagger UI |

### Running Tests

```bash
# (after installing torch + requirements-api.txt)
pytest tests/ -v --cov=app
```

### CI/CD (GitHub Actions)

Three-stage pipeline defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

1. **Lint** – `ruff` checks all Python files
2. **Test** – `pytest` with coverage, using CPU-only torch wheel
3. **Docker Build** – Builds the image and hits `/health` as a smoke test

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TL;DR:** An end-to-end Deep Learning system for automated skin lesion analysis and classification from dermatoscopic imagery. Built to tackle real-world medical imaging challenges including severe class imbalance, high inter-class visual similarity, and clinical evaluation sensitivity.

---

## 📌 Executive Summary

Early detection of malignant melanoma and other skin carcinomas significantly improves patient survival rates. This project implements a modular deep learning workflow capable of triaging dermatoscopic images into distinct lesion categories. 

### Why this project stands out to hiring managers:
- **Production-Style Modularity:** Decoupled data preprocessing, model definitions (`model.py`), augmentation workflows, and evaluation logic.
- **Handling Extreme Class Imbalance:** Engineered targeted data augmentation (`augmentor.ipynb`) to oversample rare malignant classes without dataset leakage.
- **Clinically Oriented Metrics:** Evaluated beyond raw accuracy, prioritizing sensitivity/recall, specificity, ROC-AUC, and multi-class confusion matrix dynamics (`evaluator_v2.ipynb`).

---

## 🛠️ Tech Stack & Tooling

| Domain | Technologies / Libraries |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Frameworks & Modeling** | PyTorch / TensorFlow, Keras, Transfer Learning |
| **Computer Vision** | OpenCV, PIL (Pillow), Albumentations |
| **Data & Metrics** | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| **Environment** | Jupyter Notebooks, Virtualenv / Conda |

---

## 🏗️ Architecture & Pipeline Flow

```text
[ Raw Dermoscopic Images & Metadata ]
                  │
                  ▼
         Data/final_sort.py
   (Stratified Splitting & Directory Structuring)
                  │
                  ▼
       models/augmentor.ipynb
   (Targeted Class Balancing & Transformations)
                  │
                  ▼
         models/model.py  ◄── Transfer Learning Backbones & CNN Heads
                  │
                  ▼
       models/Main_Model_v2.ipynb
   (Training Loop, Loss Optimization, Checkpointing)
                  │
                  ▼
       models/evaluator_v2.ipynb
   (ROC-AUC, Precision-Recall, Confusion Matrices)
```

---

## 📂 Repository Structure

```text
skin-cancer/
├── Data/
│   └── final_sort.py          # Automated sorting script for dataset reorganization
├── models/
│   ├── model.py               # Core modular neural network architectures
│   ├── Main_Model_v2.ipynb    # Primary training and hyperparameter tuning pipeline
│   ├── new_model.ipynb        # Experimental sandbox for architectural iterations
│   ├── augmentor.ipynb        # Targeted augmentation for minority class synthesis
│   └── evaluator_v2.ipynb     # Clinical evaluation, ROC curves, and error analysis
├── .gitignore                 # Exclusion rules for checkpoints, weights, and caches
├── requirements.txt           # Explicit environment dependencies
└── README.md                  # Project overview and technical documentation
```

---

## 💡 Key Engineering Decisions

### 1. Stratified Data Organization (`Data/final_sort.py`)
* Automatically parses raw lesion metadata and restructures unorganized image collections into standard directory structures compatible with batch generators.
* Prevents data leakage between train, validation, and test splits by isolating patient IDs where applicable.

### 2. Targeted Augmentation Pipeline (`models/augmentor.ipynb`)
* Medical datasets often suffer from drastic overrepresentation of benign nevi.
* Applied affine rotations, color jitter, cropping, and flips specifically tuned to dermatoscopic properties to balance rare minority classes (e.g., Melanoma, Dermatofibroma) without introducing geometric artifacts.

### 3. Model Decoupling & Rapid Prototyping (`models/model.py` & `new_model.ipynb`)
* Defined base convolutional models and custom classification heads in reusable Python modules rather than monolithic notebook scripts.
* Facilitated rapid testing of transfer learning backbones (e.g., ResNet, EfficientNet) against baseline CNN architectures.

### 4. Rigorous Diagnostic Evaluation (`models/evaluator_v2.ipynb`)
* In medical triage, **false negatives carry severe clinical consequences**.
* The evaluation suite inspects:
  * **Macro & Per-Class F1-Scores**
  * **Sensitivity / Recall on Malignant Lesions**
  * **ROC-AUC curves** for multi-class discriminative capacity

---

## 🚀 Quickstart & Setup

### 1. Clone & Setup Environment
```bash
git clone https://github.com/vishalvemula1/skin-cancer.git
cd skin-cancer

# Create virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
cd Data
python final_sort.py
cd ..
```

### 3. Train & Evaluate
- Open `models/Main_Model_v2.ipynb` to run the training process.
- Open `models/evaluator_v2.ipynb` to inspect classification metrics and generate visual reports.

---

## 📊 Sample Performance Summary

> *Tip: Populate this section with your actual test split metrics.*

| Metric | Target / Baseline | Result |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | > 85% | `[e.g., 88.4%]` |
| **Macro F1-Score** | > 0.80 | `[e.g., 0.83]` |
| **Melanoma Sensitivity (Recall)** | > 85% | `[e.g., 89.1%]` |
| **ROC-AUC (Macro)** | > 0.90 | `[e.g., 0.94]` |
