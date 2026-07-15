# SVM Handwritten Digit Classifier

A complete, self-contained image classification project using **Support Vector
Machines (SVM)** to recognize handwritten digits (0–9).

## Overview

- **Dataset**: scikit-learn's built-in `digits` dataset — 1,797 grayscale
  images, 8×8 pixels each. No download required.
- **Model**: `sklearn.svm.SVC`, tuned via `GridSearchCV` over kernel type
  (linear / rbf / poly), `C`, and `gamma`.
- **Result**: ~98% test accuracy.

## Project structure

```
svm_digit_project/
├── svm_digit_classifier.py   # main script: load, train, tune, evaluate, save
├── requirements.txt
├── README.md
└── outputs/                  # generated after running the script
    ├── 01_sample_digits.png
    ├── 02_confusion_matrix.png
    ├── 03_misclassified.png
    └── svm_digit_model.joblib
```

## How it works

1. **Load & explore** — loads the digit images and plots a few samples.
2. **Preprocess** — flattens each 8×8 image into a 64-value feature vector
   and standardizes features with `StandardScaler` (SVMs are distance-based,
   so scaling matters).
3. **Baseline model** — trains a default RBF-kernel SVM to establish a
   reference accuracy.
4. **Hyperparameter tuning** — runs 5-fold cross-validated grid search over:
   - kernel: `rbf`, `linear`, `poly`
   - `C`: regularization strength
   - `gamma`: kernel coefficient (for rbf/poly)
5. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion
   matrix heatmap.
6. **Error analysis** — visualizes which digits get misclassified and what
   the model predicted instead.
7. **Save** — persists the trained model + scaler to `svm_digit_model.joblib`
   so it can be reloaded without retraining.

## Running it

```bash
pip install -r requirements.txt
python svm_digit_classifier.py
```

All plots and the saved model land in `outputs/`.

## Using the saved model on new data

```python
import joblib
import numpy as np

bundle = joblib.load("outputs/svm_digit_model.joblib")
model, scaler = bundle["model"], bundle["scaler"]

# image must be an 8x8 grayscale array, pixel values roughly 0-16
# (same format as sklearn.datasets.load_digits)
def predict_digit(image_8x8):
    flat = np.array(image_8x8).reshape(1, -1)
    flat_scaled = scaler.transform(flat)
    return model.predict(flat_scaled)[0]
```

## Ideas to extend this project

- Swap in the full **MNIST** dataset (28×28, 70k images) for a harder
  benchmark — expect longer training times with SVM at that scale.
- Try `sklearn.decomposition.PCA` before the SVM to speed up training on
  larger images.
- Add a simple **Tkinter/Streamlit** drawing canvas so you can hand-draw a
  digit and classify it live.
- Compare SVM against other classifiers (k-NN, Random Forest, a small CNN)
  on the same data/split.
