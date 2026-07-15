"""
SVM Digit Classification Project
=================================
Classifies handwritten digit images (0-9) using Support Vector Machines.

Dataset: sklearn's built-in `digits` dataset — 1,797 images, 8x8 pixels each,
grayscale, representing handwritten digits 0 through 9. It ships with
scikit-learn so no external download is required.

What this script does:
  1. Loads and visualizes sample digit images
  2. Splits data into train/test sets
  3. Trains a baseline SVM classifier
  4. Tunes hyperparameters (kernel, C, gamma) with GridSearchCV
  5. Evaluates the best model (accuracy, classification report, confusion matrix)
  6. Visualizes misclassified digits
  7. Saves the trained model to disk

Run:
    python svm_digit_classifier.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; we save figures to files
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load the digits dataset and return images + flattened feature vectors."""
    digits = datasets.load_digits()
    print(f"Loaded {len(digits.images)} images, each {digits.images[0].shape}")
    print(f"Classes: {np.unique(digits.target)}")
    return digits


def plot_sample_images(digits, n=10, filename="01_sample_digits.png"):
    """Save a figure showing example digit images with their true labels."""
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.5))
    for ax, image, label in zip(axes, digits.images[:n], digits.target[:n]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{label}", fontsize=10)
    fig.suptitle("Sample digits from the dataset")
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved sample digit grid -> {path}")


def prepare_features(digits):
    """Flatten 8x8 images into 64-length feature vectors and split train/test."""
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))  # (1797, 64)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42, stratify=y
    )

    # Feature scaling generally helps SVMs converge better / perform better
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_baseline(X_train, y_train):
    """Train a simple default SVM as a baseline for comparison."""
    clf = svm.SVC(kernel="rbf", gamma="scale")
    clf.fit(X_train, y_train)
    return clf


def tune_hyperparameters(X_train, y_train):
    """Use grid search with cross-validation to find good SVM hyperparameters."""
    param_grid = [
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, "scale"]},
        {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
        {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]},
    ]
    grid = GridSearchCV(
        svm.SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def evaluate(clf, X_test, y_test, label="Model"):
    """Print accuracy + classification report, return predictions."""
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"\n--- {label} evaluation ---")
    print(f"Test accuracy: {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred, digits=3))
    return y_pred, acc


def plot_confusion_matrix(y_test, y_pred, filename="02_confusion_matrix.png"):
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix - Tuned SVM")
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def plot_misclassified(digits, X_test_idx_map, y_test, y_pred, X_test_raw,
                        filename="03_misclassified.png", max_show=10):
    """Show a grid of misclassified test images alongside true/predicted labels."""
    wrong = np.where(y_test != y_pred)[0]
    print(f"\nMisclassified: {len(wrong)} out of {len(y_test)} test samples")
    if len(wrong) == 0:
        print("No misclassifications to plot!")
        return

    n = min(max_show, len(wrong))
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.8))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, wrong[:n]):
        image = X_test_raw[idx].reshape(8, 8)
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"T:{y_test[idx]} P:{y_pred[idx]}", fontsize=9, color="red")
    fig.suptitle("Misclassified digits (True vs Predicted)")
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved misclassified digits grid -> {path}")


def main():
    print("=" * 60)
    print("SVM Digit Classification Project")
    print("=" * 60)

    # 1. Load & explore
    digits = load_data()
    plot_sample_images(digits)

    # 2. Prepare features (unscaled raw version kept for plotting original pixels)
    n_samples = len(digits.images)
    X_raw = digits.images.reshape((n_samples, -1))
    y = digits.target
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, shuffle=True, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 3. Baseline model
    baseline_clf = train_baseline(X_train, y_train)
    _, baseline_acc = evaluate(baseline_clf, X_test, y_test, label="Baseline SVM (RBF, default params)")

    # 4. Hyperparameter tuning
    print("\nRunning grid search (this may take a little while)...")
    best_clf, best_params = tune_hyperparameters(X_train, y_train)
    y_pred, tuned_acc = evaluate(best_clf, X_test, y_test, label="Tuned SVM (best params from grid search)")

    # 5. Visualize results
    plot_confusion_matrix(y_test, y_pred)
    plot_misclassified(digits, None, y_test, y_pred, X_test_raw)

    # 6. Save model + scaler
    model_path = os.path.join(OUTPUT_DIR, "svm_digit_model.joblib")
    joblib.dump({"model": best_clf, "scaler": scaler, "params": best_params}, model_path)
    print(f"\nSaved trained model -> {model_path}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline accuracy : {baseline_acc:.4f}")
    print(f"Tuned accuracy    : {tuned_acc:.4f}")
    print(f"Best hyperparams  : {best_params}")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
