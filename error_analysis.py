import numpy as np
import matplotlib.pyplot as plt
from typing import List
from joblib import load as joblib_load
from sklearn.metrics import accuracy_score

from features_run import build_feature_matrices


def plot_feature_importances(model, feat_names: List[str], top_k: int = 25, title: str = "Top feature importances"):
    if hasattr(model.named_steps["clf"], "feature_importances_"):
        importances = model.named_steps["clf"].feature_importances_
        idx = np.argsort(importances)[::-1][:top_k]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(idx)), importances[idx][::-1])
        plt.yticks(range(len(idx)), [feat_names[i] for i in idx][::-1], fontsize=8)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not provide feature_importances_")


def per_class_per_snr_accuracy(y_true: np.ndarray, y_pred: np.ndarray, snr_arr: np.ndarray, num_classes: int):
    snr_values = sorted(list(set([float(s) for s in snr_arr])))
    mat = np.zeros((num_classes, len(snr_values)))
    for j, s in enumerate(snr_values):
        idx = np.where(np.isclose(snr_arr, s))[0]
        yt = y_true[idx]
        yp = y_pred[idx]
        for c in range(num_classes):
            mask = yt == c
            if mask.sum() > 0:
                mat[c, j] = (yp[mask] == yt[mask]).mean()
            else:
                mat[c, j] = np.nan
    return snr_values, mat


def plot_heatmap(mat: np.ndarray, class_names: List[str], snr_values: List[float], title: str):
    plt.figure(figsize=(1.2*len(snr_values), 0.4*len(class_names) + 3))
    plt.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(snr_values)), [str(int(s)) for s in snr_values])
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Class")
    plt.title(title)
    plt.tight_layout()
    plt.show()
