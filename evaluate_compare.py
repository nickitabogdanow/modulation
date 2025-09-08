import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
from joblib import load as joblib_load

from features_run import build_feature_matrices
from dl_prep import load_npz, make_dataset_1d
from config_and_utils import SignalConfig


def evaluate_ml_per_snr(train_path: str, val_path: str, test_path: str, fs: float, ml_model_path: str):
    (F_tr, y_tr), (F_va, y_va), (F_te, y_te), feat_names, labels = build_feature_matrices(
        train_path, val_path, test_path, fs
    )
    test = np.load(test_path, allow_pickle=True)
    snr_te = test["snr_db"]

    clf = joblib_load(ml_model_path)
    y_pred = clf.predict(F_te)

    overall = {
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_te, y_pred, average="micro")),
    }

    snr_values = sorted(list(set([float(s) for s in snr_te])))
    per_snr = {}
    for s in snr_values:
        idx = np.where(np.isclose(snr_te, s))[0]
        if len(idx) == 0:
            continue
        acc = accuracy_score(y_te[idx], y_pred[idx])
        f1m = f1_score(y_te[idx], y_pred[idx], average="macro")
        f1u = f1_score(y_te[idx], y_pred[idx], average="micro")
        per_snr[s] = (float(acc), float(f1m), float(f1u))

    return per_snr, overall, labels, y_te, y_pred


def evaluate_dl_per_snr(test_path: str, dl_model_path: str, num_classes: int, batch_size: int = 128):
    test = load_npz(test_path)
    y_te = test["y"].astype(np.int64)
    snr_te = test["snr_db"].astype(np.float32)
    labels = test["mod_labels"]

    ds_te = make_dataset_1d(test_path, num_classes=num_classes, batch_size=batch_size, shuffle=False, augment=False)
    model = tf.keras.models.load_model(dl_model_path)

    y_proba = model.predict(ds_te, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    overall = {
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_te, y_pred, average="micro")),
    }

    snr_values = sorted(list(set([float(s) for s in snr_te])))
    per_snr = {}
    for s in snr_values:
        idx = np.where(np.isclose(snr_te, s))[0]
        if len(idx) == 0:
            continue
        acc = accuracy_score(y_te[idx], y_pred[idx])
        f1m = f1_score(y_te[idx], y_pred[idx], average="macro")
        f1u = f1_score(y_te[idx], y_pred[idx], average="micro")
        per_snr[s] = (float(acc), float(f1m), float(f1u))

    return per_snr, overall, labels, y_te, y_pred


def plot_metrics_vs_snr(per_snr_ml: Dict[float, Tuple[float, float, float]],
                        per_snr_dl: Dict[float, Tuple[float, float, float]],
                        title_prefix: str = ""):
    snrs = sorted(list(set(list(per_snr_ml.keys()) + list(per_snr_dl.keys()))))

    acc_ml = [per_snr_ml[s][0] if s in per_snr_ml else np.nan for s in snrs]
    f1m_ml = [per_snr_ml[s][1] if s in per_snr_ml else np.nan for s in snrs]
    acc_dl = [per_snr_dl[s][0] if s in per_snr_dl else np.nan for s in snrs]
    f1m_dl = [per_snr_dl[s][1] if s in per_snr_dl else np.nan for s in snrs]

    plt.figure(figsize=(8,4))
    plt.plot(snrs, acc_ml, "o--", label="ML Acc")
    plt.plot(snrs, acc_dl, "o--", label="DL Acc")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(snrs, f1m_ml, "o--", label="ML F1-macro")
    plt.plot(snrs, f1m_dl, "o--", label="DL F1-macro")
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1-macro")
    plt.title(f"{title_prefix} F1-macro vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusions_by_snr(y_true, y_pred, snr_arr, class_names, chosen_snrs: List[float], suptitle: str):
    cols = len(chosen_snrs)
    plt.figure(figsize=(5*cols, 4))
    for i, s in enumerate(chosen_snrs):
        idx = np.where(np.isclose(snr_arr, s))[0]
        cm = confusion_matrix(y_true[idx], y_pred[idx], labels=range(len(class_names)))
        plt.subplot(1, cols, i+1)
        plt.imshow(cm, cmap="Blues")
        plt.title(f"SNR={s} dB")
        plt.colorbar()
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(ticks, class_names, fontsize=8)
        plt.tight_layout()
    plt.suptitle(suptitle)
    plt.show()


if __name__ == "__main__":
    cfg = SignalConfig()
    train_path = "data/train_v1.npz"
    val_path   = "data/val_v1.npz"
    test_path  = "data/test_v1.npz"

    ml_model_path = "artifacts/ml_best.joblib"
    dl_model_path = "checkpoints_1d/best.keras"

    per_snr_ml, overall_ml, labels, y_te_ml, y_pred_ml = evaluate_ml_per_snr(
        train_path, val_path, test_path, fs=cfg.fs, ml_model_path=ml_model_path
    )
    print("ML overall:", overall_ml)

    num_classes = len(labels)
    per_snr_dl, overall_dl, labels_dl, y_te_dl, y_pred_dl = evaluate_dl_per_snr(
        test_path, dl_model_path, num_classes=num_classes
    )
    print("DL overall:", overall_dl)

    plot_metrics_vs_snr(per_snr_ml, per_snr_dl, title_prefix="ML vs DL")

    snrs_all = sorted(list(per_snr_ml.keys()))
    if len(snrs_all) >= 3:
        chosen = [snrs_all[0], snrs_all[len(snrs_all)//2], snrs_all[-1]]
    elif len(snrs_all) > 0:
        chosen = snrs_all
    else:
        chosen = []

    test_npz = np.load(test_path, allow_pickle=True)
    snr_te = test_npz["snr_db"]

    if chosen:
        plot_confusions_by_snr(y_te_ml, y_pred_ml, snr_te, list(labels), chosen, suptitle="ML confusions by SNR")
        plot_confusions_by_snr(y_te_dl, y_pred_dl, snr_te, list(labels), chosen, suptitle="DL confusions by SNR")
