import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump as joblib_dump

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from features_run import build_feature_matrices
from config_and_utils import SignalConfig
from sklearn.ensemble import HistGradientBoostingClassifier


def plot_confusion(cm: np.ndarray, labels: List[str], title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def train_and_eval_ml(train_path: str, val_path: str, test_path: str, fs: float):
    (F_tr, y_tr), (F_va, y_va), (F_te, y_te), feat_names, labels = build_feature_matrices(
        train_path, val_path, test_path, fs
    )

    if HAS_LGBM:
        base_model = LGBMClassifier(
            objective="multiclass", num_class=len(labels), random_state=42, n_estimators=500
        )
        param_dist = {
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 8, 12],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_child_samples": [10, 20, 50],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
    elif HAS_XGB:
        base_model = XGBClassifier(
            objective="multi:softprob", num_class=len(labels), random_state=42,
            n_estimators=600, tree_method="hist", eval_metric="mlogloss"
        )
        param_dist = {
            "max_depth": [4, 6, 10],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 5, 10],
        }
    else:
        # Robust sklearn fallback: HistGradientBoostingClassifier (supports multiclass)
        base_model = HistGradientBoostingClassifier(random_state=42)
        param_dist = {
            "max_depth": [None, 6, 10, 14],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_leaf_nodes": [31, 63, 127],
            "l2_regularization": [0.0, 0.01, 0.1],
        }

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", base_model)
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions={f"clf__{k}": v for k, v in param_dist.items()},
        n_iter=12,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(F_tr, y_tr)
    best_model = search.best_estimator_

    print("Best params:", search.best_params_)
    print("CV best score:", search.best_score_)

    yv_pred = best_model.predict(F_va)
    print("Validation:")
    print(f"Acc={accuracy_score(y_va, yv_pred):.4f} | F1-macro={f1_score(y_va, yv_pred, average='macro'):.4f} | F1-micro={f1_score(y_va, yv_pred, average='micro'):.4f}")
    print(classification_report(y_va, yv_pred, target_names=list(labels)))

    yt_pred = best_model.predict(F_te)
    print("Test:")
    print(f"Acc={accuracy_score(y_te, yt_pred):.4f} | F1-macro={f1_score(y_te, yt_pred, average='macro'):.4f} | F1-micro={f1_score(y_te, yt_pred, average='micro'):.4f}")
    print(classification_report(y_te, yt_pred, target_names=list(labels)))

    cm_te = confusion_matrix(y_te, yt_pred)
    plot_confusion(cm_te, list(labels), title="Confusion Matrix (Test)")

    os.makedirs("artifacts", exist_ok=True)
    joblib_dump_path = "artifacts/ml_best.joblib"
    joblib_dump(best_model, joblib_dump_path)
    print("Saved ML model to:", joblib_dump_path)

    return best_model


if __name__ == "__main__":
    cfg = SignalConfig()
    train_path = "data/train_v1.npz"
    val_path   = "data/val_v1.npz"
    test_path  = "data/test_v1.npz"
    _ = train_and_eval_ml(train_path, val_path, test_path, fs=cfg.fs)
