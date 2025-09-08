import numpy as np
from typing import Dict, Tuple
from features import extract_features_batch


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def twochan_to_complex(X_2ch: np.ndarray) -> np.ndarray:
    return X_2ch[...,0].astype(np.float32) + 1j * X_2ch[...,1].astype(np.float32)


def build_feature_matrices(train_path: str, val_path: str, test_path: str, fs: float):
    train = load_npz(train_path)
    val   = load_npz(val_path)
    test  = load_npz(test_path)

    Xc_tr = twochan_to_complex(train["X"])
    Xc_va = twochan_to_complex(val["X"])
    Xc_te = twochan_to_complex(test["X"])

    F_tr, keys = extract_features_batch(Xc_tr, fs)
    F_va, _    = extract_features_batch(Xc_va, fs)
    F_te, _    = extract_features_batch(Xc_te, fs)

    y_tr, y_va, y_te = train["y"], val["y"], test["y"]
    labels = train["mod_labels"]

    return (F_tr, y_tr), (F_va, y_va), (F_te, y_te), keys, labels
