import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import argparse
import logging

from config_and_utils import SignalConfig, configure_logging, set_seed
from dl_prep import make_dataset_1d, make_dataset_2d_spectrogram, load_npz
from models_dl import build_cnn1d, build_cnn2d_spectrogram


def evaluate_model(model: tf.keras.Model, ds, y_true, class_names, reports_dir: str, prefix: str = "val"):
    y_pred_proba = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1u = f1_score(y_true, y_pred, average="micro")
    with open(os.path.join(reports_dir, f"dl_{prefix}_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy={acc:.4f} | F1-macro={f1m:.4f} | F1-micro={f1u:.4f}\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"dl_{prefix}_confusion.png"), dpi=150)
    plt.close()


def train_1d(
    train_path: str, val_path: str, test_path: str,
    cfg: SignalConfig,
    batch_size: int = 128,
    epochs: int = 30,
    out_dir: str = "checkpoints_1d",
    reports_dir: str = "reports",
    seed: int = 42,
    logger: logging.Logger = None
):
    if logger is None:
        logger = logging.getLogger("modulation")
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    train = load_npz(train_path)
    val = load_npz(val_path)
    test = load_npz(test_path)

    num_classes = len(train["mod_labels"])
    T = train["X"].shape[1]
    C = train["X"].shape[2]

    ds_tr = make_dataset_1d(train_path, num_classes=num_classes, batch_size=batch_size, shuffle=True, augment=True)
    ds_va = make_dataset_1d(val_path,   num_classes=num_classes, batch_size=batch_size, shuffle=False, augment=False)
    ds_te = make_dataset_1d(test_path,  num_classes=num_classes, batch_size=batch_size, shuffle=False, augment=False)

    model = build_cnn1d(input_len=T, num_channels=C, num_classes=num_classes)
    model.summary(print_fn=lambda s: logger.info(s))

    ckpt_path = os.path.join(out_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]

    history = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Validation evaluation:")
    evaluate_model(model, ds_va, y_true=val["y"], class_names=list(train["mod_labels"]), reports_dir=reports_dir, prefix="val")

    logger.info("Test evaluation:")
    evaluate_model(model, ds_te, y_true=test["y"], class_names=list(train["mod_labels"]), reports_dir=reports_dir, prefix="test")

    return model, history


def train_2d_spectrogram(
    train_path: str, val_path: str, test_path: str,
    cfg: SignalConfig,
    batch_size: int = 64,
    epochs: int = 25,
    out_dir: str = "checkpoints_2d",
    reports_dir: str = "reports",
    seed: int = 42,
    logger: logging.Logger = None
):
    if logger is None:
        logger = logging.getLogger("modulation")
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    train = load_npz(train_path)
    val = load_npz(val_path)
    test = load_npz(test_path)

    num_classes = len(train["mod_labels"])

    ds_tr = make_dataset_2d_spectrogram(train_path, num_classes=num_classes, batch_size=batch_size, shuffle=True)
    ds_va = make_dataset_2d_spectrogram(val_path,   num_classes=num_classes, batch_size=batch_size, shuffle=False)
    ds_te = make_dataset_2d_spectrogram(test_path,  num_classes=num_classes, batch_size=batch_size, shuffle=False)

    for xb, yb in ds_tr.take(1):
        input_shape = xb.shape[1:]
    model = build_cnn2d_spectrogram(input_shape, num_classes=num_classes)
    model.summary(print_fn=lambda s: logger.info(s))

    ckpt_path = os.path.join(out_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ]

    history = model.fit(
        ds_tr, validation_data=ds_va, epochs=epochs, callbacks=callbacks, verbose=1
    )

    logger.info("Validation evaluation (2D):")
    evaluate_model(model, ds_va, y_true=val["y"], class_names=list(train["mod_labels"]), reports_dir=reports_dir, prefix="val2d")

    logger.info("Test evaluation (2D):")
    evaluate_model(model, ds_te, y_true=test["y"], class_names=list(train["mod_labels"]), reports_dir=reports_dir, prefix="test2d")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DL models (1D IQ and optional 2D spectrogram)")
    parser.add_argument("--train", type=str, default="data/train_v1.npz")
    parser.add_argument("--val", type=str, default="data/val_v1.npz")
    parser.add_argument("--test", type=str, default="data/test_v1.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="checkpoints_1d")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="reports")
    parser.add_argument("--mode", type=str, choices=["1d", "2d"], default="1d")
    args = parser.parse_args()

    logger = configure_logging(log_dir=args.log_dir, file_prefix="train_dl")
    cfg = SignalConfig()
    if args.mode == "1d":
        _ = train_1d(args.train, args.val, args.test, cfg, batch_size=args.batch_size, epochs=args.epochs, out_dir=args.out, reports_dir=args.reports, seed=args.seed, logger=logger)
    else:
        _ = train_2d_spectrogram(args.train, args.val, args.test, cfg, batch_size=max(64, args.batch_size//2), epochs=max(25, args.epochs), out_dir="checkpoints_2d", reports_dir=args.reports, seed=args.seed, logger=logger)
