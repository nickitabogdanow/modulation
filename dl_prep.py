import numpy as np
import tensorflow as tf
from typing import Dict
from scipy import signal

AUTOTUNE = tf.data.AUTOTUNE


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def standardize_iq(example: np.ndarray) -> np.ndarray:
    x = example.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / std


def tf_standardize(x, y):
    x = tf.numpy_function(standardize_iq, [x], tf.float32)
    x.set_shape([None, 2])
    return x, y


def augment_iq_small(x, y):
    scale = tf.random.uniform([], 0.9, 1.1)
    x = x * scale
    noise = tf.random.normal(tf.shape(x), stddev=0.02)
    x = x + noise
    return x, y


def make_dataset_1d(
    path: str,
    num_classes: int,
    batch_size: int = 128,
    shuffle: bool = True,
    augment: bool = True
) -> tf.data.Dataset:
    data = load_npz(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    N = X.shape[0]

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(10_000, N), reshuffle_each_iteration=True)

    # Use sparse integer labels directly to save memory and CPU
    ds = ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)

    ds = ds.map(tf_standardize, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(augment_iq_small, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ===== 2D Spectrogram (optional) =====

def compute_spectrogram(x: np.ndarray, nperseg: int = 256, noverlap: int = 128) -> np.ndarray:
    xc = x[:,0] + 1j * x[:,1]
    f, t, Sxx = signal.spectrogram(xc, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
    Sxx = 20*np.log10(Sxx + 1e-9)
    Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-6)
    return Sxx.astype(np.float32)


def tf_spectrogram_stft(x, y, frame_length: int = 256, frame_step: int = 128):
    # x: [T, 2] float32
    # Convert to complex IQ
    real = x[:, 0]
    imag = x[:, 1]
    xc = tf.complex(real, imag)
    stft = tf.signal.stft(xc, frame_length=frame_length, frame_step=frame_step, window_fn=tf.signal.hann_window)
    mag = tf.abs(stft)  # [frames, freq]
    mag_db = 20.0 * tf.math.log(mag + 1e-9) / tf.math.log(10.0)
    mean = tf.reduce_mean(mag_db)
    std = tf.math.reduce_std(mag_db) + 1e-6
    spec = (mag_db - mean) / std
    spec = tf.expand_dims(spec, axis=-1)  # [frames, freq, 1]
    return spec, y


def make_dataset_2d_spectrogram(
    path: str,
    num_classes: int,
    batch_size: int = 64,
    shuffle: bool = True
) -> tf.data.Dataset:
    data = load_npz(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    N = X.shape[0]

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(10_000, N), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (x, y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(tf_standardize, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: tf_spectrogram_stft(x, y, 256, 128), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
