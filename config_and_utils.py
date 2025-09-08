from dataclasses import dataclass
from typing import Tuple, Any, Dict
import numpy as np
import os
import random
import logging
from functools import lru_cache
import yaml


def db2lin(db: float) -> float:
    return 10 ** (db / 10.0)


def lin2db(lin: float) -> float:
    return 10 * np.log10(np.maximum(lin, 1e-12))


def set_seed(seed: int = 42, deterministic_tf: bool = True) -> None:
    """Set seeds for Python, NumPy, and TensorFlow (if available).

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic_tf : bool
        Whether to enable deterministic operations in TensorFlow when possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf  # Lazy import to avoid hard dependency
        tf.random.set_seed(seed)
        if deterministic_tf:
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
            try:
                # TF 2.13+
                tf.config.experimental.enable_op_determinism(True)
            except Exception:
                pass
        # Reduce noisy TF logs by default
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    except Exception:
        # TensorFlow not installed or determinism not available; ignore
        pass


def configure_logging(log_dir: str = "reports", file_prefix: str = "run", level: int = logging.INFO) -> logging.Logger:
    """Configure a root project logger that logs to both stdout and a rotating file.

    Parameters
    ----------
    log_dir : str
        Directory to store log files.
    file_prefix : str
        Prefix for the log file name.
    level : int
        Logging level, e.g., logging.INFO.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("modulation")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f"{file_prefix}_{ts}.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass
class SignalConfig:
    fs: float = 1_000_000.0       # Sampling rate (Hz)
    sps: int = 8                  # Samples per symbol
    window_size: int = 4096       # Window length per example
    snr_db_grid: Tuple[int, ...] = (-6, 0, 6, 10, 14, 18, 20)
    # Impairments ranges
    cfo_ppm_range: Tuple[float, float] = (-20_000.0, 20_000.0)  # ppm of fs
    timing_offset_range: Tuple[float, float] = (-0.3, 0.3)      # fraction of sps
    iq_amp_imbalance_db_range: Tuple[float, float] = (-1.0, 1.0) # dB
    iq_phase_imbalance_deg_range: Tuple[float, float] = (-5.0, 5.0) # degrees
    phase_noise_std_range: Tuple[float, float] = (0.0, 0.01)    # rad/sample (random walk std)
    # FSK
    fsk_h: float = 0.5
    gaussian_bt: float = 0.5
    # AM/FM
    am_mod_index: float = 0.5
    fm_deviation_hz: float = 75_000.0   # WBFM deviation
    # Classes
    classes: Tuple[str, ...] = (
        "BPSK", "QPSK", "PSK8",
        "QAM16", "QAM64",
        "CPFSK", "GFSK",
        "AM-DSB", "AM-SSB", "WBFM"
    )


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def make_signal_config_from_yaml(path: str) -> SignalConfig:
    raw = load_config_from_yaml(path)
    # Only set known fields
    fields = {f: raw.get(f, getattr(SignalConfig, f)) for f in SignalConfig.__annotations__.keys()}
    # Convert lists to tuples for tuple-typed fields
    for key in ["snr_db_grid", "classes"]:
        if key in fields and isinstance(fields[key], list):
            fields[key] = tuple(fields[key])
    return SignalConfig(**fields)
@lru_cache(maxsize=64)
def rrc_filter(num_taps: int, beta: float, sps: int) -> np.ndarray:
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    eps = 1e-10
    numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    denominator = np.pi * t * (1 - (4 * beta * t) ** 2) + eps
    h = numerator / denominator
    h /= np.sqrt(np.sum(h ** 2))
    return h


@lru_cache(maxsize=64)
def gaussian_pulse(sps: int, bt: float, span_symbols: int = 4) -> np.ndarray:
    t = np.linspace(-span_symbols/2, span_symbols/2, span_symbols * sps)
    alpha = np.sqrt(np.log(2)) / (bt)
    h = np.exp(-(np.pi * t / alpha) ** 2)
    h /= np.sum(h)
    return h


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    return up


def add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    p_sig = np.mean(np.abs(x) ** 2)
    x_norm = x / np.sqrt(p_sig + 1e-12)
    snr_lin = db2lin(snr_db)
    noise_power = 1.0 / snr_lin
    noise = (np.random.normal(scale=np.sqrt(noise_power/2), size=x.shape) +
             1j * np.random.normal(scale=np.sqrt(noise_power/2), size=x.shape))
    return x_norm + noise


def apply_cfo(x: np.ndarray, fs: float, cfo_ppm: float) -> np.ndarray:
    cfo_hz = fs * cfo_ppm * 1e-6
    n = np.arange(len(x))
    ph = 2 * np.pi * cfo_hz * n / fs
    return x * np.exp(1j * ph)


def apply_phase_noise(x: np.ndarray, std_per_sample: float) -> np.ndarray:
    if std_per_sample <= 0:
        return x
    dphi = np.random.normal(0.0, std_per_sample, size=len(x))
    phi = np.cumsum(dphi)
    return x * np.exp(1j * phi)


def apply_timing_offset(x: np.ndarray, offset: float) -> np.ndarray:
    if abs(offset) < 1e-6:
        return x
    N = 31
    n = np.arange(-N//2, N//2 + 1)
    h = np.sinc(n - offset)
    h *= np.hamming(len(h))
    h /= np.sum(h)
    return np.convolve(x, h, mode='same')


def apply_iq_imbalance(x: np.ndarray, amp_db: float, phase_deg: float) -> np.ndarray:
    amp_scale = 10 ** (amp_db / 20.0)
    phi = np.deg2rad(phase_deg)
    i = np.real(x) * amp_scale
    q = np.imag(x)
    y = (i + 1j * (q * np.cos(phi) + i * np.sin(phi)))
    return y
