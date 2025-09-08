from dataclasses import dataclass
from typing import Tuple
import numpy as np


def db2lin(db: float) -> float:
    return 10 ** (db / 10.0)


def lin2db(lin: float) -> float:
    return 10 * np.log10(np.maximum(lin, 1e-12))


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


def rrc_filter(num_taps: int, beta: float, sps: int) -> np.ndarray:
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    eps = 1e-10
    numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    denominator = np.pi * t * (1 - (4 * beta * t) ** 2) + eps
    h = numerator / denominator
    h /= np.sqrt(np.sum(h ** 2))
    return h


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
