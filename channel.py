import numpy as np
from typing import Optional
from scipy import signal

from config_and_utils import (
    SignalConfig,
    add_awgn, apply_cfo, apply_phase_noise,
    apply_timing_offset, apply_iq_imbalance
)


def random_multipath_ir(max_taps: int = 5, max_delay: int = 12, exp_decay: float = 0.7) -> np.ndarray:
    n_taps = np.random.randint(1, max_taps + 1)
    delays = np.random.choice(np.arange(0, max_delay + 1), size=n_taps, replace=False)
    ir_len = delays.max() + 1
    h = np.zeros(ir_len, dtype=complex)
    for d in delays:
        amp = (exp_decay ** d) * (0.5 + 0.5 * np.random.rand())
        phase = np.random.uniform(-np.pi, np.pi)
        h[d] += amp * np.exp(1j * phase)
    h = h / np.sqrt(np.sum(np.abs(h) ** 2) + 1e-12)
    return h


def apply_multipath(x: np.ndarray, ir: Optional[np.ndarray] = None) -> np.ndarray:
    if ir is None:
        ir = random_multipath_ir()
    y = signal.lfilter(ir, [1.0], x)
    if len(y) >= len(x):
        y = y[:len(x)]
    else:
        pad = np.zeros(len(x) - len(y), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=0)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-12)
    return y


def pass_through_channel(
    x: np.ndarray,
    cfg: SignalConfig,
    snr_db: float,
    use_multipath: bool = True
) -> np.ndarray:
    toff = np.random.uniform(*cfg.timing_offset_range)
    y = apply_timing_offset(x, toff)
    cfo_ppm = np.random.uniform(*cfg.cfo_ppm_range)
    y = apply_cfo(y, cfg.fs, cfo_ppm)
    pn_std = np.random.uniform(*cfg.phase_noise_std_range)
    y = apply_phase_noise(y, pn_std)
    iq_amp_db = np.random.uniform(*cfg.iq_amp_imbalance_db_range)
    iq_ph_deg = np.random.uniform(*cfg.iq_phase_imbalance_deg_range)
    y = apply_iq_imbalance(y, iq_amp_db, iq_ph_deg)
    if use_multipath:
        y = apply_multipath(y)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-12)
    y = add_awgn(y, snr_db)
    return y
