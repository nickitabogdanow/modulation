import numpy as np
from scipy import signal
from typing import Dict, Tuple

from config_and_utils import (
    SignalConfig, rrc_filter, upsample, add_awgn, apply_cfo,
    apply_phase_noise, apply_timing_offset, apply_iq_imbalance,
    gaussian_pulse
)


def rand_bits(n: int) -> np.ndarray:
    return np.random.randint(0, 2, size=n)


def psk_symbols(M: int, n_sym: int) -> np.ndarray:
    if M == 2:
        b = rand_bits(n_sym)
        return (2*b - 1).astype(complex)
    else:
        syms = np.random.randint(0, M, size=n_sym)
        ph = 2 * np.pi * syms / M
        return np.exp(1j * ph)


def qam_symbols(M: int, n_sym: int) -> np.ndarray:
    m_side = int(np.sqrt(M))
    assert m_side * m_side == M
    levels = np.arange(-m_side+1, m_side, 2)
    I = np.random.choice(levels, size=n_sym)
    Q = np.random.choice(levels, size=n_sym)
    const = I + 1j * Q
    const = const / np.sqrt(np.mean(np.abs(const) ** 2))
    return const


def cpfsk_phase(symbols: np.ndarray, h: float, sps: int) -> np.ndarray:
    up = upsample(symbols, sps).astype(float)
    freq = h * up
    phase = 2 * np.pi * np.cumsum(freq) / sps
    return np.exp(1j * phase)


def gfsk_phase(symbols: np.ndarray, h: float, sps: int, bt: float) -> np.ndarray:
    up = upsample(symbols, sps).astype(float)
    h_gauss = gaussian_pulse(sps=sps, bt=bt, span_symbols=4)
    freq_shaped = np.convolve(up, h_gauss, mode='same') * h
    phase = 2 * np.pi * np.cumsum(freq_shaped) / sps
    return np.exp(1j * phase)


def am_dsb(baseband: np.ndarray, mod_index: float) -> np.ndarray:
    m = np.real(baseband)
    x = (1.0 + mod_index * m).astype(float)
    return x.astype(complex)


def am_ssb(baseband: np.ndarray, mod_index: float, side: str = "USB") -> np.ndarray:
    analytic = signal.hilbert(np.real(baseband))
    if side.upper() == "USB":
        ssb = np.real(analytic).astype(float)
    else:
        ssb = np.real(np.conj(analytic)).astype(float)
    x = (mod_index * ssb).astype(float)
    return x.astype(complex)


def wbfm(baseband: np.ndarray, fs: float, dev_hz: float) -> np.ndarray:
    m = np.real(baseband)
    kf = 2 * np.pi * dev_hz / fs
    phi = kf * np.cumsum(m)
    return np.exp(1j * phi)


def pulse_shape_and_trim(up: np.ndarray, filt: np.ndarray, out_len: int) -> np.ndarray:
    y = np.convolve(up, filt, mode='same')
    if len(y) >= out_len:
        return y[:out_len]
    else:
        pad = np.zeros(out_len - len(y), dtype=y.dtype)
        return np.concatenate([y, pad], axis=0)


def synthesize_modulation(mod: str, cfg: SignalConfig) -> np.ndarray:
    n_sym = int(np.ceil(cfg.window_size / cfg.sps)) + 4
    if mod == "BPSK":
        syms = psk_symbols(2, n_sym)
        filt = rrc_filter(num_taps=8*cfg.sps+1, beta=0.35, sps=cfg.sps)
        y = pulse_shape_and_trim(upsample(syms, cfg.sps), filt, cfg.window_size)
    elif mod == "QPSK":
        syms = psk_symbols(4, n_sym)
        filt = rrc_filter(num_taps=8*cfg.sps+1, beta=0.35, sps=cfg.sps)
        y = pulse_shape_and_trim(upsample(syms, cfg.sps), filt, cfg.window_size)
    elif mod == "PSK8":
        syms = psk_symbols(8, n_sym)
        filt = rrc_filter(num_taps=8*cfg.sps+1, beta=0.35, sps=cfg.sps)
        y = pulse_shape_and_trim(upsample(syms, cfg.sps), filt, cfg.window_size)
    elif mod == "QAM16":
        syms = qam_symbols(16, n_sym)
        filt = rrc_filter(num_taps=8*cfg.sps+1, beta=0.25, sps=cfg.sps)
        y = pulse_shape_and_trim(upsample(syms, cfg.sps), filt, cfg.window_size)
    elif mod == "QAM64":
        syms = qam_symbols(64, n_sym)
        filt = rrc_filter(num_taps=8*cfg.sps+1, beta=0.2, sps=cfg.sps)
        y = pulse_shape_and_trim(upsample(syms, cfg.sps), filt, cfg.window_size)
    elif mod == "CPFSK":
        bits = rand_bits(n_sym)
        symbols = 2*bits - 1
        y = cpfsk_phase(symbols, h=cfg.fsk_h, sps=cfg.sps)
        y = y[:cfg.window_size]
    elif mod == "GFSK":
        bits = rand_bits(n_sym)
        symbols = 2*bits - 1
        y = gfsk_phase(symbols, h=cfg.fsk_h, sps=cfg.sps, bt=cfg.gaussian_bt)
        y = y[:cfg.window_size]
    elif mod == "AM-DSB":
        m = np.random.normal(size=cfg.window_size)
        b = signal.firwin(numtaps=129, cutoff=0.05)
        m = signal.lfilter(b, [1.0], m)
        y = am_dsb(m.astype(complex), mod_index=cfg.am_mod_index)
    elif mod == "AM-SSB":
        m = np.random.normal(size=cfg.window_size)
        b = signal.firwin(numtaps=129, cutoff=0.05)
        m = signal.lfilter(b, [1.0], m)
        y = am_ssb(m.astype(complex), mod_index=cfg.am_mod_index, side="USB")
    elif mod == "WBFM":
        m = np.random.normal(size=cfg.window_size)
        b = signal.firwin(numtaps=129, cutoff=0.05)
        m = signal.lfilter(b, [1.0], m)
        y = wbfm(m.astype(complex), fs=cfg.fs, dev_hz=cfg.fm_deviation_hz)
    else:
        raise ValueError(f"Unknown modulation: {mod}")
    y = y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-12)
    return y


def apply_random_impairments(x: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    cfo_ppm = np.random.uniform(*cfg.cfo_ppm_range)
    toff = np.random.uniform(*cfg.timing_offset_range)
    iq_amp_db = np.random.uniform(*cfg.iq_amp_imbalance_db_range)
    iq_ph_deg = np.random.uniform(*cfg.iq_phase_imbalance_deg_range)
    pn_std = np.random.uniform(*cfg.phase_noise_std_range)

    y = x
    y = apply_timing_offset(y, toff)
    y = apply_cfo(y, cfg.fs, cfo_ppm)
    y = apply_phase_noise(y, pn_std)
    y = apply_iq_imbalance(y, iq_amp_db, iq_ph_deg)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-12)
    return y


def generate_example(mod: str, snr_db: float, cfg: SignalConfig, impair: bool = True):
    x = synthesize_modulation(mod, cfg)
    if impair:
        x = apply_random_impairments(x, cfg)
    y = add_awgn(x, snr_db)
    meta = {"mod": mod, "snr_db": snr_db}
    return y, meta
