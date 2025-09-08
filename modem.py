"""
Educational modem utilities: modulation/demodulation for several schemes.

Supported (baseband complex, Fs from SignalConfig):
- BPSK, QPSK, PSK8
- QAM16, QAM64
- CPFSK (binary), GFSK (binary)
- AM-DSB, AM-SSB (analytic baseband)
- WBFM (analytic baseband)

Notes
- These are simplified, instructional implementations to illustrate ideas.
- For digital schemes we return symbols/bits; for AM/FM we return an example recovered message (normalized).
- Pulse shaping: RRC for linear modulations, Gaussian for GFSK.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import signal

from config_and_utils import (
    SignalConfig,
    rrc_filter,
    gaussian_pulse,
)


def _bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    assert bits.ndim == 1 and len(bits) % k == 0
    return bits.reshape(-1, k).dot(1 << np.arange(k)[::-1])


def _gray_encode(x: np.ndarray) -> np.ndarray:
    return x ^ (x >> 1)


def _gray_decode(g: np.ndarray) -> np.ndarray:
    # iterative Gray decode
    x = g.copy()
    shift = 1
    while (g >> shift).any():
        x ^= (g >> shift)
        shift += 1
    return x


def _pam_levels(M: int) -> np.ndarray:
    # Odd-even centered levels, normalized to unit average power when used with QAM
    levels = np.arange(-(M - 1), M, 2)
    return levels.astype(float)


def _norm_qam_constellation(M: int) -> Tuple[np.ndarray, float]:
    # Build square QAM (M=16,64) with Gray coding per-axis; normalize average power to 1
    m_side = int(np.sqrt(M))
    assert m_side * m_side == M
    lv = _pam_levels(m_side)
    const = (lv[:, None] + 1j * lv[None, :]).reshape(-1)
    Es_avg = np.mean(np.abs(const) ** 2)
    scale = 1.0 / np.sqrt(Es_avg)
    const *= scale
    return const, scale


def modulate(bits: np.ndarray,
             modulation: str,
             cfg: SignalConfig,
             message: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """
    Modulate a bitstream (or message for AM/FM) into complex baseband samples.

    Returns (x, info) where x is complex np.ndarray and info has metadata (e.g., mapping).
    For AM/FM you may pass `message` as a real-valued array in [-1,1]. If None, a test tone is used.
    """
    mod = modulation.upper()
    sps = cfg.sps
    info: Dict = {"modulation": mod, "sps": sps}

    if mod in {"BPSK", "QPSK", "PSK8"}:
        # PSK family
        if mod == "BPSK":
            M = 2; k = 1
        elif mod == "QPSK":
            M = 4; k = 2
        else:
            M = 8; k = 3
        ints = _bits_to_ints(bits.astype(int), k)
        # Gray mapping on phases 0..M-1
        gray = _gray_encode(ints)
        phases = 2 * np.pi * gray / M
        symbols = np.exp(1j * phases)
        # RRC pulse shaping
        up = np.zeros(len(symbols) * sps, dtype=complex)
        up[::sps] = symbols
        h = rrc_filter(num_taps=8 * sps + 1, beta=0.35, sps=sps)
        x = signal.lfilter(h, 1.0, up)
        # Normalize avg power ~1
        x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
        info.update({"M": M, "k": k})
        return x.astype(np.complex64), info

    if mod in {"QAM16", "QAM64"}:
        M = 16 if mod == "QAM16" else 64
        k = int(np.log2(M))
        ints = _bits_to_ints(bits.astype(int), k)
        # Gray per-axis mapping
        m_side = int(np.sqrt(M))
        i_bits = (ints >> (k // 2)) & (m_side - 1)
        q_bits = (ints & (m_side - 1))
        i_gray = _gray_encode(i_bits)
        q_gray = _gray_encode(q_bits)
        lv = _pam_levels(m_side)
        # Map Gray to indices then to levels
        def gray_to_level(g):
            # inverse gray: brute map for side size
            idx = _gray_decode(g)
            return lv[idx]
        I = gray_to_level(i_gray)
        Q = gray_to_level(q_gray)
        const, scale = _norm_qam_constellation(M)
        # But our per-axis mapping uses unscaled lv; apply same scale
        symbols = (I + 1j * Q) * (1.0 / np.sqrt(np.mean((lv ** 2)) * 2))
        # RRC shaping
        up = np.zeros(len(symbols) * sps, dtype=complex)
        up[::sps] = symbols
        h = rrc_filter(num_taps=8 * sps + 1, beta=0.35, sps=sps)
        x = signal.lfilter(h, 1.0, up)
        x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
        info.update({"M": M, "k": k})
        return x.astype(np.complex64), info

    if mod in {"CPFSK", "GFSK"}:
        # Assume binary CPFSK/GFSK: bits -> {-1,+1} frequency deviation
        a = (bits.astype(int) * 2 - 1).astype(float)
        # Upsample to samples
        a_up = np.repeat(a, sps)
        if mod == "GFSK":
            # Gaussian filter symbols before integrating phase
            g = gaussian_pulse(bt=0.5, sps=sps, span=4)
            a_up = np.convolve(a_up, g, mode='same')
        # Integrate phase
        df = 0.3  # normalized deviation (fraction of symbol rate)
        phi = 2 * np.pi * np.cumsum(a_up) * (df / sps)
        x = np.exp(1j * phi)
        x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
        info.update({"binary": True})
        return x.astype(np.complex64), info

    if mod in {"AM-DSB", "AM-SSB"}:
        m = message
        if m is None:
            # test tone
            t = np.arange(cfg.window_size) / cfg.fs
            m = np.sin(2 * np.pi * 1000.0 * t)
        m = np.clip(m, -1.0, 1.0)
        if mod == "AM-DSB":
            x = (1.0 + 0.7 * m).astype(np.float32)
            x = signal.lfilter([1.0], [1.0], x)  # identity (placeholder)
            x = x.astype(np.complex64)  # analytic baseband (I-only here)
        else:
            # AM-SSB via Hilbert (create analytic message and shift sign)
            m_analytic = signal.hilbert(m)
            x = np.real(m_analytic).astype(np.float32) + 1j * np.imag(m_analytic).astype(np.float32)
        return x.astype(np.complex64), {"message_used": True}

    if mod == "WBFM":
        m = message
        if m is None:
            t = np.arange(cfg.window_size) / cfg.fs
            m = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
        m = np.clip(m, -1.0, 1.0)
        kf = 2.5  # frequency deviation factor (example)
        phi = np.cumsum(m) * (kf)
        x = np.exp(1j * phi)
        x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
        return x.astype(np.complex64), {"message_used": True}

    raise ValueError(f"Unsupported modulation: {modulation}")


def demodulate(x: np.ndarray,
               modulation: str,
               cfg: SignalConfig,
               expected_bits: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    Demodulate complex baseband samples.

    For digital mods returns (bits, info); for AM/FM returns (message, info).
    """
    mod = modulation.upper()
    sps = cfg.sps
    info: Dict = {"modulation": mod}

    if mod in {"BPSK", "QPSK", "PSK8"}:
        # Matched filter + symbol timing at multiples of sps
        h = rrc_filter(num_taps=8 * sps + 1, beta=0.35, sps=sps)
        y = signal.lfilter(h[::-1].conj(), 1.0, x)
        # Total group delay tx(RRC)+rx(RRC) ≈ (len(h)-1)
        delay = (len(h) - 1)
        if len(y) <= delay:
            return np.array([], dtype=int), info
        sym = y[delay::sps]
        M = 2 if mod == "BPSK" else 4 if mod == "QPSK" else 8
        # Estimate and remove common phase for M-PSK using Mth power
        if M == 2:
            # Use 2nd-power estimator to remove 180° ambiguity in BPSK
            theta = 0.5 * np.angle(np.mean(sym ** 2) + 1e-12)
        else:
            mth = sym ** M
            theta = np.angle(np.mean(mth) + 1e-12) / M
        sym_corr = sym * np.exp(-1j * theta)
        phases = np.mod(np.angle(sym_corr), 2 * np.pi)
        idx = np.round(phases / (2 * np.pi / M)) % M
        gray = idx.astype(int)
        ints = _gray_decode(gray)
        k = int(np.log2(M))
        bits = (((ints[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
        return bits[:expected_bits] if expected_bits is not None else bits, info

    if mod in {"QAM16", "QAM64"}:
        h = rrc_filter(num_taps=8 * sps + 1, beta=0.35, sps=sps)
        y = signal.lfilter(h[::-1].conj(), 1.0, x)
        delay = (len(h) - 1)
        if len(y) <= delay:
            return np.array([], dtype=int), info
        sym = y[delay::sps]
        M = 16 if mod == "QAM16" else 64
        m_side = int(np.sqrt(M))
        # Estimate and remove common phase (4th power works well for square QAM)
        theta = 0.25 * np.angle(np.mean(sym ** 4) + 1e-12)
        sym = sym * np.exp(-1j * theta)
        # Normalize average power to 1 for decision
        sym /= np.sqrt(np.mean(np.abs(sym) ** 2) + 1e-12)
        # Hard decision on per-axis levels
        lv = _pam_levels(m_side)
        scale = 1.0 / np.sqrt(np.mean(lv ** 2) * 2)
        I = np.real(sym) / scale
        Q = np.imag(sym) / scale
        # nearest level indices
        I_idx = np.clip(np.argmin(np.abs(I[:, None] - lv[None, :]), axis=1), 0, m_side - 1)
        Q_idx = np.clip(np.argmin(np.abs(Q[:, None] - lv[None, :]), axis=1), 0, m_side - 1)
        # decode Gray indices back to binary before composing symbol index
        i_bin = _gray_decode(I_idx)
        q_bin = _gray_decode(Q_idx)
        ints = (i_bin << (int(np.log2(M)) // 2)) | q_bin
        k = int(np.log2(M))
        bits = (((ints[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
        return bits[:expected_bits] if expected_bits is not None else bits, info

    if mod in {"CPFSK", "GFSK"}:
        # Simple binary detector using instantaneous frequency sign averaged per symbol
        phase = np.unwrap(np.angle(x))
        finst = np.diff(phase) * (cfg.fs / (2 * np.pi))
        # Pad to align length
        finst = np.r_[finst, finst[-1]]
        # Average per symbol
        T = sps
        Nsym = len(finst) // T
        finst = finst[:Nsym * T].reshape(Nsym, T).mean(axis=1)
        bits = (finst > 0).astype(int)
        return bits[:expected_bits] if expected_bits is not None else bits, info

    if mod == "AM-DSB":
        # Envelope detector
        env = np.abs(x)
        # simple LPF
        b, a = signal.butter(4, 0.02)
        m_rec = signal.filtfilt(b, a, env)
        m_rec = m_rec - np.mean(m_rec)
        m_rec /= np.max(np.abs(m_rec)) + 1e-12
        return m_rec.astype(np.float32), info

    if mod == "AM-SSB":
        # Take real part of analytic baseband then LPF (demo)
        b, a = signal.butter(4, 0.05)
        m_rec = signal.filtfilt(b, a, np.real(x))
        m_rec /= np.max(np.abs(m_rec)) + 1e-12
        return m_rec.astype(np.float32), info

    if mod == "WBFM":
        # Differentiate unwrapped phase
        phase = np.unwrap(np.angle(x))
        dm = np.diff(phase)
        dm = np.r_[dm, dm[-1]]
        # LPF to audio band
        b, a = signal.butter(4, 0.05)
        m_rec = signal.filtfilt(b, a, dm)
        m_rec /= np.max(np.abs(m_rec)) + 1e-12
        return m_rec.astype(np.float32), info

    raise ValueError(f"Unsupported modulation for demodulate: {modulation}")
