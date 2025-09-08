import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple


def safe_mean(x): return float(np.mean(x))

def safe_std(x): return float(np.std(x))

def safe_skew(x):
    # Cast to float64 to avoid scipy placing float64 into float32 arrays (TypeError)
    x64 = np.asarray(x, dtype=np.float64)
    return float(stats.skew(x64, bias=False)) if np.std(x64) > 0 else 0.0

def safe_kurtosis(x):
    # Cast to float64 for numerical stability and to prevent dtype casting errors
    x64 = np.asarray(x, dtype=np.float64)
    return float(stats.kurtosis(x64, fisher=True, bias=False)) if np.std(x64) > 0 else 0.0


def circ_stats(angles):
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    mean_angle = np.arctan2(s, c)
    R = np.sqrt(c*c + s*s)
    return float(mean_angle), float(R)


def instantaneous_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    z = x.astype(np.complex64)
    amp = np.abs(z)
    phase = np.angle(z)
    unph = np.unwrap(phase)
    inst_freq = np.diff(unph) * fs / (2*np.pi)
    inst_freq = inst_freq if len(inst_freq) > 0 else np.array([0.0])

    feats = {
        "amp_mean": safe_mean(amp),
        "amp_std": safe_std(amp),
        "amp_skew": safe_skew(amp),
        "amp_kurt": safe_kurtosis(amp),

        "if_mean": safe_mean(inst_freq),
        "if_std": safe_std(inst_freq),
        "if_skew": safe_skew(inst_freq),
        "if_kurt": safe_kurtosis(inst_freq),
    }

    ph_mean, ph_R = circ_stats(phase)
    feats["phase_mean"] = ph_mean
    feats["phase_concentration"] = ph_R

    r = 0.5
    re = np.real(z); im = np.imag(z)
    close = (np.abs(re - np.round(re)) < r) & (np.abs(im - np.round(im)) < r)
    feats["constellation_density"] = float(np.mean(close))

    return feats


def spectral_features(x: np.ndarray, fs: float, nfft: int = 2048, n_bands: int = 8) -> Dict[str, float]:
    """
    Compute spectral features for complex IQ by using full complex FFT.
    We sort by frequency (using fftshift-equivalent) and split into n_bands across the full band.
    """
    N = len(x)
    nfft = min(nfft, N)
    X = np.fft.fft(x, n=nfft)
    P = np.abs(X) ** 2
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    # Sort by frequency (equivalent to fftshift ordering)
    order = np.argsort(freqs)
    freqs = freqs[order]
    P = P[order]
    # Normalize power to 1
    P /= np.sum(P) + 1e-12

    # Bandpowers across full spectrum
    bands = np.array_split(np.arange(len(P)), n_bands)
    feats = {}
    for i, idx in enumerate(bands):
        feats[f"bandpower_{i}"] = float(np.sum(P[idx]))

    # Spectral centroid and rolloff (85%) over full spectrum
    centroid = float(np.sum(freqs * P))
    cumsum = np.cumsum(P)
    roll_idx = np.searchsorted(cumsum, 0.85)
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])
    feats["spec_centroid"] = centroid
    feats["spec_rolloff85"] = rolloff

    # Effective bandwidth
    Ef = centroid
    Ef2 = float(np.sum((freqs ** 2) * P))
    bw_eff = np.sqrt(max(Ef2 - Ef ** 2, 0.0))
    feats["spec_bw_eff"] = float(bw_eff)

    return feats


def cumulants_complex(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(np.complex64)
    x2 = x**2
    m20 = np.mean(x2)
    m21 = np.mean(np.abs(x)**2)
    m40 = np.mean(x**4)
    m42 = np.mean(np.abs(x)**4)

    c20 = m20
    c21 = m21
    c40 = m40 - 3*(m20**2)
    c42 = m42 - np.abs(m20)**2 - 2*(m21**2)

    denom = (m21**2) + 1e-12
    feats = {
        "c20_re": float(np.real(c20)/denom),
        "c20_im": float(np.imag(c20)/denom),
        "c40_re": float(np.real(c40)/denom),
        "c40_im": float(np.imag(c40)/denom),
        "c42": float(np.real(c42)/denom),
    }
    return feats


def iq_stats(x: np.ndarray) -> Dict[str, float]:
    I = np.real(x); Q = np.imag(x)
    feats = {
        "I_mean": safe_mean(I), "I_std": safe_std(I), "I_skew": safe_skew(I), "I_kurt": safe_kurtosis(I),
        "Q_mean": safe_mean(Q), "Q_std": safe_std(Q), "Q_skew": safe_skew(Q), "Q_kurt": safe_kurtosis(Q),
        "IQ_corr": float(np.corrcoef(I, Q)[0,1]) if np.std(I)>0 and np.std(Q)>0 else 0.0,
        "IQ_power_ratio": float((np.mean(I**2)+1e-12)/(np.mean(Q**2)+1e-12)),
    }
    return feats


def extract_features_one(x: np.ndarray, fs: float) -> Dict[str, float]:
    feats = {}
    feats.update(instantaneous_features(x, fs))
    feats.update(spectral_features(x, fs))
    feats.update(cumulants_complex(x))
    feats.update(iq_stats(x))
    return feats


def extract_features_batch(X_complex: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    all_feats: List[Dict[str, float]] = []
    for x in X_complex:
        all_feats.append(extract_features_one(x, fs))
    keys = sorted(all_feats[0].keys())
    F = np.array([[f[k] for k in keys] for f in all_feats], dtype=np.float32)
    return F, keys
