import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate


def ber(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    n = min(len(bits_true), len(bits_pred))
    if n == 0:
        return 1.0
    return float(np.mean(bits_true[:n] != bits_pred[:n]))


def run_roundtrip(mod: str, snr_db: float, n_bits: int = 4096) -> Tuple[float, int]:
    cfg = SignalConfig()
    # Ensure n_bits multiple of k for symbol mapping
    modU = mod.upper()
    k_map = {"BPSK": 1, "QPSK": 2, "PSK8": 3, "QAM16": 4, "QAM64": 6, "CPFSK": 1, "GFSK": 1}
    k = k_map.get(modU, 1)
    n_bits = (n_bits // k) * k
    bits = np.random.randint(0, 2, size=n_bits, dtype=int)

    x, _ = modulate(bits, modU, cfg)
    y = add_awgn(x, snr_db=snr_db)
    bits_hat, _ = demodulate(y, modU, cfg, expected_bits=n_bits)
    return ber(bits, bits_hat), n_bits


def save_ber_plot(results: Dict[str, List[float]], snrs: List[int], out_path: str) -> None:
    plt.figure(figsize=(7,4))
    for mod, bers in results.items():
        plt.semilogy(snrs, bers, marker='o', label=mod)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Round-trip BER vs SNR (educational modem)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_constellation_snapshot(mod: str, snr_db: float, out_path: str, n_bits: int = 2048) -> None:
    cfg = SignalConfig()
    # make bits multiple of k handled in run_roundtrip
    k_map = {"BPSK": 1, "QPSK": 2, "PSK8": 3, "QAM16": 4, "QAM64": 6, "CPFSK": 1, "GFSK": 1}
    k = k_map.get(mod.upper(), 1)
    n_bits = (n_bits // k) * k
    bits = np.random.randint(0, 2, size=n_bits, dtype=int)
    x, _ = modulate(bits, mod, cfg)
    y = add_awgn(x, snr_db=snr_db)
    # Thin for plotting
    yi = y[::10]
    plt.figure(figsize=(4,4))
    plt.scatter(np.real(yi), np.imag(yi), s=5, alpha=0.6)
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True, alpha=0.3)
    plt.title(f"{mod} @ SNR={snr_db} dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    mods = ["BPSK", "QPSK", "QAM16"]
    snrs = [0, 6, 12, 18, 24]
    print("Quick modem demo (round-trip BER)")
    results: Dict[str, List[float]] = {}
    for mod in mods:
        print(f"\n{mod}:")
        bers: List[float] = []
        for snr in snrs:
            b, n = run_roundtrip(mod, snr_db=snr, n_bits=8192)
            bers.append(b)
            print(f"  SNR={snr:>2} dB | BER={b:.4f} (n={n})")
        results[mod] = bers

    # Save BER plot
    os.makedirs('img', exist_ok=True)
    save_ber_plot(results, snrs, out_path='img/ber_curves.png')

    # Save constellation snapshots at SNR=12 dB
    for mod in mods:
        out = f"img/constellation_{mod}_12dB.png"
        save_constellation_snapshot(mod, snr_db=12, out_path=out)

    # FM/AM demonstration: message recovery length
    cfg = SignalConfig()
    t = np.arange(cfg.window_size) / cfg.fs
    message = 0.7 * np.sin(2 * np.pi * 1000.0 * t)

    for mod in ["AM-DSB", "WBFM"]:
        x, _ = modulate(np.array([], dtype=int), mod, cfg, message=message)
        y = add_awgn(x, snr_db=20)
        m_rec, _ = demodulate(y, mod, cfg)
        print(f"\n{mod}: recovered message len={len(m_rec)} samples, std={np.std(m_rec):.3f}")


if __name__ == "__main__":
    main()
