import numpy as np
import matplotlib.pyplot as plt

from config_and_utils import SignalConfig
from generators import generate_example

from visualization import (
    plot_time_iq, plot_constellation, plot_spectrum
)

if __name__ == "__main__":
    cfg = SignalConfig()

    mods_to_show = ["BPSK", "QPSK", "QAM16", "CPFSK", "GFSK", "AM-DSB", "WBFM"]
    snr_db = 10

    for m in mods_to_show:
        x, meta = generate_example(m, snr_db=snr_db, cfg=cfg, impair=True)
        print(f"Example: {m}, SNR={snr_db} dB")
        plot_time_iq(x, fs=cfg.fs, n=1500, title=f"{m} — time (I/Q)")
        plot_constellation(x, step=20, title=f"{m} — constellation")
        plot_spectrum(x, fs=cfg.fs, title=f"{m} — spectrum")
