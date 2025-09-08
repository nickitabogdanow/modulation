import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plot_time_iq(x: np.ndarray, fs: float, n: int = 1000, title: str = "Time-domain IQ"):
    t = np.arange(min(n, len(x))) / fs
    plt.figure(figsize=(10, 3))
    plt.plot(t, np.real(x[:n]), label="I")
    plt.plot(t, np.imag(x[:n]), label="Q", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_constellation(x: np.ndarray, step: int = 10, title: str = "Constellation"):
    xi = np.real(x[::step])
    xq = np.imag(x[::step])
    plt.figure(figsize=(4, 4))
    plt.scatter(xi, xq, s=3, alpha=0.5)
    plt.gca().set_aspect("equal", "box")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(x: np.ndarray, fs: float, title: str = "Spectrum"):
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0/fs))
    plt.figure(figsize=(8, 3))
    plt.plot(f/1e3, 20*np.log10(np.abs(X)+1e-12))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(x: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 128, title: str = "Spectrogram"):
    f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="spectrum", mode="magnitude")
    plt.figure(figsize=(8, 3))
    plt.pcolormesh(t, f/1e3, 20*np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()
