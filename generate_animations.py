import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config_and_utils import SignalConfig, apply_cfo, apply_phase_noise, add_awgn, apply_iq_imbalance
from generators import synthesize_modulation


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def animate_cfo(cfg: SignalConfig, mod: str, out_path: str, frames: int = 60, step_ppm: float = 300.0):
    x = synthesize_modulation(mod, cfg)
    idx = slice(0, len(x), 10)

    fig, ax = plt.subplots(figsize=(4,4))
    sc = ax.scatter(np.real(x[idx]), np.imag(x[idx]), s=5, alpha=0.6)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f"CFO rotation: 0 ppm")

    def init():
        sc.set_offsets(np.c_[np.real(x[idx]), np.imag(x[idx])])
        title.set_text("CFO rotation: 0 ppm")
        return sc, title

    def update(f):
        cfo_ppm = f * step_ppm
        y = apply_cfo(x, cfg.fs, cfo_ppm=cfo_ppm)
        pts = np.c_[np.real(y[idx]), np.imag(y[idx])]
        sc.set_offsets(pts)
        title.set_text(f"CFO rotation: {int(cfo_ppm)} ppm")
        return sc, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=80, blit=True)
    anim.save(out_path, writer='pillow', fps=12)
    plt.close(fig)


def animate_awgn(cfg: SignalConfig, mod: str, out_path: str, frames: int = 40, start_snr_db: float = 30.0, step_db: float = -1.0):
    """Animate constellation as SNR decreases (AWGN grows)."""
    x = synthesize_modulation(mod, cfg)
    idx = slice(0, len(x), 10)

    fig, ax = plt.subplots(figsize=(4,4))
    sc = ax.scatter(np.real(x[idx]), np.imag(x[idx]), s=5, alpha=0.6)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f"AWGN: SNR={start_snr_db:.1f} dB")

    def init():
        y = add_awgn(x, snr_db=start_snr_db)
        sc.set_offsets(np.c_[np.real(y[idx]), np.imag(y[idx])])
        title.set_text(f"AWGN: SNR={start_snr_db:.1f} dB")
        return sc, title

    def update(f):
        snr = start_snr_db + f * step_db
        y = add_awgn(x, snr_db=snr)
        sc.set_offsets(np.c_[np.real(y[idx]), np.imag(y[idx])])
        title.set_text(f"AWGN: SNR={snr:.1f} dB")
        return sc, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=100, blit=True)
    anim.save(out_path, writer='pillow', fps=10)
    plt.close(fig)


def animate_iq_imbalance(cfg: SignalConfig, mod: str, out_path: str, frames: int = 40, amp_db_range=(0.0, 2.0), phase_deg_range=(0.0, 10.0)):
    """Animate constellation under growing IQ imbalance."""
    x = synthesize_modulation(mod, cfg)
    idx = slice(0, len(x), 10)

    fig, ax = plt.subplots(figsize=(4,4))
    sc = ax.scatter(np.real(x[idx]), np.imag(x[idx]), s=5, alpha=0.6)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.3)
    title = ax.set_title("IQ imbalance: amp=0.0 dB, phase=0.0°")

    def init():
        sc.set_offsets(np.c_[np.real(x[idx]), np.imag(x[idx])])
        return sc, title

    def update(f):
        amp = amp_db_range[0] + (amp_db_range[1] - amp_db_range[0]) * (f / (frames - 1))
        ph = phase_deg_range[0] + (phase_deg_range[1] - phase_deg_range[0]) * (f / (frames - 1))
        y = apply_iq_imbalance(x, amp_db=amp, phase_deg=ph)
        sc.set_offsets(np.c_[np.real(y[idx]), np.imag(y[idx])])
        title.set_text(f"IQ imbalance: amp={amp:.1f} dB, phase={ph:.1f}°")
        return sc, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=100, blit=True)
    anim.save(out_path, writer='pillow', fps=10)
    plt.close(fig)

def animate_phase_noise(cfg: SignalConfig, mod: str, out_path: str, frames: int = 60, std_per_sample: float = 0.002):
    x = synthesize_modulation(mod, cfg)
    idx = slice(0, len(x), 10)

    fig, ax = plt.subplots(figsize=(4,4))
    sc = ax.scatter(np.real(x[idx]), np.imag(x[idx]), s=5, alpha=0.6)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.3)
    title = ax.set_title("Phase noise: t=0")

    y = x.copy()

    def init():
        sc.set_offsets(np.c_[np.real(x[idx]), np.imag(x[idx])])
        title.set_text("Phase noise: t=0")
        return sc, title

    def update(f):
        # accumulate phase noise progressively to simulate random walk over time
        nonlocal y
        y = apply_phase_noise(y, std_per_sample=std_per_sample)
        pts = np.c_[np.real(y[idx]), np.imag(y[idx])]
        sc.set_offsets(pts)
        title.set_text(f"Phase noise: step {f+1}")
        return sc, title

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=80, blit=True)
    anim.save(out_path, writer='pillow', fps=12)
    plt.close(fig)


def main():
    cfg = SignalConfig()
    out_dir = 'img'
    ensure_dir(out_dir)

    cfo_gif = os.path.join(out_dir, 'cfo_animation.gif')
    animate_cfo(cfg, mod='QPSK', out_path=cfo_gif, frames=60, step_ppm=400.0)

    pn_gif = os.path.join(out_dir, 'phase_noise_animation.gif')
    animate_phase_noise(cfg, mod='QPSK', out_path=pn_gif, frames=60, std_per_sample=0.004)

    # AWGN animation (decreasing SNR)
    awgn_gif = os.path.join(out_dir, 'awgn_animation.gif')
    animate_awgn(cfg, mod='QPSK', out_path=awgn_gif, frames=40, start_snr_db=30.0, step_db=-1.0)

    # IQ imbalance animation
    iqimb_gif = os.path.join(out_dir, 'iq_imbalance_animation.gif')
    animate_iq_imbalance(cfg, mod='QPSK', out_path=iqimb_gif, frames=40, amp_db_range=(0.0, 3.0), phase_deg_range=(0.0, 12.0))

    print('Animations saved to:', out_dir)


if __name__ == '__main__':
    main()
