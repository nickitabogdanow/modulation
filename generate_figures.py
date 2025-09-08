import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from config_and_utils import SignalConfig, rrc_filter
from generators import synthesize_modulation
from channel import random_multipath_ir


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_constellations(cfg: SignalConfig, mods, path: str):
    cols = 5
    rows = int(np.ceil(len(mods) / cols))
    plt.figure(figsize=(3.2*cols, 3.0*rows))
    for i, m in enumerate(mods):
        x = synthesize_modulation(m, cfg)
        xi = np.real(x[::10])
        xq = np.imag(x[::10])
        plt.subplot(rows, cols, i+1)
        plt.scatter(xi, xq, s=4, alpha=0.5)
        plt.gca().set_aspect('equal', 'box')
        plt.title(m)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_iq_block_diagram(path: str):
    """Render a simple IQ downconversion block diagram with matplotlib."""
    plt.figure(figsize=(9, 3))
    ax = plt.gca()
    ax.axis('off')

    def box(x, y, w, h, text):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center')
        return (x, y, w, h)

    # Blocks
    b_rf   = box(0.1, 0.4, 0.18, 0.2, 'RF сигнал\ncos(2πf_ct+φ)')
    b_lo_c = box(0.4, 0.65, 0.18, 0.2, 'LO: cos(2πf_ct)')
    b_lo_s = box(0.4, 0.15, 0.18, 0.2, 'LO: sin(2πf_ct)')
    b_mult1= box(0.65, 0.65, 0.18, 0.2, '× (микшер)')
    b_mult2= box(0.65, 0.15, 0.18, 0.2, '× (микшер)')
    b_lpf1 = box(0.86, 0.65, 0.12, 0.2, 'LPF\n→ I')
    b_lpf2 = box(0.86, 0.15, 0.12, 0.2, 'LPF\n→ Q')

    # Arrows
    def arrow(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', lw=2))

    arrow(b_rf[0]+b_rf[2], 0.5, b_mult1[0], 0.75)
    arrow(b_rf[0]+b_rf[2], 0.5, b_mult2[0], 0.25)
    arrow(b_lo_c[0]+b_lo_c[2], 0.75, b_mult1[0], 0.75)
    arrow(b_lo_s[0]+b_lo_s[2], 0.25, b_mult2[0], 0.25)
    arrow(b_mult1[0]+b_mult1[2], 0.75, b_lpf1[0], 0.75)
    arrow(b_mult2[0]+b_mult2[2], 0.25, b_lpf2[0], 0.25)

    ax.text(0.12, 0.67, '↓ перенос в baseband', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_constellation_impairments(cfg: SignalConfig, path: str, mod: str = 'QPSK'):
    """Plot ideal constellation and effects of impairments: AWGN, CFO, phase noise, IQ imbalance."""
    from config_and_utils import add_awgn, apply_cfo, apply_phase_noise, apply_iq_imbalance
    x = synthesize_modulation(mod, cfg)
    idx = slice(0, len(x), 10)

    panels = []
    titles = []

    # Ideal
    panels.append(x[idx])
    titles.append('Идеал')

    # +AWGN
    panels.append(add_awgn(x, snr_db=10)[idx])
    titles.append('+AWGN (10 дБ)')

    # +CFO
    panels.append(apply_cfo(x, cfg.fs, cfo_ppm=15000)[idx])
    titles.append('+CFO (15k ppm)')

    # +Phase noise
    panels.append(apply_phase_noise(x, std_per_sample=0.01)[idx])
    titles.append('+Фазовый шум')

    # +IQ imbalance
    panels.append(apply_iq_imbalance(x, amp_db=0.8, phase_deg=5)[idx])
    titles.append('+I/Q несбаланс')

    cols = 5
    rows = 1
    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i, (p, t) in enumerate(zip(panels, titles)):
        plt.subplot(rows, cols, i+1)
        plt.scatter(np.real(p), np.imag(p), s=4, alpha=0.6)
        plt.gca().set_aspect('equal', 'box')
        plt.title(t)
        plt.grid(True, alpha=0.3)
    plt.suptitle(f'Созвездие {mod}: влияние искажений', y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_rrc_impulse(cfg: SignalConfig, path: str):
    h = rrc_filter(num_taps=8*cfg.sps+1, beta=0.35, sps=cfg.sps)
    t = np.arange(len(h)) - (len(h)-1)/2
    plt.figure(figsize=(7,3))
    plt.stem(t, h)
    plt.title('Импульсная характеристика RRC (beta=0.35)')
    plt.xlabel('Отсчёт')
    plt.ylabel('Амплитуда')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_multipath_ir(path: str):
    h = random_multipath_ir(max_taps=5, max_delay=15, exp_decay=0.7)
    t = np.arange(len(h))
    plt.figure(figsize=(7,3))
    plt.stem(t, np.abs(h))
    plt.title('Пример импульсной характеристики мультипутевого канала (модуль)')
    plt.xlabel('Задержка (отсчёты)')
    plt.ylabel('|h[n]|')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_spectrum_examples(cfg: SignalConfig, mods, path: str):
    cols = 3
    rows = int(np.ceil(len(mods) / cols))
    plt.figure(figsize=(4.2*cols, 3.2*rows))
    for i, m in enumerate(mods):
        x = synthesize_modulation(m, cfg)
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0/cfg.fs))
        plt.subplot(rows, cols, i+1)
        plt.plot(f/1e3, 20*np.log10(np.abs(X)+1e-12))
        plt.title(m)
        plt.xlabel('Частота (кГц)')
        plt.ylabel('Ампл. (дБ)')
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_spectrogram_wbfm(cfg: SignalConfig, path: str):
    # генерируем WBFM и считаем спектрограмму
    x = synthesize_modulation('WBFM', cfg)
    f, t, Sxx = signal.spectrogram(x, fs=cfg.fs, nperseg=256, noverlap=128, scaling='spectrum', mode='magnitude')
    plt.figure(figsize=(7,3))
    plt.pcolormesh(t, f/1e3, 20*np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    plt.title('Спектрограмма WBFM')
    plt.ylabel('Частота (кГц)')
    plt.xlabel('Время (с)')
    plt.colorbar(label='дБ')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_psd_examples(cfg: SignalConfig, mods, path: str):
    cols = 3
    rows = int(np.ceil(len(mods) / cols))
    plt.figure(figsize=(4.2*cols, 3.2*rows))
    for i, m in enumerate(mods):
        x = synthesize_modulation(m, cfg)
        f, Pxx = signal.welch(x, fs=cfg.fs, nperseg=512)
        plt.subplot(rows, cols, i+1)
        plt.semilogy(f/1e3, Pxx + 1e-18)
        plt.title(m)
        plt.xlabel('Частота (кГц)')
        plt.ylabel('PSD')
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    cfg = SignalConfig()
    out_dir = 'img'
    ensure_dir(out_dir)

    # 1) Созвездия
    const_path = os.path.join(out_dir, 'constellations.png')
    save_constellations(cfg, [
        'BPSK', 'QPSK', 'PSK8',
        'QAM16', 'QAM64',
        'CPFSK', 'GFSK',
        'AM-DSB', 'AM-SSB', 'WBFM'], const_path)

    # 2) RRC импульсный отклик
    rrc_path = os.path.join(out_dir, 'rrc_impulse.png')
    save_rrc_impulse(cfg, rrc_path)

    # 3) Мультипуть IR
    mp_path = os.path.join(out_dir, 'multipath_ir.png')
    save_multipath_ir(mp_path)

    # 4) Спектры разных модуляций
    spec_path = os.path.join(out_dir, 'spectrum_examples.png')
    save_spectrum_examples(cfg, ['BPSK','QAM16','CPFSK','GFSK','AM-SSB','WBFM'], spec_path)

    # 5) Спектрограмма WBFM
    specgram_path = os.path.join(out_dir, 'spectrogram_wbfm.png')
    save_spectrogram_wbfm(cfg, specgram_path)

    # 6) PSD (Welch) для примеров
    psd_path = os.path.join(out_dir, 'psd_examples.png')
    save_psd_examples(cfg, ['BPSK','QAM16','GFSK','AM-DSB','AM-SSB','WBFM'], psd_path)

    # 7) Блок-схема IQ
    iq_block_path = os.path.join(out_dir, 'iq_block_diagram.png')
    save_iq_block_diagram(iq_block_path)

    # 8) Созвездие с искажениями
    const_imp_path = os.path.join(out_dir, 'constellation_impairments.png')
    save_constellation_impairments(cfg, const_imp_path, mod='QPSK')

    print('Figures saved to:', out_dir)


if __name__ == '__main__':
    main()
