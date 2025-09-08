# Путеводитель по модуляциям

Навигация: [Документация](../index.md) › Теория › Этот документ

---

В этом документе подробно описаны поддерживаемые модуляции, их математические модели, созвездия/спектры, особенности канала и на интуитивном уровне — как их демодулировать. Для практики см. `modem.py` и пример `quick_modem_demo.py`.

См. изображения:
- Созвездия (сводно): `../img/constellations.png`
- Примеры спектров: `../img/spectrum_examples.png`
- Спектрограмма WBFM: `../img/spectrogram_wbfm.png`

---

## 1) PSK (BPSK, QPSK, 8PSK)

- Идея: информация в фазе. Амплитуда почти постоянна; точки созвездия на окружности.
- Модель (baseband, по символам): `s_k = A · exp(j · 2π m_k/M)` (часто используется Gray-кодировка индексов `m_k`).
- Передача: символы `s_k` фильтруют импульсом (например, RRC) и апсемплируют до `Fs`.
- Демодуляция (интуиция):
  - Снять фазу: `φ̂ = atan2(Q, I)`; разбить круг на `M` секторов (решающие области).
  - Для робастности к общей фазе оценить и убрать `φ0` (например, для M-PSK: `φ0 ≈ (1/M) · angle(mean(r^M))`).
- Особенности канала: CFO → вращение созвездия; фазовый шум → размывание по углу.

BPSK:
- Двоичная фаза {0, π}. Решение по знаку I (или по φ в {−π/2, π/2}).

QPSK:
- Четыре фазовых состояния (0°, 90°, 180°, 270°). Решение по знаку I и Q (с учётом поворота).

8PSK:
- Восемь секторов; чувствительнее к фазовым ошибкам (меньше угловой зазор).

Подсказки к изображениям:
- Смотрите `../img/constellations.png`: окружности с равномерным угловым шагом — это PSK. Чем больше M, тем ближе точки по углу.
- В `../img/constellation_impairments.png` видно, как CFO вращает все точки, а фазовый шум «смазывает» их по окружности.

Пример (Python):

```python
import numpy as np
from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate

cfg = SignalConfig()
bits = np.random.randint(0, 2, size=4096)
x, _ = modulate(bits, 'QPSK', cfg)
y = add_awgn(x, snr_db=12)
bits_hat, _ = demodulate(y, 'QPSK', cfg, expected_bits=len(bits))
ber = np.mean(bits != bits_hat[:len(bits)])
print('QPSK BER @12dB =', ber)
```

---

## 2) QAM (16/64)

- Идея: информация в амплитуде и фазе одновременно. Созвездия — прямоугольные решётки по I и Q.
- Модель (по символам): `s_k = I_m + j Q_m` (с уровнями, зависящими от порядка модуляции), нормируют на среднюю мощность.
- Демодуляция (интуиция):
  - Оценить общий фазовый угол (часто ≈0 в baseband), нормировать масштаб и квантизовать I и Q по ближайшим уровням.
  - Gray-кодировка уменьшает битовые ошибки при близких точках.
- Особенности канала: требовательна к SNR, чувствительна к I/Q-несбалансу и нелинейностям.

Подсказки к изображениям:
- В `../img/constellations.png` QAM — прямоугольная решётка (4×4 для 16‑QAM, 8×8 для 64‑QAM). Чем выше порядок, тем меньше расстояние между соседями.
- В `../img/constellation_impairments.png` при I/Q‑несбалансе решётка становится эллиптической и наклонённой, что смещает границы решений.
- Снимки с шумом: `../img/constellation_QAM16_12dB.png` — видно, как кластеры распухают с SNR≈12 дБ.

Пример (Python):

```python
import numpy as np
from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate

cfg = SignalConfig()
bits = np.random.randint(0, 2, size=6*1024)  # кратно 4 для 16-QAM
x, _ = modulate(bits, 'QAM16', cfg)
y = add_awgn(x, snr_db=18)
bits_hat, _ = demodulate(y, 'QAM16', cfg, expected_bits=len(bits))
print('QAM16 BER @18dB =', np.mean(bits != bits_hat[:len(bits)]))
```

---

## 3) CPFSK / GFSK

- Идея: информация в частоте (и фазе, как интеграле частоты). CPFSK — непрерывность фазы; GFSK — предварительная гауссова фильтрация символов.
- Модель (binary FSK в baseband):
  - `φ[n] = φ[n-1] + 2π · (f0 + Δf · a_k) / Fs`, где `a_k ∈ {−1, +1}` на интервале символа.
  - `x[n] = exp(j · φ[n])`.
- Демодуляция (интуиция):
  - Частотный детектор: `f_inst ≈ Δφ · Fs / (2π)`, усреднить по интервалу символа и порог по знаку.
- Особенности канала: хорошо различима по распределению мгновенной частоты; спектр — более «размазанный».

Подсказки к изображениям:
- В `../img/constellations.png` FSK/GFSK выглядят менее «узловыми» на плоскости IQ — это нормально, т.к. информация закодирована в частоте, а не в фиксированных точках.
- Смотрите спектры в `../img/spectrum_examples.png` — у FSK/GFSK полоса шире, боковые лепестки сглажены (особенно у GFSK).

Пример (Python):

```python
import numpy as np
from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate

cfg = SignalConfig()
bits = np.random.randint(0, 2, size=4096)
x, _ = modulate(bits, 'GFSK', cfg)
y = add_awgn(x, snr_db=10)
bits_hat, _ = demodulate(y, 'GFSK', cfg, expected_bits=len(bits))
print('GFSK BER @10dB =', np.mean(bits != bits_hat[:len(bits)]))
```

---

## 4) AM (DSB/SSB)

- AM-DSB (двухполосная): `x(t) = (1 + m · m(t))`, где `m(t)` — сообщение (нормированное), в baseband — комплексная огибающая с реальной модуляцией амплитуды.
  - Демодуляция: `|x(t)|` (обводка/энвелоп) с НЧ-фильтром.
- AM-SSB (однополосная): формируют аналитический сигнал сообщения и смещают спектр; в baseband-симуляциях можно использовать фильтрацию знака Гильберта, чтобы удалить одну полосу.
  - Демодуляция: преобразовать обратно через согласованные операции (грубо — взять действительную часть после обратного сдвига; в baseband — восстановление сообщения фильтром).
- Особенности: очень наглядна в спектре (одна/две боковые полосы).

Подсказки к изображениям:
- Сравните `../img/spectrum_examples.png` для AM‑DSB vs AM‑SSB: у SSB одна боковая полоса подавлена.
- В созвездиях AM‑модуляции не имеют стабильных «узлов» как PSK/QAM — ориентируйтесь на спектр/PSD.

Пример (Python):

```python
import numpy as np
from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate

cfg = SignalConfig()
t = np.arange(cfg.window_size) / cfg.fs
message = 0.6*np.sin(2*np.pi*1000*t)
x, _ = modulate(np.array([], dtype=int), 'AM-DSB', cfg, message=message)
y = add_awgn(x, snr_db=20)
m_rec, _ = demodulate(y, 'AM-DSB', cfg)
print('AM-DSB rec std:', np.std(m_rec))
```

---

## 5) WBFM (широкополосная ЧМ)

- Модель (baseband): `x[n] = exp(j · k_f · cumsum(m[n]) / Fs)`, где `m[n]` — сообщение (например, аудио), `k_f` — девиация.
- Демодуляция: `f_inst[n] = unwrap(angle(x))[n] − unwrap(angle(x))[n−1]`, затем НЧ-фильтр/масштаб к аудио.
- Особенности: очень характерная спектрограмма (см. `img/spectrogram_wbfm.png`).

Подсказки к изображениям:
- `../img/spectrogram_wbfm.png`: широкая полоса и «шероховатая» структура по времени — признак частотной модуляции.
- В анимациях `IQ_BASICS.md` видно, как фазовый шум и CFO влияют на любую угловую модуляцию.

Пример (Python):

```python
import numpy as np
from config_and_utils import SignalConfig, add_awgn
from modem import modulate, demodulate

cfg = SignalConfig()
t = np.arange(cfg.window_size) / cfg.fs
message = 0.5*np.sin(2*np.pi*1200*t)
x, _ = modulate(np.array([], dtype=int), 'WBFM', cfg, message=message)
y = add_awgn(x, snr_db=20)
m_rec, _ = demodulate(y, 'WBFM', cfg)
print('WBFM rec std:', np.std(m_rec))
```

---

## Практические советы по демодуляции в канале

- Для PSK: оцените и компенсируйте общий фазовый сдвиг (CFO), затем делайте решение по секторам/решётке.
- Для QAM: нормируйте масштаб по мощности; можно оценить фазу по ближайшим решениям (итеративно).
- Для FSK/FM: используйте мгновенную частоту с усреднением по символу.
- Для AM: используйте огибающую (|·|) + НЧ-фильтрацию.

См. реализацию «учебных» алгоритмов в `modem.py`.

---

## Приложение: как читать BER‑кривые

- В `img/ber_curves.png` показана ошибка по битам (BER) vs SNR для BPSK, QPSK, QAM16.
- Ожидаемо BPSK наиболее устойчив к шуму (наибольшее расстояние между решениями), затем QPSK, затем QAM16.
- На высоких порядках (QAM16) из‑за меньших расстояний между соседями BER убывает медленнее. В реальных каналах добавляются CFO/фазовый шум/несбаланс I/Q, что ещё сильнее ухудшает QAM.
