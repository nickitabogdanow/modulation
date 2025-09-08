# Путеводитель для новичка (без радиотехники и ML)

Цель: за 30–60 минут понять смысл проекта, запустить код, увидеть результаты и осознать базовую математику.

---

## Что вы построите

- Сгенерируете маленький синтетический датасет радиосигналов.
- Обучите два классификатора: классический ML и DL (1D‑CNN по IQ).
- Сравните качество на разных уровнях шума (SNR) и посмотрите, что именно путается.

---

## 1) Мини‑ликбез (совсем просто)

- Сигнал описываем парой чисел на каждом шаге времени: `I` и `Q`. Вместе это «комплексное число» `x = I + jQ`. Отсюда легко получить амплитуду `|x|` и фазу `arg(x)`.
- Типы модуляций: где «лежит» информация:
  - PSK: в фазе; QAM: в амплитуде+фазе; FSK/FM: в частоте; AM: в огибающей.
- Канал портит сигнал: шум (SNR), вращение фазы (CFO), фазовый шум, сдвиги тайминга, несбаланс I/Q, мультипуть.

Чуть подробнее — в: `docs/theory/THEORY_BEGINNER.md` (разделы 2, 4, 5).

---

## 2) Быстрый запуск

Подготовка окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Сгенерировать маленький датасет (быстрее):

```bash
python dataset_builder.py --examples 100 --use-multipath --save-dir data --tag v1 --seed 42
```

Обучить ML‑модель на признаках:

```bash
python train_ml.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --seed 42
```

Обучить DL (1D‑CNN) на «сыром» IQ:

```bash
python train_dl.py --mode 1d --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --epochs 10 --batch-size 128 --seed 42
```

Сравнить ML vs DL и сохранить графики:

```bash
python evaluate_compare.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz
```

Куда смотреть:
- Логи и отчёты: `reports/`
- Чекпойнты DL: `checkpoints_1d/best.keras`
- ML‑модель: `artifacts/ml_best.joblib`

---

## 3) Куда смотреть глазами (три «вида»)

- Созвездие (IQ‑плоскость): PSK — точки на окружности; QAM — квадратная решётка; FSK/FM — размытые облака.
- Спектр (|FFT|): AM‑SSB — одна боковая полоса; AM‑DSB — две; QAM/PSK — компактная полоса вокруг нуля.
- Спектрограмма (STFT): у WBFM — широкая «шероховатая» картинка.

Скрипт «быстрого взгляда»: `python quick_check.py`.

---

## 4) Как связаны код и теория

- Генерация/искажения: `generators.py`, `channel.py` → теория: `theory/THEORY_BEGINNER.md` (§5, §17, §18)
- Признаки: `features.py` → теория: `theory/THEORY_BEGINNER.md` (§6, §21, §22)
- DL входы: `dl_prep.py` → теория: `theory/THEORY_BEGINNER.md` (§20), `IQ_BASICS.md`
- Модели: `models_dl.py` → `DL_GUIDE.md`
- Обучение и сравнение: `train_ml.py`, `train_dl.py`, `evaluate_compare.py` → `EVALUATION.md`

Полная карта соответствий — в `docs/CODE_TO_THEORY_MAP.md`.

---

## 5) Что чаще всего идёт не так

- Перепутали `Fs` (частоту дискретизации) — подписи частот неверны, спектры «странные».
- Не снят CFO — созвездие «крутится» и признаки фазы становятся шумными.
- Без стратификации по SNR — метрика красивая, а в шуме разваливается.
- Неправильная нормализация IQ — амплитудные признаки бесполезны.

См. «Частые ошибки» в `theory/THEORY_BEGINNER.md`.

---

## 6) Что дальше

- Пройдите туториалы в `docs/tutorials/`.
- Поиграйте ползунками (SNR, CFO) в `quick_check.py`.
- Попробуйте свои признаки в `features.py` и сравните ML vs DL.

Если возник вопрос — ищите раздел в `docs/` по названию модуля.