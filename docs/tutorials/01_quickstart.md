# Туториал 01: Быстрый старт с картинками

В этом туториале вы за 10–20 минут получите первый результат и визуально увидите данные и модели.

---

## 1) Подготовка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Сгенерируйте мини-датасет

```bash
python dataset_builder.py --examples 100 --use-multipath --save-dir data --tag v1 --seed 42
```

Файлы появятся в `data/`: `train_v1.npz`, `val_v1.npz`, `test_v1.npz`.

---

## 3) Быстрый взгляд на сигналы

```bash
python quick_check.py
```

Смотрите:
- Созвездия: `img/constellations.png`, `img/constellation_impairments.png`.
- Спектры: `img/spectrum_examples.png`.
- Спектрограмму WBFM: `img/spectrogram_wbfm.png`.

---

## 4) Обучите ML и DL

```bash
python train_ml.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --seed 42
python train_dl.py --mode 1d --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --epochs 10 --batch-size 128 --seed 42
```

Результаты ищите в `reports/` и `artifacts/`, `checkpoints_1d/`.

---

## 5) Сравните ML и DL

```bash
python evaluate_compare.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz
```

Получите графики Accuracy/F1 vs SNR и матрицы ошибок в `reports/`.

---

## 6) Дальше

- Измените `--examples`, `--epochs`, `--seed` и повторите.
- Поищите «трудные» SNR — где модели начинают путаться.
- Зайдите в `docs/BEGINNER_TOUR.md` и `docs/PIPELINE.md` для продолжения.