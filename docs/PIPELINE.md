# Сквозной пайплайн: от синтетики до сравнения моделей

Эта заметка связывает все шаги в одну историю и подсказывает, где смотреть.

---

## 1) Генерация данных

- Что: синтетические окна сигнала разных модуляций и уровней SNR.
- Где в коде: `dataset_builder.py` → `generators.py` + `channel.py`.
- Что получить: `data/train_*.npz`, `val_*.npz`, `test_*.npz` с `(X, y, snr_db, mod_labels)`.

Команда:
```bash
python dataset_builder.py --examples 200 --use-multipath --save-dir data --tag v1 --seed 42
```

---

## 2) Два подхода к классификации

- ML на признаках: `features.py` → `train_ml.py`.
  - Плюсы: быстрее, интерпретируемее (важности признаков).
- DL на «сыром» IQ: `dl_prep.py` + `models_dl.py` → `train_dl.py`.
  - Плюсы: лучше обобщает, меньше ручного тюнинга признаков.

Команды:
```bash
python train_ml.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --seed 42
python train_dl.py --mode 1d --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --epochs 30 --batch-size 128 --seed 42
```

---

## 3) Оценка и сравнение

- Что: метрики Accuracy/F1, матрицы ошибок, кривые по SNR.
- Где: `evaluate_compare.py`, `error_analysis.py`.
- Выводы: у каких классов путаница; на каких SNR модели «сыпятся».

Команда:
```bash
python evaluate_compare.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz
```

Результаты: `reports/` (графики, матрицы, отчеты).

---

## 4) Как копать дальше

- Сложнее канал: Rayleigh/Rician, разные профили импульсных характеристик.
- Больше признаков: циклостационарные, энтропийные, автокорреляции.
- Более мощные модели: ResNet1D/TCN/Conformer (IQ), EfficientNet (спектрограммы).
- Анализ ошибок: выгружайте «трудные» примеры и смотрите их созвездия/спектры.