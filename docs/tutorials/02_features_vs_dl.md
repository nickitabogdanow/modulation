# Туториал 02: Признаки vs DL — когда и что лучше

Цель: понять, почему классический ML и DL дают разные результаты и как интерпретировать ошибки.

---

## 1) Подготовьте данные

```bash
python dataset_builder.py --examples 200 --use-multipath --save-dir data --tag v1 --seed 42
```

---

## 2) Обучите обе ветки

```bash
python train_ml.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --seed 42
python train_dl.py --mode 1d --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz --epochs 30 --batch-size 128 --seed 42
```

---

## 3) Сравните по SNR

```bash
python evaluate_compare.py --train data/train_v1.npz --val data/val_v1.npz --test data/test_v1.npz
```

Посмотрите на кривые Accuracy/F1 vs SNR в `reports/`.

- Если ML > DL на высоких SNR: признаки дают сильные инварианты (кумулянты/мгновенная частота).
- Если DL > ML на низких SNR: CNN лучше извлекает слабые паттерны без ручных признаков.

---

## 4) Разбор ошибок

- Матрицы ошибок (по выбранным SNR) в `reports/`.
- Сравните пары, где путаница максимальна: QAM16↔QAM64, PSK8↔QAM16, GFSK↔WBFM, AM-DSB↔AM-SSB.
- Откройте соответствующие созвездия/спектры и попытайтесь объяснить, что именно схлопывается.

---

## 5) Эксперименты для закрепления

- Увеличьте окно `window_size` — как меняются кумулянты/метрики?
- Уберите мультипуть (`--no-multipath`) — кто выиграл и почему?
- Добавьте аугментации в DL (`dl_prep.py`) — стабилизировалась ли оценка на низких SNR?

Подсказки: теория в `docs/theory/THEORY_BEGINNER.md` (§6, §20–22, §26), `docs/MATH_PRIMER.md`.