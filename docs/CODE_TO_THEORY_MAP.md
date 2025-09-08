# Карта соответствий: код ↔ теория

Помогает быстро найти, какая теория покрывает каждый модуль кода.

- config_and_utils.py
  - Конфигурация и базовые функции: теория не требуется
  - dB/мощности: THEORY_BEGINNER.md (§13)
  - Фильтры RRC/Gaussian: THEORY_BEGINNER.md (§15)
- generators.py
  - PSK/QAM/FSK/AM/FM: THEORY_BEGINNER.md (§16), MODULATIONS_GUIDE.md
- channel.py
  - Импейрменты (AWGN, CFO, фазовый шум, тайминг, I/Q, мультипуть): THEORY_BEGINNER.md (§5, §17, §18)
- dataset_builder.py
  - Стратификация по классам и SNR: DATASETS.md (§3), THEORY_BEGINNER.md (§26)
- visualization.py
  - Констелляции/спектры/спектрограммы: THEORY_BEGINNER.md (§20, §21)
- features.py, features_run.py
  - Мгновенные признаки: THEORY_BEGINNER.md (§6)
  - Спектральные/PSD: THEORY_BEGINNER.md (§21)
  - Кумулянты: THEORY_BEGINNER.md (§22)
- dl_prep.py
  - Стандартизация, STFT: THEORY_BEGINNER.md (§20)
- models_dl.py
  - 1D/2D CNN: DL_GUIDE.md
- train_ml.py, train_dl.py
  - Процедуры обучения/валидации: EVALUATION.md
- evaluate_compare.py, error_analysis.py
  - Метрики/матрицы ошибок/по SNR: EVALUATION.md, THEORY_BEGINNER.md (§26)

См. также: BEGINNER_TOUR.md — пошаговый вход для новичка.