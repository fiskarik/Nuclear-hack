# Adaptive wavefront: обученная версия

**Цель Pearson ≥0.8 и SSIM ≥0.8 пока не достигнута.**

На одинаковом test из 32 четвёрок, один общий ДОЭ и прежняя сетка Френеля:

| Метод | Pearson | SSIM |
|---|---:|---:|
| Прежняя чистая нейросеть | 0.57364 | 0.41852 |
| Новая adaptive нейросеть, один forward pass | **0.60747** | **0.43629** |
| GS, 80 итераций | 0.65675 | 0.49291 |

Обучены wavefront U-Net (20 эпох) и adaptive сеть с 12 оптическими слоями
(16 эпох). Новая сеть качественнее прежней и тяжелее: около 0.152 против 0.033
с/четвёрку в пакетном CPU-измерении. Эпоха выбрана по validation loss.

- [PDF](reports/report.pdf), [подробный отчёт](reports/report.md),
  [CSV](reports/measurements.csv), [JSON](reports/summary.json).
- [Веса adaptive](best.pt), [обучение](training.json),
  [одинаковый тест](benchmark_test.json), [веса wavefront](wavefront_parent.pt).
- [ДОЭ и реконструкции чистой сети](user_neural/),
  [дополнительные 3000 шагов общей фазы](user_refined_3000/).
- [Подготовленный приватный Kaggle notebook](kaggle/notebook/adaptive_gpu.ipynb),
  [инструкции запуска](../../../docs/adaptive-training.md),
  [полный архив](submission.zip).

На выбранной четвёрке adaptive: **0.36223/0.29678**; с 3000 шагами общей
фазы: **0.59091/0.45464**. Последний результат хуже v2 QAT + 3000, поэтому
[прежний гибрид](../v2/README.md) сохранён. Его test + 600 остаётся
0.67111/0.60637; это отдельный итеративный метод.

В отчёте приведены и неудачные попытки. Опыт с более частой сеткой камеры
меняет физический шаг и поле зрения и не включён в сравнение моделей выше.
Первые 4 validation-четвёрки использованы для исследования; это не новый test.
Неудачные опыты не доказывают невозможности порога 0.8.

26 тестов пройдены. Все 6 ячеек Kaggle notebook выполнены локально в quick-режиме.
Полный GPU job (24 слоя, 80 эпох, 2048 train-четвёрок) подготовлен, но **не отправлен**:
нет привязанного секрета Kaggle и применённого доступа к доменам Kaggle.
Результат 0.8 для этого будущего запуска не обещается.

```bash
python -m doe predict --checkpoint deliverables/quality_research/v3/best.pt \
  --images examples/noise_quartet/plane_1.png examples/noise_quartet/plane_2.png \
           examples/noise_quartet/plane_3.png examples/noise_quartet/plane_4.png \
  --output artifacts/adaptive-prediction
```

Архив содержит исходники, полный подготовленный набор данных, веса и историю
обоих обучений, PNG/NPY результатов, параметры и фазы исследовательских опытов.
`run/result` — чистая adaptive сеть, `run/user_refined_3000` — гибрид.
`run/best.pt` — adaptive, `run/parent.pt` — wavefront, `run/texture_parent.pt` —
исходная texture-сеть. Kaggle metadata приватные; входной ZIP для загрузки
собирается из исходного пользовательского файла скриптом подготовки.
