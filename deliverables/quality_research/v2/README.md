# Качество: цель Pearson ≥0.8 и SSIM ≥0.8

**Оба порога при одном общем ДОЭ пока не достигнуты.**

- [PDF](reports/report.pdf), [полный отчёт](reports/report.md),
  [CSV](reports/measurements.csv), [JSON](reports/summary.json).
- [Тест](reports/test_quality.png), [контроль ограничения](reports/constraint_control.png),
  [ваши изображения до/после](reports/user_comparison.png).
- [Нейросеть](user_neural/), [600 шагов уточнения](user_refined_600/),
  [3000 шагов](user_refined_3000/).
- [Кандидат QAT](candidate_qat.pt), [обучение](training.json),
  [benchmark](benchmark_test.json), [полный архив](submission.zip).

| Один общий ДОЭ, тот же test из 32 четвёрок | Pearson | SSIM |
|---|---:|---:|
| Прежняя дообученная нейросеть | 0.57364 | 0.41853 |
| Кандидат с perceptual/QAT loss | 0.57268 | 0.41655 |
| Кандидат + 600 шагов Adam по фазе | **0.67111** | **0.60637** |

Новый loss сам по себе не улучшил test чистой сети: production-веса оставлены
прежними. Итеративное уточнение улучшает качество, с ценой около 6.04 с/четвёрку
в пакетном CPU-тесте вместо примерно 0.04 с для сети. Это не один forward pass.

На выбранных Cracks/Gabor/Perlin/Voronoi: **0.31135/0.21861 → 0.59732/0.46088**
с кандидатом и 3000 шагами (Pearson/SSIM).

Контроль с отдельным ДОЭ на каждую плоскость дал **0.98968/0.95072** на первых
4 validation-четвёрках. Он не соответствует условию одного общего ДОЭ и не
считается достижением цели задания. Неудача этих методов не доказывает
математическую невозможность 0.8 при исходных условиях.

## Использование

```bash
python -m doe predict --checkpoint deliverables/quality_research/v2/candidate_qat.pt \
  --images examples/noise_quartet/plane_1.png examples/noise_quartet/plane_2.png \
           examples/noise_quartet/plane_3.png examples/noise_quartet/plane_4.png \
  --refine-steps 600 --refine-lr 0.08 --output artifacts/quality-new-result
```

`--refine-steps 0` сохраняет обычный инференс. В metrics.json записаны число шагов,
лучшая итерация и время; реконструкции рассчитаны из сохранённого PNG фазы.
Нет фильтрации, gain по target, обрезки ROI и замены исходных PNG.

В архиве `run/result` — общая фаза после 3000 шагов, `run/user_refined_600` — после
600, `run/user_neural` — без уточнения. `run/parent.pt` — предыдущие веса,
`run/best.pt` — кандидат. `run/probes` содержит validation-опыты; файлы
`independent_planes` — не соответствующий ТЗ контроль.

17 тестов пройдены. Все 6 ячеек notebook с дообучением и уточнением фазы прошли
локальный короткий интеграционный прогон. Полные результаты — CPU. Удалённые GPU/Kaggle и физический
ПВМС не проверены. Возможные изменения архитектуры (wavefront U-Net/HoloNet,
амплитудная свобода, больше optical stages) — направления исследования, не
проверенное обещание 0.8.
