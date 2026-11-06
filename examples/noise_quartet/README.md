# Четыре текстуры для расчёта ДОЭ

Из предоставленного пользователем архива Screaming Brain Studios выбраны первые
номерные варианты четырёх разных семейств, до расчёта метрик:

| Файл | Источник | Плоскость |
|---|---|---:|
| `plane_1.png` | Cracks 1 | 24.1 мм |
| `plane_2.png` | Gabor 1 | 25.3 мм |
| `plane_3.png` | Perlin 1 | 26.5 мм |
| `plane_4.png` | Voronoi 1 | 27.7 мм |

Входы — PNG, 128×128, 8-битный grayscale L. Преобразование: Pillow
`alpha_composite` на чёрный фон и `convert('L')`, без изменения размера,
растяжения контраста и фильтрации. У всех четырёх исходных изображений alpha=255,
поэтому фон на результат преобразования не влияет.

В `originals` сохранены точные байты выбранных файлов из архива; SHA256 исходников
и преобразованных PNG, имена внутри архива и его контрольная сумма — в
`manifest.json`. В этом каталоге находится только выбранная четвёрка. Данные для
отдельного дообучения описаны в `docs/texture-finetuning.md` и входят в архив
дообучения. Входящий в архив `License.txt` указывает CC0/Public Domain.

Расчёт выполняется готовой сетью без переобучения:

```bash
python -m doe predict --checkpoint deliverables/best.pt \
  --images examples/noise_quartet/plane_1.png examples/noise_quartet/plane_2.png \
           examples/noise_quartet/plane_3.png examples/noise_quartet/plane_4.png \
  --output artifacts/noise-quartet-new-run
```

Подготовка из исходного ZIP при необходимости:

```bash
python scripts/prepare_noise_quartet.py --archive PATH_TO_ZIP --output NEW_DIRECTORY
```

[Результаты](../../deliverables/noise_quartet/README.md) рассчитаны из сохранённого
квантованного ДОЭ. Веса сети не менялись. Это пользовательский тестовый набор
текстур, не изображения, извлечённые из документов с заданием.
