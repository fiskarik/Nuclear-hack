# Дообучение на пользовательских текстурах

Дообучаются существующие веса `deliverables/best.pt`, а не новая случайно
инициализированная сеть. Архитектура сохраняется: 6 оптических слоёв, CNN width=12.
Для fine-tuning добавлен параметр `--initial-checkpoint`; оптимизатор Adam
инициализируется заново. В checkpoint и training.json записывается SHA256 родителя.

## Данные

Доступен загруженный пользователем ZIP с 262 PNG 128×128. Лицензия внутри архива —
CC0/Public Domain, Screaming Brain Studios. RGB/RGBA преобразуются в L через
alpha-composite на чёрный фон; размер и контраст не меняются.

Разделение фиксировано до обучения, на уровне семейств:

| Назначение | Семейства | Исходных PNG | Четвёрок |
|---|---|---:|---:|
| train | Craters, Grainy, Manifold, Marble, Melt, Milky, Spokes, Super Noise, Swirl, Techno | 140 | 256 |
| validation | Super Perlin, Turbulence | 28 | 32 |
| test | Streak, Vein | 28 | 32 |
| резерв для пользовательской четвёрки | Cracks, Gabor, Perlin, Voronoi | 66 | отдельный расчёт |

Источник не попадает в разные split; дополнительно проверяются точные дубликаты
grayscale-пикселей. Train-аугментации: отражение, поворот на 90°, циклический сдвиг.
Каждый исходник и преобразование записываются в manifest. Test и validation не
аугментируются. 32 тестовые четвёрки используют 28 исходных изображений повторно,
а не представляют 128 независимых оригиналов.

## Воспроизведение

```bash
python scripts/prepare_texture_dataset.py --archive PATH_TO_ZIP --output data/textures-new
python -m doe train --data data/textures-new --output artifacts/textures-new \
  --initial-checkpoint deliverables/best.pt --epochs 30 --learning-rate 0.001 \
  --batch-size 4 --stages 6 --width 12 --device auto
```

Затем оцените исходную и дообученную модели на **том же** test split командой
`doe evaluate`. Примените обе к одной четвёрке командой `doe predict` и соберите
отчёт:

```bash
python scripts/report_texture_finetuning.py \
  --training artifacts/textures-new/training.json --dataset data/textures-new \
  --before-benchmark BEFORE_BENCHMARK_JSON --after-benchmark AFTER_BENCHMARK_JSON \
  --before-result BEFORE_RESULT_DIR --after-result AFTER_RESULT_DIR --output NEW_REPORT_DIR
```

Создаются PDF, Markdown, CSV, JSON и графики. Генератор проверяет SHA256 родителя,
соответствие checkpoint и одинаковые входы до/после. Результат включает фактические
ухудшения, если они есть; улучшение не предполагается заранее.

В Kaggle notebook задайте `TEXTURE_ZIP` и `INITIAL_CHECKPOINT` на файлы,
прикреплённые как Kaggle Input. Он автоматически наследует архитектуру checkpoint,
готовит family-disjoint набор, дообучает сеть, сравнивает до/после и собирает отчёт.
Внешний интернет внутри notebook не требуется. Короткий интеграционный прогон
проверяет этот режим отдельно от полного обучения.

## Предоставленные ссылки

- [TensorFlow README](https://github.com/Jemuna/Image-Classification-Using-Deep-Learning-in-TensorFlow/blob/main/README.md)
  прочитан: это описание классификации grayscale CIFAR-10. Из него не взяты данные
  для обучения; задача синтеза фазового ДОЭ не подменяется классификацией.
- [700 Noise Textures](https://opengameart.org/content/700-noise-textures) и
  [иллюстрация на ResearchGate](https://www.researchgate.net/figure/Aliasing-free-area-in-pixel-units-of-the-chirp-functions-under-the-calculation-condition_fig5_236235124)
  недоступны по текущей политике сети. Домены добавлены в черновик настроек среды,
  но сохранение черновика само по себе не применяет разрешение. Содержимое этих
  страниц не подтверждено. Полный набор из 700 не скачивался; использован только
  реально доступный пользовательский ZIP с 262 PNG.

## Дискретизация Френеля

Независимо от недоступной иллюстрации добавлена диагностика обеих chirp-функций.
Консервативная граница входной chirp: `z_critical=N*du²/λ≈24.06015 мм`.
Все расстояния задачи чуть выше этой границы и проходят проверку входной chirp.
Соответствующие отношения к Nyquist для выходной chirp, напротив, выше единицы:
это важно для интерполяции комплексной фазы. В нашем одно-FFT расчёте выходная
chirp только поточечно умножает поле на фазовый множитель единичного модуля и не
меняет сохраняемый модуль. Утверждение, что обе chirp-функции полностью свободны
от aliasing во всём поле, не делается. Диагностика включена в новые metrics.json.
