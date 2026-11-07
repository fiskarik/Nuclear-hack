# Обученный adaptive wavefront и запуск Kaggle

Результаты: [v3, PDF, веса и изображения](../deliverables/quality_research/v3/README.md).
Чистая сеть улучшилась на прежнем test из 32 четвёрок до **0.60747 Pearson /
0.43629 SSIM**. Оба 0.8 пока не достигнуты. Более тяжёлая архитектура и большее
число эпох сами по себе не гарантируют этот порог.

## Воспроизведение CPU-обучения

Используйте новый пустой output-каталог для каждой команды.
Сначала подготовьте набор из пользовательского SBS ZIP:

```bash
python scripts/prepare_texture_dataset.py --archive /path/to/sbs-textures.zip \
  --output data/textures_v1 --train 256 --validation 32 --test 32
python -m doe train --data data/textures_v1 --output artifacts/wavefront_v1 \
  --initial-checkpoint deliverables/texture_finetuned/best.pt \
  --architecture wavefront --stages 6 --width 12 --initializer-width 8 \
  --initializer-depth 3 --objective perceptual --epochs 20 --batch-size 4 \
  --learning-rate 0.001 --device cpu --threads 2
python -m doe train --data data/textures_v1 --output artifacts/adaptive_v1 \
  --initial-checkpoint artifacts/wavefront_v1/best.pt \
  --architecture adaptive --stages 12 --width 12 --initializer-width 8 \
  --initializer-depth 4 --objective perceptual --epochs 16 --batch-size 4 \
  --learning-rate 0.0005 --device cpu --threads 2
python scripts/compare_models.py --data data/textures_v1 \
  --checkpoint baseline=deliverables/texture_finetuned/best.pt \
  --checkpoint adaptive=artifacts/adaptive_v1/best.pt \
  --output artifacts/adaptive-comparison.json --threads 1 --batch-size 4
```

Квантование фазы и реконструкций в 8 бит одинаково для всех методов.
Улучшение сети сравнивается только при одинаковой оптике и входных изображениях.
Старые checkpoint без поля `kind` поддерживаются. Переход к U-Net и увеличение
числа слоёв сохраняют прежние параметры; новые блоки инициализируются отдельно.
Adam начинает с нового состояния, как указано в training.json.

## Подготовка приватного Kaggle job

```bash
python scripts/stage_kaggle_job.py --archive /path/to/sbs-textures.zip \
  --checkpoint deliverables/quality_research/v3/best.pt \
  --output artifacts/kaggle-v3 --username fiskarik
```

Скрипт не обращается к Kaggle. Он готовит dataset с исходным ZIP и весами,
самодостаточный notebook, приватные metadata и SHA256. В репозиторий включена
копия notebook/metadata, а входные бинарные файлы собираются командой выше.
SBS ZIP содержит CC0/Public Domain; для весов новая публичная лицензия не выдаётся.
Dataset metadata использует `other` с пояснением, dataset и notebook приватные.

Полная конфигурация: 24 оптических слоя, 4 уровня U-Net, 80 эпох, batch 8,
Adam lr=0.0003, perceptual loss, 2048 train-четвёрок. Это больше комбинаций
прежних train-текстур, а не 2048 независимых исходных наборов. Seed и 32
validation/test-четвёрки прежние. Без GPU полный job останавливается с понятной
ошибкой. Нужны доступная GPU-квота и подтверждённый аккаунт Kaggle.

Перед отправкой подключите KAGGLE_API_TOKEN через защищённые настройки среды,
разрешите HTTPS к api.kaggle.com и www.kaggle.com, сохраните и опубликуйте настройки.
Токен не нужен внутри notebook и не должен попадать в код, Git или чат.
Откройте сгенерированный README: там команды проверки входа, создания приватного
dataset, отправки job, статуса и скачивания результатов. Если указанный dataset
или notebook уже занят другим проектом, сначала выберите новый slug.

Полный GPU-запуск пока не отправлен. Сохранение файлов и локальный quick-прогон
не являются подтверждением удалённого обучения. После выполнения проверяются
фактические benchmark.json и PNG, а не только статус job.

## Отдельные исследовательские режимы

`probe_progressive.py`, `probe_raar.py`, `probe_homotopy.py`, `probe_quality.py`
и `probe_neural_fit.py` содержат диагностические эксперименты. Подгонка весов
по одному target не является измерением обобщения сети.

`doe/camera_optics.py` и `probe_camera.py` задают отдельную физическую сетку
камеры через нулевое дополнение перед FFT. Меняются шаг и поле зрения; эти
оценки вынесены в отдельные строки отчёта и не выдаются за улучшение нейросети
на прежней сетке. В основном CLI и checkpoint используется прежний Fresnel.
