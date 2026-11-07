"""Build a standalone, inspectable Kaggle notebook with embedded project source."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = [*sorted((ROOT / "doe").glob("*.py")), *sorted((ROOT / "tests").glob("*.py")),
             *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "docs").glob("*.md")),
             ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "requirements-dev.txt", ROOT / "README.md"]
    sources = {str(path.relative_to(ROOT)): path.read_text() for path in files}
    cells = [nbf.v4.new_markdown_cell(r"""# Нейросетевой синтез четырёхплоскостного ДОЭ

Задание хакатона: 4 одноканальных PNG 128×128 → фазовый ДОЭ 128×128 →
4 реконструкции методом Френеля; λ=532 нм, пиксель ПВМС 10 мкм,
z=[24.1,25.3,26.5,27.7] мм. Экспорт: 8-битные PNG, Pearson и SSIM.

**Run All** запускает обучение, тестовую оценку и экспорт. Включите
Settings → Accelerator → GPU T4 ×2 при доступной квоте; код также работает на CPU.
Интернет и Kaggle API-токен внутри notebook не требуются.

В документах нет четырёх исходных объектных PNG: по умолчанию используется явно
обозначенный процедурный демонстрационный набор. Для своих данных укажите четыре
пути в `INPUT_IMAGES` ниже. Их порядок соответствует возрастающему z.

Для дообучения на текстурах загрузите ZIP и исходные веса как Kaggle Input,
заполните `TEXTURE_ZIP` и `INITIAL_CHECKPOINT`. Тогда notebook разделит данные
по семействам, исключит Cracks/Gabor/Perlin/Voronoi из обучения и сформирует
PDF/CSV-отчёт с результатами до и после.

Метод — фиксированная развёрнутая нейросеть с оптическими проекциями и обучаемыми
CNN-коррекциями. Обучение без эталонных голограмм; при инференсе нет шагов оптимизатора.
Тестовые изображения не участвуют в обучении и выборе checkpoint.
"""), nbf.v4.new_code_cell(
        "# Embedded source: inspectable, no Git clone or external download.\n"
        "from pathlib import Path\nimport json, os, sys\n"
        "BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'notebook-workdir'\n"
        "PROJECT = BASE / 'nuclear-hack-source'\nPROJECT.mkdir(parents=True, exist_ok=True)\n"
        + "SOURCE_FILES = " + repr(sources) + "\n"
        "for name, source in SOURCE_FILES.items():\n"
        "    path = PROJECT / name\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if path.exists() and path.read_text() != source:\n"
        "        raise RuntimeError(f'Existing source differs: {path}; preserve it and use a fresh work directory')\n"
        "    path.write_text(source)\n"
        "sys.path.insert(0, str(PROJECT))\n"
        "import torch, numpy as np, PIL, skimage, matplotlib\n"
        "print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'numpy': np.__version__})\n"
        "if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))\n"
    ), nbf.v4.new_markdown_cell("""## Настройки

Четыре входных файла должны быть PNG, режим L, 8 бит, 128×128. Пустая плоскость
несовместима с фазовым ДОЭ при равномерном освещении и не принимается.
`INPUT_IMAGES=[]` означает демонстрацию, а не выполнение на официальных данных.
CPU использует меньший учебный набор; на GPU можно увеличить число эпох и примеров.
При повторном запуске создаётся новый каталог, предыдущие результаты сохраняются.
"""), nbf.v4.new_code_cell("""from datetime import datetime, timezone
from doe.engine import TrainConfig, train, evaluate, predict
from doe.data import generate_dataset
import runpy

INPUT_IMAGES = []  # ['/kaggle/input/YOUR_DATA/image1.png', ..., 'image4.png']
REFINE_STEPS = int(os.environ.get('DOE_REFINE_STEPS', '0'))  # Optional, slower phase solver; raw network outputs remain separate
MODEL_KIND = os.environ.get('DOE_MODEL_KIND', 'inherit')  # 'wavefront' or 'adaptive' upgrades a matching-width checkpoint
TEXTURE_ZIP = os.environ.get('DOE_TEXTURE_ZIP') or None  # '/kaggle/input/YOUR_DATA/sbs...128x128.zip'
INITIAL_CHECKPOINT = os.environ.get('DOE_INITIAL_CHECKPOINT') or None  # '/kaggle/input/YOUR_MODEL/best.pt'
if TEXTURE_ZIP and not INITIAL_CHECKPOINT:
    raise ValueError('Texture fine-tuning requires INITIAL_CHECKPOINT')
QUICK_CHECK = os.environ.get('DOE_QUICK_CHECK') == '1'  # integration validation only
GPU = torch.cuda.is_available()
RUN = BASE / ('doe-run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
RUN.mkdir(parents=True)
DATA = RUN / 'data'
TRAINING = RUN / 'training'
if QUICK_CHECK:
    counts = (8, 4, 4)
    config = TrainConfig(epochs=2, batch_size=2, stages=2, width=4)
else:
    counts = (512, 64, 32) if GPU else (96, 16, 16)
    config = TrainConfig(epochs=20 if GPU else 24, batch_size=8 if GPU else 4,
                         stages=8 if GPU else 6, width=16 if GPU else 12)
if TEXTURE_ZIP:
    counts = (8, 4, 4) if QUICK_CHECK else (256, 32, 32)
    config.epochs = 2 if QUICK_CHECK else 30
    config.learning_rate = 0.001
if INITIAL_CHECKPOINT:
    parent = torch.load(INITIAL_CHECKPOINT, map_location='cpu', weights_only=True)
    config.stages = parent['architecture']['stages']
    config.width = parent['architecture']['width']
    config.architecture = parent['architecture'].get('kind','fixed')
    config.initializer_width = parent['architecture'].get('initializer_width',8)
    config.initializer_depth = parent['architecture'].get('initializer_depth',3)
    config.initial_checkpoint = INITIAL_CHECKPOINT
if MODEL_KIND != 'inherit':
    if MODEL_KIND not in ('fixed','wavefront','adaptive'):
        raise ValueError('Unknown model architecture')
    config.architecture = MODEL_KIND
    if MODEL_KIND == 'adaptive':
        config.stages = max(config.stages, 12)
        config.initializer_depth = 4
print('Run:', RUN, '\\nTraining:', config)
"""), nbf.v4.new_markdown_cell("""## Данные и обучение

Разные seed для train/validation/test; сеть минимизирует ошибку амплитуд после
дифференцируемого одно-FFT оператора Френеля. Лучшая модель выбирается по validation loss.
`initial.pt` нужен для контроля того, чему научилась сеть. `best.pt` — для инференса.
"""), nbf.v4.new_code_cell("""if TEXTURE_ZIP:
    prepare = runpy.run_path(str(PROJECT / 'scripts/prepare_texture_dataset.py'))['prepare']
    prepare(TEXTURE_ZIP, DATA, train=counts[0], validation=counts[1], test=counts[2])
else:
    generate_dataset(DATA, train=counts[0], validation=counts[1], test=counts[2])
training_report = train(DATA, TRAINING, config)
assert training_report['parameter_l2_change_last_vs_initial'] > 0
print('Best validation epoch:', training_report['best_epoch'])
"""), nbf.v4.new_markdown_cell("""## Независимая тестовая оценка

Сравнение с той же сетью до обучения и многоплоскостным Герчбергом–Сэкстоном
при одинаковом и большем числе оптических шагов. Все методы оцениваются после
квантования фазы и амплитуд в 8 бит. Время отражает batched throughput на текущем
устройстве; ускорение относительно классического метода не предполагается заранее.
"""), nbf.v4.new_code_cell("""benchmark = evaluate(DATA, TRAINING / 'best.pt', RUN / 'benchmark.json',
                     batch_size=config.batch_size, gs_steps=12 if QUICK_CHECK else 80)
print(json.dumps(benchmark['summary'], indent=2))
if INITIAL_CHECKPOINT:
    before_benchmark = evaluate(DATA, INITIAL_CHECKPOINT, RUN / 'benchmark-before.json',
                                batch_size=config.batch_size, gs_steps=12 if QUICK_CHECK else 80)
"""), nbf.v4.new_markdown_cell("""## ДОЭ и восстановленные изображения

Сеть выдаёт фазу, кодируемую как `0 → −π`, `255 → +π` с округлением.
Реконструкции рассчитываются заново **из сохранённого PNG ДОЭ**. Сохраняется модуль
комплексного поля, не квадрат модуля. Для отображения каждый модуль делится на свой
максимум; без вычитания фона и без подгонки к целевому изображению.
Метрики считаются по фактическим сохранённым 8-битным PNG. В `.npy` сохранены также
модули до масштабирования и комплексные поля. Шаг выходной сетки зависит от z:
Δx_out = λz/(NΔu); изображения разных плоскостей задаются на соответствующих сетках.
"""), nbf.v4.new_code_cell("""if INPUT_IMAGES:
    if len(INPUT_IMAGES) != 4:
        raise ValueError('Specify exactly four paths in increasing distance order')
    input_paths = [Path(p) for p in INPUT_IMAGES]
    data_source = 'user-provided quartet'
elif TEXTURE_ZIP:
    names = runpy.run_path(str(PROJECT / 'scripts/prepare_noise_quartet.py'))['SELECTION']
    input_paths = [DATA / 'pool' / Path(name).parent.name.replace(' ', '_') / Path(name).name for name in names]
    data_source = 'reserved texture families: Cracks, Gabor, Perlin, Voronoi; not used in training or validation'
else:
    input_paths = [DATA / 'test/sample_0000' / f'plane_{i}.png' for i in range(1, 5)]
    data_source = 'synthetic held-out demonstration, not official hackathon inputs'
result = predict(TRAINING / 'best.pt', input_paths, RUN / 'result', source=data_source)
print(json.dumps(result['planes'], indent=2))
from IPython.display import display, Image as DisplayImage
display(DisplayImage(filename=str(RUN / 'result/comparison.png')))
display(DisplayImage(filename=str(RUN / 'result/phase_preview.png')))
if REFINE_STEPS:
    refined = predict(TRAINING / 'best.pt', input_paths, RUN / 'result-refined', source=data_source,
                      refine_steps=REFINE_STEPS, refine_lr=0.08)
    print('Neural initializer + phase optimizer (not raw network):', refined['mean_pearson'], refined['mean_ssim'])
    display(DisplayImage(filename=str(RUN / 'result-refined/comparison.png')))
if INITIAL_CHECKPOINT:
    before_result = predict(INITIAL_CHECKPOINT, input_paths, RUN / 'result-before', source=data_source)
if TEXTURE_ZIP:
    report = runpy.run_path(str(PROJECT / 'scripts/report_texture_finetuning.py'))['generate']
    report(TRAINING / 'training.json', DATA, RUN / 'benchmark-before.json', RUN / 'benchmark.json',
           RUN / 'result-before', RUN / 'result', RUN / 'reports')
"""), nbf.v4.new_markdown_cell("""## Материалы для экспертов

Архив содержит код, полный сгенерированный набор, веса, историю обучения,
тестовое сравнение, входные PNG, фазовый ДОЭ, четыре восстановленных PNG,
метрики и исходные комплексные поля. Качество на официальной четвёрке необходимо
измерить отдельно после её предоставления. Это численное моделирование; оптическая
проверка на ПВМС здесь не выполняется.
"""), nbf.v4.new_code_cell("""import shutil
submission = RUN / 'submission'
submission.mkdir()
for name, source in SOURCE_FILES.items():
    path = submission / 'source' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
for name in ('data', 'training', 'result'):
    shutil.copytree(RUN / name, submission / name)
for name in ('result-before', 'reports', 'result-refined'):
    if (RUN / name).exists(): shutil.copytree(RUN / name, submission / name)
if INITIAL_CHECKPOINT:
    shutil.copy2(INITIAL_CHECKPOINT, submission / 'parent.pt')
    shutil.copy2(RUN / 'benchmark-before.json', submission / 'benchmark-before.json')
shutil.copy2(RUN / 'benchmark.json', submission / 'benchmark.json')
(submission / 'runtime.json').write_text(json.dumps({'torch': torch.__version__, 'numpy': np.__version__,
    'pillow': PIL.__version__, 'skimage': skimage.__version__, 'device': benchmark['device'],
    'input_source': data_source, 'quick_check': QUICK_CHECK}, indent=2))
archive = shutil.make_archive(str(RUN / 'submission'), 'zip', submission)
print('Submission archive:', archive)
print('Saved results:', RUN / 'result')
""")]
    for index, cell in enumerate(cells):
        cell.id = f"section-{index:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": False},
    })
    nbf.validate(notebook)
    out = ROOT / "notebooks/fresnel_doe_kaggle.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, out)
    metadata = {"id": "fiskarik/nuclear-hack-fresnel-doe", "title": "Nuclear Hack Fresnel DOE",
                "code_file": out.name, "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out.parent / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    build()
