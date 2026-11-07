# Передача обучения: как продолжить обучать модель

Текущая модель по умолчанию — **r8** (`weights/doe_net.pt`): развёрнутая сеть на 8 стадий,
4,3 млн параметров. История: g4 → ftmm → s8 (расширение до 8 стадий) → r8 (4 ч дообучения на
реальных фото и текстурах). Подробности и все цифры — в [REPORT.md](REPORT.md).

## 1. Что нужно новому человеку

1. **Доступ к репозиторию:** GitHub → Settings → Collaborators → Add people (или форк).
2. **Свой аккаунт Kaggle** с подтверждённым телефоном (иначе ядро получит CPU вместо GPU).
   GPU-квота у каждого аккаунта своя, 30 ч в неделю.
3. **API-токен Kaggle:** Kaggle → Settings → API → Create New Token. Файл `kaggle.json` положить в
   `~/.kaggle/`, затем `pip install kaggle`. Токен никому не пересылать.
4. **Python 3.11 и зависимости:** `pip install -r requirements.txt` из корня репозитория.

## 2. Данные и стартовые веса на Kaggle

Код уходит в ядро внутри скрипта (≈ 340 КБ), а кэши данных и стартовые веса — через
Kaggle-датасет. Есть два варианта.

**а) Доступ к нашему датасету** `fiskarik/doe-nn-data`: владелец открывает его страницу на
Kaggle → Settings → Sharing → добавляет аккаунт нового человека. Внутри лежат кэши
`textures128.npz`, `textures512.npz`, `cifar_gray.npz` и веса g4, ftmm и s8.
**Весов r8 там нет:** их нужно добавить новой версией датасета или выбрать вариант б.

**б) Свой датасет** (из папки `unrolled/`):

```bash
python scripts/prepare_data.py            # текстуры 512 (opengameart.org) + CIFAR-10 (Kaggle) → data/cache
mkdir -p ~/doe-nn-data && cp data/cache/*.npz ~/doe-nn-data/
cp weights/doe_net.pt ~/doe-nn-data/doe_net_r8.pt
kaggle datasets init -p ~/doe-nn-data     # в dataset-metadata.json задать title и id: <ваш_логин>/doe-nn-data
kaggle datasets create -p ~/doe-nn-data   # датасет по умолчанию приватный
```

В ядре все `*.npz` из подключённых датасетов попадают в `data/cache/`, а все `*.pt` — в `weights/`.
Поэтому стартовые веса указываются как `--init weights/<имя файла>.pt`.

Внешние датасеты подключаются по имени, скачивать их не нужно:

| Датасет | Зачем |
|---|---|
| `jmexpert/describable-textures-dataset-dtd` | реальные текстуры DTD |
| `jhoward/imagenette-160-px` | фото (архив распаковывается в ядре) |
| `balraj98/berkeley-segmentation-dataset-500-bsds500` | фото |
| `sherylmehta/kodak-dataset` | контроль на фото Kodak |
| `zalando-research/fashionmnist`, `crawford/emnist`, `anokas/kuzushiji` | символы и силуэты |
| `jessicali9530/lfw-dataset` | лица (добавляются к фото) |
| `pascalbliem/handwritten-chinese-character-hanzi-datasets` | рукописные иероглифы CASIA (14 ГБ, необязательно) |

## 3. Запуск обучения

Из папки `unrolled/`. `KAGGLE_USERNAME` — логин нового человека (по умолчанию в скрипте `fiskarik`).

```bash
export KAGGLE_USERNAME=<ваш_логин>
COMMON="--init weights/doe_net_r8.pt --grad-feat --pad zeros --stages 8 --batch 24 --bank 120000 \
 --lr 5e-5 --w-minmax 0.5 --w-hotspot 1.0 --real-root /kaggle/input,/kaggle/tmp/extracted \
 --eval-every 500 --log-every 100 --hours 4.0"
python scripts/kaggle_run.py push --slug doe-nn-next --require-gpu --timeout 15600 \
  --dataset <ваш_логин>/doe-nn-data \
  --dataset jmexpert/describable-textures-dataset-dtd --dataset jhoward/imagenette-160-px \
  --dataset balraj98/berkeley-segmentation-dataset-500-bsds500 --dataset sherylmehta/kodak-dataset \
  --dataset zalando-research/fashionmnist --dataset crawford/emnist --dataset anokas/kuzushiji \
  --dataset jessicali9530/lfw-dataset \
  --job "glyph:$COMMON --mix 0.15,0.1,0.15,0,0.2,0.15,0.25" \
  --job "real:$COMMON --mix 0.2,0.1,0.2,0,0.25,0.25"
python scripts/kaggle_run.py status --slug doe-nn-next
python scripts/kaggle_run.py pull --slug doe-nn-next --dest runs/kaggle_doe-nn-next
```

Две задачи (`--job`) идут параллельно на двух T4, каждая на своей видеокарте.

`--mix` задаёт доли источников в банке изображений: синтетика, текстуры 128, текстуры 512,
CIFAR-10, фото, текстуры DTD, символы. Семь долей включают символы. Источник, которого нет в
данных, заменяется синтетикой.

В `runs/<имя>/` появляются:
- `doe_net.pt` — лучший по автоматическому критерию;
- `ckpt_last.pt` — последний чекпойнт;
- `train_log.jsonl`;
- `stdout.log`.

**Что уже известно — не повторять:**
- Шаг 2e-4 на 8+ стадиях неустойчив: нормы градиента 10⁴–10⁵, лучший чекпойнт — уже на шаге
  1000, дальше метрики проседают. Вариант s8hi с шагом 3e-4 разошёлся (NaN). При 5e-5 обучение r8
  шло устойчиво до конца.
- Автоматический критерий выбора чекпойнта шумный из-за синтетики. Сравнивать нужно
  `ckpt_last.pt` и `doe_net.pt` вручную, см. раздел 4.
- fp16 (`--amp`) обучается быстрее, но получился чуть хуже fp32.
- 12 стадий (s12) и длинное, 8,4 ч, дообучение 8 стадий (s8L) с шагом 2e-4 дали хуже, чем s8.
  12 стадий с малым шагом ещё не пробовали.
- `--w-hotspot` не убирать. Без штрафа сеть учится «горячей точке»: почти чёрный экран с одной
  яркой точкой. ИСС на разреженных картинках от этого растёт, но изображения не видно
  (REPORT.md, раздел 7а).
- Запуск `doe-nn-glyph` (символы, старт с s8) не дал результатов: закончилась GPU-квота.
  Следующий логичный шаг — тот же запуск, но со стартом от r8 (команда выше).

## 4. Проверка: лучше ли новая модель

Сравнение в режиме по умолчанию (сеть + 50 итераций доводки), по правилу организаторов:

```bash
python scripts/compare_models.py --alpha fast --refine 50 \
  --weights r8=weights/doe_net.pt new=runs/kaggle_doe-nn-next/runs/glyph/ckpt_last.pt \
  --glyph-root path/to/glyph_test --expert-dir path/to/expert_runs \
  --out results/model_comparison_new.json
```

Аргумент `--glyph-root` необязательный. Это папка с test-частями символьных датасетов:

```bash
kaggle datasets download zalando-research/fashionmnist -f t10k-images-idx3-ubyte -p glyph_test/fashion --unzip
kaggle datasets download anokas/kuzushiji -f k49-test-imgs.npz -p glyph_test/k49
kaggle datasets download crawford/emnist -f emnist_source_files/emnist-balanced-test-images-idx3-ubyte -p glyph_test/emnist --unzip
```

`.npz` должен остаться как есть: если скачался `.zip`, распакуйте его. Аргумент `--expert-dir` —
присланные `expert_runs`, в репозитории их нет, тоже необязательно. Без них сравниваются
отложенные текстуры, CIFAR-10, синтетика и наша четвёрка.

**Правило замены:** модель заменяется, только если на текстурах, CIFAR и кейсах экспертов ИСС
выше без потери КК. Разница должна держаться на большинстве комплектов, а не только в среднем.
У r8 против s8, например, ИСС на текстурах вырос на +0,007 и был выше во всех 24 комплектах.

## 5. Замена модели по умолчанию

```bash
python - <<'EOF'
import torch
ck = torch.load("runs/kaggle_doe-nn-next/runs/glyph/ckpt_last.pt", map_location="cpu", weights_only=False)
torch.save({k: ck[k] for k in ("model", "config", "step", "eval")}, "weights/doe_net.pt")  # без состояния оптимизатора
EOF
python -m pytest -q tests
python predict.py --input examples --output /tmp/check --no-extras     # самопроверка, ~0,7 с на комплект
cd .. && docker build -t doe-nn . && docker run --rm --network none \
  -v "$PWD/in:/data/input:ro" -v "$PWD/out:/data/output" doe-nn
```

Чекпойнт с состоянием оптимизатора весит ~52 МБ, без него ~17,5 МБ. В Docker-образ попадает
только `weights/doe_net.pt` (см. `.dockerignore`). Архитектура читается из `config` внутри
файла, поэтому сеть с другим числом стадий тоже загрузится без правок кода. После замены
обновите таблицы в README.md и REPORT.md.

Для отдельных опытов есть `--resume runs/.../ckpt_last.pt` (продолжить с тем же оптимизатором) и
`--init` (новое обучение с этих весов, в том числе с другим числом стадий).
