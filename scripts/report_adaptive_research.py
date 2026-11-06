"""Publish measured adaptive-network results and separate sampling experiments."""
import argparse
import csv
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

plt.rcParams["pdf.fonttype"] = 42


def read(path):
    return json.loads(Path(path).read_text())


def generate(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = read("artifacts/quality_research/adaptive_test.json")
    training = {name: read(f"artifacts/{name}_v1/training.json") for name in ("wavefront", "adaptive")}
    user = {steps: read(f"artifacts/adaptive_v1/user_{steps}/metrics.json") for steps in (0, 3000)}
    rows = []

    def add(split, method, pearson, ssim, grid="original", seconds=None):
        rows.append(dict(split=split, method=method, pearson=pearson, ssim=ssim,
                         sampling_grid=grid, seconds_per_quartet_batched=seconds,
                         both_at_least_0_8=pearson >= .8 and ssim >= .8))

    for name, value in benchmark["summary"].items():
        add("test_32", name, value["pearson"], value["ssim"],
            seconds=value["seconds_per_quartet_batched"])
    for steps, value in user.items():
        add("reserved_user_quartet", f"adaptive + {steps} phase Adam steps",
            value["mean_pearson"], value["mean_ssim"])

    probes = Path("artifacts/quality_research")
    original_names = ["progressive_direct", "progressive_fitwarm", "raar_085", "raar_095",
                      "homotopy_geometry", "homotopy_targets", "robust_from_expert", "softpeak_from_expert"]
    for name in original_names:
        probe = read(probes / f"{name}.json")
        value = probe if "mean_pearson" in probe else probe["history"][-1]
        add("validation_first_4", name, value.get("mean_pearson", value.get("pearson")),
            value.get("mean_ssim", value.get("ssim")))
    diagnostic = read(probes / "target_specific_wavefront.json")["history"][-1]
    add("validation_first_1_fitted_diagnostic", "target-specific neural weights; NOT generalization",
        diagnostic["pearson"], diagnostic["ssim"])
    for path in sorted(probes.glob("camera_*.json")):
        probe = read(path)
        add("validation_first_4", path.stem, probe["pearson"], probe["ssim"],
            grid=f"declared finer camera, padding={probe['config']['padding']}")
    assert not any(row["both_at_least_0_8"] for row in rows)
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(goal=dict(pearson=.8, ssim=.8, one_shared_doe=True), goal_achieved=False,
                   benchmark=benchmark, training=training, measurements=rows,
                   validation=dict(pytest_passed=26, notebook_code_cells_executed=6,
                                   notebook_mode="local quick integration; not full GPU training"),
                   kaggle_status="staged, not submitted: no credential binding or applied Kaggle host access",
                   interpretation="Compare model gains only on identical test images and original sampling grid. "
                                  "Finer camera probes change the physical field of view and are separate research.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    methods = ["baseline", "adaptive", "gs_80"]
    for ax, metric in zip(axes, ("pearson", "ssim")):
        bars = ax.bar(range(3), [benchmark["summary"][n][metric] for n in methods],
                      color=["#8798a5", "#237ab0", "#c48c49"])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.axhline(.8, color="#b03838", linestyle="--", label="Goal 0.8")
        ax.set_xticks(range(3), ["Original NN", "Adaptive NN", "GS 80"])
        ax.set(ylim=(0, 1), title=metric.upper(), ylabel="Same test: 32 quartets")
        ax.legend()
    fig.savefig(output / "test_quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, run) in zip(axes, training.items()):
        ax.plot([r["epoch"] for r in run["history"]], [r["validation_loss"] for r in run["history"]], label="validation")
        trained = run["history"][1:]
        ax.plot([r["epoch"] for r in trained], [r["train_loss"] for r in trained], label="train")
        ax.axvline(run["best_epoch"], color="gray", linestyle="--")
        ax.set(title=name, xlabel="Epoch", ylabel="Perceptual loss")
        ax.legend()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    for i in range(4):
        paths = [Path(f"artifacts/adaptive_v1/user_0/target_{i+1}.png")] + [
            Path(f"artifacts/adaptive_v1/user_{n}/reconstruction_{i+1}.png") for n in (0, 3000)]
        for j, (ax, path) in enumerate(zip(axes[i], paths)):
            with Image.open(path) as im:
                ax.imshow(np.array(im), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Target, z={user[0]['planes'][i]['distance_mm']:.1f} mm", fontsize=9)
            else:
                plane = user[(0, 3000)[j-1]]["planes"][i]
                ax.set_title(f"{('Adaptive NN', '+ Adam 3000')[j-1]}\nr={plane['pearson']:.3f}; SSIM={plane['ssim']:.3f}", fontsize=9)
    fig.savefig(output / "user_comparison.png", dpi=150)
    plt.close(fig)

    test = benchmark["summary"]
    delta = {key: test["adaptive"][key] - test["baseline"][key] for key in ("pearson", "ssim")}
    table = "| Набор | Метод | Pearson | SSIM | Сетка |\n|---|---|---:|---:|---|\n"
    for row in rows:
        table += f"| {row['split']} | {row['method']} | {row['pearson']:.5f} | {row['ssim']:.5f} | {row['sampling_grid']} |\n"
    text = f"""# Adaptive wavefront: результаты обучения

**Pearson ≥0.8 и SSIM ≥0.8 одновременно пока не достигнуты.**
Новая чистая сеть на том же test из 32 четвёрок: **{test['adaptive']['pearson']:.5f} / {test['adaptive']['ssim']:.5f}**.
Прежняя сеть в том же измерении: {test['baseline']['pearson']:.5f} / {test['baseline']['ssim']:.5f}.
Прибавка: +{delta['pearson']:.5f} Pearson и +{delta['ssim']:.5f} SSIM.
GS80 здесь качественнее обеих сетей; порог 0.8 также не достигнут.

![Один test и одна физическая сетка](test_quality.png)

## Что обучено

1. Wavefront U-Net: 20 эпох, 6 оптических слоёв, инициализатор ширины 8 с тремя
   уровнями. Старт из прежних texture-весов, perceptual loss с 8-битным STE.
   Validation loss {training['wavefront']['initial_validation_loss']:.6f} → {training['wavefront']['best_validation_loss']:.6f},
   лучшая эпоха {training['wavefront']['best_epoch']}; CPU {training['wavefront']['elapsed_seconds']:.1f} с.
2. Adaptive: 16 эпох, 12 оптических слоёв, U-Net с четырьмя уровнями и обучаемая
   пространственная амплитудная свобода/фазовая поправка каждого слоя. Модуль
   итогового единственного ДОЭ всегда единица. Старт из выбранного wavefront checkpoint.
   Validation loss {training['adaptive']['initial_validation_loss']:.6f} → {training['adaptive']['best_validation_loss']:.6f},
   лучшая эпоха {training['adaptive']['best_epoch']}; CPU {training['adaptive']['elapsed_seconds']:.1f} с.

![Истории обучения](training.png)

Обучение использовало прежние 256 train-четвёрок, validation — 32. Семейства
train/validation/test различны; Cracks/Gabor/Perlin/Voronoi зарезервированы.
Checkpoint выбирался по validation loss. Test не использовался в loss,
оптимизации весов или выборе эпохи. Повторные исследовательские сравнения на
одном test означают, что его нельзя считать новым, ранее не виденным аудитом.
Для итогового подтверждения новой архитектуры нужен дополнительный независимый набор.
На CPU, 1 поток, batch=4: исходная сеть {test['baseline']['seconds_per_quartet_batched']:.4f},
adaptive {test['adaptive']['seconds_per_quartet_batched']:.4f}, GS80 {test['gs_80']['seconds_per_quartet_batched']:.4f} с/четвёрку.
Это время синтеза в пакете, без IO и расчёта метрик; новая сеть тяжелее прежней.

## Выбранные пользовательские изображения

Adaptive без оптимизатора: {user[0]['mean_pearson']:.5f}/{user[0]['mean_ssim']:.5f}.
С 3000 шагами общей фазы: {user[3000]['mean_pearson']:.5f}/{user[3000]['mean_ssim']:.5f}.
Последний результат хуже опубликованного v2 QAT + 3000: 0.59732/0.46088.
Поэтому v2 сохранён, новый гибрид не объявлен улучшением.
Лучший опубликованный test-гибрид v2 + 600 остаётся 0.67111/0.60637;
его нельзя напрямую сопоставлять со скоростью одного forward pass.

![Реконструкции выбранных изображений](user_comparison.png)

## Журнал дополнительных попыток

{table}

Progressive phase, RAAR, homotopy и новые loss использовали первые 4 validation-четвёрки.
Robust/softpeak продолжают прежние 3000 шагов ещё на 1800; это больше вычислений.
RAAR при обоих beta вернулся к одной сохранённой лучшей инициализации.
Target-specific neural fitting меняет веса на одном входе: это диагностика, не
обобщение сети. Параметры опытов, история и итоговые фазы лежат в `../probes`.

## Отдельное исследование сетки камеры

В приложенном ТЗ указан шаг ПВМС, но не физический шаг камеры. Проверена явно
заданная более частая сетка: нулевое дополнение **до** единственного FFT,
затем центральные 128×128 отсчётов. Управляемый ДОЭ всё ещё 128×128; оцениваются
все 128×128 пикселей каждого исходного target. Выходной шаг и физическое поле
зрения меняются относительно прежней сетки λz/(128Δu). Эти результаты **не являются
улучшением модели при прежней дискретизации**, не включены в основной benchmark
и не используются в production-инференсе. Учтена также энергия вне камеры.
Подгонка gain применялась только внутри MRAF-инициализации; конечная оценка
использует прежнюю нормировку модуля по собственному максимуму без target gain.
Ни одна проверенная сетка не дала оба показателя 0.8.

## Следующий GPU-эксперимент

Готов приватный Kaggle job: 24 оптических слоя, 4 уровня U-Net, 80 эпох,
2048 train-четвёрок, те же 32 validation и 32 test. Исходные train-текстуры прежние:
увеличено число комбинаций и аугментаций, не число независимых источников.
Notebook самодостаточен и не требует Интернета внутри Kaggle. Целиком выполнены
6 ячеек локального quick-прогона; это проверка интеграции, не GPU-обучение.
Удалённый запуск **не отправлен**: нет KAGGLE_API_TOKEN в секретах и применённого
доступа к api.kaggle.com/www.kaggle.com. Подготовленные metadata и инструкции
в `../kaggle`; скрипт подготовки `scripts/stage_kaggle_job.py`.
Больший запуск не гарантирует 0.8. После него требуются фактические метрики.

## Протокол и проверки

Один общий фазовый ДОЭ, λ=532 нм, шаг ПВМС 10 мкм, z=24.1/25.3/26.5/27.7 мм.
Для основной таблицы физика и выходные сетки прежние. Оценка по реальным
8-битным реконструкциям из квантованной фазы, SSIM skimage с окном 7 и
data_range=1, Pearson по всем пикселям. Нет фильтрации и подмены входных изображений.
26 тестов прошли, включая прямую квадратуру Fresnel, сопряжённость и энергию
camera-модели, градиенты нового инициализатора и checkpoint round trip.
Физический ПВМС не проверялся. Неудачные опыты не доказывают невозможность 0.8.
"""
    (output / "report.md").write_text(text)
    with PdfPages(output / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .94, "Adaptive wavefront: измеренные результаты", fontsize=16, weight="bold")
        paragraphs = [
            "Цель: один общий фазовый ДОЭ, Pearson и SSIM не ниже 0.8. Цель пока не достигнута.",
            f"Одинаковый test, 32 четвёрки: прежняя сеть {test['baseline']['pearson']:.5f}/{test['baseline']['ssim']:.5f}; adaptive {test['adaptive']['pearson']:.5f}/{test['adaptive']['ssim']:.5f}; GS80 {test['gs_80']['pearson']:.5f}/{test['gs_80']['ssim']:.5f} (Pearson/SSIM).",
            "Обучены U-Net wavefront (20 эпох) и adaptive сеть с 12 оптическими слоями (16 эпох). Выбор checkpoint по 32 validation-четвёркам. Train: 256 четвёрок, семейства раздельны. Повторный test не является новым независимым аудитом.",
            f"На выбранной пользовательской четвёрке adaptive: {user[0]['mean_pearson']:.5f}/{user[0]['mean_ssim']:.5f}, +3000 шагов общей фазы: {user[3000]['mean_pearson']:.5f}/{user[3000]['mean_ssim']:.5f}. Старый v2-гибрид лучше; он сохранён.",
            "Дополнительно измерены progressive phase, RAAR, homotopy, robust loss и distributed max gradient. Порог не достигнут. Подгонка весов на одном target обозначена только как диагностика.",
            "Более частая camera-сетка меняет физический шаг и поле зрения. Её оценки вынесены отдельно, не являются выигрышем сети на прежней сетке. Все пиксели targets оценены; энергия вне камеры учтена. Оба 0.8 также не достигнуты.",
            "Подготовлен приватный Kaggle GPU job: 24 слоя, 80 эпох, 2048 train-четвёрок. Шесть ячеек notebook прошли локальный quick-прогон. Полный GPU job не отправлен: нет секрета Kaggle и применённого сетевого доступа. Обещания 0.8 нет.",
            "26 тестов пройдены. Полный Markdown/CSV/JSON и исходные истории включены в поставку. Основные метрики: реальный 8-битный ДОЭ и PNG амплитуд, прежняя физика, без фильтрации и target gain. Физический эксперимент не выполнен. Неудачи не доказывают невозможности цели."
        ]
        y = .88
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, 90)
            fig.text(.07, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.4)
            y -= len(lines) * .019 + .025
        pdf.savefig(fig)
        plt.close(fig)
        for name in ("test_quality.png", "training.png", "user_comparison.png"):
            fig, ax = plt.subplots(figsize=(8.27, 11.69) if "user" in name else (10, 6))
            with Image.open(output / name) as im:
                ax.imshow(np.array(im))
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps({"output": str(output), "goal_achieved": False, "adaptive_test": test["adaptive"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    generate(parser.parse_args().output)
