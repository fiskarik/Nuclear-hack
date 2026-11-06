"""Create a PDF, Markdown and CSV report from actual before/after run outputs."""
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

plt.rcParams["pdf.fonttype"] = 42  # Embed searchable TrueType text, including Cyrillic.


def read(path):
    return json.loads(Path(path).read_text())


def generate(training, dataset, before_benchmark, after_benchmark, before_result, after_result, output):
    training, dataset = read(training), read(Path(dataset) / "manifest.json")
    before, after = read(before_benchmark), read(after_benchmark)
    before_result, after_result, output = Path(before_result), Path(after_result), Path(output)
    old, new = read(before_result / "metrics.json"), read(after_result / "metrics.json")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if before["sample_count"] != after["sample_count"]:
        raise ValueError("benchmark sample counts differ")
    if training["parent_checkpoint"]["sha256"] != old["metadata"]["checkpoint_sha256"]:
        raise ValueError("before result is not the fine-tuning parent")
    if new["metadata"]["checkpoint_sha256"] != after["checkpoint_sha256"]:
        raise ValueError("after benchmark and result use different checkpoints")
    if old["inputs"] != new["inputs"]:
        raise ValueError("before/after quartet inputs differ")
    base, fine = before["summary"]["neural"], after["summary"]["neural"]
    rows = []

    def comparison(label, a, b):
        row = {"case": label}
        for key in ("pearson", "ssim"):
            row[f"before_{key}"] = a[key]
            row[f"after_{key}"] = b[key]
            row[f"delta_{key}"] = b[key] - a[key]
        rows.append(row)

    comparison("independent_test_mean", {k: base[f"mean_{k}"] for k in ("pearson", "ssim")},
               {k: fine[f"mean_{k}"] for k in ("pearson", "ssim")})
    comparison("reserved_user_quartet_mean", {k: old[f"mean_{k}"] for k in ("pearson", "ssim")},
               {k: new[f"mean_{k}"] for k in ("pearson", "ssim")})
    for a, b in zip(old["planes"], new["planes"]):
        comparison(f"reserved_z_{b['distance_mm']:.1f}_mm", a, b)
    with (output / "before_after.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"training": {k: training[k] for k in ("config", "parent_checkpoint", "initial_validation_loss",
                 "best_validation_loss", "best_epoch", "elapsed_seconds", "parameter_l2_change_last_vs_initial")},
               "image_counts": dataset["image_counts"], "families": dataset["families"],
               "test_count": after["sample_count"], "comparisons": rows,
               "sampling": new["sampling"], "device": after["device"],
               "before_checkpoint_sha256": before["checkpoint_sha256"],
               "after_checkpoint_sha256": after["checkpoint_sha256"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    history = training["history"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs[1:], [row["train_loss"] for row in history[1:]], label="Train")
    ax.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    ax.axvline(training["best_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
    ax.set(xlabel="Fine-tuning epoch", ylabel="Physics loss", title="Training history (selection uses validation only)")
    ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    gs_keys = sorted((k for k in after["summary"] if k.startswith("gs_")), key=lambda k: int(k.split("_")[1]))
    labels = ["Before", "Fine-tuned", *[key.upper().replace("_", " ") for key in gs_keys]]
    records = [base, fine, *[after["summary"][key] for key in gs_keys]]
    for ax, key in zip(axes, ("pearson", "ssim")):
        values = [r[f"mean_{key}"] for r in records]
        bars = ax.bar(labels, values, color=["#8a939d", "#3179b8", "#70a46c", "#db8c44"])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set(ylim=(min(0, min(values) - 0.05), 1.0), ylabel=key.upper(), title=f"Independent textures: {key}")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "holdout_quality.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(9, 10), constrained_layout=True)
    for i in range(4):
        paths = [after_result / f"target_{i+1}.png", before_result / f"reconstruction_{i+1}.png", after_result / f"reconstruction_{i+1}.png"]
        for ax, path in zip(axes[i], paths):
            with Image.open(path) as im:
                ax.imshow(np.array(im), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        axes[i, 0].set_title(f"Target, z={new['planes'][i]['distance_mm']:.1f} mm")
        axes[i, 1].set_title(f"Before: r={old['planes'][i]['pearson']:.3f}, SSIM={old['planes'][i]['ssim']:.3f}", fontsize=9)
        axes[i, 2].set_title(f"After: r={new['planes'][i]['pearson']:.3f}, SSIM={new['planes'][i]['ssim']:.3f}", fontsize=9)
    fig.savefig(output / "before_after.png", dpi=160)
    plt.close(fig)

    table = "| Набор/плоскость | Pearson до | После | SSIM до | После |\n|---|---:|---:|---:|---:|\n"
    for row in rows:
        table += f"| {row['case']} | {row['before_pearson']:.5f} | {row['after_pearson']:.5f} | {row['before_ssim']:.5f} | {row['after_ssim']:.5f} |\n"
    markdown = f"""# Отчёт по дообучению на текстурах

Реальное дообучение весов исходной модели, а не обучение новой сети с нуля.
Устройство: **{after['device']}**. Эпох: {training['config']['epochs']}; выбран checkpoint
эпохи **{training['best_epoch']}** по минимуму validation loss. Adam запущен с новой
историей оптимизатора. Время обучения: {training['elapsed_seconds']:.1f} с.

Validation loss: **{training['initial_validation_loss']:.6f} → {training['best_validation_loss']:.6f}**.

## Данные и отсутствие утечки

Использован предоставленный ZIP: **262** текстуры 128×128, а не полный набор из 700.
Обучение: {dataset['image_counts']['train']} исходных изображений;
validation: {dataset['image_counts']['validation']}; test: {dataset['image_counts']['test']};
резервные семейства: {dataset['image_counts']['reserved']}.
Составлено {len(dataset['splits']['train'])}/{len(dataset['splits']['validation'])}/{len(dataset['splits']['test'])}
четвёрок. Семейства между split не пересекаются, проверены дубликаты пикселей.
Cracks/Gabor/Perlin/Voronoi полностью исключены из обучения и выбора checkpoint.
Train-аугментации: повороты на 90°, отражение, циклические сдвиги; все записаны в manifest.

## Результаты до и после

Одна и та же независимая выборка: **{after['sample_count']} четвёрки**.
Одинаковые входы, оптическая модель, 8-битное квантование ДОЭ и амплитуд.
Выбранная пользовательская четвёрка не участвовала в обучении.

{table}

![Обучение](learning_curves.png)

![Независимая проверка](holdout_quality.png)

![Четыре изображения: до и после](before_after.png)

## Физика, метрики и ограничения

λ=532 нм, входной шаг 10 мкм, 128×128, z=24.1/25.3/26.5/27.7 мм.
Один FFT на плоскость, сохраняется модуль поля, а не интенсивность.
Все реконструкции рассчитаны из сохранённого квантованного ДОЭ.
PNG каждой плоскости масштабирован по собственному максимуму без подгонки по target.
Pearson и локальный SSIM (окно 7×7) рассчитаны по фактическим PNG.
Время benchmark — batched throughput после прогрева, не статистическая оценка latency.
Повторяющиеся исходные изображения в разных тестовых четвёрках не считаются
независимыми новыми источниками: это 28 исходников из двух семейств, не 128 уникальных.
Высокое качество и преимущество над GS не гарантируются; сравнение показывает
фактическое изменение, включая ухудшения. Оптический эксперимент не проводился.

## Указанные пользователем ссылки

- [TensorFlow README](https://github.com/Jemuna/Image-Classification-Using-Deep-Learning-in-TensorFlow):
  прочитан; это описание классификации CIFAR-10, не набор голограмм. Код модели и
  данные CIFAR-10 не копировались и не использовались для этого дообучения.
- [OpenGameArt](https://opengameart.org/content/700-noise-textures):
  доступ из среды заблокирован сетевой политикой; использован только загруженный
  пользователем ZIP с 262 текстурами. Полный набор из 700 не скачивался.
- [ResearchGate](https://www.researchgate.net/figure/Aliasing-free-area-in-pixel-units-of-the-chirp-functions-under-the-calculation-condition_fig5_236235124):
  страница недоступна по сетевой политике, её содержимое не проверено.
  В код независимо добавлена диагностика частоты обеих квадратичных chirp-функций.
  Входная chirp проходит консервативную проверку Nyquist; выходная требует
  осторожности при интерполяции комплексной фазы. Она является поточечным множителем
  единичного модуля и не меняет сохраняемую амплитуду. Полная aliasing-free
  дискретизация обеих chirp-функций не заявляется.

Удалённый запуск в Kaggle не выполнен: в среде нет настроенной авторизации.
Лицензия входящего ZIP указана в его License.txt как CC0/Public Domain.

SHA256 до: `{before['checkpoint_sha256']}`.

SHA256 после: `{after['checkpoint_sha256']}`.
"""
    (output / "report.md").write_text(markdown)
    with PdfPages(output / "report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "Дообучение FresnelNet на текстурах", fontsize=18, weight="bold")
        paragraphs = [
            f"CPU: {after['device']}; {training['config']['epochs']} эпох; лучший checkpoint: эпоха {training['best_epoch']}. Время обучения: {training['elapsed_seconds']:.1f} с.",
            f"Validation physics loss: {training['initial_validation_loss']:.6f} → {training['best_validation_loss']:.6f}.",
            "Данные: 262 изображения из загруженного ZIP. 140 train, 28 validation, 28 test, 66 в резерве. Семейства не пересекаются. Полный набор из 700 не скачивался.",
            "Независимая проверка: 32 четвёрки из test. Пользовательские Cracks/Gabor/Perlin/Voronoi полностью исключены из обучения и выбора checkpoint.",
            f"Test Pearson: {base['mean_pearson']:.5f} → {fine['mean_pearson']:.5f}; SSIM: {base['mean_ssim']:.5f} → {fine['mean_ssim']:.5f}.",
            f"Выбранная четвёрка: Pearson {old['mean_pearson']:.5f} → {new['mean_pearson']:.5f}; SSIM {old['mean_ssim']:.5f} → {new['mean_ssim']:.5f}.",
            "Реконструкции из сохранённого 8-битного ДОЭ, модуль поля, без фильтрации и подгонки gain по target. Pearson и локальный SSIM измерены по реальным PNG.",
            "TensorFlow README описывает классификацию CIFAR-10; эти данные не использованы. OpenGameArt и ResearchGate недоступны по политике сети. Проверка chirp-дискретизации выполнена независимо по формулам модели.",
            "Качество не гарантируется; улучшения и ухудшения приведены без отбора. Тестовые четвёрки используют повторно 28 исходных текстур. GPU/Kaggle и эксперимент на ПВМС не проверены.",
        ]
        y = 0.89
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, width=91)
            fig.text(0.07, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.5)
            y -= len(lines) * 0.020 + 0.025
        pdf.savefig(fig); plt.close(fig)
        for filename, title, size in (("learning_curves.png", "История обучения", (8.27, 6)),
                                      ("holdout_quality.png", "Независимая проверка", (10, 6)),
                                      ("before_after.png", "Выбранная четвёрка: до и после", (8.27, 11.69))):
            fig, ax = plt.subplots(figsize=size)
            with Image.open(output / filename) as im:
                ax.imshow(np.array(im))
            ax.axis("off"); ax.set_title(title, fontsize=14)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(json.dumps({"reports": str(output), "comparisons": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "dataset", "before-benchmark", "after-benchmark", "before-result", "after-result", "output"):
        parser.add_argument("--" + name, required=True)
    generate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
