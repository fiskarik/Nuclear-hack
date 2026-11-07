"""Синтез фазового ДОЭ по 4 объектным изображениям обученной нейросетью.

Вход: 4 PNG (128×128, 8 бит, оттенки серого) — изображения для плоскостей
24,1; 25,3; 26,5; 27,7 мм (в этом порядке).
Выход (в папке --out):
    doe.png                 — фазовый ДОЭ 128×128, 8 бит (0 ↔ −π, 255 ↔ +π);
    recon_1.png … recon_4.png — модули полей, восстановленных из doe.png
                              прямым расчётом дифракции Френеля (одно БПФ);
    metrics.csv / metrics.json — КК Пирсона и ИСС (SSIM) для каждой пары;
    preview.png             — сводная картинка.

Примеры:
    python synthesize.py examples/digits/object_{1,2,3,4}.png --out results/digits
    python synthesize.py --input-dir examples/letters --out results/letters --alpha 0.5
    python synthesize.py --input-dir my_images --method gs     # классический ГС для сравнения
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from doe.imageio import load_gray
from doe.pipeline import DEFAULT_WEIGHTS, load_model, save_result, synthesize


def collect_inputs(args) -> list[Path]:
    if args.images:
        files = [Path(p) for p in args.images]
    elif args.input_dir:
        d = Path(args.input_dir)
        files = [d / f"object_{i}.png" for i in range(1, 5)]
        if not all(f.exists() for f in files):
            files = sorted(d.glob("*.png"))[:4]
    else:
        sys.exit("укажите 4 PNG-файла или --input-dir")
    if len(files) != 4:
        sys.exit(f"нужно ровно 4 изображения, получено {len(files)}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="4 PNG-файла в порядке плоскостей 1…4")
    ap.add_argument("--input-dir", help="папка с object_1.png…object_4.png (или любыми 4 PNG)")
    ap.add_argument("--out", default="results/output")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--method", choices=["nn", "gs", "adam"], default="nn",
                    help="nn — нейросеть (по умолчанию); gs/adam — итерационные методы для сравнения")
    ap.add_argument("--alpha", default="auto",
                    help="компромисс метрик: 0 — максимум КК, 1 — максимум ИСС (тёмный фон); "
                         "auto (по умолчанию) — сеть считает 5 вариантов α за один проход и выбирает "
                         "ДОЭ с максимальным min(КК, ИСС) по физической модели")
    ap.add_argument("--refine", type=int, default=0,
                    help="число итераций дополнительной градиентной доводки после сети (0 — чистая сеть)")
    ap.add_argument("--select", choices=["weighted", "min", "sum"], default="weighted",
                    help="критерий --alpha auto: weighted — максимум (1−w)·КК + w·ИСС (по умолчанию, w=0,65: "
                         "метрики оцениваются независимо, ИСС чуть важнее); min — максимум min(КК, ИСС); sum")
    ap.add_argument("--ssim-weight", type=float, default=0.65, help="вес ИСС для --select weighted")
    ap.add_argument("--objective", choices=["tradeoff", "minmax"], default="minmax",
                    help="цель доводки: minmax — максимум худшей из метрик, tradeoff — взвешенная по α")
    ap.add_argument("--gs-iters", type=int, default=200)
    ap.add_argument("--norm", choices=["max", "fit", "rms"], default="max",
                    help="сохранение модуля: max — нормировка на максимум; fit — масштаб МНК к исходному; "
                         "rms — RMS восстановления = RMS исходного (как в expert_runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.alpha != "auto":
        args.alpha = float(args.alpha)

    files = collect_inputs(args)
    images = [load_gray(f) for f in files]
    model = load_model(args.weights, args.device) if args.method == "nn" else None
    if model is not None:  # прогрев (первый вызов медленнее из-за инициализации)
        synthesize(images, "nn", model, 0.5, device=args.device)
    res = synthesize(images, args.method, model, alpha=args.alpha, refine_iters=args.refine,
                     gs_iters=args.gs_iters, norm=args.norm, device=args.device, objective=args.objective,
                     select=args.select, ssim_weight=args.ssim_weight)
    save_result(res, args.out, images)

    print(f"Метод: {args.method}" + (f" (α={res.alpha:g}{' — выбрано автоматически' if args.alpha == 'auto' else ''}, "
                                     f"доводка={args.refine})" if args.method == "nn" else ""))
    print(f"Время синтеза ДОЭ: {res.seconds * 1000:.1f} мс ({args.device})")
    print(f"{'плоскость':>9} {'z, мм':>6} {'КК Пирсона':>11} {'ИСС(SSIM)':>10} {'ИСС 7×7':>9} {'ИСС глоб.':>10}  файл")
    zs = [24.1, 25.3, 26.5, 27.7]
    for f, m, z in zip(files, res.metrics, zs):
        print(f"{m['plane']:>9} {z:>6} {m['pearson']:>11.4f} {m['ssim']:>10.4f} {m['ssim_7x7']:>9.4f} "
              f"{m['ssim_global']:>10.4f}  {f.name}")
    mean = {k: sum(m[k] for m in res.metrics) / 4 for k in ("pearson", "ssim", "ssim_7x7", "ssim_global")}
    print(f"{'среднее':>9} {'':>6} {mean['pearson']:>11.4f} {mean['ssim']:>10.4f} {mean['ssim_7x7']:>9.4f} "
          f"{mean['ssim_global']:>10.4f}")
    print(f"Результаты сохранены в {args.out}/")


if __name__ == "__main__":
    main()
