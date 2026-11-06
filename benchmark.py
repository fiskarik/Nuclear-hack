"""Сравнение нейросети с классическими итерационными методами.

Для каждого тестового набора (examples/*) и для N случайных синтетических
четвёрок (не участвовавших в обучении) считает средние КК и ИСС по 4
плоскостям и время синтеза ДОЭ. Метрики вычисляются по сохранённым 8-битным
PNG (ДОЭ квантуется в 256 уровней, восстановление нормируется на максимум).

    python benchmark.py --out results/benchmark --random 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doe.data import build_bank
from doe.imageio import load_gray
from doe.physics import FresnelSystem
from doe.pipeline import DEFAULT_WEIGHTS, load_model, save_result, synthesize

METHODS = {
    "НС, α=0 (макс. КК)": dict(method="nn", alpha=0.0),
    "НС, α=0.5": dict(method="nn", alpha=0.5),
    "НС, α=1 (макс. ИСС)": dict(method="nn", alpha=1.0),
    "НС α=0.5 + 100 ит. доводки": dict(method="nn", alpha=0.5, refine_iters=100),
    "ГС, 50 итераций": dict(method="gs", gs_iters=50),
    "ГС, 200 итераций": dict(method="gs", gs_iters=200),
    "Adam, 500 итераций, α=0.5": dict(method="adam", alpha=0.5, refine_iters=500),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--random", type=int, default=50, help="число случайных синтетических четвёрок")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--methods", default="", help="подстрока для отбора методов")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sets: dict[str, list[list[np.ndarray]]] = {}
    for d in sorted(Path(args.examples).iterdir()):
        files = [d / f"object_{i}.png" for i in range(1, 5)]
        if all(f.exists() for f in files):
            sets[d.name] = [[load_gray(f) for f in files]]
    if args.random:
        bank = build_bank(args.random * 4, seed=31337)  # seed не пересекается с обучением
        sets["random"] = [list(bank[i * 4:(i + 1) * 4]) for i in range(args.random)]

    model = load_model(args.weights, args.device)
    system = FresnelSystem().to(args.device)
    synthesize(sets[next(iter(sets))][0], "nn", model, device=args.device, system=system)  # прогрев

    methods = {k: v for k, v in METHODS.items() if args.methods in k}
    table = {}
    for mname, kw in methods.items():
        table[mname] = {}
        for sname, quads in sets.items():
            pcc, ssim, gss, secs = [], [], [], []
            for qi, imgs in enumerate(quads):
                res = synthesize(imgs, model=model, device=args.device, system=system, **kw)
                pcc.append(np.mean([m["pearson"] for m in res.metrics]))
                ssim.append(np.mean([m["ssim"] for m in res.metrics]))
                gss.append(np.mean([m["ssim_global"] for m in res.metrics]))
                secs.append(res.seconds)
                if sname != "random" and mname == "НС, α=0.5":
                    save_result(res, out / sname, imgs)
            table[mname][sname] = {"pearson": float(np.mean(pcc)), "ssim": float(np.mean(ssim)),
                                   "ssim_global": float(np.mean(gss)), "seconds": float(np.mean(secs))}
            print(f"{mname:30s} {sname:8s} КК={np.mean(pcc):.3f} ИСС={np.mean(ssim):.3f} "
                  f"ИССг={np.mean(gss):.3f} t={np.mean(secs) * 1000:.0f} мс", flush=True)
    (out / "benchmark.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))

    names = list(sets)
    lines = [f"Устройство: {args.device}. Значения — средние по 4 плоскостям (и по {args.random} четвёркам для random).",
             "", "| Метод | " + " | ".join(names) + " | время, мс |",
             "|---|" + "---|" * len(names) + "---|"]
    for mname, row in table.items():
        cells = [f"{row[s]['pearson']:.3f} / {row[s]['ssim']:.3f}" for s in names]
        t = np.mean([row[s]["seconds"] for s in names]) * 1000
        lines.append(f"| {mname} | " + " | ".join(cells) + f" | {t:.0f} |")
    lines.append("")
    lines.append("В ячейках: КК Пирсона / ИСС (SSIM, окно Гаусса 11×11).")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
