"""Сравнение нескольких весов на одних и тех же отложенных данных (правило организаторов).

Метрики считаются по 8-битным PNG; исходное и восстановленное изображения перед
сравнением нормируются на максимум (doe.metrics.evaluate). Для каждой модели —
чистый прямой проход с --alpha auto.

    python scripts/compare_models.py --weights g4=weights/doe_net.pt ft=runs/.../doe_net.pt \
        --expert-dir path/to/expert_runs --out results/model_comparison.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doe.data import build_bank, load_cache, load_textures  # noqa: E402
from doe.imageio import load_gray  # noqa: E402
from doe.physics import FresnelSystem  # noqa: E402
from doe.pipeline import load_model, synthesize  # noqa: E402


def groups(n: int, expert_dir: str | None) -> dict[str, list[list[np.ndarray]]]:
    rng = np.random.default_rng(2026)
    out = {}
    tex = load_textures("test")
    if len(tex):
        out["текстуры (отлож.)"] = [list(tex[rng.choice(len(tex), 4, replace=False)]) for _ in range(n)]
    cif = load_cache("cifar_gray", "test")
    if len(cif):
        up = lambda a: np.asarray(Image.fromarray(a).resize((128, 128), Image.BICUBIC))  # noqa: E731
        out["CIFAR-10 (отлож.)"] = [[up(cif[j]) for j in rng.choice(len(cif), 4, replace=False)] for _ in range(n)]
    synth = build_bank(n * 4, seed=31337)
    out["синтетика"] = [list(synth[i * 4:(i + 1) * 4]) for i in range(n)]
    if expert_dir and os.path.isdir(expert_dir):
        cases = sorted(glob.glob(os.path.join(expert_dir, "case_*")))
        out["кейсы экспертов"] = [[load_gray(f"{c}/input/target_{i}.png") for i in range(1, 5)] for c in cases]
    q = ROOT / "examples" / "noise_quartet"
    if q.exists():
        out["ваша четвёрка"] = [[load_gray(q / f"object_{i}.png") for i in range(1, 5)]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="имя=путь")
    ap.add_argument("--expert-dir")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--select", choices=["weighted", "min", "sum"], default="weighted")
    ap.add_argument("--out", default=str(ROOT / "results" / "model_comparison.json"))
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    system = FresnelSystem()
    gs = groups(a.n, a.expert_dir)
    table = {}
    for spec in a.weights:
        name, path = spec.split("=", 1)
        model = load_model(path)
        table[name] = {}
        for g, quads in gs.items():
            res = [synthesize(q, "nn", model, "auto", system=system, select=a.select) for q in quads]
            p = float(np.mean([np.mean([m["pearson"] for m in r.metrics]) for r in res]))
            s = float(np.mean([np.mean([m["ssim"] for m in r.metrics]) for r in res]))
            s7 = float(np.mean([np.mean([m["ssim_7x7"] for m in r.metrics]) for r in res]))
            table[name][g] = {"pearson": p, "ssim": s, "ssim_7x7": s7, "n": len(quads)}
            print(f"{name:8s} {g:20s} КК={p:.3f} ИСС={s:.3f} ИСС7={s7:.3f} (n={len(quads)})", flush=True)
        main_groups = [g for g in gs if g not in ("синтетика", "ваша четвёрка")]
        table[name]["_score_min"] = float(np.mean([min(table[name][g]["pearson"], table[name][g]["ssim"])
                                                   for g in main_groups]))
        table[name]["_score_sum"] = float(np.mean([table[name][g]["pearson"] + table[name][g]["ssim"]
                                                   for g in main_groups]))
        print(f"{name:8s} score_min={table[name]['_score_min']:.4f} score_sum={table[name]['_score_sum']:.4f}",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
