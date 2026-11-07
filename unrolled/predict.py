"""Пакетный синтез фазовых ДОЭ — точка входа Docker-образа.

Вход (--input): папка, в которой
  * либо подпапки-комплекты, в каждой ровно 4 PNG 128×128 (8 бит, оттенки серого);
  * либо сами 4 PNG (один комплект).
Файлы комплекта сортируются по имени (естественный порядок: 1, 2, …, 10) и
сопоставляются расстояниям 24,1 / 25,3 / 26,5 / 27,7 мм.

Выход (--output), для каждого комплекта <имя>/:
  doe.png                — фазовый ДОЭ 128×128, 8 бит, grayscale (0 ↔ −π, 255 ↔ +π);
  recon_1.png…recon_4.png — модули полей, восстановленных из doe.png (Френель, одно БПФ);
  metrics.json, metrics.csv, preview.png;
и общий summary.csv: КК, ИСС и время синтеза по каждому комплекту.

    python predict.py --input /data/input --output /data/output
Без входной папки (или если она пуста) обрабатываются встроенные примеры examples/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from doe.imageio import load_gray  # noqa: E402
from doe.physics import FresnelSystem  # noqa: E402
from doe.pipeline import DEFAULT_WEIGHTS, load_model, save_result, synthesize  # noqa: E402

IMG_EXT = (".png",)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def find_sets(root: Path) -> list[tuple[str, list[Path]]]:
    """Комплекты: подпапки с ≥4 PNG или сама папка с ≥4 PNG."""
    if not root.is_dir():
        return []
    own = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT], key=natural_key)
    sets = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key):
        files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT], key=natural_key)
        if len(files) >= 4:
            sets.append((d.name, files[:4]))
            if len(files) > 4:
                print(f"  {d.name}: найдено {len(files)} PNG, взяты первые 4 по имени", flush=True)
        elif files:
            print(f"  {d.name}: пропущено — нужно 4 PNG, найдено {len(files)}", flush=True)
    if len(own) >= 4:
        sets.insert(0, (root.name or "set", own[:4]))
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.environ.get("DOE_INPUT", "/data/input"))
    ap.add_argument("--output", default=os.environ.get("DOE_OUTPUT", "/data/output"))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--alpha", default="fast",
                    help="fast (по умолчанию) — один проход при α=0,5 с защитой от «горячей точки»; "
                         "auto — 5 вариантов α и выбор лучшего; или число 0…1")
    ap.add_argument("--refine", type=int, default=0, help="итераций доводки после сети (0 — чистая сеть)")
    ap.add_argument("--threads", type=int, default=0, help="потоков CPU (0 — все)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-extras", action="store_true", help="сохранять только doe.png")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    alpha = a.alpha if a.alpha in ("fast", "auto") else float(a.alpha)

    inp, out = Path(a.input), Path(a.output)
    sets = find_sets(inp)
    if not sets:
        print(f"Во входной папке {inp} нет комплектов из 4 PNG — обрабатываю встроенные примеры", flush=True)
        sets = find_sets(ROOT / "examples")
        sets = [(n, f) for n, f in sets if n in ("noise_quartet", "textures_a", "textures_b", "textures_c",
                                                   "natural")] or sets
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = load_model(a.weights, a.device)
    system = FresnelSystem().to(a.device)
    synthesize([np.zeros((128, 128), np.uint8) + 1] * 4, "nn", model, 0.5, device=a.device, system=system)  # прогрев
    t_load = time.perf_counter() - t0
    print(f"Модель: {a.weights} ({model.config.get('stages')} стадий), загрузка и прогрев {t_load:.2f} с, "
          f"устройство {a.device}, потоков {torch.get_num_threads()}", flush=True)

    rows = []
    for name, files in sets:
        imgs = [load_gray(f) for f in files]
        res = synthesize(imgs, "nn", model, alpha, refine_iters=a.refine, device=a.device, system=system)
        dst = out / name
        if a.no_extras:
            from doe.imageio import save_gray

            save_gray(dst / "doe.png", res.doe)
        else:
            save_result(res, dst, imgs)
        (dst / "inputs.json").write_text(json.dumps(
            {"files_in_plane_order": [f.name for f in files], "distances_mm": [24.1, 25.3, 26.5, 27.7]},
            ensure_ascii=False, indent=1))
        p = float(np.mean([m["pearson"] for m in res.metrics]))
        s = float(np.mean([m["ssim"] for m in res.metrics]))
        s7 = float(np.mean([m["ssim_7x7"] for m in res.metrics]))
        rows.append({"set": name, "pearson": p, "ssim": s, "ssim_7x7": s7, "alpha": res.alpha,
                     "synthesis_ms": res.seconds * 1000})
        print(f"  {name:20s} КК={p:.3f}  ИСС={s:.3f}  ИСС7×7={s7:.3f}  α={res.alpha:g}  "
              f"{res.seconds * 1000:.0f} мс  → {dst / 'doe.png'}", flush=True)

    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("pearson", "ssim", "ssim_7x7", "synthesis_ms")}
        print(f"Среднее по {len(rows)} комплектам: КК={mean['pearson']:.3f}  ИСС={mean['ssim']:.3f}  "
              f"ИСС7×7={mean['ssim_7x7']:.3f}  время синтеза {mean['synthesis_ms']:.0f} мс/комплект", flush=True)
        print("КК и ИСС здесь — самоконтроль по нашей модели Френеля (оба изображения нормируются на максимум).")
    print(f"Готово: {out}")


if __name__ == "__main__":
    main()
