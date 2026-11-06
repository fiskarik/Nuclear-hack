"""Формирует отчёт о работе модели: результаты, сравнение, графики (results/).

    python scripts/make_report.py --weights weights/doe_net.pt --log weights/train_log.jsonl

Создаёт:
  results/<набор>/            — doe.png, recon_1…4.png, metrics.csv/json, preview.png (НС, α=0.5)
  results/benchmark.{md,json} — НС против Герчберга–Сэкстона и Adam (качество и время)
  results/tradeoff.png        — кривая компромисса КК/ИСС при изменении α
  results/training.png        — кривые обучения
  results/aliasing.png        — проверка условий дискретизации чирп-функций
  results/gallery_<набор>.png — исходные / НС / ГС
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doe.data import build_bank, load_cache, load_textures  # noqa: E402
from doe.imageio import load_gray  # noqa: E402
from doe.physics import FresnelSystem, OpticalParams  # noqa: E402
from doe.pipeline import load_model, save_result, synthesize  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def example_sets(root: Path) -> dict[str, list[np.ndarray]]:
    out = {}
    for d in sorted(root.iterdir()):
        files = [d / f"object_{i}.png" for i in range(1, 5)]
        if all(f.exists() for f in files):
            out[d.name] = [load_gray(f) for f in files]
    return out


def random_quads(n: int) -> dict[str, list[list[np.ndarray]]]:
    """Контрольные четвёрки, не участвовавшие в обучении."""
    rng = np.random.default_rng(2026)
    groups = {}
    synth = build_bank(n * 4, seed=31337)
    groups["синтетика"] = [list(synth[i * 4:(i + 1) * 4]) for i in range(n)]
    tex = load_textures("test")
    if len(tex):
        groups["текстуры (отлож.)"] = [list(tex[rng.choice(len(tex), 4, replace=False)]) for _ in range(n)]
    cif = load_cache("cifar_gray", "test")
    if len(cif):
        from PIL import Image

        up = lambda a: np.asarray(Image.fromarray(a).resize((128, 128), Image.BICUBIC))  # noqa: E731
        groups["CIFAR-10 (отлож.)"] = [[up(cif[j]) for j in rng.choice(len(cif), 4, replace=False)] for _ in range(n)]
    return groups


def mean_metrics(res_list):
    p = np.mean([np.mean([m["pearson"] for m in r.metrics]) for r in res_list])
    s = np.mean([np.mean([m["ssim"] for m in r.metrics]) for r in res_list])
    g = np.mean([np.mean([m["ssim_global"] for m in r.metrics]) for r in res_list])
    t = np.mean([r.seconds for r in res_list])
    return float(p), float(s), float(g), float(t)


def aliasing_figure(path: Path):
    p = OpticalParams()
    z = np.linspace(15e-3, 35e-3, 400)
    edge = (p.n / 2) * p.pixel
    f_edge = edge * p.pixel / (p.wavelength * z)              # циклов на пиксел у края апертуры
    free_half = p.wavelength * z / (2 * p.pixel ** 2)          # полуширина безалиасной зоны, пикселы
    z_crit = p.n * p.pixel ** 2 / p.wavelength
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(z * 1e3, f_edge, label="частота чирпа у края ДОЭ")
    ax[0].axhline(0.5, color="r", ls="--", label="предел Найквиста (0,5)")
    for zi in p.distances:
        fi = edge * p.pixel / (p.wavelength * zi)
        ax[0].plot(zi * 1e3, fi, "ko")
        ax[0].annotate(f"{zi * 1e3:.1f} мм\n{fi:.3f}", (zi * 1e3, fi), textcoords="offset points", xytext=(4, 6),
                       fontsize=7)
    ax[0].set_xlabel("z, мм")
    ax[0].set_ylabel("циклов / пиксел")
    ax[0].set_title("Локальная частота exp(jπr²/λz) на краю ДОЭ")
    ax[0].legend(fontsize=7)
    ax[1].plot(z * 1e3, np.minimum(free_half * 2, 400), label="ширина безалиасной зоны чирпа")
    ax[1].axhline(p.n, color="r", ls="--", label=f"размер ДОЭ N = {p.n}")
    ax[1].axvline(z_crit * 1e3, color="gray", ls=":", label=f"z_крит = N·Δu²/λ = {z_crit * 1e3:.2f} мм")
    for zi in p.distances:
        ax[1].plot(zi * 1e3, p.wavelength * zi / p.pixel ** 2, "ko")
    ax[1].set_xlabel("z, мм")
    ax[1].set_ylabel("пикселы")
    ax[1].set_title("Безалиасная область чирп-функции (λz/Δu²)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return z_crit


def training_figure(log_path: Path, path: Path):
    rows, evals = [], []
    for line in open(log_path):
        d = json.loads(line)
        (evals if "eval" in d else rows).append(d.get("eval", d))
    if not rows:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    st = np.array([r["step"] for r in rows])

    def smooth(v, k=9):
        v = np.asarray(v, float)
        return np.convolve(v, np.ones(k) / k, mode="valid") if len(v) > k else v

    ax[0].plot(st[:len(smooth([r["loss"] for r in rows]))], smooth([r["loss"] for r in rows]))
    ax[0].set_title("Функция потерь (обучение)")
    ax[0].set_xlabel("шаг")
    for key, lab in (("pcc", "КК"), ("ssim", "ИСС")):
        v = smooth([r[key] for r in rows])
        ax[1].plot(st[:len(v)], v, label=lab)
    ax[1].set_title("Метрики на обучающих батчах (α случайное)")
    ax[1].set_xlabel("шаг")
    ax[1].legend()
    if evals:
        es = [e["step"] for e in evals]
        keys = [k for k in evals[-1]["val"] if k.endswith("_0.5")]
        for k in keys:
            ax[2].plot(es, [e["val"][k]["pcc"] for e in evals], "-o", ms=2, label=f"КК {k[:-4]}")
            ax[2].plot(es, [e["val"][k]["ssim"] for e in evals], "--o", ms=2, label=f"ИСС {k[:-4]}")
        ax[2].set_title("Контроль (α = 0,5)")
        ax[2].set_xlabel("шаг")
        ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gallery(path: Path, name: str, imgs, res_nn, res_gs):
    fig, ax = plt.subplots(3, 5, figsize=(10, 6.3))
    for i in range(4):
        z = OpticalParams().distances[i] * 1e3
        ax[0, i].imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(f"объект {i + 1} ({z:.1f} мм)", fontsize=8)
        ax[1, i].imshow(res_nn.recons[i], cmap="gray", vmin=0, vmax=255)
        m = res_nn.metrics[i]
        ax[1, i].set_title(f"НС: КК {m['pearson']:.3f}, ИСС {m['ssim']:.3f}", fontsize=8)
        ax[2, i].imshow(res_gs.recons[i], cmap="gray", vmin=0, vmax=255)
        m = res_gs.metrics[i]
        ax[2, i].set_title(f"ГС-200: КК {m['pearson']:.3f}, ИСС {m['ssim']:.3f}", fontsize=8)
    ax[1, 4].imshow(res_nn.doe, cmap="twilight", vmin=0, vmax=255)
    ax[1, 4].set_title(f"ДОЭ (НС), {res_nn.seconds * 1e3:.0f} мс", fontsize=8)
    ax[2, 4].imshow(res_gs.doe, cmap="twilight", vmin=0, vmax=255)
    ax[2, 4].set_title(f"ДОЭ (ГС), {res_gs.seconds * 1e3:.0f} мс", fontsize=8)
    ax[0, 4].axis("off")
    ax[0, 4].text(0.0, 0.5, f"набор: {name}\nλ = 532 нм, Δu = 10 мкм\nz = 24,1 / 25,3 / 26,5 / 27,7 мм",
                  fontsize=8, va="center")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "doe_net.pt"))
    ap.add_argument("--log", default=str(ROOT / "weights" / "train_log.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--examples", default=str(ROOT / "examples"))
    ap.add_argument("--n", type=int, default=24, help="число контрольных четвёрок в каждой группе")
    ap.add_argument("--adam-iters", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    system = FresnelSystem().to(dev)
    model = load_model(a.weights, dev)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    summary = {"weights": str(a.weights), "config": ck["config"], "train_step": ck.get("step"),
               "params": int(sum(p.numel() for p in model.parameters())), "device": dev,
               "torch_threads": torch.get_num_threads()}

    # --- физика: дискретизация ---
    summary["z_crit_mm"] = aliasing_figure(out / "aliasing.png") * 1e3
    if Path(a.log).exists():
        training_figure(Path(a.log), out / "training.png")

    # --- тестовые наборы ---
    sets = example_sets(Path(a.examples))
    synthesize(next(iter(sets.values())), "nn", model, device=dev, system=system)  # прогрев
    per_set = {}
    for name, imgs in sets.items():
        r_nn = synthesize(imgs, "nn", model, 0.5, device=dev, system=system)
        r_gs = synthesize(imgs, "gs", gs_iters=200, device=dev, system=system)
        save_result(r_nn, out / name, imgs)
        gallery(out / f"gallery_{name}.png", name, imgs, r_nn, r_gs)
        per_set[name] = {"nn": r_nn.metrics, "gs": r_gs.metrics, "nn_ms": r_nn.seconds * 1e3,
                         "gs_ms": r_gs.seconds * 1e3}
        print(name, "НС", mean_metrics([r_nn])[:2], "ГС", mean_metrics([r_gs])[:2], flush=True)
    summary["examples"] = per_set

    # --- сравнение методов на контрольных группах ---
    groups = random_quads(a.n)
    methods = {
        "НС, α=0 (макс. КК)": dict(method="nn", model=model, alpha=0.0),
        "НС, α=0.5": dict(method="nn", model=model, alpha=0.5),
        "НС, α=1 (макс. ИСС)": dict(method="nn", model=model, alpha=1.0),
        "НС α=0.5 + 100 ит. Adam": dict(method="nn", model=model, alpha=0.5, refine_iters=100),
        "ГС, 50 итераций": dict(method="gs", gs_iters=50),
        "ГС, 200 итераций": dict(method="gs", gs_iters=200),
        f"Adam, {a.adam_iters} итераций, α=0.5": dict(method="adam", alpha=0.5, refine_iters=a.adam_iters),
    }
    bench = {}
    for mname, kw in methods.items():
        bench[mname] = {}
        for gname, quads in groups.items():
            qs = quads if "Adam" not in mname or "НС" in mname else quads[: max(4, len(quads) // 3)]
            res = [synthesize(q, device=dev, system=system, **kw) for q in qs]
            p, s, g, t = mean_metrics(res)
            bench[mname][gname] = {"pearson": p, "ssim": s, "ssim_global": g, "ms": t * 1e3, "n": len(qs)}
            print(f"{mname:28s} {gname:20s} КК={p:.3f} ИСС={s:.3f} t={t * 1e3:.0f} мс", flush=True)
    summary["benchmark"] = bench
    gn = list(groups)
    lines = ["| Метод | " + " | ".join(gn) + " | время на ДОЭ, мс |", "|---|" + "---|" * len(gn) + "---|"]
    for mname, row in bench.items():
        cells = [f"{row[g]['pearson']:.3f} / {row[g]['ssim']:.3f}" for g in gn]
        lines.append(f"| {mname} | " + " | ".join(cells) + f" | {np.mean([row[g]['ms'] for g in gn]):.0f} |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n\nВ ячейках: КК Пирсона / ИСС (SSIM).\n")

    # --- кривая компромисса α ---
    fig, ax = plt.subplots(1, len(gn), figsize=(4 * len(gn), 3.4))
    ax = np.atleast_1d(ax)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    summary["tradeoff"] = {}
    for gi, g in enumerate(gn):
        qs = groups[g][:12]
        pts = [mean_metrics([synthesize(q, "nn", model, al, device=dev, system=system) for q in qs]) for al in alphas]
        summary["tradeoff"][g] = {str(al): {"pearson": p[0], "ssim": p[1]} for al, p in zip(alphas, pts)}
        ax[gi].plot([p[0] for p in pts], [p[1] for p in pts], "-o", label="НС (α = 0 … 1)")
        for al, p in zip(alphas, pts):
            ax[gi].annotate(f"α={al}", (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for mname, mk in (("ГС, 200 итераций", "s"), (f"Adam, {a.adam_iters} итераций, α=0.5", "^"),
                          ("НС α=0.5 + 100 ит. Adam", "D")):
            r = bench[mname][g]
            ax[gi].plot(r["pearson"], r["ssim"], mk, label=mname)
        ax[gi].set_xlabel("КК Пирсона")
        ax[gi].set_ylabel("ИСС (SSIM)")
        ax[gi].set_title(g)
        ax[gi].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png")
    plt.close(fig)

    summary["generated"] = time.strftime("%Y-%m-%d %H:%M")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("отчёт:", out)


if __name__ == "__main__":
    main()
