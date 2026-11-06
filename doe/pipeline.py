"""Общий конвейер: 4 изображения → ДОЭ → восстановление → метрики."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .imageio import amplitude_to_uint8, save_gray
from .iterative import gerchberg_saxton
from .losses import tradeoff_loss
from .metrics import evaluate
from .model import DOENet
from .physics import FresnelSystem, phase_to_uint8, uint8_to_phase

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "doe_net.pt"


def load_model(path: str | Path = DEFAULT_WEIGHTS, device: str = "cpu") -> DOENet:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = DOENet(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def refine(system: FresnelSystem, target: torch.Tensor, psi: torch.Tensor, iters: int, alpha: float,
           lr: float = 0.03) -> torch.Tensor:
    """Дополнительная оптимизация фазы (Adam) от решения сети — опционально."""
    if iters <= 0:
        return psi
    a = torch.full((target.shape[0],), float(alpha), device=target.device)
    p = psi.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        tradeoff_loss(system.reconstruct(p), target, a)["loss"].backward()
        opt.step()
    return p.detach()


@dataclass
class Result:
    doe: np.ndarray                 # uint8 128×128, 0 ↔ −π, 255 ↔ +π
    recons: list[np.ndarray]        # 4 × uint8 128×128
    metrics: list[dict]
    seconds: float                  # время синтеза ДОЭ (без ввода-вывода)
    method: str


def synthesize(images: list[np.ndarray], method: str = "nn", model: DOENet | None = None,
               alpha: float = 0.5, refine_iters: int = 0, gs_iters: int = 200, norm: str = "max",
               device: str = "cpu", system: FresnelSystem | None = None) -> Result:
    system = system or FresnelSystem().to(device)
    target = torch.tensor(np.stack(images)[None] / 255.0, dtype=torch.float32, device=device)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    if method == "nn":
        if model is None:
            raise ValueError("нужна модель")
        with torch.no_grad():
            psi = model.synthesize(target, system, alpha)
        psi = refine(system, target, psi, refine_iters, alpha)
    elif method == "gs":
        psi = gerchberg_saxton(system, target, iters=gs_iters)
    elif method == "adam":
        psi = gerchberg_saxton(system, target, iters=50)
        psi = refine(system, target, psi, refine_iters or 500, alpha)
    else:
        raise ValueError(method)
    sync()
    seconds = time.perf_counter() - t0

    # ДОЭ в физическом виде → 8 бит; восстановление делается из сохранённого 8-битного ДОЭ
    doe = phase_to_uint8(system.to_physical(psi))[0].cpu()
    rec = system.reconstruct_physical(uint8_to_phase(doe)[None].to(device))[0].cpu().numpy()
    recons = [amplitude_to_uint8(rec[i], images[i], norm) for i in range(len(images))]
    return Result(doe.numpy(), recons, evaluate(images, recons), seconds, method)


def save_result(res: Result, out: str | Path, originals: list[np.ndarray] | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_gray(out / "doe.png", res.doe)
    for i, r in enumerate(res.recons, 1):
        save_gray(out / f"recon_{i}.png", r)
    import csv
    import json

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["plane", "distance_mm", "pearson", "ssim", "ssim_global"])
        w.writeheader()
        for row, z in zip(res.metrics, FresnelSystem().params.distances):
            w.writerow({"plane": row["plane"], "distance_mm": round(z * 1e3, 1),
                        **{k: f"{row[k]:.4f}" for k in ("pearson", "ssim", "ssim_global")}})
    summary = {
        "method": res.method,
        "synthesis_seconds": res.seconds,
        "planes": res.metrics,
        "mean": {k: float(np.mean([r[k] for r in res.metrics])) for k in ("pearson", "ssim", "ssim_global")},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if originals is not None:
        save_preview(out / "preview.png", originals, res)


def save_preview(path: str | Path, originals: list[np.ndarray], res: Result) -> None:
    """Сводная картинка: исходные / восстановленные / ДОЭ (для отчёта)."""
    from PIL import Image, ImageDraw

    n = originals[0].shape[0]
    pad = 4
    cols = len(originals)
    w = cols * (n + pad) + pad + n + pad
    h = 2 * (n + pad) + pad + 28
    canvas = Image.new("L", (w, h), 40)
    d = ImageDraw.Draw(canvas)
    for i, (o, r, m) in enumerate(zip(originals, res.recons, res.metrics)):
        x = pad + i * (n + pad)
        canvas.paste(Image.fromarray(o), (x, pad))
        canvas.paste(Image.fromarray(r), (x, 2 * pad + n))
        d.text((x, 2 * (n + pad) + 2), f"r={m['pearson']:.3f}", fill=255)
        d.text((x, 2 * (n + pad) + 14), f"SSIM={m['ssim']:.3f}", fill=255)
    x = pad + cols * (n + pad)
    canvas.paste(Image.fromarray(res.doe), (x, pad + (n + pad) // 2))
    d.text((x, 2 * (n + pad) + 2), "DOE", fill=255)
    canvas.save(path)
