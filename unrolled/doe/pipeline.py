"""Общий конвейер: 4 изображения → ДОЭ → восстановление → метрики."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .imageio import amplitude_to_uint8, save_gray
from .iterative import gerchberg_saxton
from .losses import display_norm, pearson_t, ssim_map, tradeoff_loss
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


def minmax_loss(recon: torch.Tensor, target: torch.Tensor, sharpness: float = 20.0, norm: str = "max"
                ) -> torch.Tensor:
    """−softmin(КК, ИСС): максимизирует худшую из двух метрик (в шкале сохранения PNG)."""
    r = display_norm(recon, target, norm)
    t = target if norm == "rms" else target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    pcc = pearson_t(r, t).mean()
    ssim = ssim_map(r, t).mean()
    return torch.logsumexp(torch.stack([-pcc, -ssim]) * sharpness, 0) / sharpness


def refine(system: FresnelSystem, target: torch.Tensor, psi: torch.Tensor, iters: int, alpha: float,
           lr: float = 0.03, objective: str = "tradeoff", norm: str = "max") -> torch.Tensor:
    """Дополнительная оптимизация фазы (Adam) от решения сети — опционально.

    objective="tradeoff" — (1−α)(1−КК) + α(1−ИСС) + 0,1·СКО; "minmax" — максимум худшей метрики.
    """
    if iters <= 0:
        return psi
    a = torch.full((target.shape[0],), float(alpha), device=target.device)
    p = psi.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        rec = system.reconstruct(p)
        loss = (minmax_loss(rec, target, norm=norm) if objective == "minmax"
                else tradeoff_loss(rec, target, a, ssim_norm=norm)["loss"])
        loss.backward()
        opt.step()
    return p.detach()


ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def nn_auto_alpha(model: DOENet, system: FresnelSystem, target: torch.Tensor,
                  grid: tuple[float, ...] = ALPHA_GRID, norm: str = "max", select: str = "weighted",
                  ssim_weight: float = 0.65) -> tuple[torch.Tensor, float]:
    """Один батч-проход сети при нескольких α; выбирается ДОЭ с максимальным min(КК, ИСС).

    Цели известны на этапе синтеза (это входные изображения), поэтому выбор делается по
    физической модели — без оптимизации и без подгонки результата.
    """
    k = len(grid)
    t = target.expand(k, -1, -1, -1)
    psi = model.synthesize(t, system, torch.tensor(grid, device=target.device))
    phi = system.to_physical(psi)
    phi = uint8_to_phase(phase_to_uint8(phi))  # оцениваем ровно то, что будет сохранено
    rec = system.reconstruct(system.to_dechirped(phi))
    tq = torch.round(t * 255) / 255
    rec = torch.round(display_norm(rec, tq, "rms" if norm == "rms" else "max") * 255) / 255
    if norm != "rms":
        tq = tq / tq.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    pcc, ssim = pearson_t(rec, tq).mean(1), ssim_map(rec, tq).mean(dim=(-3, -2, -1))
    if select == "min":
        score = torch.minimum(pcc, ssim)
    elif select == "sum":
        score = pcc + ssim
    else:  # weighted: КК и ИСС оцениваются независимо, ИСС «чуть важнее» (ответ эксперта)
        score = (1 - ssim_weight) * pcc + ssim_weight * ssim
    best = int(torch.argmax(score))
    return psi[best:best + 1], float(grid[best])


@dataclass
class Result:
    doe: np.ndarray                 # uint8 128×128, 0 ↔ −π, 255 ↔ +π
    recons: list[np.ndarray]        # 4 × uint8 128×128
    metrics: list[dict]
    seconds: float                  # время синтеза ДОЭ (без ввода-вывода)
    method: str
    alpha: float | None = None      # использованное α (для --alpha auto — выбранное)


def synthesize(images: list[np.ndarray], method: str = "nn", model: DOENet | None = None,
               alpha: float | str = 0.5, refine_iters: int = 0, gs_iters: int = 200, norm: str = "max",
               device: str = "cpu", system: FresnelSystem | None = None, objective: str = "tradeoff",
               select: str = "weighted", ssim_weight: float = 0.65) -> Result:
    """alpha — число из [0, 1] или "auto" (выбор по min(КК, ИСС), только для method="nn")."""
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
        if alpha == "auto":
            psi, alpha = nn_auto_alpha(model, system, target, norm=norm, select=select, ssim_weight=ssim_weight)
        else:
            with torch.no_grad():
                psi = model.synthesize(target, system, float(alpha))
        psi = refine(system, target, psi, refine_iters, float(alpha), objective=objective, norm=norm)
    elif method == "gs":
        psi = gerchberg_saxton(system, target, iters=gs_iters)
        alpha = None
    elif method == "adam":
        alpha = 0.5 if alpha == "auto" else float(alpha)
        psi = gerchberg_saxton(system, target, iters=50)
        psi = refine(system, target, psi, refine_iters or 500, alpha, objective=objective, norm=norm)
    else:
        raise ValueError(method)
    sync()
    seconds = time.perf_counter() - t0

    # ДОЭ в физическом виде → 8 бит; восстановление делается из сохранённого 8-битного ДОЭ
    doe = phase_to_uint8(system.to_physical(psi))[0].cpu()
    rec = system.reconstruct_physical(uint8_to_phase(doe)[None].to(device))[0].cpu().numpy()
    recons = [amplitude_to_uint8(rec[i], images[i], norm) for i in range(len(images))]
    return Result(doe.numpy(), recons, evaluate(images, recons), seconds, method,
                  None if alpha is None else float(alpha))


def save_result(res: Result, out: str | Path, originals: list[np.ndarray] | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_gray(out / "doe.png", res.doe)
    for i, r in enumerate(res.recons, 1):
        save_gray(out / f"recon_{i}.png", r)
    import csv
    import json

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["plane", "distance_mm", "pearson", "ssim", "ssim_7x7", "ssim_global"])
        w.writeheader()
        for row, z in zip(res.metrics, FresnelSystem().params.distances):
            w.writerow({"plane": row["plane"], "distance_mm": round(z * 1e3, 1),
                        **{k: f"{row[k]:.4f}" for k in ("pearson", "ssim", "ssim_7x7", "ssim_global")}})
    summary = {
        "method": res.method,
        "alpha": res.alpha,
        "synthesis_seconds": res.seconds,
        "planes": res.metrics,
        "mean": {k: float(np.mean([r[k] for r in res.metrics])) for k in ("pearson", "ssim", "ssim_7x7", "ssim_global")},
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
