"""Классические итерационные методы — базовая линия для сравнения с нейросетью.

* ``gerchberg_saxton`` — многоплоскостной (глобальный) алгоритм Герчберга–Сэкстона:
  на каждой итерации поле распространяется во все P плоскостей, амплитуда
  заменяется целевой (фаза сохраняется), поля возвращаются в плоскость ДОЭ,
  усредняются, и остаётся только фаза.
* ``gradient_descent`` — прямая оптимизация фазы ДОЭ методом Adam
  через ту же дифференцируемую модель и ту же функцию потерь, что и у сети
  (верхняя оценка достижимого качества при данном числе итераций).
"""
from __future__ import annotations

import torch

from .losses import holo_loss
from .physics import FresnelSystem


def _unit(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return z / (z.abs() + eps)


@torch.no_grad()
def gerchberg_saxton(system: FresnelSystem, target: torch.Tensor, iters: int = 200,
                     psi0: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    """target — амплитуды (B,P,N,N) в [0,1]; возвращает дечирпированную фазу ψ (B,N,N)."""
    b, _, n, _ = target.shape
    if psi0 is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        psi0 = (torch.rand(b, n, n, generator=g) * 2 - 1).mul(torch.pi).to(target.device)
    slm = torch.polar(torch.ones_like(psi0), psi0)
    # нормируем цели на одинаковую энергию (FFT ortho сохраняет энергию)
    amp = target / target.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
    for _ in range(iters):
        fields = system.forward_fields(torch.angle(slm))
        fields = amp * _unit(fields)
        slm = _unit(system.backward_fields(fields).sum(1))
    return torch.angle(slm)


def gradient_descent(system: FresnelSystem, target: torch.Tensor, iters: int = 300, lr: float = 0.05,
                     psi0: torch.Tensor | None = None, seed: int = 0, loss_kwargs: dict | None = None
                     ) -> torch.Tensor:
    b, _, n, _ = target.shape
    if psi0 is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        psi0 = (torch.rand(b, n, n, generator=g) * 2 - 1).mul(torch.pi).to(target.device)
    psi = psi0.clone().requires_grad_(True)
    opt = torch.optim.Adam([psi], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = holo_loss(system.reconstruct(psi), target, **(loss_kwargs or {}))["loss"]
        loss.backward()
        opt.step()
    return psi.detach()
