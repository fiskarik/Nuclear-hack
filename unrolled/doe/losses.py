"""Дифференцируемые функции потерь для обучения без учителя.

Восстановленное изображение сравнивается с целевым с точностью до общего
масштаба (яркость восстановления произвольна и при сохранении нормируется),
поэтому сначала находится оптимальный масштаб s = <R,A>/<R,R> для каждой
плоскости, затем считаются нормированная СКО, (1 − КК) и (1 − ИСС).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_C1, _C2 = 0.01 ** 2, 0.03 ** 2


def _gauss(win: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    x = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim_map(a: torch.Tensor, b: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM (Wang 2004) для тензоров (B,P,N,N) со шкалой [0,1]; окно — гауссово 11×11."""
    bsz, p, n, m = a.shape
    g = _gauss(win, sigma, a.device, a.dtype)
    kx = g.view(1, 1, 1, win).expand(p, 1, 1, win)
    ky = g.view(1, 1, win, 1).expand(p, 1, win, 1)

    def blur(x):
        return F.conv2d(F.conv2d(x, kx, groups=p), ky, groups=p)

    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a ** 2
    s_bb = blur(b * b) - mu_b ** 2
    s_ab = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + _C1) * (2 * s_ab + _C2)) / ((mu_a ** 2 + mu_b ** 2 + _C1) * (s_aa + s_bb + _C2))


def pearson_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """КК Пирсона по последним двум осям → (B,P)."""
    da = a - a.mean(dim=(-2, -1), keepdim=True)
    db = b - b.mean(dim=(-2, -1), keepdim=True)
    num = (da * db).sum(dim=(-2, -1))
    den = (da.pow(2).sum(dim=(-2, -1)) * db.pow(2).sum(dim=(-2, -1))).clamp_min(1e-12).sqrt()
    return num / den


def display_norm(recon: torch.Tensor, target: torch.Tensor, norm: str = "max") -> torch.Tensor:
    """Восстановление в шкале [0,1] так, как оно будет сохранено в PNG.

    max — деление на максимум; rms — RMS восстановления = RMS цели с насыщением на 1
    (правило из expert_runs); fit — масштаб по МНК без насыщения.
    """
    if norm == "max":
        return recon / recon.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    if norm == "rms":
        k = target.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() / \
            recon.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        return (recon * k).clamp(max=1.0)
    return fit_scale(recon, target)


def fit_scale(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    s = (recon * target).sum(dim=(-2, -1), keepdim=True) / recon.pow(2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return recon * s


def holo_loss(recon: torch.Tensor, target: torch.Tensor, w_mse: float = 1.0, w_pcc: float = 0.5,
              w_ssim: float = 0.5) -> dict[str, torch.Tensor]:
    """recon — модули полей (B,P,N,N); target — амплитуды в [0,1]."""
    r = fit_scale(recon, target)
    mse = (r - target).pow(2).mean(dim=(-2, -1)) / target.pow(2).mean(dim=(-2, -1)).clamp_min(1e-6)
    pcc = pearson_t(recon, target)
    ssim = ssim_map(r, target).mean(dim=(-2, -1))
    loss = w_mse * mse.mean() + w_pcc * (1 - pcc).mean() + w_ssim * (1 - ssim).mean()
    return {"loss": loss, "mse": mse.mean().detach(), "pcc": pcc.mean().detach(), "ssim": ssim.mean().detach()}


def tradeoff_loss(recon: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, w_mse: float = 0.1,
                  ssim_norm: str = "max") -> dict[str, torch.Tensor]:
    """Потери с параметром компромисса α (B,): (1−α)(1−КК) + α(1−ИСС) + w_mse·СКО.

    ssim_norm — правило сохранения PNG (см. display_norm): max | rms | fit. Для rms
    КК и СКО тоже считаются по насыщенному изображению — ровно как при оценке.
    """
    rs = display_norm(recon, target, ssim_norm)
    r = rs if ssim_norm == "rms" else fit_scale(recon, target)
    # официальное правило: исходное тоже нормируется на максимум перед сравнением (для ИСС)
    t_ssim = target if ssim_norm == "rms" else target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    mse = ((r - target).pow(2).mean(dim=(-2, -1)) / target.pow(2).mean(dim=(-2, -1)).clamp_min(1e-6)).mean(1)
    pcc = pearson_t(rs if ssim_norm == "rms" else recon, target).mean(1)
    ssim = ssim_map(rs, t_ssim).mean(dim=(-3, -2, -1))
    a = alpha.to(recon.dtype)
    loss = ((1 - a) * (1 - pcc) + a * (1 - ssim) + w_mse * mse).mean()
    return {"loss": loss, "mse": mse.mean().detach(), "pcc": pcc.mean().detach(), "ssim": ssim.mean().detach()}


def minmax_term(recon: torch.Tensor, target: torch.Tensor, sharpness: float = 20.0, norm: str = "max"
                ) -> torch.Tensor:
    """1 − softmin(КК, ИСС) для каждого примера (в шкале сохранения PNG), среднее по батчу."""
    r = display_norm(recon, target, norm)
    t = target if norm == "rms" else target / target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    pcc = pearson_t(r, t).mean(1)
    ssim = ssim_map(r, t).mean(dim=(-3, -2, -1))
    soft_min = -torch.logsumexp(torch.stack([-pcc, -ssim]) * sharpness, 0) / sharpness
    return (1 - soft_min).mean()
