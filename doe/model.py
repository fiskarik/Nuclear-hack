"""Нейросеть синтеза многоплоскостного фазового ДОЭ.

Архитектура — «физически информированная развёрнутая сеть» (deep unfolding):
модель Френеля встроена в сеть как фиксированный дифференцируемый слой,
а свёрточные сети решают только «локальные» подзадачи в плоскостях
изображений, где они работают лучше всего.

    цели A_1..A_4, α ──► U-Net-инициализатор ──► фазы θ_i в плоскостях изображений
                         │
                         ▼
        ψ⁰ = arg Σ_i P_i⁻¹[A_i e^{jθ_i}]                 (обратное распространение)
                         │
          ┌──────────────┴───────── × K стадий ──────────────────────┐
          │  F_i = P_i[e^{jψ}]                 (прямое распространение) │
          │  (w_i, m_i, δ_i) = CNN_k(A_i, |F_i|, A_i − |F_i|, α)        │
          │  G_i = (A_i e^{w_i} + σ(m_i)|F_i|) · e^{j(arg F_i + δ_i)}   │
          │  ψ = arg Σ_i P_i⁻¹[G_i]                                     │
          └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
                φ = wrap(ψ − π r²/(λ z_ref))   → ДОЭ 128×128

P_i — распространение до i-й плоскости (FFT с квадратичной фазой, см. physics.py).
Каждая стадия — обученный аналог итерации Герчберга–Сэкстона: сеть сама
решает, как перевзвесить целевые амплитуды (w), где оставить «свободу
амплитуды» для рассеяния шума (m, аналог MRAF) и как скорректировать фазу (δ).

α ∈ [0, 1] — параметр компромисса между метриками, подаётся на вход как
дополнительный канал: α = 0 — максимум коэффициента корреляции,
α = 1 — максимум индекса структурного сходства (тёмный фон).
Сеть обучается без учителя: функция потерь сравнивает |P_i[e^{jψ}]| с A_i.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .losses import fit_scale, tradeoff_loss
from .physics import FresnelSystem


def unit(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / (z.abs() + eps)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        groups = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, cin: int, cout: int, base: int = 32, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth)]
        self.inc = ConvBlock(cin, chs[0])
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(nn.Sequential(nn.AvgPool2d(2), ConvBlock(chs[i - 1], chs[i])))
        for i in range(depth - 1, 0, -1):
            self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.up_convs.append(ConvBlock(chs[i] + chs[i - 1], chs[i - 1]))
        self.out = nn.Conv2d(chs[0], cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        y = skips.pop()
        for up, conv in zip(self.ups, self.up_convs):
            y = conv(torch.cat([up(y), skips.pop()], 1))
        return self.out(y)


def forward_unit(system: FresnelSystem, slm: torch.Tensor) -> torch.Tensor:
    """Прямое распространение поля ДОЭ, заданного комплексной экспонентой (B,N,N) → (B,P,N,N)."""
    f = slm.unsqueeze(1) * system.rel_kernel
    return torch.fft.fftshift(torch.fft.fft2(f, norm="ortho"), dim=(-2, -1))


class DOENet(nn.Module):
    def __init__(self, num_planes: int = 4, base: int = 32, depth: int = 4, stages: int = 4,
                 stage_base: int = 24, stage_depth: int = 3, grad_feat: bool = False):
        super().__init__()
        p = num_planes
        self.config = dict(num_planes=p, base=base, depth=depth, stages=stages,
                           stage_base=stage_base, stage_depth=stage_depth, grad_feat=grad_feat)
        self.num_planes = p
        self.grad_feat = grad_feat  # подавать стадиям ∂L/∂|F_i| — «обучаемый градиентный спуск»
        self.use_amp = False  # fp16 только для свёрток (FFT всегда в fp32)
        self.init_net = UNet(p + 1, 2 * p, base, depth)
        cin = (4 if grad_feat else 3) * p + 1
        self.stage_nets = nn.ModuleList([UNet(cin, 3 * p, stage_base, stage_depth) for _ in range(stages)])
        # стартовое смещение «маски свободы амплитуды»: σ(−3) ≈ 0.05
        for net in self.stage_nets:
            nn.init.constant_(net.out.bias[p:2 * p], -3.0)

    def _run(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_amp and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                return net(x).float()
        return net(x)

    @staticmethod
    def loss_grad(mag: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Нормированный градиент функции потерь по модулям полей ∂L/∂|F_i| (B,P,N,N), без 2-го порядка."""
        with torch.enable_grad():
            m = mag.detach().float().requires_grad_(True)
            loss = tradeoff_loss(m, target.float(), alpha)["loss"] * m.shape[0]
            (g,) = torch.autograd.grad(loss, m)
        g = g / g.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-24).sqrt()
        return (g.clamp(-6, 6) / 2).detach()

    @staticmethod
    def norm_target(target: torch.Tensor) -> torch.Tensor:
        """Нормировка целей на единичную среднюю интенсивность (как у поля при FFT ortho)."""
        return target / target.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8).sqrt()

    def forward(self, target: torch.Tensor, system: FresnelSystem, alpha: torch.Tensor | float = 0.5
                ) -> list[torch.Tensor]:
        """target — амплитуды (B,P,N,N) в [0,1].

        Возвращает список комплексных полей ДОЭ e^{jψ} (B,N,N) после инициализации
        и после каждой стадии; итоговый ДОЭ — ``torch.angle(out[-1])``.
        """
        p = self.num_planes
        b, _, n, m = target.shape
        if not torch.is_tensor(alpha):
            alpha = torch.full((b,), float(alpha), device=target.device)
        a_map = (alpha.to(target.dtype).view(b, 1, 1, 1) * 2 - 1).expand(b, 1, n, m)
        amp = self.norm_target(target)
        tin = target * 2 - 1
        # --- инициализация: фазы в плоскостях изображений ---
        out = self._run(self.init_net, torch.cat([tin, a_map], 1))
        theta = unit(torch.complex(out[:, :p] + 1.0, out[:, p:]))  # нулевой выход ⇒ θ = 0
        slm = unit(system.backward_fields(amp * theta).sum(1))
        history = [slm]
        # --- обученные итерации ---
        for net in self.stage_nets:
            fields = forward_unit(system, slm)
            mag = fields.abs()
            mag_fit = fit_scale(mag, amp)
            feats = [tin, mag_fit / 4, (amp - mag_fit) / 4]
            if self.grad_feat:
                feats.append(self.loss_grad(mag, target, alpha))
            o = self._run(net, torch.cat(feats + [a_map], 1))
            w = 2.0 * torch.tanh(o[:, :p] / 2.0)
            mask = torch.sigmoid(o[:, p:2 * p])
            delta = torch.pi * torch.tanh(o[:, 2 * p:] / torch.pi)
            new_amp = amp * torch.exp(w) + mask * mag_fit
            g = new_amp * unit(fields) * torch.polar(torch.ones_like(delta), delta)
            slm = unit(system.backward_fields(g).sum(1))
            history.append(slm)
        return history

    @torch.no_grad()
    def synthesize(self, target: torch.Tensor, system: FresnelSystem, alpha: float = 0.5) -> torch.Tensor:
        """Дечирпированная фаза ψ (B,N,N) — результат одного прямого прохода сети."""
        return torch.angle(self.forward(target, system, alpha)[-1])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
