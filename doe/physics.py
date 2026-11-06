"""Численная модель восстановления изображений из фазового ДОЭ.

Прямой расчёт дифракции Френеля с одним преобразованием Фурье (слайд 2 задания):

    K(x, y) = exp(jkz)/(jλz) · exp(jπ(x²+y²)/(λz)) · FFT[ H(u, v) · exp(jπ(u²+v²)/(λz)) ]

Для фазового ДОЭ H(u, v) = exp(jφ(u, v)). Множители перед FFT — постоянная и
чисто фазовая функция, поэтому модуль поля в плоскости восстановления равен

    |K_i| = |FFT[ exp(jφ) · Q_i ]|,   Q_i = exp(jπ(u²+v²)/(λ z_i)).

Шаг сетки в плоскости восстановления dx_i = λ z_i / (N du); для z = 24,1…27,7 мм
это ≈ 10,0…11,5 мкм, т.е. каждое восстановленное изображение имеет 128×128 отсчётов.

«Дечирпированное» представление
--------------------------------
Квадратичная фаза Q_i на краю ДОЭ достигает ~100 рад и близка к частоте Найквиста
при z = 24,1 мм. Чтобы нейросети не приходилось воспроизводить эту быстро
осциллирующую функцию, вводится опорная плоскость 1/z_ref = mean(1/z_i):

    φ = ψ − π r²/(λ z_ref)          (wrap в [−π, π)),
    |K_i| = |FFT[ exp(jψ) · D_i ]|,  D_i = exp(jπ r²/λ · (1/z_i − 1/z_ref)).

Фаза D_i на краю апертуры не превышает ~±9 рад, поэтому ψ — гладкая
функция, удобная для свёрточных сетей и итерационных методов. Переход
φ ↔ ψ точный, итоговый ДОЭ всегда сохраняется в физическом виде φ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class OpticalParams:
    """Физические параметры задачи (слайд 5 / разд. 1 задания)."""

    n: int = 128                      # размер ДОЭ и изображений, пикселы
    pixel: float = 10e-6              # линейный размер пиксела ПВМС, м
    wavelength: float = 532e-9        # длина волны лазера, м
    distances: Sequence[float] = field(
        default_factory=lambda: (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3))  # м

    @property
    def num_planes(self) -> int:
        return len(self.distances)

    def output_pixel(self) -> list[float]:
        """Шаг сетки в каждой плоскости восстановления, м."""
        return [self.wavelength * z / (self.n * self.pixel) for z in self.distances]


def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """Приведение фазы к диапазону [−π, π)."""
    return torch.remainder(phase + math.pi, TWO_PI) - math.pi


class FresnelSystem(torch.nn.Module):
    """Дифференцируемая модель «ДОЭ → 4 плоскости восстановления».

    Все методы работают с батчами: фаза (B, N, N), поля (B, P, N, N).
    Используется нормировка FFT ``ortho`` — энергия сохраняется, поэтому
    средняя интенсивность в каждой плоскости равна 1 для фазового ДОЭ.
    """

    def __init__(self, params: OpticalParams | None = None):
        super().__init__()
        self.params = params or OpticalParams()
        p = self.params
        n = p.n
        coords = (torch.arange(n, dtype=torch.float64) - n // 2) * p.pixel
        r2 = coords[:, None] ** 2 + coords[None, :] ** 2
        z = torch.tensor(list(p.distances), dtype=torch.float64)
        inv_ref = (1.0 / z).mean()
        # Полная квадратичная фаза каждой плоскости: π r² / (λ z_i)
        chirp = math.pi * r2[None] / (p.wavelength * z[:, None, None])
        ref = math.pi * r2 * inv_ref / p.wavelength
        rel = chirp - ref[None]
        self.z_ref = float(1.0 / inv_ref)
        self.register_buffer("chirp_phase", chirp.float(), persistent=False)
        self.register_buffer("ref_phase", ref.float(), persistent=False)
        self.register_buffer("rel_phase", rel.float(), persistent=False)
        self.register_buffer("rel_kernel", torch.polar(torch.ones_like(rel), rel).to(torch.complex64),
                             persistent=False)

    # ------------------------------------------------------------------ #
    # Переход между физической фазой φ и дечирпированной ψ
    # ------------------------------------------------------------------ #
    def to_physical(self, psi: torch.Tensor) -> torch.Tensor:
        return wrap_phase(psi - self.ref_phase)

    def to_dechirped(self, phi: torch.Tensor) -> torch.Tensor:
        return wrap_phase(phi + self.ref_phase)

    # ------------------------------------------------------------------ #
    # Прямое и обратное распространение
    # ------------------------------------------------------------------ #
    def forward_fields(self, psi: torch.Tensor) -> torch.Tensor:
        """ψ (B,N,N) → комплексные поля в P плоскостях (B,P,N,N)."""
        slm = torch.polar(torch.ones_like(psi), psi).unsqueeze(1) * self.rel_kernel
        return torch.fft.fftshift(torch.fft.fft2(slm, norm="ortho"), dim=(-2, -1))

    def backward_fields(self, fields: torch.Tensor) -> torch.Tensor:
        """Поля в плоскостях (B,P,N,N) → вклады каждой плоскости в поле ДОЭ (B,P,N,N)."""
        slm = torch.fft.ifft2(torch.fft.ifftshift(fields, dim=(-2, -1)), norm="ortho")
        return slm * self.rel_kernel.conj()

    def reconstruct(self, psi: torch.Tensor) -> torch.Tensor:
        """Модули полей |K_i| для дечирпированной фазы ψ."""
        return self.forward_fields(psi).abs()

    @torch.no_grad()
    def reconstruct_physical(self, phi: torch.Tensor) -> torch.Tensor:
        """Прямой расчёт по формуле задания для физической фазы φ (B,N,N) → |K_i| (B,P,N,N).

        Используется для финальной проверки: вычисляется без дечирпирования,
        с полной квадратичной фазой exp(jπ(u²+v²)/(λz_i)).
        """
        h = torch.polar(torch.ones_like(phi), phi).unsqueeze(1)
        q = torch.polar(torch.ones_like(self.chirp_phase), self.chirp_phase)
        k = torch.fft.fftshift(torch.fft.fft2(h * q, norm="ortho"), dim=(-2, -1))
        return k.abs()


# ---------------------------------------------------------------------- #
# Квантование фазы в 8-битное изображение и обратно (разд. 2 задания)
# ---------------------------------------------------------------------- #
def phase_to_uint8(phi: torch.Tensor) -> torch.Tensor:
    """φ ∈ [−π, π] → 0…255 (0 ↔ −π, 255 ↔ +π)."""
    phi = wrap_phase(phi)
    return torch.round((phi + math.pi) / TWO_PI * 255.0).clamp(0, 255).to(torch.uint8)


def uint8_to_phase(img: torch.Tensor) -> torch.Tensor:
    """0…255 → φ ∈ [−π, π]."""
    return img.float() / 255.0 * TWO_PI - math.pi
