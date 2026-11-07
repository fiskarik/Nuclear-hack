"""Метрики качества восстановленных изображений (слайд 3 задания).

* ``pearson`` — коэффициент корреляции Пирсона (КК).
* ``ssim`` — индекс структурного сходства (ИСС) в стандартной форме
  Wang et al., 2004: локальные статистики в гауссовом окне 11×11, σ = 1,5,
  K1 = 0,01, K2 = 0,03, L = 255; итог — среднее по изображению.
  Совпадает с ``skimage.metrics.structural_similarity(..., gaussian_weights=True,
  sigma=1.5, use_sample_covariance=False, data_range=255)`` и MATLAB ``ssim``.
* ``ssim_7x7`` — SSIM как в ``skimage`` по умолчанию (окно 7×7, выборочная ковариация).
* ``ssim_global`` — формула со слайда 3, где средние, дисперсии и
  ковариация берутся по всему изображению (одно «окно» на весь кадр).

Все функции принимают 8-битные (или вещественные в шкале 0…255) массивы.

Официальное правило сравнения (ответ эксперта в чате хакатона, 06.11.2026):
сравниваются амплитуды |K|, и **каждое изображение перед сравнением нормируется
на максимум** (максимум становится 1). ``evaluate`` делает это и для исходного,
и для восстановленного изображения (в шкале 0…255: максимум → 255). На КК
нормировка не влияет, на ИСС — влияет, если максимум исходного меньше 255.
"""
from __future__ import annotations

import numpy as np

_K1, _K2, _L = 0.01, 0.03, 255.0
_C1, _C2 = (_K1 * _L) ** 2, (_K2 * _L) ** 2


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da, db = a - a.mean(), b - b.mean()
    den = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / den) if den > 0 else 0.0


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return float((2 * ma * mb + _C1) * (2 * cov + _C2) / ((ma ** 2 + mb ** 2 + _C1) * (va + vb + _C2)))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _filter_valid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Сепарабельная свёртка с отбрасыванием краёв (как в skimage)."""
    tmp = np.stack([np.convolve(row, g, mode="valid") for row in img], axis=0)
    return np.stack([np.convolve(col, g, mode="valid") for col in tmp.T], axis=1)


def ssim(a: np.ndarray, b: np.ndarray, win: int = 11, sigma: float = 1.5) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = _gaussian_kernel(win, sigma)
    mu_a = _filter_valid(a, g)
    mu_b = _filter_valid(b, g)
    s_aa = _filter_valid(a * a, g) - mu_a ** 2
    s_bb = _filter_valid(b * b, g) - mu_b ** 2
    s_ab = _filter_valid(a * b, g) - mu_a * mu_b
    num = (2 * mu_a * mu_b + _C1) * (2 * s_ab + _C2)
    den = (mu_a ** 2 + mu_b ** 2 + _C1) * (s_aa + s_bb + _C2)
    # skimage обрезает поле на (win-1)//2 с каждой стороны — «valid» уже это делает
    return float((num / den).mean())


def ssim_7x7(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """SSIM как ``skimage.metrics.structural_similarity`` по умолчанию: равномерное окно 7×7,
    выборочная ковариация (N/(N−1)), L = 255. Для сопоставимости с другими протоколами."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = np.full(win, 1.0 / win)
    cov_norm = win * win / (win * win - 1.0)
    mu_a, mu_b = _filter_valid(a, g), _filter_valid(b, g)
    s_aa = cov_norm * (_filter_valid(a * a, g) - mu_a ** 2)
    s_bb = cov_norm * (_filter_valid(b * b, g) - mu_b ** 2)
    s_ab = cov_norm * (_filter_valid(a * b, g) - mu_a * mu_b)
    num = (2 * mu_a * mu_b + _C1) * (2 * s_ab + _C2)
    den = (mu_a ** 2 + mu_b ** 2 + _C1) * (s_aa + s_bb + _C2)
    return float((num / den).mean())


def normalize_max(img: np.ndarray) -> np.ndarray:
    """Нормировка на максимум (максимум → 255), как требуют организаторы."""
    img = np.asarray(img, dtype=np.float64)
    m = img.max()
    return img * (255.0 / m) if m > 0 else img


def evaluate(originals: list[np.ndarray], recons: list[np.ndarray]) -> list[dict]:
    """Попарное сравнение исходных и восстановленных изображений (оба нормируются на максимум)."""
    rows = []
    for i, (a, b) in enumerate(zip(originals, recons)):
        a, b = normalize_max(a), normalize_max(b)
        rows.append({
            "plane": i + 1,
            "pearson": pearson(a, b),
            "ssim": ssim(a, b),
            "ssim_global": ssim_global(a, b),
            "ssim_7x7": ssim_7x7(a, b),
        })
    return rows
