"""Проверки физической модели, квантования и метрик: python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doe.metrics import pearson, ssim, ssim_7x7, ssim_global  # noqa: E402
from doe.physics import FresnelSystem, phase_to_uint8, uint8_to_phase  # noqa: E402


def test_dechirped_equals_direct_formula():
    s = FresnelSystem()
    psi = (torch.rand(2, 128, 128) * 2 - 1) * math.pi
    a = s.reconstruct(psi)
    b = s.reconstruct_physical(s.to_physical(psi))
    assert torch.allclose(a, b, atol=1e-4)


def test_energy_conservation():
    s = FresnelSystem()
    psi = (torch.rand(1, 128, 128) * 2 - 1) * math.pi
    inten = s.reconstruct(psi).pow(2).mean(dim=(-2, -1))
    assert torch.allclose(inten, torch.ones_like(inten), atol=1e-4)


def test_phase_quantization_mapping():
    assert phase_to_uint8(torch.tensor([-math.pi]))[0] == 0
    assert phase_to_uint8(torch.tensor([math.pi - 1e-6]))[0] == 255
    assert abs(uint8_to_phase(torch.tensor([0], dtype=torch.uint8))[0] + math.pi) < 1e-6
    assert abs(uint8_to_phase(torch.tensor([255], dtype=torch.uint8))[0] - math.pi) < 1e-6


def test_output_sampling():
    s = FresnelSystem()
    dx = s.params.output_pixel()
    assert abs(dx[0] - 10.017e-6) < 1e-8 and abs(dx[-1] - 11.513e-6) < 1e-8


def test_metrics_identity_and_skimage():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    assert pearson(a, a) == pytest.approx(1.0)
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim_global(a, a) == pytest.approx(1.0)
    skm = pytest.importorskip("skimage.metrics")
    b = np.clip(a.astype(int) + rng.integers(-40, 40, a.shape), 0, 255).astype(np.uint8)
    ref = skm.structural_similarity(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                    data_range=255)
    assert ssim(a, b) == pytest.approx(ref, abs=1e-9)
    assert pearson(a, b) == pytest.approx(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert ssim_7x7(a, b) == pytest.approx(skm.structural_similarity(a, b, data_range=255), abs=1e-9)


def test_evaluate_normalizes_both_images_to_max():
    """Правило организаторов: перед сравнением каждое изображение нормируется на максимум."""
    from doe.metrics import evaluate
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (128, 128)).astype(np.float64)
    dim = np.round(a * 0.5).astype(np.uint8)          # тот же объект, но максимум ≈ 128
    row = evaluate([dim], [a.astype(np.uint8)])[0]
    assert row["pearson"] == pytest.approx(1.0, abs=1e-3)
    assert row["ssim"] > 0.99 and row["ssim_7x7"] > 0.99
