"""Independent quadrature and gradient checks, beyond testing FFT against itself."""
import math

import numpy as np
import pytest
import torch

from doe.model import FresnelNet, physics_loss
from doe.optics import Fresnel, OpticsConfig


def test_fresnel_matches_direct_diffraction_integral():
    config = OpticsConfig(size=8)
    optics = Fresnel(config, dtype=torch.float64)
    rng = np.random.default_rng(17)
    phase = rng.uniform(-math.pi, math.pi, (8, 8))
    calculated = optics.physical(torch.tensor(phase)[None])[0].numpy()
    axis = (np.arange(8) - 4) * config.pixel_pitch_m
    u, v = np.meshgrid(axis, axis)
    for i, z in enumerate(config.distances_m):
        out_axis = (np.arange(8) - 4) * config.output_pixel_pitches_m[i]
        x, y = np.meshgrid(out_axis, out_axis)
        distance_squared = (x[:, :, None, None] - u)**2 + (y[:, :, None, None] - v)**2
        integrand = np.exp(1j * phase) * np.exp(1j * math.pi * distance_squared / (config.wavelength_m * z))
        direct = (integrand.sum(axis=(-2, -1)) * config.pixel_pitch_m**2
                  * np.exp(2j * math.pi * z / config.wavelength_m) / (1j * config.wavelength_m * z))
        np.testing.assert_allclose(calculated[i], direct, rtol=1e-8, atol=1e-8)


def test_adjoint_and_physical_power_with_different_output_pixel_sizes():
    torch.manual_seed(7)
    config = OpticsConfig(size=16)
    optics = Fresnel(config, dtype=torch.float64)
    field = torch.randn(2, 16, 16, dtype=torch.complex128)
    propagated = optics.field(field)
    arbitrary = torch.randn_like(propagated)
    lhs = (propagated.conj() * arbitrary).sum()
    rhs = (field[:, None].conj() * optics.adjoint(arbitrary)).sum()
    torch.testing.assert_close(lhs, rhs)
    phase = torch.rand(2, 16, 16, dtype=torch.float64)
    power = optics.physical(phase).abs().square().sum(dim=(-2, -1))
    pitches = torch.tensor(config.output_pixel_pitches_m, dtype=torch.float64)
    np.testing.assert_allclose((power * pitches.square()).numpy(), config.size**2 * config.pixel_pitch_m**2, rtol=1e-9)


def test_phase_gradient_matches_finite_difference():
    torch.manual_seed(20)
    optics = Fresnel(OpticsConfig(size=8), dtype=torch.float64)
    phase = torch.randn(1, 8, 8, dtype=torch.float64, requires_grad=True)
    targets = torch.rand(1, 4, 8, 8, dtype=torch.float64) + 0.1
    physics_loss(optics, phase, targets).backward()
    index = (0, 2, 3)
    step = 1e-5
    plus, minus = phase.detach().clone(), phase.detach().clone()
    plus[index] += step
    minus[index] -= step
    numerical = (physics_loss(optics, plus, targets) - physics_loss(optics, minus, targets)) / (2 * step)
    torch.testing.assert_close(phase.grad[index], numerical, rtol=1e-5, atol=1e-7)


def test_neural_weights_receive_finite_nonzero_gradients():
    torch.manual_seed(5)
    model = FresnelNet(OpticsConfig(size=16), stages=2, width=4)
    targets = torch.rand(2, 4, 16, 16) + 0.1
    loss = physics_loss(model.optics, model(targets), targets)
    loss.backward()
    gradient = model.corrector[-1].weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0
    assert torch.isfinite(model.plane_logits.grad).all()
    with pytest.raises(ValueError, match="black"):
        model(torch.zeros_like(targets))
