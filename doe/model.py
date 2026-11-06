"""A fixed-depth, trainable, physics-informed unrolled network."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .optics import Fresnel, OpticsConfig, normalize_targets, unit_complex


def initialization(optics, targets):
    # Deterministic random output phases reduce coherent bias. Same seed at test time.
    generator = torch.Generator().manual_seed(1701)
    phase = (torch.rand((4, optics.config.size, optics.config.size), generator=generator) * 2 - 1) * math.pi
    phase = phase.to(device=targets.device, dtype=targets.dtype)
    fields = targets * torch.polar(torch.ones_like(phase), phase)
    return unit_complex(optics.adjoint(fields).mean(dim=1))


class FresnelNet(nn.Module):
    """Six-to-eight optical projection layers with learned CNN phase corrections.

    Inference is one fixed-depth network forward pass, without optimizer steps.
    This is an unrolled neural method, not a claim of zero Fourier transforms.
    """

    def __init__(self, config=OpticsConfig(), stages=6, width=12):
        super().__init__()
        if stages < 1 or width < 1:
            raise ValueError("stages and width must be positive")
        self.optics = Fresnel(config)
        self.stages = stages
        self.width = width
        self.plane_logits = nn.Parameter(torch.zeros(stages, 4))
        # Start at the classical projection (rho=1); learn under/over-relaxation in (0,2).
        self.relax_logits = nn.Parameter(torch.zeros(stages))
        self.corrector = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def forward(self, targets):
        amplitude = normalize_targets(targets)
        field = initialization(self.optics, amplitude)
        for stage in range(self.stages):
            reconstructed = self.optics.field(field)
            constrained = amplitude * unit_complex(reconstructed)
            back = self.optics.adjoint(constrained)
            weights = self.plane_logits[stage].softmax(dim=0)[None, :, None, None]
            projected = unit_complex((weights * back).sum(dim=1))
            features = torch.stack((field.real, field.imag, projected.real, projected.imag), dim=1)
            correction = 0.35 * torch.tanh(self.corrector(features)[:, 0])
            relax = 2 * self.relax_logits[stage].sigmoid()
            field = unit_complex((1 - relax) * field + relax * projected)
            field = field * torch.polar(torch.ones_like(correction), correction)
        return torch.angle(field)


@torch.no_grad()
def gerchberg_saxton(optics, targets, steps=80):
    """Multi-plane alternating-projection reference, using the identical physics."""
    if steps < 1:
        raise ValueError("steps must be positive")
    amplitude = normalize_targets(targets)
    field = initialization(optics, amplitude)
    for _ in range(steps):
        reconstructed = optics.field(field)
        field = unit_complex(optics.adjoint(amplitude * unit_complex(reconstructed)).mean(dim=1))
    return torch.angle(field)


def physics_loss(optics, phase, targets):
    """Supervision-free amplitude loss; no ground-truth holograms are needed."""
    desired = normalize_targets(targets)
    amplitude = optics(phase).abs()
    loss = F.mse_loss(amplitude, desired)
    for scale in (2, 4):
        loss = loss + 0.15 * F.mse_loss(F.avg_pool2d(amplitude, scale), F.avg_pool2d(desired, scale))
    centered_a = amplitude - amplitude.mean(dim=(-2, -1), keepdim=True)
    centered_b = desired - desired.mean(dim=(-2, -1), keepdim=True)
    corr = (centered_a * centered_b).mean(dim=(-2, -1)) / (
        centered_a.square().mean(dim=(-2, -1)) * centered_b.square().mean(dim=(-2, -1))
    ).clamp_min(1e-10).sqrt()
    return loss + 0.1 * (1 - corr).mean()
