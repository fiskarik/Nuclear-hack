"""Centered single-FFT Fresnel propagation on its natural output grids."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class OpticsConfig:
    size: int = 128
    wavelength_m: float = 532e-9
    pixel_pitch_m: float = 10e-6
    distances_m: tuple[float, ...] = (0.0241, 0.0253, 0.0265, 0.0277)

    def __post_init__(self):
        if self.size < 8 or self.size % 2:
            raise ValueError("size must be even and >= 8")
        if not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0:
            raise ValueError("wavelength must be finite and positive")
        if not math.isfinite(self.pixel_pitch_m) or self.pixel_pitch_m <= 0:
            raise ValueError("pixel pitch must be finite and positive")
        if len(self.distances_m) != 4 or any(not math.isfinite(z) or z <= 0 for z in self.distances_m):
            raise ValueError("exactly four finite positive distances are required")

    @property
    def output_pixel_pitches_m(self):
        return tuple(self.wavelength_m * z / (self.size * self.pixel_pitch_m) for z in self.distances_m)

    def to_dict(self):
        return asdict(self)


def fft2c(field):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(field):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class Fresnel(nn.Module):
    """Unitary preconditioned Fresnel operator; physical() restores its prefactor.

    Input: phase [B,H,W], uniform unit-amplitude illumination.
    Output: complex field [B,4,H,W]. Each plane has its own output pixel pitch.
    The unitary normalization changes only one constant per plane, not its image.
    """

    def __init__(self, config=OpticsConfig(), dtype=torch.float32):
        super().__init__()
        self.config = config
        n, dx, lam = config.size, config.pixel_pitch_m, config.wavelength_m
        # Compute oscillatory chirps in double precision before optional casting.
        axis = (torch.arange(n, dtype=torch.float64) - n // 2) * dx
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        z = torch.tensor(config.distances_m, dtype=torch.float64)[:, None, None]
        input_phase = math.pi * (xx.square() + yy.square())[None] / (lam * z)
        output_axis = (torch.arange(n, dtype=torch.float64) - n // 2)[None] * (lam * z[:, 0, 0] / (n * dx))[:, None]
        output_phase = math.pi * (output_axis[:, :, None].square() + output_axis[:, None, :].square()) / (lam * z)
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.register_buffer("input_chirp", torch.polar(torch.ones_like(input_phase), input_phase).to(complex_dtype))
        self.register_buffer("output_chirp", torch.polar(torch.ones_like(output_phase), output_phase).to(complex_dtype))
        global_phase = 2 * math.pi * z / lam - math.pi / 2
        factor = n * dx**2 / (lam * z) * torch.polar(torch.ones_like(global_phase), global_phase)
        self.register_buffer("physical_factor", factor.to(complex_dtype))

    def field(self, field):
        if field.shape[-2:] != (self.config.size, self.config.size):
            raise ValueError("field size does not match optics configuration")
        return self.output_chirp * fft2c(field[:, None] * self.input_chirp)

    def forward(self, phase):
        return self.field(torch.polar(torch.ones_like(phase), phase))

    def adjoint(self, fields):
        """Per-plane unitary adjoint, without summing over planes."""
        return ifft2c(fields * self.output_chirp.conj()) * self.input_chirp.conj()

    def physical(self, phase):
        return self(phase) * self.physical_factor


def unit_complex(field, eps=1e-7):
    # Stable fallback at zeros, including blank regions in target back-projection.
    return torch.where(field.abs() > eps, field / field.abs().clamp_min(eps), torch.ones_like(field))


def normalize_targets(targets):
    if targets.ndim != 4 or targets.shape[1] != 4:
        raise ValueError("targets must have shape [batch,4,height,width]")
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("targets must be finite nonnegative amplitudes")
    rms = targets.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    if (rms <= 1e-8).any():
        raise ValueError("a completely black plane is incompatible with unit-amplitude phase-only illumination")
    return targets / rms
