"""Differentiable counterparts of the reported Pearson and local SSIM metrics."""
import math

import torch
from torch.nn import functional as F


def ssim_per_plane(a, b, window=7):
    """Match skimage SSIM: uniform valid window, sample covariance, data_range=1."""
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError("SSIM expects equally sized [B,planes,H,W] tensors")
    if window % 2 != 1 or min(a.shape[-2:]) < window:
        raise ValueError("window must be odd and fit the images")
    shape = a.shape[:2]
    x, y = a.flatten(0, 1)[:, None], b.flatten(0, 1)[:, None]
    mean_x, mean_y = F.avg_pool2d(x, window, stride=1), F.avg_pool2d(y, window, stride=1)
    correction = window**2 / (window**2 - 1)
    var_x = correction * (F.avg_pool2d(x*x, window, stride=1) - mean_x.square())
    var_y = correction * (F.avg_pool2d(y*y, window, stride=1) - mean_y.square())
    cov = correction * (F.avg_pool2d(x*y, window, stride=1) - mean_x*mean_y)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2*mean_x*mean_y+c1) * (2*cov+c2) /
             ((mean_x.square()+mean_y.square()+c1) * (var_x+var_y+c2)))
    return score.mean(dim=(1, 2, 3)).reshape(shape)


def pearson_per_plane(a, b):
    x = a - a.mean(dim=(-2, -1), keepdim=True)
    y = b - b.mean(dim=(-2, -1), keepdim=True)
    numerator = (x*y).mean(dim=(-2, -1))
    denominator = (x.square().mean(dim=(-2, -1)) * y.square().mean(dim=(-2, -1))).clamp_min(1e-16).sqrt()
    return numerator / denominator


def quantized_phase_ste(phase):
    """Forward: actual 8-bit phase. Backward: straight-through derivative."""
    wrapped = torch.atan2(phase.sin(), phase.cos())
    precise = wrapped.to(torch.float64).clamp(-math.pi, math.pi)
    code = torch.round((precise + math.pi) * (255 / (2*math.pi)))
    decoded = (code * (2*math.pi/255) - math.pi).to(phase.dtype)
    return wrapped + (decoded-wrapped).detach()


def display_amplitudes(optics, phase, quantize=True):
    actual_phase = quantized_phase_ste(phase) if quantize else phase
    amplitude = optics.physical(actual_phase).abs()
    display = amplitude / amplitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    if quantize:
        rounded = (display*255).round() / 255
        display = display + (rounded-display).detach()
    return display


def perceptual_loss_per_item(optics, phase, targets, quantize=True):
    display = display_amplitudes(optics, phase, quantize=quantize)
    ssim = ssim_per_plane(display, targets)
    pearson = pearson_per_plane(display, targets)
    # Constant target planes have undefined Pearson; omit its term for those planes.
    varying = targets.var(dim=(-2, -1), unbiased=False) > 1e-12
    correlation_loss = ((1-pearson)*varying).sum(dim=1) / varying.sum(dim=1).clamp_min(1)
    mse = (display-targets).square().mean(dim=(1, 2, 3))
    return 0.5*(1-ssim).mean(dim=1) + 0.5*correlation_loss + 0.05*mse
