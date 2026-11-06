"""Differentiable counterparts of the reported Pearson and local SSIM metrics."""
import math

import torch
from torch.nn import functional as F


class _SoftPeakGradient(torch.autograd.Function):
    """Exact maximum forward; distributed logsumexp surrogate in backward only."""
    @staticmethod
    def forward(ctx,amplitude,temperature):
        weights=(amplitude.flatten(2)/temperature).softmax(dim=-1).reshape_as(amplitude)
        ctx.save_for_backward(weights)
        return amplitude.amax(dim=(-2,-1),keepdim=True)

    @staticmethod
    def backward(ctx,gradient):
        weights,=ctx.saved_tensors
        return gradient*weights,None


def ssim_per_plane(a, b, window=7, return_map=False):
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
    if return_map:return score[:,0].reshape(*shape,*score.shape[-2:])
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


def display_amplitudes(optics, phase, quantize=True,max_temperature=0):
    actual_phase = quantized_phase_ste(phase) if quantize else phase
    amplitude = optics.physical(actual_phase).abs()
    peak=(_SoftPeakGradient.apply(amplitude,max_temperature) if max_temperature>0
          else amplitude.amax(dim=(-2,-1),keepdim=True))
    display = amplitude / peak.clamp_min(1e-8)
    if quantize:
        rounded = (display*255).round() / 255
        display = display + (rounded-display).detach()
    return display


def perceptual_loss_per_item(optics, phase, targets, quantize=True,max_temperature=0):
    display = display_amplitudes(optics, phase, quantize=quantize,max_temperature=max_temperature)
    ssim = ssim_per_plane(display, targets)
    pearson = pearson_per_plane(display, targets)
    # Constant target planes have undefined Pearson; omit its term for those planes.
    varying = targets.var(dim=(-2, -1), unbiased=False) > 1e-12
    correlation_loss = ((1-pearson)*varying).sum(dim=1) / varying.sum(dim=1).clamp_min(1)
    mse = (display-targets).square().mean(dim=(1, 2, 3))
    return 0.5*(1-ssim).mean(dim=1) + 0.5*correlation_loss + 0.05*mse


def robust_quality_loss_per_item(optics,phase,targets,quantize=True,max_temperature=0):
    """Logarithmic quality penalties emphasize weak-contrast and badly matched windows.

    Used only as an optimization objective; final PNG metrics are unchanged.
    Softplus prevents zero gradients or undefined logs for negative initial SSIM.
    """
    display=display_amplitudes(optics,phase,quantize=quantize,max_temperature=max_temperature)
    local=ssim_per_plane(display,targets,return_map=True)
    correlation=pearson_per_plane(display,targets)
    varying=targets.var(dim=(-2,-1),unbiased=False)>1e-12
    pcc_penalty=-torch.log(F.softplus(10*correlation)/10+1e-8)
    pcc_loss=(pcc_penalty*varying).sum(dim=1)/varying.sum(dim=1).clamp_min(1)
    ssim_loss=-torch.log(F.softplus(20*local)/20+1e-8).mean(dim=(1,2,3))
    mse=(display-targets).square().mean(dim=(1,2,3))
    return .5*pcc_loss+.5*ssim_loss+.05*mse
