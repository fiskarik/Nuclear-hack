import math

import numpy as np
import torch

from doe.camera_optics import CameraConfig,CameraFresnel
from doe.optics import Fresnel,OpticsConfig


def test_padding_one_matches_original_model():
    torch.manual_seed(60)
    a=Fresnel(OpticsConfig(size=16),dtype=torch.float64)
    b=CameraFresnel(CameraConfig(size=16,fft_padding=1),dtype=torch.float64)
    phase=torch.rand(2,16,16,dtype=torch.float64)
    torch.testing.assert_close(a.physical(phase),b.physical(phase),rtol=0,atol=0)


def test_finer_camera_matches_independent_direct_diffraction_sum():
    cfg=CameraConfig(size=8,fft_padding=2)
    optics=CameraFresnel(cfg,dtype=torch.float64)
    phase=np.random.default_rng(61).uniform(-math.pi,math.pi,(8,8))
    actual=optics.physical(torch.tensor(phase)[None])[0].numpy()
    axis=(np.arange(8)-4)*cfg.pixel_pitch_m
    u,v=np.meshgrid(axis,axis)
    for i,z in enumerate(cfg.distances_m):
        camera=(np.arange(8)-4)*cfg.output_pixel_pitches_m[i]
        x,y=np.meshgrid(camera,camera)
        r2=(x[:,:,None,None]-u)**2+(y[:,:,None,None]-v)**2
        direct=(np.exp(1j*phase)*np.exp(1j*math.pi*r2/(cfg.wavelength_m*z))).sum((-2,-1))
        direct*=cfg.pixel_pitch_m**2*np.exp(2j*math.pi*z/cfg.wavelength_m)/(1j*cfg.wavelength_m*z)
        np.testing.assert_allclose(actual[i],direct,rtol=1e-8,atol=1e-8)


def test_camera_adjoint_and_full_energy_are_correct():
    torch.manual_seed(62)
    cfg=CameraConfig(size=16,fft_padding=2)
    optics=CameraFresnel(cfg,dtype=torch.float64)
    source=torch.randn(2,16,16,dtype=torch.complex128)
    observed=torch.randn(2,4,16,16,dtype=torch.complex128)
    lhs=(optics.field(source).conj()*observed).sum()
    rhs=(source[:,None].conj()*optics.adjoint(observed)).sum()
    torch.testing.assert_close(lhs,rhs)
    phase=torch.rand(2,16,16,dtype=torch.float64)
    field=optics.full_physical(phase)
    pitches=torch.tensor(cfg.output_pixel_pitches_m,dtype=torch.float64)
    power=field.abs().square().sum((-2,-1))*pitches.square()
    torch.testing.assert_close(power,torch.full_like(power,cfg.size**2*cfg.pixel_pitch_m**2))
    fraction=optics.camera_energy_fraction(phase)
    assert ((fraction>0)&(fraction<=1)).all()
    assert cfg.sampling_report()['fft_size']==32
