"""Finite-aperture single-FFT Fresnel model with a declared camera sampling grid."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .optics import OpticsConfig,fft2c,ifft2c


@dataclass(frozen=True)
class CameraConfig(OpticsConfig):
    fft_padding: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.fft_padding<1 or not isinstance(self.fft_padding,int):
            raise ValueError('FFT padding must be a positive integer')

    @property
    def fft_size(self):return self.size*self.fft_padding

    @property
    def output_pixel_pitches_m(self):
        return tuple(self.wavelength_m*z/(self.fft_size*self.pixel_pitch_m) for z in self.distances_m)

    def sampling_report(self):
        report=super().sampling_report()
        report.update(fft_size=self.fft_size,controlled_slm_size=self.size,camera_size=self.size,
                      camera_definition=f'central {self.size}x{self.size} samples of zero-padded Fresnel FFT; fixed before synthesis; all camera pixels scored')
        for p in report['planes']:
            p['output_full_grid_nyquist_ratio']=p['output_chirp_nyquist_ratio']/self.fft_padding
            p['output_chirp_nyquist_ratio']/=self.fft_padding**2
        report['interpretation']='Declared finer camera grid; input pupil and phase remain 128x128. Light outside the camera is retained in full-field outputs and energy accounting.'
        return report


class CameraFresnel(nn.Module):
    def __init__(self,config=CameraConfig(),dtype=torch.float32):
        super().__init__();self.config=config
        n,m=config.size,config.fft_size
        dx,lam=config.pixel_pitch_m,config.wavelength_m
        z=torch.tensor(config.distances_m,dtype=torch.float64)[:,None,None]
        axis=(torch.arange(n,dtype=torch.float64)-n//2)*dx
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        phase=math.pi*(xx.square()+yy.square())[None]/(lam*z)
        out_axis=(torch.arange(m,dtype=torch.float64)-m//2)[None]*(lam*z[:,0,0]/(m*dx))[:,None]
        out_phase=math.pi*(out_axis[:,:,None].square()+out_axis[:,None,:].square())/(lam*z)
        cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        self.register_buffer('input_chirp',torch.polar(torch.ones_like(phase),phase).to(cdtype))
        self.register_buffer('output_chirp',torch.polar(torch.ones_like(out_phase),out_phase).to(cdtype))
        factor=m*dx**2/(lam*z)*torch.polar(torch.ones_like(z),2*math.pi*z/lam-math.pi/2)
        self.register_buffer('physical_factor',factor.to(cdtype))

    @property
    def margin(self):return (self.config.fft_size-self.config.size)//2

    def crop(self,x):
        p=self.margin;n=self.config.size
        return x[...,p:p+n,p:p+n]

    def embed(self,x):
        p=self.margin
        return F.pad(x,(p,p,p,p))

    def full_field(self,field):
        return self.output_chirp*fft2c(self.embed(field[:,None]*self.input_chirp))

    def field(self,field):return self.crop(self.full_field(field))

    def forward(self,phase):return self.field(torch.polar(torch.ones_like(phase),phase))

    def full_physical(self,phase):
        return self.full_field(torch.polar(torch.ones_like(phase),phase))*self.physical_factor

    def physical(self,phase):return self(phase)*self.physical_factor

    def full_adjoint(self,fields):
        return self.crop(ifft2c(fields*self.output_chirp.conj()))*self.input_chirp.conj()

    def adjoint(self,fields):return self.full_adjoint(self.embed(fields))

    def camera_energy_fraction(self,phase):
        full=self.full_physical(phase).abs().square()
        return self.crop(full).sum((-2,-1))/full.sum((-2,-1)).clamp_min(1e-12)
