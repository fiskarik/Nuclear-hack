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

    def architecture(self):
        return {"kind":"fixed","stages":self.stages,"width":self.width}

    def initial_field(self,amplitude):
        return initialization(self.optics,amplitude)

    def constrained_fields(self,amplitude,reconstructed,stage):
        return amplitude*unit_complex(reconstructed)

    def forward(self, targets):
        amplitude = normalize_targets(targets)
        field = self.initial_field(amplitude)
        for stage in range(self.stages):
            reconstructed = self.optics.field(field)
            constrained = self.constrained_fields(amplitude,reconstructed,stage)
            back = self.optics.adjoint(constrained)
            weights = self.plane_logits[stage].softmax(dim=0)[None, :, None, None]
            projected = unit_complex((weights * back).sum(dim=1))
            features = torch.stack((field.real, field.imag, projected.real, projected.imag), dim=1)
            correction = 0.35 * torch.tanh(self.corrector(features)[:, 0])
            relax = 2 * self.relax_logits[stage].sigmoid()
            field = unit_complex((1 - relax) * field + relax * projected)
            field = field * torch.polar(torch.ones_like(correction), correction)
        return torch.angle(field)


class WavefrontBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,padding_mode='circular'),nn.GroupNorm(4,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,padding=1,padding_mode='circular'),nn.GroupNorm(4,cout),nn.SiLU())
    def forward(self,x):
        return self.net(x)


class WavefrontFresnelNet(FresnelNet):
    """Learn four object-plane phases with a nonlocal U-Net before shared-DOE synthesis."""
    def __init__(self,config=OpticsConfig(),stages=6,width=12,initializer_width=8,initializer_depth=3):
        super().__init__(config,stages,width)
        if initializer_width<4 or initializer_width%4:
            raise ValueError('initializer width must be a positive multiple of 4')
        if initializer_depth not in (3,4):raise ValueError('initializer depth must be 3 or 4')
        w=initializer_width;self.initializer_width=w;self.initializer_depth=initializer_depth
        self.phase_initializer=nn.ModuleDict({
            'first':WavefrontBlock(10,w),'down1':WavefrontBlock(w,2*w),
            'down2':WavefrontBlock(2*w,4*w),'up1':WavefrontBlock(6*w,2*w),
            'up0':WavefrontBlock(3*w,w),'head':nn.Conv2d(w,4,1)})
        nn.init.zeros_(self.phase_initializer['head'].weight)
        nn.init.zeros_(self.phase_initializer['head'].bias)
        if initializer_depth==4:
            self.phase_initializer['down3']=WavefrontBlock(4*w,8*w)
            self.phase_initializer['up2']=WavefrontBlock(12*w,4*w)
            self.deep_scale=nn.Parameter(torch.zeros(()))
        n=config.size
        axis=torch.arange(n,dtype=torch.float32)-n//2
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        beta=sum(math.pi*config.pixel_pitch_m**2/(config.wavelength_m*z) for z in config.distances_m)/4
        carrier=(math.pi/n-beta)*(xx.square()+yy.square())
        with torch.no_grad():reference=unit_complex(self.optics(carrier[None]))[0]
        self.register_buffer('reference_phase',reference)
        self.register_buffer('coordinate_grid',torch.stack((xx/(n/2),yy/(n/2)),dim=0))

    def architecture(self):
        return {'kind':'wavefront','stages':self.stages,'width':self.width,'initializer_width':self.initializer_width,
                'initializer_depth':self.initializer_depth}

    def initial_field(self,amplitude):
        peak=amplitude.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-8)
        coordinates=self.coordinate_grid[None].expand(len(amplitude),-1,-1,-1)
        x=torch.cat((amplitude,amplitude/peak,coordinates),dim=1)
        layers=self.phase_initializer
        a=layers['first'](x)
        b=layers['down1'](F.avg_pool2d(a,2))
        c=layers['down2'](F.avg_pool2d(b,2))
        if self.initializer_depth==4:
            deep=layers['down3'](F.avg_pool2d(c,2))
            correction=layers['up2'](torch.cat((F.interpolate(deep,size=c.shape[-2:],mode='bilinear',align_corners=False),c),dim=1))
            c=c+self.deep_scale*correction
        b2=layers['up1'](torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),dim=1))
        a2=layers['up0'](torch.cat((F.interpolate(b2,size=a.shape[-2:],mode='bilinear',align_corners=False),a),dim=1))
        theta=math.pi*torch.tanh(layers['head'](a2))
        fields=amplitude*self.reference_phase*torch.polar(torch.ones_like(theta),theta)
        return unit_complex(self.optics.adjoint(fields).mean(dim=1))


class AdaptiveWavefrontNet(WavefrontFresnelNet):
    """Learn the amplitude freedom and phase correction of each optical projection."""
    def __init__(self,config=OpticsConfig(),stages=12,width=12,initializer_width=8,initializer_depth=4):
        super().__init__(config,stages,width,initializer_width,initializer_depth)
        self.amplitude_modulators=nn.ModuleList()
        for _ in range(stages):
            net=nn.Sequential(nn.Conv2d(20,8,3,padding=1,padding_mode='circular'),nn.SiLU(),
                              nn.Conv2d(8,8,3,padding=1,padding_mode='circular'),nn.SiLU(),nn.Conv2d(8,12,1))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            with torch.no_grad():net[-1].bias[4:8].fill_(-4)
            self.amplitude_modulators.append(net)

    def architecture(self):
        arch=super().architecture();arch['kind']='adaptive';return arch

    def constrained_fields(self,amplitude,reconstructed,stage):
        magnitude=reconstructed.abs()
        direction=unit_complex(reconstructed)
        features=torch.cat((amplitude,magnitude,amplitude-magnitude,direction.real,direction.imag),dim=1)
        modulation=self.amplitude_modulators[stage](features)
        desired=amplitude*torch.exp(.75*torch.tanh(modulation[:,:4]))
        freedom=modulation[:,4:8].sigmoid()*magnitude
        phase=math.pi*torch.tanh(modulation[:,8:])
        return (desired+freedom)*direction*torch.polar(torch.ones_like(phase),phase)


def build_model(config,architecture):
    args=dict(architecture)
    kind=args.pop('kind','fixed')
    if kind=='fixed':return FresnelNet(config,**args)
    if kind=='wavefront':return WavefrontFresnelNet(config,**args)
    if kind=='adaptive':return AdaptiveWavefrontNet(config,**args)
    raise ValueError(f'unknown architecture kind: {kind}')


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
