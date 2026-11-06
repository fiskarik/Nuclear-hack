"""Validation-only experiments to assess achievable phase quality before model changes."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from doe.data import QuartetDataset
from doe.engine import load_model, setup
from doe.model import physics_loss
from doe.optics import normalize_targets, unit_complex, fft2c
from doe.quality import perceptual_loss_per_item,robust_quality_loss_per_item
from doe.results import phase_metrics


class IndependentPlaneOptics:
    """A noncompliant control: a different phase-only DOE for every target plane."""
    def __init__(self,base):
        self.base=base
        self.config=base.config

    def __call__(self,phase):
        return self.base.output_chirp*fft2c(torch.polar(torch.ones_like(phase),phase)*self.base.input_chirp)

    def physical(self,phase):
        return self(phase)*self.base.physical_factor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/textures_v1')
    p.add_argument('--checkpoint',default='deliverables/texture_finetuned/best.pt')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--lr',type=float,default=0.05)
    p.add_argument('--objective',choices=['perceptual','physics','robust'],default='perceptual')
    p.add_argument('--initialization',choices=['neural','chirp','chirp_projected','zero_projected','random'],default='neural')
    p.add_argument('--chirp-order',type=float,default=1)
    p.add_argument('--optimizer',choices=['adam','lbfgs'],default='adam')
    p.add_argument('--continuous',action='store_true',help='Optimize continuous phase; still evaluate after 8-bit quantization')
    p.add_argument('--report-every',type=int,default=50)
    p.add_argument('--initial-phase',help='Resume a diagnostic NPZ on exactly the same validation targets')
    p.add_argument('--independent-planes',action='store_true',help='NONCOMPLIANT ablation: four separate DOEs instead of one')
    p.add_argument('--softmax-temp',type=float,default=0,help='Annealed surrogate for maximum gradient only; PNG forward/scoring unchanged')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    model,_=load_model(args.checkpoint,device)
    model.requires_grad_(False)
    data=QuartetDataset(args.data,'validation')
    target=torch.stack([data[i] for i in range(args.count)]).to(device)
    with torch.no_grad():
        if args.initialization=='neural':
            initial=model(target)
        elif args.initialization=='random':
            initial=(torch.rand((len(target),128,128),device=device)*2-1)*math.pi
        elif args.initialization=='zero_projected':
            initial=torch.angle(model.optics.adjoint(normalize_targets(target).to(torch.complex64)).mean(dim=1))
        else:
            cfg=model.optics.config
            axis=torch.arange(cfg.size,device=device)-cfg.size//2
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            beta=sum(math.pi*cfg.pixel_pitch_m**2/(cfg.wavelength_m*z) for z in cfg.distances_m)/4
            carrier=(args.chirp_order*math.pi/cfg.size-beta)*(xx.square()+yy.square())
            initial=carrier[None].expand(len(target),-1,-1).clone()
            if args.initialization=='chirp_projected':
                fields=model.optics(initial)
                initial=torch.angle(model.optics.adjoint(normalize_targets(target)*unit_complex(fields)).mean(dim=1))
    optics=model.optics
    if args.independent_planes:
        optics=IndependentPlaneOptics(model.optics)
        initial=initial[:,None].expand(-1,4,-1,-1).clone()
    if args.initial_phase:
        saved=np.load(args.initial_phase)
        np.testing.assert_array_equal(saved['targets'],target.cpu().numpy())
        initial=torch.from_numpy(saved['phase']).to(device)
    phase=torch.nn.Parameter(initial.clone())
    if args.optimizer=='adam':
        optimizer=torch.optim.Adam([phase],lr=args.lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.lr/20)
    else:
        optimizer=torch.optim.LBFGS([phase],lr=args.lr,max_iter=10,history_size=20,line_search_fn='strong_wolfe')
        scheduler=None
    evaluations=0
    records=[]; started=time.perf_counter()
    for step in range(args.steps+1):
        if step%args.report_every==0 or step==args.steps:
            wrapped=torch.atan2(phase.sin(),phase.cos())
            values=phase_metrics(optics,wrapped,target)
            row={'step':step,'mean_pearson':sum(v['pearson'] for v in values)/len(values),
                 'mean_ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
            records.append(row)
            print({k:row[k] for k in ('step','mean_pearson','mean_ssim')},flush=True)
        if step==args.steps: break
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            temp=args.softmax_temp*(.01+.99*(1-step/max(1,args.steps)))
            if args.objective=='perceptual':loss=perceptual_loss_per_item(optics,phase,target,quantize=not args.continuous,max_temperature=temp).mean()
            elif args.objective=='robust':loss=robust_quality_loss_per_item(optics,phase,target,quantize=not args.continuous,max_temperature=temp).mean()
            else:loss=physics_loss(optics,phase,target)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward(); evaluations+=1
            return loss
        optimizer.step(closure)
        if scheduler is not None: scheduler.step()
    output.write_text(json.dumps({'config':vars(args),'split':'validation','seconds':time.perf_counter()-started,
                                 'kind':('NONCOMPLIANT control: four independent DOEs' if args.independent_planes else 'per-input phase optimization diagnostic, not a trained network improvement'),
                                 'objective_evaluations':evaluations,'history':records},indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),phase=torch.atan2(phase.sin(),phase.cos()).detach().cpu().numpy(),targets=target.cpu().numpy())


if __name__=='__main__': main()
