"""Validation-only coarse-to-fine common-phase optimization, with unchanged scoring."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from doe.data import QuartetDataset
from doe.engine import load_model,setup
from doe.quality import perceptual_loss_per_item,ssim_per_plane,pearson_per_plane
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/textures_v1')
    p.add_argument('--checkpoint',default='deliverables/texture_finetuned/best.pt')
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--stage-steps',type=int,default=500)
    p.add_argument('--lr',type=float,default=0.08)
    p.add_argument('--chirp-order',type=float,default=1)
    p.add_argument('--fit-start',action='store_true',help='Intermediate loss fits amplitude gain; reported PNGs never use fitted gain')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    model,_=load_model(args.checkpoint,device);model.requires_grad_(False)
    data=QuartetDataset(args.data,'validation')
    targets=torch.stack([data[i] for i in range(args.count)]).to(device)
    cfg=model.optics.config;n=cfg.size
    axis=torch.arange(n,device=device)-n//2
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    beta=sum(math.pi*cfg.pixel_pitch_m**2/(cfg.wavelength_m*z) for z in cfg.distances_m)/4
    carrier=((args.chirp_order*math.pi/n-beta)*(xx.square()+yy.square()))[None].expand(len(targets),-1,-1)
    residual=torch.zeros(len(targets),1,8,8,device=device)
    history=[];started=time.perf_counter();total=0
    for resolution in (8,16,32,64,128):
        parameter=torch.nn.Parameter(F.interpolate(residual,size=(resolution,resolution),mode='bilinear',align_corners=False).clone())
        optimizer=torch.optim.Adam([parameter],lr=args.lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.stage_steps,eta_min=args.lr/10)
        for step in range(args.stage_steps+1):
            phase=carrier+F.interpolate(parameter,size=(n,n),mode='bilinear',align_corners=False)[:,0]
            if step%100==0 or step==args.stage_steps:
                values=phase_metrics(model.optics,torch.atan2(phase.sin(),phase.cos()),targets)
                record={'resolution':resolution,'stage_step':step,'total_step':total,
                        'pearson':sum(v['pearson'] for v in values)/len(values),
                        'ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
                history.append(record)
                print({k:record[k] for k in ('resolution','stage_step','pearson','ssim')},flush=True)
            if step==args.stage_steps:break
            optimizer.zero_grad(set_to_none=True)
            if args.fit_start and resolution<128:
                amplitude=model.optics(phase).abs()
                gain=(amplitude*targets).sum((-2,-1),keepdim=True)/amplitude.square().sum((-2,-1),keepdim=True).clamp_min(1e-8)
                fitted=amplitude*gain
                loss=.5*(1-pearson_per_plane(fitted,targets)).mean()+.5*(1-ssim_per_plane(fitted,targets)).mean()+.05*(fitted-targets).square().mean()
            else:
                loss=perceptual_loss_per_item(model.optics,phase,targets).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();optimizer.step();scheduler.step();total+=1
        residual=parameter.detach()
    output.write_text(json.dumps({'config':vars(args),'split':'validation','kind':'coarse-to-fine optimization of ONE common phase; scoring unchanged',
                                 'seconds':time.perf_counter()-started,'history':history},indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),phase=torch.atan2(phase.sin(),phase.cos()).detach().cpu().numpy(),targets=targets.cpu().numpy())


if __name__=='__main__':main()
