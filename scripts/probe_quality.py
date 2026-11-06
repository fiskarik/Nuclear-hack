"""Validation-only experiments to assess achievable phase quality before model changes."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from doe.data import QuartetDataset
from doe.engine import load_model, setup
from doe.model import physics_loss
from doe.quality import perceptual_loss_per_item
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/textures_v1')
    p.add_argument('--checkpoint',default='deliverables/texture_finetuned/best.pt')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--lr',type=float,default=0.05)
    p.add_argument('--objective',choices=['perceptual','physics'],default='perceptual')
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    model,_=load_model(args.checkpoint,device)
    model.requires_grad_(False)
    data=QuartetDataset(args.data,'validation')
    target=torch.stack([data[i] for i in range(args.count)]).to(device)
    with torch.no_grad(): initial=model(target)
    phase=torch.nn.Parameter(initial.clone())
    optimizer=torch.optim.Adam([phase],lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.lr/20)
    records=[]; started=time.perf_counter()
    for step in range(args.steps+1):
        if step%50==0 or step==args.steps:
            wrapped=torch.atan2(phase.sin(),phase.cos())
            values=phase_metrics(model.optics,wrapped,target)
            row={'step':step,'mean_pearson':sum(v['pearson'] for v in values)/len(values),
                 'mean_ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
            records.append(row)
            print({k:row[k] for k in ('step','mean_pearson','mean_ssim')},flush=True)
        if step==args.steps: break
        optimizer.zero_grad(set_to_none=True)
        loss=(perceptual_loss_per_item(model.optics,phase,target).mean() if args.objective=='perceptual'
              else physics_loss(model.optics,phase,target))
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward()
        optimizer.step(); scheduler.step()
    output.write_text(json.dumps({'config':vars(args),'split':'validation','seconds':time.perf_counter()-started,
                                 'kind':'per-input phase optimization diagnostic, not a trained network improvement',
                                 'history':records},indent=2)+'\n')


if __name__=='__main__': main()
