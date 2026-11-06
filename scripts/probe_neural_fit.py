"""Diagnostic target-specific network optimization, not an amortized test score."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from doe.data import QuartetDataset
from doe.engine import load_model,setup
from doe.quality import perceptual_loss_per_item
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='artifacts/wavefront_v1/best.pt')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--count',type=int,default=1)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    model,_=load_model(args.checkpoint,device);model.train()
    data=QuartetDataset('data/textures_v1','validation')
    targets=torch.stack([data[i] for i in range(args.count)]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.lr/20)
    history=[];started=time.perf_counter();best_loss=float('inf');best=None
    for step in range(args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        phase=model(targets)
        loss=perceptual_loss_per_item(model.optics,phase,targets).mean()
        if float(loss.detach())<best_loss:
            best_loss=float(loss.detach());best=phase.detach().clone()
        if step%100==0 or step==args.steps:
            values=phase_metrics(model.optics,best,targets)
            r={'step':step,'pearson':sum(v['pearson'] for v in values)/len(values),
               'ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
            history.append(r);print({k:r[k] for k in ('step','pearson','ssim')},flush=True)
        if step==args.steps:break
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
    out.write_text(json.dumps({'config':vars(args),'split':'validation','seconds':time.perf_counter()-started,
                              'kind':'target-specific network weight fitting; NOT an independent generalization score',
                              'history':history},indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),phase=best.cpu().numpy(),targets=targets.cpu().numpy())


if __name__=='__main__':main()
