"""Reach the original four-plane inverse problem through an easier continuation path."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from doe.data import QuartetDataset
from doe.engine import setup
from doe.model import gerchberg_saxton
from doe.optics import Fresnel,OpticsConfig
from doe.refinement import refine_phase
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--steps-per-stage',type=int,default=300)
    p.add_argument('--target-only',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    cfg=OpticsConfig();original=Fresnel(cfg).to(device)
    dataset=QuartetDataset('data/textures_v1','validation')
    target=torch.stack([dataset[i] for i in range(args.count)]).to(device)
    common=target.mean(dim=1,keepdim=True).expand(-1,4,-1,-1)
    mean_z=sum(cfg.distances_m)/4
    initial_cfg=cfg if args.target_only else OpticsConfig(distances_m=(mean_z,)*4)
    virtual=Fresnel(initial_cfg).to(device)
    phase=gerchberg_saxton(virtual,common,steps=100)
    history=[];started=time.perf_counter()
    for alpha in np.linspace(0,1,11):
        distances=cfg.distances_m if args.target_only else tuple(mean_z+float(alpha)*(z-mean_z) for z in cfg.distances_m)
        virtual=Fresnel(OpticsConfig(distances_m=distances)).to(device)
        intermediate=(1-float(alpha))*common+float(alpha)*target
        phase,trace=refine_phase(virtual,phase,intermediate,steps=args.steps_per_stage,lr=.05)
        values=phase_metrics(original,phase,target)
        record={'alpha':float(alpha),'scoring':'original optics and original PNG targets, always',
                'pearson':sum(v['pearson'] for v in values)/len(values),
                'ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values,'refinement':trace}
        history.append(record);print({k:record[k] for k in ('alpha','pearson','ssim')},flush=True)
    out.write_text(json.dumps({'config':vars(args),'split':'validation','seconds':time.perf_counter()-started,
                              'kind':'homotopy optimization; scoring only on ORIGINAL four-plane task',
                              'history':history},indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),phase=phase.cpu().numpy(),targets=target.cpu().numpy())


if __name__=='__main__':main()
