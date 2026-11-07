"""Validation-only relaxed alternating reflections with one common phase screen."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from doe.data import QuartetDataset
from doe.engine import load_model,setup
from doe.optics import normalize_targets,unit_complex
from doe.quality import perceptual_loss_per_item
from doe.refinement import refine_phase
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--beta',type=float,default=.85)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--refine-steps',type=int,default=1200)
    p.add_argument('--count',type=int,default=4)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    model,_=load_model('deliverables/texture_finetuned/best.pt',device);model.requires_grad_(False)
    data=QuartetDataset('data/textures_v1','validation')
    target=torch.stack([data[i] for i in range(args.count)]).to(device)
    history=[];started=time.perf_counter()
    with torch.no_grad():
        initial=model(target)
        x=model.optics(initial)
        desired=normalize_targets(target)
        best=initial.clone()
        best_loss=perceptual_loss_per_item(model.optics,best,target)
        for step in range(args.steps+1):
            if step%100==0 or step==args.steps:
                values=phase_metrics(model.optics,best,target)
                r={'step':step,'pearson':sum(v['pearson'] for v in values)/len(values),
                   'ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
                history.append(r);print({k:r[k] for k in ('step','pearson','ssim')},flush=True)
            if step==args.steps:break
            a=desired*unit_complex(x)
            field=unit_complex(model.optics.adjoint(2*a-x).mean(dim=1))
            b=model.optics.field(field)
            x=args.beta*b+args.beta*x+(1-2*args.beta)*a
            if step%10==0:
                phase=torch.angle(field)
                loss=perceptual_loss_per_item(model.optics,phase,target)
                improved=loss<best_loss
                best[improved]=phase[improved]
                best_loss=torch.minimum(best_loss,loss)
        phase,trace=refine_phase(model.optics,best,target,steps=args.refine_steps,lr=.05)
        final=phase_metrics(model.optics,phase,target)
    result={'config':vars(args),'split':'validation','kind':'RAAR + Adam; ONE shared phase, original PNG scoring',
            'seconds':time.perf_counter()-started,'history':history,'refinement':trace,'final':final,
            'mean_pearson':sum(v['pearson'] for v in final)/len(final),
            'mean_ssim':sum(v['ssim'] for v in final)/len(final)}
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),phase=phase.cpu().numpy(),targets=target.cpu().numpy())
    print({'final_pearson':result['mean_pearson'],'final_ssim':result['mean_ssim']},flush=True)


if __name__=='__main__':main()
