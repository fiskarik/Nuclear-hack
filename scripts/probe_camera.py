"""Validate a declared camera sampling grid; DOE stays 128x128 and all image pixels are scored."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from doe.camera_optics import CameraConfig,CameraFresnel
from doe.data import QuartetDataset
from doe.engine import setup
from doe.model import initialization
from doe.optics import normalize_targets,unit_complex
from doe.quality import perceptual_loss_per_item
from doe.refinement import refine_phase
from doe.results import phase_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--padding',type=int,default=2)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--gs-steps',type=int,default=500)
    p.add_argument('--refine-steps',type=int,default=1200)
    p.add_argument('--minimum-gain',type=float,default=.15)
    p.add_argument('--initial-phase')
    p.add_argument('--initialization',choices=('random','flat','minimum-x'),default='random')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    device=setup(device='cpu',threads=2)
    cfg=CameraConfig(fft_padding=args.padding);optics=CameraFresnel(cfg).to(device)
    data=QuartetDataset('data/textures_v1','validation')
    targets=torch.stack([data[i] for i in range(args.count)]).to(device)
    amplitude=normalize_targets(targets)
    started=time.perf_counter();history=[]
    with torch.no_grad():
        if args.initialization=='random':source=initialization(optics,amplitude)
        else:
            camera_chirp=optics.crop(optics.output_chirp)[None]
            if args.initialization=='flat':desired_phase=torch.ones_like(amplitude,dtype=torch.complex64)
            else:
                n=cfg.size
                frequency=torch.fft.fftfreq(n,device=device)*n
                sign=frequency.sign();sign[0]=0;sign[n//2]=0
                log_amplitude=amplitude.clamp_min(.02).log()
                hilbert=torch.fft.ifft(torch.fft.fft(log_amplitude,dim=-1)*(-1j*sign),dim=-1).real
                axis=torch.arange(n,device=device)-n//2
                carrier=-2*torch.pi*(n/(4*args.padding))*axis/n
                angle=hilbert+carrier
                desired_phase=torch.polar(torch.ones_like(angle),angle)
            source=unit_complex(optics.adjoint(amplitude*camera_chirp*desired_phase).mean(dim=1))
        phase=torch.angle(source);best=phase.clone()
        if args.initial_phase:
            saved=np.load(args.initial_phase)
            np.testing.assert_array_equal(saved['targets'],targets.cpu().numpy())
            phase=torch.from_numpy(saved['phase']).to(device)
            source=torch.polar(torch.ones_like(phase),phase);best=phase.clone()
        best_loss=perceptual_loss_per_item(optics,best,targets)
        for step in range(args.gs_steps+1):
            if step%100==0 or step==args.gs_steps:
                values=phase_metrics(optics,best,targets)
                record={'step':step,'pearson':sum(v['pearson'] for v in values)/len(values),
                        'ssim':sum(v['ssim'] for v in values)/len(values),'per_sample':values}
                history.append(record);print({k:record[k] for k in ('step','pearson','ssim')},flush=True)
            if step==args.gs_steps:break
            fields=optics.full_field(source)
            camera=optics.crop(fields)
            scale=(camera.abs()*amplitude).mean((-2,-1),keepdim=True).clamp(args.minimum_gain,.8)
            desired=amplitude*scale*unit_complex(camera)
            constrained=fields.clone()
            start=optics.margin;n=cfg.size
            constrained[...,start:start+n,start:start+n]=desired
            source=unit_complex(optics.full_adjoint(constrained).mean(dim=1))
            if step%10==0:
                phase=torch.angle(source)
                loss=perceptual_loss_per_item(optics,phase,targets)
                improve=loss<best_loss;best[improve]=phase[improve];best_loss=torch.minimum(best_loss,loss)
        phase,trace=refine_phase(optics,best,targets,steps=args.refine_steps,lr=.05)
        values=phase_metrics(optics,phase,targets)
        fractions=optics.camera_energy_fraction(phase).cpu().tolist()
    report={'config':vars(args),'optics':cfg.to_dict(),'sampling':cfg.sampling_report(),'split':'validation',
            'kind':'ONE shared 128x128 phase DOE; explicitly finer camera sampling, NOT the old full-period grid',
            'seconds':time.perf_counter()-started,'history':history,'refinement':trace,'per_sample':values,
            'camera_energy_fraction':fractions,
            'pearson':sum(v['pearson'] for v in values)/len(values),'ssim':sum(v['ssim'] for v in values)/len(values)}
    out.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),phase=phase.cpu().numpy(),targets=targets.cpu().numpy())
    print({'final_pearson':report['pearson'],'final_ssim':report['ssim'],'energy_fraction':fractions},flush=True)


if __name__=='__main__':main()
