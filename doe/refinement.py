"""Optional per-input phase optimization, explicitly separate from network inference."""
import math
import time

import torch

from .optics import normalize_targets
from .quality import perceptual_loss_per_item


def refine_phase(optics, initial_phase, targets, steps=600, lr=0.08, check_every=25):
    if steps < 0 or lr <= 0 or check_every < 1:
        raise ValueError("steps must be >=0; lr and check_every must be positive")
    if initial_phase.ndim != 3 or targets.shape != (len(initial_phase), 4, *initial_phase.shape[-2:]):
        raise ValueError("one shared phase [B,H,W] and four target planes are required")
    normalize_targets(targets)
    if not torch.isfinite(initial_phase).all():
        raise ValueError("initial phase must be finite")
    started=time.perf_counter()
    history=[]
    with torch.enable_grad():
        phase=torch.nn.Parameter(initial_phase.detach().clone())
        target=targets.detach()
        optimizer=torch.optim.Adam([phase],lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,steps),eta_min=lr/20)
        with torch.no_grad():
            best_loss=perceptual_loss_per_item(optics,phase,target)
            best=phase.detach().clone()
            best_step=torch.zeros(len(phase),dtype=torch.long,device=phase.device)
            history.append({'step':0,'loss':float(best_loss.mean())})
        for step in range(1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss=perceptual_loss_per_item(optics,phase,target).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite refinement loss")
            loss.backward()
            optimizer.step();scheduler.step()
            if step%check_every==0 or step==steps:
                with torch.no_grad():
                    current=perceptual_loss_per_item(optics,phase,target)
                    improve=current<best_loss
                    best[improve]=phase.detach()[improve]
                    best_step[improve]=step
                    best_loss=torch.minimum(best_loss,current)
                    history.append({'step':step,'loss':float(best_loss.mean())})
    if initial_phase.is_cuda:
        torch.cuda.synchronize(initial_phase.device)
    phase=torch.atan2(best.sin(),best.cos()).detach()
    report={'method':'per-input Adam on shared phase, with straight-through 8-bit quantization',
            'optimizer_steps':steps,'learning_rate':lr,'best_step_per_input':best_step.cpu().tolist(),
            'seconds':time.perf_counter()-started,'history':history,
            'warning':'This is a neural initializer plus iterative phase solver, not a single network forward pass.'}
    return phase,report
