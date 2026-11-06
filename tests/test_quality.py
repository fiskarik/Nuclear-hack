import math
from pathlib import Path
import runpy

import numpy as np
import torch
from skimage.metrics import structural_similarity

from doe.quality import pearson_per_plane, quantized_phase_ste, ssim_per_plane
from doe.results import decode_phase, encode_phase
from doe.optics import Fresnel,OpticsConfig
from doe.refinement import refine_phase


def test_differentiable_quality_matches_reported_definitions():
    rng=np.random.default_rng(44)
    a=rng.uniform(0,1,(2,4,16,16))
    b=rng.uniform(0,1,(2,4,16,16))
    ssim=ssim_per_plane(torch.tensor(a),torch.tensor(b)).numpy()
    corr=pearson_per_plane(torch.tensor(a),torch.tensor(b)).numpy()
    for i in range(2):
        for j in range(4):
            np.testing.assert_allclose(ssim[i,j],structural_similarity(a[i,j],b[i,j],data_range=1.0,win_size=7),atol=1e-12)
            np.testing.assert_allclose(corr[i,j],np.corrcoef(a[i,j].ravel(),b[i,j].ravel())[0,1],atol=1e-12)


def test_ste_quantization_matches_saved_phase_encoding():
    phi=torch.linspace(-math.pi,math.pi,1001,dtype=torch.float64,requires_grad=True)
    # Interior values avoid the equivalent -pi/+pi endpoint wrapping ambiguity.
    phi=phi[1:-1].detach().requires_grad_(True)
    actual=quantized_phase_ste(phi)
    expected=decode_phase(encode_phase(phi.detach().numpy()))
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=1e-14)
    actual.sum().backward()
    torch.testing.assert_close(phi.grad,torch.ones_like(phi))


def test_ssim_gradient_matches_finite_difference():
    torch.manual_seed(10)
    a=torch.rand(1,4,9,9,dtype=torch.float64,requires_grad=True)
    b=torch.rand_like(a)
    ssim_per_plane(a,b).mean().backward()
    idx=(0,1,4,5)
    eps=1e-5
    plus,minus=a.detach().clone(),a.detach().clone()
    plus[idx]+=eps;minus[idx]-=eps
    fd=(ssim_per_plane(plus,b).mean()-ssim_per_plane(minus,b).mean())/(2*eps)
    torch.testing.assert_close(a.grad[idx],fd,atol=1e-8,rtol=1e-6)


def test_refinement_keeps_one_shared_phase_and_works_under_no_grad():
    torch.manual_seed(9)
    optics=Fresnel(OpticsConfig(size=16))
    initial=torch.rand(1,16,16)*2*math.pi-math.pi
    targets=torch.rand(1,4,16,16)+0.01
    with torch.no_grad():
        phase,report=refine_phase(optics,initial,targets,steps=3,check_every=1)
    assert phase.shape==initial.shape and not phase.requires_grad
    assert torch.isfinite(phase).all() and phase.abs().max()<=math.pi+1e-6
    assert report['optimizer_steps']==3
    assert report['history'][-1]['loss']<=report['history'][0]['loss']
    assert len(report['best_step_per_input'])==1


def test_independent_plane_control_uses_the_same_physics_per_plane():
    scope=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/probe_quality.py'))
    optics=Fresnel(OpticsConfig(size=16))
    control=scope['IndependentPlaneOptics'](optics)
    phase=torch.rand(2,4,16,16)*2*math.pi-math.pi
    actual=control.physical(phase)
    for plane in range(4):
        expected=optics.physical(phase[:,plane])[:,plane]
        torch.testing.assert_close(actual[:,plane],expected)
