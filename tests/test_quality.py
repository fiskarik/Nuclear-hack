import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

from doe.quality import pearson_per_plane, quantized_phase_ste, ssim_per_plane
from doe.results import decode_phase, encode_phase


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
