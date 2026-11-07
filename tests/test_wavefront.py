import hashlib

import torch

from doe.data import generate_dataset
from doe.engine import TrainConfig,load_model,train
from doe.model import FresnelNet,WavefrontFresnelNet,AdaptiveWavefrontNet,physics_loss
from doe.optics import OpticsConfig


def test_wavefront_initializer_receives_nonzero_gradients():
    torch.manual_seed(41)
    model=WavefrontFresnelNet(OpticsConfig(size=16),stages=2,width=4,initializer_width=4)
    targets=torch.rand(2,4,16,16)+.1
    phase=model(targets)
    assert phase.shape==(2,16,16)
    physics_loss(model.optics,phase,targets).backward()
    gradient=model.phase_initializer['head'].weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    assert phase.abs().max()<=torch.pi+1e-6


def test_wavefront_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    model=WavefrontFresnelNet(OpticsConfig(size=16),stages=1,width=4,initializer_width=4).eval()
    target=torch.rand(1,4,16,16)+.1
    path=tmp_path/'wavefront.pt'
    torch.save({'format_version':1,'architecture':model.architecture(),'optics':model.optics.config.to_dict(),
                'model':model.state_dict(),'epoch':0},path)
    loaded,_=load_model(path,torch.device('cpu'))
    assert isinstance(loaded,WavefrontFresnelNet)
    with torch.no_grad():torch.testing.assert_close(loaded(target),model(target),rtol=0,atol=0)


def test_fixed_checkpoint_can_upgrade_without_losing_existing_weights(tmp_path):
    torch.manual_seed(43)
    base=FresnelNet(stages=1,width=4)
    path=tmp_path/'parent.pt'
    torch.save({'format_version':1,'architecture':{'stages':1,'width':4},'optics':base.optics.config.to_dict(),
                'model':base.state_dict(),'epoch':3},path)
    generate_dataset(tmp_path/'data',train=1,validation=1,test=1)
    report=train(tmp_path/'data',tmp_path/'run',TrainConfig(epochs=1,batch_size=1,stages=1,width=4,
        initializer_width=4,architecture='wavefront',objective='perceptual',initial_checkpoint=str(path),device='cpu'))
    initial=torch.load(tmp_path/'run/initial.pt',weights_only=True)
    for name,value in base.state_dict().items():
        torch.testing.assert_close(initial['model'][name],value,rtol=0,atol=0)
    assert initial['architecture']['kind']=='wavefront'
    assert report['parent_checkpoint']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['parameter_l2_change_last_vs_initial']>0


def test_adaptive_projection_and_deeper_initializer_have_gradients():
    torch.manual_seed(48)
    model=AdaptiveWavefrontNet(OpticsConfig(size=16),stages=2,width=4,initializer_width=4,initializer_depth=4)
    target=torch.rand(2,4,16,16)+.1
    phase=model(target)
    physics_loss(model.optics,phase,target).backward()
    grad=model.amplitude_modulators[0][-1].weight.grad
    assert torch.isfinite(grad).all() and grad.abs().max()>0
    # The phase head is zero-initialized. After its first update, the gated
    # deeper branch must receive a signal without perturbing the warm start.
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    physics_loss(model.optics,model(target),target).backward()
    assert torch.isfinite(model.deep_scale.grad) and model.deep_scale.grad.abs()>0
    assert phase.shape==(2,16,16)
