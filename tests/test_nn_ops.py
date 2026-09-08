"""Independent forward/adjoint/state checks for reusable neural-network operations."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import munet as mu
from munet.core import operation


def parity(native,reference,arrays,device,*,wrt=None,rtol=2e-4,atol=3e-5):
    arrays=[np.asarray(a,np.float32) for a in arrays]
    wrt=list(range(len(arrays))) if wrt is None else wrt
    weights=None
    tx=[torch.tensor(a,requires_grad=i in wrt) for i,a in enumerate(arrays)]
    expected=reference(*tx)
    weights=np.random.default_rng(515).normal(size=tuple(expected.shape)).astype(np.float32)
    (expected*torch.tensor(weights)).sum().backward()
    def function(*xs):
        value=native(*xs)
        return (value,*mu.grad((value*weights).sum(),[xs[i] for i in wrt]))
    result=mu.compile(function,device=device)(*arrays)
    np.testing.assert_allclose(result[0].numpy(),expected.detach().numpy(),rtol=rtol,atol=atol)
    for r,i in zip(result[1:],wrt): np.testing.assert_allclose(r.numpy(),tx[i].grad.numpy(),rtol=rtol,atol=atol)


def test_grouped_dilated_conv_forward_backward(device):
    rng=np.random.default_rng(8)
    arrays=[rng.normal(size=(2,4,6,7)),rng.normal(size=(6,2,2,3))]
    kw=dict(stride=(2,1),padding=(1,2),dilation=(2,1),groups=2)
    parity(lambda x,w:mu.nn.functional.conv2d(x,w,**kw),lambda x,w:F.conv2d(x,w,**kw),arrays,device)


@pytest.mark.parametrize('average,ceil,pad,count',[(False,False,1,False),(False,True,1,False),(True,False,1,False),(True,True,1,True),(True,True,0,True)])
def test_pooling_forward_backward(device,average,ceil,pad,count):
    x=np.random.default_rng(9).normal(size=(2,2,5,6))
    kw=dict(kernel_size=3,stride=2,padding=pad,ceil_mode=ceil)
    if average:
        kw['count_include_pad']=count
        parity(lambda x:mu.nn.functional.avg_pool2d(x,**kw),lambda x:F.avg_pool2d(x,**kw),[x],device)
    else: parity(lambda x:mu.nn.functional.max_pool2d(x,**kw),lambda x:F.max_pool2d(x,**kw),[x],device)


def test_bilinear_sampling_values_and_coordinates(device):
    rng=np.random.default_rng(12)
    x=rng.normal(size=(2,3,3,5));grid=rng.uniform(-1.8,1.8,size=(2,4,3,2))
    grid[0,0]=[[-1,-1],[1,1],[-1.21,0.31]]
    parity(mu.nn.functional.grid_sample,lambda x,g:F.grid_sample(x,g,align_corners=False),[x,grid],device)


def test_sampling_coordinate_finite_difference():
    rng=np.random.default_rng(15)
    x=rng.normal(size=(1,2,3,5)).astype(np.float32);grid=np.array([[[[-0.37,0.21],[0.41,-0.32]]]],np.float32)
    f=mu.compile(lambda a,g:(mu.nn.functional.grid_sample(a,g).square().sum(),mu.grad(mu.nn.functional.grid_sample(a,g).square().sum(),[g])[0]),device='cpu')
    _,gradient=f(x,grid);actual=gradient.numpy()
    for index in np.ndindex(grid.shape):
        plus,minus=grid.copy(),grid.copy();plus[index]+=1e-3;minus[index]-=1e-3
        hi=f(x,plus)[0].item();lo=f(x,minus)[0].item()
        np.testing.assert_allclose(actual[index],(hi-lo)/2e-3,rtol=2e-3,atol=4e-4)


def test_batched_matmul_broadcast_and_views(device):
    rng=np.random.default_rng(7)
    a=rng.normal(size=(2,1,3,4));b=rng.normal(size=(1,5,4,2))
    parity(lambda a,b:(a@b).permute(0,2,1,3),lambda a,b:(a@b).permute(0,2,1,3),[a,b],device)


def test_slices_concat_and_resize(device):
    x=np.random.default_rng(13).normal(size=(2,3,5,6))
    parity(lambda x:mu.cat([x[:,:,1:5:2,::-2],x[:,:,1:5:2,::-2]],1),lambda x:torch.cat([x[:,:,1:5:2].flip(-1)[:,:,:,::2]]*2,1),[x],device)
    parity(lambda x:mu.nn.functional.interpolate(x,size=(8,4)),lambda x:F.interpolate(x,size=(8,4),mode='nearest'),[x],device)


def test_gather_scatter_repeated_and_padding(device):
    rng=np.random.default_rng(5);x=rng.normal(size=(2,4,3))
    indices=np.array([[[2,0,2],[2,3,1]],[[0,2,1],[0,2,3]]],np.float32)
    parity(lambda x,i:x.gather(1,i),lambda x,i:torch.gather(x,1,i.long()),[x,indices],device,wrt=[0])
    table=rng.normal(size=(5,3));idx=np.array([[2,2,-1],[0,4,2]],np.float32)
    parity(lambda x,i:x.take(i),lambda x,i:x[i.long().clamp(min=0)]*(i>=0).unsqueeze(-1),[table,idx],device,wrt=[0])


def test_stable_softmax_activations_and_topk(device):
    x=np.random.default_rng(3).normal(size=(2,4,7)).astype(np.float32)*3
    parity(lambda x:x.gelu().softmax(-1),lambda x:F.gelu(x).softmax(-1),[x],device)
    f=mu.compile(lambda x:x.topk(3,-1),device=device)
    values,indices=f(x);tv,ti=torch.topk(torch.tensor(x),3,-1)
    np.testing.assert_allclose(values.numpy(),tv.numpy());np.testing.assert_array_equal(indices.numpy(),ti.numpy())
    ties=mu.compile(lambda x:x.topk(2),device=device)(np.ones((1,4),np.float32))[1].numpy()
    np.testing.assert_array_equal(ties,[[0,1]])


def test_assignment_matches_scipy_including_empty_targets(device):
    rng=np.random.default_rng(34)
    costs=rng.normal(size=(3,7,4)).astype(np.float32)
    valid=np.array([[1,1,1,1],[1,0,1,0],[0,0,0,0]],np.float32)
    result=mu.compile(lambda c,m:operation('assignment',c,m),device=device)(costs,valid).numpy().astype(int)
    for b in range(3):
        targets=np.flatnonzero(valid[b]);actual=result[b,targets]
        if len(targets):
            qi,ti=linear_sum_assignment(costs[b][:,targets])
            np.testing.assert_allclose(costs[b,actual,targets].sum(),costs[b,qi,targets[ti]].sum(),atol=2e-6)
            assert len(set(actual))==len(actual)
        assert np.all(result[b,valid[b]==0]==-1)


def test_batchnorm_buffers_gradients_and_eval(device):
    rng=np.random.default_rng(9);model=mu.nn.BatchNorm2d(3);reference=torch.nn.BatchNorm2d(3)
    f=mu.compile(lambda x:(model(x),),device=device)
    for _ in range(3):
        x=rng.normal(size=(2,3,3,4)).astype(np.float32)
        actual=f(x)[0].numpy();expected=reference(torch.tensor(x)).detach().numpy()
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    state=model.state_dict()
    for k,v in reference.state_dict().items(): np.testing.assert_allclose(state[k],v.numpy(),rtol=2e-5,atol=2e-6)
    model.eval();reference.eval()
    np.testing.assert_allclose(f(x)[0].numpy(),reference(torch.tensor(x)).detach().numpy(),rtol=2e-5,atol=2e-6)
    model.train();reference.train()
    parity(lambda x:model(x),lambda x:reference(x),[x],device)


def test_layernorm_and_self_attention(device):
    rng=np.random.default_rng(42);x=rng.normal(size=(2,4,8)).astype(np.float32)
    norm=mu.nn.LayerNorm(8);tnorm=torch.nn.LayerNorm(8)
    parity(norm,tnorm,[x],device)
    model=mu.nn.MultiheadAttention(8,2);reference=torch.nn.MultiheadAttention(8,2,batch_first=True)
    model.load_state_dict(reference.state_dict())
    mask=np.triu(np.ones((4,4),np.float32),1)
    parity(lambda x:model(x,x,x,attn_mask=mu.as_tensor(mask))[0],lambda x:reference(x,x,x,attn_mask=torch.tensor(mask,dtype=torch.bool),need_weights=False)[0],[x],device)


def test_adamw_groups_clipping_and_resume(device,tmp_path):
    rng=np.random.default_rng(8);model=mu.nn.Linear(3,2);reference=torch.nn.Linear(3,2)
    model.load_state_dict(reference.state_dict())
    optimizer=mu.optim.AdamW([{'params':[model.weight],'lr':0.01},{'params':[model.bias],'lr':0.02,'weight_decay':0}],eps=1e-6)
    torch_opt=torch.optim.AdamW([{'params':[reference.weight],'lr':0.01},{'params':[reference.bias],'lr':0.02,'weight_decay':0}],eps=1e-6)
    def step(x,y):
        optimizer.zero_grad();loss=(model(x)-y).square().mean();loss.backward();mu.optim.clip_grad_norm_(model.parameters(),0.1);optimizer.step();return loss
    program=mu.compile(step,device=device)
    x=rng.normal(size=(4,3)).astype(np.float32);y=rng.normal(size=(4,2)).astype(np.float32)
    for i in range(4):
        if i==2: optimizer.param_groups[0]['lr']=0.005;torch_opt.param_groups[0]['lr']=0.005
        actual=program(x,y).item();torch_opt.zero_grad();loss=(reference(torch.tensor(x))-torch.tensor(y)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),0.1);torch_opt.step()
        np.testing.assert_allclose(actual,loss.item(),rtol=4e-5)
        for name,value in model.state_dict().items(): np.testing.assert_allclose(value,reference.state_dict()[name].numpy(),rtol=5e-5,atol=5e-6)
    mu.save(program,tmp_path/'step.mnet');resumed=mu.load(tmp_path/'step.mnet',device=device)
    np.testing.assert_allclose(program(x,y).numpy(),resumed(x,y).numpy(),rtol=2e-5,atol=2e-6)


def test_dropout_replay_and_serialized_rng(device,tmp_path):
    layer=mu.nn.Dropout(0.4,rng=np.random.default_rng(18))
    f=mu.compile(lambda x:layer(x),device=device);x=np.ones((16,16),np.float32)
    a=f(x).numpy();b=f(x).numpy();assert not np.array_equal(a,b)
    mu.save(f,tmp_path/'rng.mnet');restored=mu.load(tmp_path/'rng.mnet',device=device)
    np.testing.assert_array_equal(f(x).numpy(),restored(x).numpy())
    layer.eval();np.testing.assert_array_equal(f(x).numpy(),x)


def test_plan_can_describe_large_arena_without_allocating_it():
    # Each tensor uses 32-bit indices, but descriptor offsets and the combined
    # plan can exceed 4 GiB. Planning must not allocate this inspection-only arena.
    from munet import _native
    g=_native.Graph();a=g.leaf('input','a',[1024,512,1024]);b=g.leaf('input','b',[1024,512,1024])
    c=g.op('add',[a,b]);plan=_native.Plan(g,[c],[],False)
    assert plan.stats()['arena_bytes']==6*1024**3
    assert plan.stats()['runs']==0
