"""Pinned upstream RT-DETR parity, including the actual R50-vd architecture."""
import numpy as np
import pytest
import torch
import munet as mu
from examples.rtdetr import RTDETR,SetCriterion,pad_targets,prepare_denoising
from .rtdetr_reference import model_for,criterion_for,load_reference


def torch_targets(targets): return [{'labels':torch.tensor(t['labels'],dtype=torch.long),'boxes':torch.tensor(np.asarray(t['boxes'],np.float32))} for t in targets]


def targets_example():
    # Avoid exact equality of predicted/target box edges: GIoU has a kink there,
    # so a one-ULP sigmoid difference legitimately selects a different derivative.
    return [{'labels':[1,0],'boxes':[[.223,.319,.137,.217],[.713,.617,.193,.113]]},{'labels':[],'boxes':np.zeros((0,4))}]


def test_criterion_upstream_losses_and_gradients(device):
    rng=np.random.default_rng(56);raw=targets_example();padded=pad_targets(raw,3)
    logits=rng.normal(size=(2,6,3)).astype(np.float32)
    boxes=rng.uniform(.05,.8,size=(2,6,4)).astype(np.float32)
    reference=criterion_for(3);tl=torch.tensor(logits,requires_grad=True);tb=torch.tensor(boxes,requires_grad=True)
    expected=reference({'pred_logits':tl,'pred_boxes':tb},torch_targets(raw));sum(expected.values()).backward()
    criterion=SetCriterion(3)
    def function(l,b,labels,truth,valid):
        losses=criterion({'pred_logits':l,'pred_boxes':b},{'labels':labels,'boxes':truth,'valid':valid})
        return losses,mu.grad(criterion.total(losses),[l,b])
    losses,grads=mu.compile(function,device=device)(logits,boxes,padded['labels'],padded['boxes'],padded['valid'])
    for name,value in expected.items(): np.testing.assert_allclose(losses[name].item(),value.item(),rtol=4e-5,atol=1e-5)
    np.testing.assert_allclose(grads[0].numpy(),tl.grad.numpy(),rtol=1e-4,atol=2e-5)
    np.testing.assert_allclose(grads[1].numpy(),tb.grad.numpy(),rtol=1e-4,atol=2e-5)


def test_deformable_attention_against_upstream(device):
    from examples.rtdetr.utils import deformable_attention_core_func
    ref=load_reference().utils.deformable_attention_core_func
    rng=np.random.default_rng(7)
    arrays=[rng.normal(size=(2,9,2,3)),rng.uniform(-.2,1.2,size=(2,4,2,3,2,2)),rng.uniform(size=(2,4,2,3,2))]
    arrays=[a.astype(np.float32) for a in arrays]
    ts=[torch.tensor(a,requires_grad=True) for a in arrays]
    expected=ref(ts[0],[(2,3),(1,2),(1,1)],ts[1],ts[2]);expected.square().sum().backward()
    def function(v,loc,w):
        result=deformable_attention_core_func(v,[(2,3),(1,2),(1,1)],loc,w)
        return result,mu.grad(result.square().sum(),[v,loc,w])
    result,grads=mu.compile(function,device=device)(*arrays)
    np.testing.assert_allclose(result.numpy(),expected.detach().numpy(),rtol=1e-4,atol=1e-5)
    for actual,want in zip(grads,ts): np.testing.assert_allclose(actual.numpy(),want.grad.numpy(),rtol=3e-4,atol=3e-5)


def test_full_r50vd_reference_inference(device):
    torch.manual_seed(7)
    model=RTDETR(80,num_queries=4,num_denoising=0).eval()
    reference=model_for(model.config).eval()
    model.load_state_dict(reference.state_dict())
    image=np.random.default_rng(77).uniform(size=(1,3,32,64)).astype(np.float32)
    with torch.no_grad():
        backbone=reference.backbone(torch.tensor(image));encoder=reference.encoder(backbone);output=reference.decoder(encoder)
    def forward(x):
        f=model.backbone(x);e=model.encoder(f)
        return {'backbone':f,'encoder':e,'prediction':model.decoder(e)}
    program=mu.compile(forward,device=device,fuse=False)
    result=program(image)
    for actual,want in zip(result['backbone'],backbone): np.testing.assert_allclose(actual.numpy(),want.numpy(),rtol=3e-4,atol=3e-5)
    for actual,want in zip(result['encoder'],encoder): np.testing.assert_allclose(actual.numpy(),want.numpy(),rtol=8e-4,atol=6e-5)
    for name,want in output.items(): np.testing.assert_allclose(result['prediction'][name].numpy(),want.numpy(),rtol=5e-4,atol=8e-5)


def test_r50vd_training_denoising_auxiliary_and_gradients(device):
    torch.manual_seed(17)
    # Full ResNet-50-vd; reduced decoder width/query count keeps this CI numerical gate short.
    # The full default-width/six-layer inference path is exercised separately above.
    model=RTDETR(3,hidden_dim=16,nhead=4,num_queries=4,num_decoder_layers=2,num_decoder_points=2,
                 dim_feedforward=32,num_denoising=4,expansion=.5,depth_mult=.34)
    reference=model_for(model.config).train();model.load_state_dict(reference.state_dict())
    raw=targets_example();targets=torch_targets(raw);padded=pad_targets(raw,3)
    dn=prepare_denoising(padded,3,4,num_denoising=4,rng=np.random.default_rng(6))
    image=np.random.default_rng(8).uniform(size=(2,3,64,64)).astype(np.float32)
    criterion=SetCriterion(3);ref_criterion=criterion_for(3)
    expected=ref_criterion(reference(torch.tensor(image),targets,denoising=dn),targets)
    sum(expected.values()).backward()
    selected=['backbone.res_layers.0.blocks.0.branch2a.conv.weight','encoder.input_proj.0.0.weight',
              'decoder.decoder.layers.0.cross_attn.sampling_offsets.weight','decoder.dec_bbox_head.1.layers.2.weight',
              'decoder.denoising_class_embed.weight']
    named=dict(model.named_parameters())
    def step(x,labels,boxes,valid,dn_labels,dn_boxes,mask):
        denoising={'labels':dn_labels,'boxes_unact':dn_boxes,'attn_mask':mask,'num_group':dn['num_group'],'max_gt':dn['max_gt']}
        out=model(x,denoising=denoising)
        losses=criterion(out,{'labels':labels,'boxes':boxes,'valid':valid})
        return losses,mu.grad(criterion.total(losses),[named[n] for n in selected])
    program=mu.compile(step,device=device,fuse=False)
    losses,grads=program(image,padded['labels'],padded['boxes'],padded['valid'],dn['labels'],dn['boxes_unact'],dn['attn_mask'])
    assert set(losses)==set(expected)
    for name,want in expected.items(): np.testing.assert_allclose(losses[name].item(),want.item(),rtol=8e-4,atol=2e-4,err_msg=name)
    ref_params=dict(reference.named_parameters())
    for name,actual in zip(selected,grads): np.testing.assert_allclose(actual.numpy(),ref_params[name].grad.numpy(),rtol=3e-3,atol=5e-4,err_msg=name)
    for name,value in model.named_buffers():
        if name.endswith(('running_mean','running_var','num_batches_tracked')):
            np.testing.assert_allclose(value.numpy(),reference.state_dict()[name].numpy(),rtol=1e-3,atol=1e-4,err_msg=name)


def test_empty_ground_truth_and_denoising_contract(device):
    p=pad_targets([{'labels':[],'boxes':np.zeros((0,4))}],3)
    assert prepare_denoising(p,3,4,rng=np.random.default_rng(1)) is None
    c=SetCriterion(3)
    def f(l,b,labels,boxes,valid):
        losses=c({'pred_logits':l,'pred_boxes':b},{'labels':labels,'boxes':boxes,'valid':valid})
        return losses,mu.grad(c.total(losses),[l,b])
    losses,grads=mu.compile(f,device=device)(np.zeros((1,4,3),np.float32),np.full((1,4,4),.2,np.float32),p['labels'],p['boxes'],p['valid'])
    assert losses['loss_bbox'].item()==0 and losses['loss_giou'].item()==0
    assert np.all(grads[1].numpy()==0) and np.isfinite(grads[0].numpy()).all()


def test_denoising_is_independent_of_matcher_padding_bucket():
    targets=targets_example()
    exact=prepare_denoising(pad_targets(targets,3),3,8,num_denoising=8,rng=np.random.default_rng(14))
    padded=prepare_denoising(pad_targets(targets,3,slots=8),3,8,num_denoising=8,rng=np.random.default_rng(14))
    for name in exact: np.testing.assert_array_equal(exact[name],padded[name])
