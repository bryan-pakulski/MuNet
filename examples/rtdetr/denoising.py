"""Host data preparation for fresh contrastive denoising inputs on every step.

The generator and its state are explicit; replay never freezes a random draw.
Image transforms and denoising are authoring/data work, not fallback model ops.
"""
import numpy as np


def pad_targets(targets,num_classes,slots=None):
    if not targets: raise ValueError("a batch must contain at least one image")
    counts=[len(t['labels']) for t in targets]
    needed=max(counts,default=0)
    slots=max(1,needed) if slots is None else int(slots)
    if slots<max(1,needed): raise ValueError("target bucket would truncate ground truth")
    labels=np.full((len(targets),slots),num_classes,np.float32)
    boxes=np.zeros((len(targets),slots,4),np.float32)
    valid=np.zeros((len(targets),slots),np.float32)
    for b,target in enumerate(targets):
        lab=np.asarray(target['labels']);box=np.asarray(target['boxes'],np.float32)
        if lab.ndim!=1 or box.shape!=(len(lab),4): raise ValueError("expected labels[N] and normalized cxcywh boxes[N,4]")
        if not np.isfinite(lab).all() or not np.isfinite(box).all() or np.any(lab!=np.floor(lab)) or np.any((lab<0)|(lab>=num_classes)): raise ValueError("invalid class labels or non-finite targets")
        if np.any((box<0)|(box>1)) or np.any(box[:,2:]<=0): raise ValueError("target boxes must be normalized with positive width/height")
        n=len(lab);labels[b,:n]=lab;boxes[b,:n]=box;valid[b,:n]=1
    return {'labels':labels,'boxes':boxes,'valid':valid}


def prepare_denoising(padded,num_classes,num_queries,*,num_denoising=100,label_noise_ratio=0.5,box_noise_scale=1.0,rng):
    if num_denoising<=0 or not np.any(padded['valid']): return None
    if not 0<=label_noise_ratio<=1 or box_noise_scale<0: raise ValueError("invalid denoising noise settings")
    batch=padded['labels'].shape[0]
    # Matcher buckets must not change the reference's denoising group count.
    # pad_targets places valid entries first, so trim only trailing bucket padding.
    slots=int(padded['valid'].sum(1).max())
    groups=max(num_denoising//slots,1)
    labels=np.tile(padded['labels'][:,:slots],(1,2*groups))
    boxes=np.tile(padded['boxes'][:,:slots],(1,2*groups,1))
    valid=np.tile(padded['valid'][:,:slots],(1,2*groups)).astype(bool)
    negative=np.tile(np.concatenate([np.zeros(slots),np.ones(slots)]),(batch,groups)).astype(np.float32)[...,None]
    if label_noise_ratio:
        noisy=rng.random(labels.shape)<label_noise_ratio*0.5
        new=rng.integers(0,num_classes,size=labels.shape)
        labels=np.where(noisy&valid,new,labels).astype(np.float32)
    if box_noise_scale:
        corners=np.concatenate([boxes[...,:2]-boxes[...,2:]*0.5,boxes[...,:2]+boxes[...,2:]*0.5],-1)
        distance=np.tile(boxes[...,2:]*0.5,(1,1,2))*box_noise_scale
        sign=rng.integers(0,2,size=boxes.shape).astype(np.float32)*2-1
        magnitude=rng.random(boxes.shape).astype(np.float32)+negative
        corners=np.clip(corners+magnitude*sign*distance,0,1)
        boxes=np.concatenate([(corners[...,:2]+corners[...,2:])*0.5,corners[...,2:]-corners[...,:2]],-1)
        # Preserve v1's inverse sigmoid convention, including the noise-disabled branch.
        boxes=np.clip(boxes,0,1)
        boxes=np.log(np.clip(boxes,1e-5,None)/np.clip(1-boxes,1e-5,None))
    count=2*slots*groups;mask=np.zeros((count+num_queries,count+num_queries),np.float32)
    mask[count:,:count]=1
    for group in range(groups):
        start,end=group*2*slots,(group+1)*2*slots
        mask[start:end,:start]=1;mask[start:end,end:count]=1
    return {'labels':labels.astype(np.float32),'boxes_unact':boxes.astype(np.float32),'attn_mask':mask,'num_group':groups,'max_gt':slots}
