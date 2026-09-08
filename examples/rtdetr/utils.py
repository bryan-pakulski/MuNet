import math
from munet import nn
from munet.core import cat, stack


def get_activation(act):
    if act is None: return nn.Identity()
    types={'relu':nn.ReLU,'silu':nn.SiLU,'gelu':nn.GELU,'sigmoid':nn.Sigmoid}
    if act not in types: raise ValueError(f"unsupported activation: {act}")
    return types[act]()


def inverse_sigmoid(x,eps=1e-5):
    x=x.clamp(min=0,max=1)
    return (x.clamp(min=eps)/(1-x).clamp(min=eps)).log()


def bias_init_with_prob(prob=0.01): return -math.log((1-prob)/prob)


def deformable_attention_core_func(value,spatial_shapes,sampling_locations,attention_weights):
    b,_,heads,channels=value.shape
    queries,levels,points=sampling_locations.shape[1],len(spatial_shapes),sampling_locations.shape[-2]
    if sum(h*w for h,w in spatial_shapes)!=value.shape[1]: raise ValueError("deformable feature shapes do not cover values")
    values=value.split([h*w for h,w in spatial_shapes],1)
    grids=2*sampling_locations-1
    sampled=[]
    for level,(h,w) in enumerate(spatial_shapes):
        image=values[level].flatten(2).permute(0,2,1).reshape(b*heads,channels,h,w)
        grid=grids[:,:,:,level].permute(0,2,1,3,4).reshape(b*heads,queries,points,2)
        sampled.append(nn.functional.grid_sample(image,grid))
    weights=attention_weights.permute(0,2,1,3,4).reshape(b*heads,1,queries,levels*points)
    result=(stack(sampled,dim=-2).flatten(-2)*weights).sum(-1).reshape(b,heads*channels,queries)
    return result.permute(0,2,1)
