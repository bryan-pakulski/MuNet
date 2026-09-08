"""Native graph layers with PyTorch-compatible layouts and persistent train/eval state."""
import math
from collections import OrderedDict
import numpy as np
from contextvars import ContextVar

_initialization_rng=ContextVar("munet_initialization_rng",default=None)
def _get_rng(rng):
    return rng if rng is not None else _initialization_rng.get() or np.random.default_rng()


def manual_seed(seed):
    """Seed subsequent MuNet layer initialization and dropout; NumPy data RNGs are separate."""
    _initialization_rng.set(np.random.default_rng(seed))

from .core import Parameter, Buffer, Tensor, _active, _trace, as_tensor, operation, cat, where, update_state, current_state


def _pair(value):
    values = tuple(value) if isinstance(value,(tuple,list)) else (value,value)
    if len(values)!=2: raise ValueError("expected two spatial dimensions")
    return values


class Module:
    training = True

    def __call__(self, *args, **kwargs):
        ctx = _active.get()
        if ctx is None:
            raise RuntimeError("MuNet models run inside a compiled function. For inference use model.eval().compile().predict(inputs); for training use munet.train_step or @munet.compile.")
        ctx.modules[self] = self.training
        return self.forward(*args, **kwargs)

    def compile(self, *, device="vulkan", fuse=True):
        """Create a compiled callable; preserves the current train/eval mode."""
        from .core import compile
        return compile(self, device=device, fuse=fuse)

    def export(self, path, example_inputs, **options):
        """Export inference in eval mode; see munet.export for naming/shader options."""
        from .api import export
        return export(self, path, example_inputs, **options)

    def __repr__(self):
        children = list(self.named_children())
        if not children:
            details = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                                if not k.startswith("_") and k != "training" and isinstance(v, (int, float, str, tuple)))
            return f"{type(self).__name__}({details})"
        lines = [f"{type(self).__name__}("]
        for name, child in children:
            lines.append(f"  ({name}): " + repr(child).replace("\n", "\n  "))
        return "\n".join([*lines, ")"])

    def _named_children(self): return vars(self).items()
    def named_children(self): return ((n,v) for n,v in self._named_children() if isinstance(v,Module))
    def register_buffer(self,name,value,persistent=True): setattr(self,name,Buffer(value,persistent=persistent))

    def _states(self):
        seen = set()
        def walk(value, prefix):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value)); yield prefix, value
            elif isinstance(value, Module):
                for name, child in value._named_children():
                    yield from walk(child, f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple, dict)):
                for name, child in (value.items() if isinstance(value,dict) else enumerate(value)):
                    yield from walk(child, f"{prefix}.{name}" if prefix else str(name))
        yield from walk(self, "")

    def named_parameters(self): return ((n,p) for n,p in self._states() if not isinstance(p,Buffer))
    def named_buffers(self): return ((n,p) for n,p in self._states() if isinstance(p,Buffer))
    def parameters(self): return [p for _,p in self.named_parameters()]
    def buffers(self): return [p for _,p in self.named_buffers()]
    def state_dict(self): return {n:p.numpy() for n,p in self._states() if not isinstance(p,Buffer) or p.persistent}

    def modules(self):
        seen=set()
        def walk(x):
            if isinstance(x,Module):
                if id(x) in seen: return
                seen.add(id(x));yield x
                for _,v in x._named_children(): yield from walk(v)
            elif isinstance(x,(list,tuple,dict)):
                for v in (x.values() if isinstance(x,dict) else x): yield from walk(v)
        yield from walk(self)

    def train(self, mode=True):
        for m in self.modules(): m.training=bool(mode)
        return self

    def eval(self): return self.train(False)

    def requires_grad_(self, enabled=True):
        for p in self.parameters(): p.requires_grad=bool(enabled)
        return self

    def load_state_dict(self, state, strict=True):
        params={n:p for n,p in self._states() if not isinstance(p,Buffer) or p.persistent}
        missing,extra=set(params)-set(state),set(state)-set(params)
        if strict and (missing or extra): raise ValueError(f"state keys differ; missing={missing}, extra={extra}")
        converted={}
        for name,value in state.items():
            if name not in params: continue
            if hasattr(value,"detach"): value=value.detach().cpu().numpy()
            value=np.asarray(value,dtype=np.float32)
            if value.shape!=params[name].shape: raise ValueError(f"state shape mismatch for {name}: {value.shape} != {params[name].shape}")
            converted[name]=value
        for name,value in converted.items(): params[name].assign(value)
        return {"missing_keys":sorted(missing),"unexpected_keys":sorted(extra)}


class Linear(Module):
    def __init__(self,in_features,out_features,bias=True,*,rng=None):
        if in_features<=0 or out_features<=0: raise ValueError("layer dimensions must be positive")
        rng=_get_rng(rng)
        limit=1/math.sqrt(in_features)
        self.in_features,self.out_features=in_features,out_features
        self.weight=Parameter(rng.uniform(-limit,limit,(out_features,in_features)))
        self.bias=Parameter(rng.uniform(-limit,limit,(out_features,))) if bias else None

    def forward(self,x):
        y=x@self.weight.T
        return y+self.bias if self.bias is not None else y


class ModuleList(Module):
    def __init__(self,layers=()): self.layers=list(layers)
    def __iter__(self): return iter(self.layers)
    def __len__(self): return len(self.layers)
    def __getitem__(self,i): return self.layers[i]
    def append(self,layer): self.layers.append(layer); return self
    def __setattr__(self,name,value):
        if name.isdigit() and "layers" in vars(self): self.layers[int(name)]=value
        else: object.__setattr__(self,name,value)
    def _named_children(self): return ((str(i),layer) for i,layer in enumerate(self.layers))


class Sequential(ModuleList):
    def __init__(self,*layers):
        self.names=list(layers[0]) if len(layers)==1 and isinstance(layers[0],dict) else [str(i) for i in range(len(layers))]
        self.layers=list(layers[0].values()) if len(layers)==1 and isinstance(layers[0],dict) else list(layers)
    def _named_children(self): return zip(self.names,self.layers)
    def __getattr__(self,name):
        names=vars(self).get("names",[])
        if name in names: return self.layers[names.index(name)]
        raise AttributeError(name)
    def __setattr__(self,name,value):
        names=vars(self).get("names",[])
        if name in names and "layers" in vars(self): self.layers[names.index(name)]=value
        else: object.__setattr__(self,name,value)
    def forward(self,x):
        for layer in self.layers: x=layer(x)
        return x


class Identity(Module):
    def forward(self,x): return x
class ReLU(Module):
    def forward(self,x): return x.relu()
class Sigmoid(Module):
    def forward(self,x): return x.sigmoid()
class SiLU(Module):
    def forward(self,x): return x*x.sigmoid()
class GELU(Module):
    def forward(self,x): return x.gelu()
class MSELoss(Module):
    def forward(self,prediction,target): return (prediction-target).square().mean()


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1): self.start_dim, self.end_dim = start_dim, end_dim
    def forward(self, x): return x.flatten(self.start_dim, self.end_dim)


class CrossEntropyLoss(Module):
    """Class-index cross entropy; use dim=-1 for (batch, sequence, classes)."""
    def __init__(self, dim=1, reduction="mean"):
        if reduction not in ("none", "mean", "sum"): raise ValueError("invalid loss reduction")
        self.dim, self.reduction = dim, reduction
    def forward(self, logits, targets):
        return functional.cross_entropy(logits, targets, self.dim, self.reduction)


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction="mean"):
        if reduction not in ("none", "mean", "sum"): raise ValueError("invalid loss reduction")
        self.reduction = reduction
    def forward(self, logits, targets):
        return functional.binary_cross_entropy_with_logits(logits, targets, self.reduction)


class Conv2d(Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,*,rng=None):
        k=_pair(kernel_size)
        if min(in_channels,out_channels,*k,groups)<=0 or in_channels%groups or out_channels%groups: raise ValueError("invalid convolution dimensions/groups")
        self.in_channels,self.out_channels=in_channels,out_channels
        self.kernel_size,self.stride,self.padding,self.dilation,self.groups=k,_pair(stride),_pair(padding),_pair(dilation),groups
        rng=_get_rng(rng)
        limit=1/math.sqrt(in_channels//groups*math.prod(k))
        self.weight=Parameter(rng.uniform(-limit,limit,(out_channels,in_channels//groups,*k)))
        self.bias=Parameter(rng.uniform(-limit,limit,out_channels)) if bias else None
    def forward(self,x): return functional.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class MaxPool2d(Module):
    def __init__(self,kernel_size,stride=None,padding=0,ceil_mode=False):
        self.kernel_size,self.stride,self.padding,self.ceil_mode=kernel_size,stride,padding,ceil_mode
    def forward(self,x): return functional.max_pool2d(x,self.kernel_size,self.stride,self.padding,self.ceil_mode)


class AvgPool2d(MaxPool2d):
    def __init__(self,kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True):
        super().__init__(kernel_size,stride,padding,ceil_mode);self.count_include_pad=count_include_pad
    def forward(self,x): return functional.avg_pool2d(x,self.kernel_size,self.stride,self.padding,self.ceil_mode,self.count_include_pad)


class BatchNorm2d(Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True):
        if num_features<=0 or eps<=0 or (momentum is not None and not 0<=momentum<=1): raise ValueError("invalid BatchNorm configuration")
        self.num_features,self.eps,self.momentum=num_features,float(eps),momentum
        self.weight=Parameter(np.ones(num_features,np.float32)) if affine else None
        self.bias=Parameter(np.zeros(num_features,np.float32)) if affine else None
        self.running_mean=Buffer(np.zeros(num_features,np.float32)) if track_running_stats else None
        self.running_var=Buffer(np.ones(num_features,np.float32)) if track_running_stats else None
        self.num_batches_tracked=Buffer(np.array(0,np.float32)) if track_running_stats else None
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=self.num_features: raise ValueError("BatchNorm2d requires NCHW with matching channels")
        if self.training or self.running_mean is None:
            count=x.shape[0]*x.shape[2]*x.shape[3]
            if count<=1: raise ValueError("training BatchNorm requires more than one value per channel")
            mean=x.mean((0,2,3),keepdim=True)
            variance=(x-mean).square().mean((0,2,3),keepdim=True)
            if self.running_mean is not None:
                batches=current_state(self.num_batches_tracked)+1
                factor=1/batches if self.momentum is None else self.momentum
                update_state(self.num_batches_tracked,batches)
                update_state(self.running_mean,(1-factor)*current_state(self.running_mean)+factor*mean.reshape(-1).detach())
                update_state(self.running_var,(1-factor)*current_state(self.running_var)+factor*variance.reshape(-1).detach()*(count/(count-1)))
        else:
            mean=self.running_mean.reshape(1,-1,1,1)
            variance=self.running_var.reshape(1,-1,1,1)
        y=(x-mean)/(variance+self.eps).sqrt()
        if self.weight is not None: y=y*self.weight.reshape(1,-1,1,1)+self.bias.reshape(1,-1,1,1)
        return y


class FrozenBatchNorm2d(Module):
    def __init__(self,num_features,eps=1e-5):
        self.eps=eps
        self.weight=Buffer(np.ones(num_features,np.float32));self.bias=Buffer(np.zeros(num_features,np.float32))
        self.running_mean=Buffer(np.zeros(num_features,np.float32));self.running_var=Buffer(np.ones(num_features,np.float32))
    def forward(self,x):
        scale=self.weight.reshape(1,-1,1,1)/(self.running_var.reshape(1,-1,1,1)+self.eps).sqrt()
        return x*scale+self.bias.reshape(1,-1,1,1)-self.running_mean.reshape(1,-1,1,1)*scale


class LayerNorm(Module):
    def __init__(self,normalized_shape,eps=1e-5,elementwise_affine=True,bias=True):
        self.normalized_shape=(normalized_shape,) if isinstance(normalized_shape,int) else tuple(normalized_shape)
        self.eps=eps
        self.weight=Parameter(np.ones(self.normalized_shape,np.float32)) if elementwise_affine else None
        self.bias=Parameter(np.zeros(self.normalized_shape,np.float32)) if elementwise_affine and bias else None
    def forward(self,x):
        if x.shape[-len(self.normalized_shape):]!=self.normalized_shape: raise ValueError("LayerNorm shape mismatch")
        axes=tuple(range(x.ndim-len(self.normalized_shape),x.ndim))
        centered=x-x.mean(axes,keepdim=True)
        y=centered/(centered.square().mean(axes,keepdim=True)+self.eps).sqrt()
        if self.weight is not None: y=y*self.weight
        return y+self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self,num_embeddings,embedding_dim,padding_idx=None,*,rng=None):
        rng=_get_rng(rng)
        self.padding_idx=padding_idx
        values=rng.normal(size=(num_embeddings,embedding_dim)).astype(np.float32)
        if padding_idx is not None: values[padding_idx]=0
        self.weight=Parameter(values)
    def forward(self,indices):
        x=self.weight.take(indices)
        return x if self.padding_idx is None else where(indices.eq(self.padding_idx).unsqueeze(-1),x.detach(),x)


class Dropout(Module):
    def __init__(self,p=0.5,*,rng=None):
        if not 0<=p<1: raise ValueError("dropout probability must lie in [0,1)")
        self.p=float(p)
        if p:
            rng=_get_rng(rng)
            self.seed=Buffer(np.array(rng.integers(0,16777216),np.float32),persistent=False)
            self.counter=Buffer(np.array(0,np.float32),persistent=False)
    def forward(self,x):
        if not self.training or not self.p: return x
        step=current_state(self.counter)
        noise=operation("random_uniform",self.seed,step,attrs=x.shape)
        update_state(self.counter,step+1)
        return where(noise>=self.p,x/(1-self.p),0)


class MultiheadAttention(Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0,bias=True,batch_first=True,*,rng=None):
        if embed_dim%num_heads or not batch_first: raise ValueError("attention requires divisible heads and batch_first=True")
        self.embed_dim,self.num_heads=embed_dim,num_heads
        rng=_get_rng(rng)
        limit=math.sqrt(6/(4*embed_dim))
        self.in_proj_weight=Parameter(rng.uniform(-limit,limit,(3*embed_dim,embed_dim)))
        self.in_proj_bias=Parameter(np.zeros(3*embed_dim,np.float32)) if bias else None
        self.out_proj=Linear(embed_dim,embed_dim,bias,rng=rng)
        if self.out_proj.bias is not None: self.out_proj.bias.assign(np.zeros(embed_dim,np.float32))
        self.dropout=Dropout(dropout,rng=rng)
    def forward(self,query,key,value,attn_mask=None,key_padding_mask=None,need_weights=False):
        b,l,e=query.shape; h=self.num_heads; d=e//h
        projected=[]
        for i,x in enumerate((query,key,value)):
            p=x@self.in_proj_weight[i*e:(i+1)*e].T
            if self.in_proj_bias is not None: p=p+self.in_proj_bias[i*e:(i+1)*e]
            projected.append(p.reshape(b,-1,h,d).permute(0,2,1,3))
        q,k,v=projected
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        if attn_mask is not None:
            if attn_mask.ndim==3: attn_mask=attn_mask.reshape(b,h,l,key.shape[1])
            scores=where(attn_mask, -1e9, scores)
        if key_padding_mask is not None: scores=where(key_padding_mask.reshape(b,1,1,-1),-1e9,scores)
        weights=self.dropout(scores.softmax(-1))
        out=(weights@v).permute(0,2,1,3).reshape(b,l,e)
        return self.out_proj(out),weights.mean(1) if need_weights else None


class functional:
    @staticmethod
    def cross_entropy(logits, targets, dim=1, reduction="mean"):
        if not -logits.ndim <= dim < logits.ndim: raise ValueError("class dimension out of range")
        dim %= logits.ndim
        if targets.shape != logits.shape[:dim] + logits.shape[dim + 1:]:
            raise ValueError("cross entropy targets must match logits with the class axis removed")
        shifted = logits - logits.amax(dim, keepdim=True).detach()
        logp = shifted - shifted.exp().sum(dim, keepdim=True).log()
        loss = -logp.gather(dim, targets.unsqueeze(dim)).squeeze(dim)
        if reduction == "none": return loss
        if reduction == "sum": return loss.sum()
        if reduction == "mean": return loss.mean()
        raise ValueError("invalid loss reduction")
    @staticmethod
    def mse_loss(prediction,target): return (prediction-target).square().mean()
    @staticmethod
    def relu(x): return x.relu()
    @staticmethod
    def sigmoid(x): return x.sigmoid()
    @staticmethod
    def silu(x): return x*x.sigmoid()
    @staticmethod
    def gelu(x): return x.gelu()
    @staticmethod
    def softmax(x,dim=-1): return x.softmax(dim)
    @staticmethod
    def conv2d(x,weight,bias=None,stride=1,padding=0,dilation=1,groups=1):
        y=operation("conv2d",x,weight,attrs=(*_pair(stride),*_pair(padding),*_pair(dilation),groups))
        return y if bias is None else y+bias.reshape(1,-1,1,1)
    @staticmethod
    def max_pool2d(x,kernel_size,stride=None,padding=0,ceil_mode=False):
        return operation("max_pool2d",x,attrs=(*_pair(kernel_size),*_pair(kernel_size if stride is None else stride),*_pair(padding),int(ceil_mode),0))
    @staticmethod
    def avg_pool2d(x,kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True):
        return operation("avg_pool2d",x,attrs=(*_pair(kernel_size),*_pair(kernel_size if stride is None else stride),*_pair(padding),int(ceil_mode),int(count_include_pad)))
    @staticmethod
    def grid_sample(x,grid,mode="bilinear",padding_mode="zeros",align_corners=False):
        if (mode,padding_mode,align_corners)!=("bilinear","zeros",False): raise ValueError("grid_sample supports bilinear/zeros/align_corners=False")
        return operation("grid_sample",x,grid)
    @staticmethod
    def interpolate(x,size=None,scale_factor=None,mode="nearest",align_corners=None):
        if mode!="nearest" or align_corners is not None: raise ValueError("interpolate supports nearest mode; use grid_sample for differentiable bilinear sampling")
        if size is None:
            if scale_factor is None: raise ValueError("interpolate requires size or scale_factor")
            factors=_pair(scale_factor);size=tuple(int(s*f) for s,f in zip(x.shape[-2:],factors))
        return operation("resize_nearest",x,attrs=_pair(size))
    @staticmethod
    def binary_cross_entropy_with_logits(logits,targets,reduction="mean"):
        loss=logits.softplus()-logits*targets
        if reduction=="none": return loss
        if reduction=="sum": return loss.sum()
        if reduction=="mean": return loss.mean()
        raise ValueError("invalid loss reduction")
