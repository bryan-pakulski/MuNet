"""Optimizers and training state compiled into native device updates."""
import math
import numpy as np
from .core import Buffer, _trace, update_state, current_state, as_tensor


def clip_grad_norm_(parameters,max_norm,norm_type=2.0):
    if norm_type!=2.0 or not math.isfinite(max_norm) or max_norm<=0: raise ValueError("clip_grad_norm_ requires a positive L2 bound")
    ctx=_trace();params=list(dict.fromkeys(parameters));squared=as_tensor(0.0)
    for p in params:
        if p.grad is not None: squared=squared+p.grad.square().sum()
    norm=squared.sqrt();scale=(max_norm/(norm+1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None: ctx.grads[p]=p.grad*scale
    return norm


class SGD:
    """SGD without momentum; persistent updates share the forward/backward replay."""
    def __init__(self,parameters,lr=0.01):
        self.parameters=list(dict.fromkeys(parameters));self.lr=float(lr)
        if not math.isfinite(self.lr) or self.lr<0: raise ValueError("learning rate must be finite and nonnegative")
    def zero_grad(self): _trace().grads.clear()
    def step(self):
        ctx=_trace()
        if ctx.optimizer_stepped: raise RuntimeError("one optimizer step per trace is supported")
        if not ctx.backward_called: raise RuntimeError("call loss.backward() before optimizer.step()")
        for p in self.parameters:
            if p.grad is not None: update_state(p,p-self.lr*p.grad)
        ctx.optimizer_stepped=True


class AdamW:
    """Float32 AdamW with per-parameter moments/steps and live parameter-group settings."""
    def __init__(self,parameters,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        values=list(parameters)
        groups=values if values and isinstance(values[0],dict) else [{"params":values}]
        self.param_groups=[];self.state={};self._hyper=[];seen=set()
        for group in groups:
            g={"lr":lr,"betas":tuple(betas),"eps":eps,"weight_decay":weight_decay,**group}
            g["params"]=list(g["params"])
            for p in g["params"]:
                if id(p) in seen: raise ValueError("parameter occurs in multiple optimizer groups")
                seen.add(id(p))
            self.param_groups.append(g)
            self._hyper.append({k:Buffer(np.array(v,np.float32)) for k,v in self._values(g).items()})
        self._sync_hyperparameters()

    @staticmethod
    def _values(g):
        b1,b2=g["betas"]
        values={"lr":float(g["lr"]),"beta1":float(b1),"beta2":float(b2),"eps":float(g["eps"]),"decay":float(g["weight_decay"])}
        if not all(math.isfinite(v) for v in values.values()) or values["lr"]<0 or values["eps"]<=0 or values["decay"]<0 or not (0<=b1<1 and 0<=b2<1): raise ValueError("invalid AdamW hyperparameters")
        return values

    def _sync_hyperparameters(self):
        for group,buffers in zip(self.param_groups,self._hyper):
            for key,value in self._values(group).items():
                if float(buffers[key]._array)!=float(np.float32(value)): buffers[key].assign(np.array(value,np.float32))

    def _state(self,p):
        if p not in self.state:
            self.state[p]={"exp_avg":Buffer(np.zeros(p.shape,np.float32)),"exp_avg_sq":Buffer(np.zeros(p.shape,np.float32)),"step":Buffer(np.array(0,np.float32))}
        return self.state[p]

    def zero_grad(self): _trace().grads.clear()

    def step(self):
        ctx=_trace()
        if not ctx.backward_called or ctx.optimizer_stepped: raise RuntimeError("AdamW requires one backward call followed by one step")
        if not hasattr(ctx,"optimizers"): ctx.optimizers=[]
        ctx.optimizers.append(self)
        for group,h in zip(self.param_groups,self._hyper):
            for p in group["params"]:
                if p.grad is None: continue
                state=self._state(p);grad=p.grad
                t=state["step"]+1
                m=h["beta1"]*state["exp_avg"]+(1-h["beta1"])*grad
                v=h["beta2"]*state["exp_avg_sq"]+(1-h["beta2"])*grad.square()
                bias1=1-(h["beta1"].log()*t).exp()
                bias2=1-(h["beta2"].log()*t).exp()
                next_p=p*(1-h["lr"]*h["decay"])-(h["lr"]/bias1)*m/((v/bias2).sqrt()+h["eps"])
                update_state(state["step"],t);update_state(state["exp_avg"],m);update_state(state["exp_avg_sq"],v);update_state(p,next_p)
        ctx.optimizer_stepped=True

    def state_dict(self):
        groups=[];state={};index=0
        for group in self.param_groups:
            ids=[]
            for p in group["params"]:
                ids.append(index)
                if p in self.state: state[str(index)]={k:v.numpy() for k,v in self.state[p].items()}
                index+=1
            groups.append({k:v for k,v in group.items() if k!="params"}|{"params":ids})
        return {"param_groups":groups,"state":state}

    def load_state_dict(self,payload):
        groups=payload["param_groups"]
        if len(groups)!=len(self.param_groups): raise ValueError("optimizer group count mismatch")
        pending=[];reset=[]
        for current,saved in zip(self.param_groups,groups):
            if len(current["params"])!=len(saved["params"]): raise ValueError("optimizer parameter count mismatch")
            self._values(saved)
            for p,index in zip(current["params"],saved["params"]):
                if str(index) not in payload["state"]:
                    if p in self.state: reset.append(p)
                    continue
                values=payload["state"][str(index)]
                if set(values)!={"step","exp_avg","exp_avg_sq"}: raise ValueError("invalid optimizer state")
                for k,v in values.items():
                    array=np.asarray(v,np.float32)
                    if array.shape!=(() if k=="step" else p.shape) or not np.isfinite(array).all(): raise ValueError("invalid optimizer tensor")
                    pending.append((p,k,array))
        for current,saved in zip(self.param_groups,groups):
            current.update({k:v for k,v in saved.items() if k!="params"})
        for p in reset:
            for value in self.state[p].values(): value.assign(np.zeros(value.shape,np.float32))
        for p,k,value in pending: self._state(p)[k].assign(value)
        self._sync_hyperparameters()


class MultiStepLR:
    def __init__(self,optimizer,milestones,gamma=0.1):
        self.optimizer=optimizer;self.milestones=sorted(int(x) for x in milestones);self.gamma=float(gamma);self.last_epoch=0
        self.base_lrs=[g["lr"] for g in optimizer.param_groups]
        if gamma<=0: raise ValueError("scheduler gamma must be positive")
    def step(self,epoch=None):
        self.last_epoch=self.last_epoch+1 if epoch is None else int(epoch)
        factor=self.gamma**sum(self.last_epoch>=m for m in self.milestones)
        for group,base in zip(self.optimizer.param_groups,self.base_lrs): group["lr"]=base*factor
        self.optimizer._sync_hyperparameters()
    def state_dict(self): return {"milestones":self.milestones,"gamma":self.gamma,"last_epoch":self.last_epoch,"base_lrs":self.base_lrs}
    def load_state_dict(self,state):
        if state["milestones"]!=self.milestones or state["gamma"]!=self.gamma or len(state["base_lrs"])!=len(self.base_lrs): raise ValueError("scheduler configuration mismatch")
        self.base_lrs=state["base_lrs"];self.step(state["last_epoch"])


class ModelEMA:
    """EMA shadow state and warm-up counter can update in the training graph."""
    def __init__(self,model,decay=0.9999,warmups=2000):
        if not 0<=decay<1 or warmups<0: raise ValueError("invalid EMA configuration")
        self.decay,self.warmups=decay,warmups
        self.steps=Buffer(np.array(0,np.float32))
        self.model=model
        self.shadow={n:Buffer(p.numpy()) for n,p in model._states() if not isinstance(p,Buffer) or p.persistent}
    def update(self):
        count=current_state(self.steps)+1
        decay=self.decay*(1-(-count/self.warmups).exp()) if self.warmups else self.decay
        update_state(self.steps,count)
        for name,p in self.model._states():
            if name in self.shadow:
                value=current_state(p)
                next_value=value if name.endswith("num_batches_tracked") else self.shadow[name]*decay+(1-decay)*value
                update_state(self.shadow[name],next_value)
    def state_dict(self): return {"steps":self.steps.numpy(),"shadow":{n:p.numpy() for n,p in self.shadow.items()},"decay":self.decay,"warmups":self.warmups}
    def load_state_dict(self,state):
        if set(state["shadow"])!=set(self.shadow) or state["decay"]!=self.decay or state["warmups"]!=self.warmups: raise ValueError("EMA configuration/state mismatch")
        for n,value in state["shadow"].items():
            if np.shape(value)!=self.shadow[n].shape: raise ValueError("EMA tensor shape mismatch")
        for n,value in state["shadow"].items(): self.shadow[n].assign(value)
        self.steps.assign(state["steps"])
    def copy_to(self,model): model.load_state_dict({n:p.numpy() for n,p in self.shadow.items()})
