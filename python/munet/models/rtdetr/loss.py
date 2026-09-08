"""RT-DETR v1 matching, varifocal/focal/BCE and box losses on native graphs."""
import numpy as np
from ... import nn
from ...core import as_tensor, cat, operation, where


def cxcywh_to_xyxy(boxes): return cat([boxes[...,:2]-boxes[...,2:]*0.5,boxes[...,:2]+boxes[...,2:]*0.5],-1)


def box_overlap(a,b):
    """Broadcasted paired IoU/GIoU; zero-area padding has finite zero derivatives."""
    a,b=cxcywh_to_xyxy(a),cxcywh_to_xyxy(b)
    lo=a[...,:2].maximum(b[...,:2]);hi=a[...,2:].minimum(b[...,2:])
    size=(hi-lo).clamp(min=0);intersection=size[...,0]*size[...,1]
    sa=a[...,2:]-a[...,:2];sb=b[...,2:]-b[...,:2]
    union=sa[...,0]*sa[...,1]+sb[...,0]*sb[...,1]-intersection
    iou=intersection/where(union>0,union,1)
    outer=b[...,2:].maximum(a[...,2:])-b[...,:2].minimum(a[...,:2])
    area=outer[...,0]*outer[...,1]
    giou=iou-(area-union)/where(area>0,area,1)
    return iou,giou


def one_hot(labels,classes):
    return labels.unsqueeze(-1).eq(np.arange(classes,dtype=np.float32))


class HungarianMatcher(nn.Module):
    def __init__(self,cost_class=2.0,cost_bbox=5.0,cost_giou=2.0,alpha=0.25,gamma=2.0):
        self.cost_class,self.cost_bbox,self.cost_giou=cost_class,cost_bbox,cost_giou
        self.alpha,self.gamma=alpha,gamma
    def forward(self,outputs,targets):
        logits,boxes=outputs['pred_logits'].detach(),outputs['pred_boxes'].detach()
        probabilities=logits.sigmoid()
        selected=probabilities@one_hot(targets['labels'],logits.shape[-1]).transpose(-2,-1)
        negative=(1-self.alpha)*(selected**self.gamma)*(-(1-selected+1e-8).log())
        positive=self.alpha*((1-selected)**self.gamma)*(-(selected+1e-8).log())
        pred,truth=boxes.unsqueeze(2),targets['boxes'].unsqueeze(1)
        l1=(pred-truth).abs().sum(-1)
        _,giou=box_overlap(pred,truth)
        cost=self.cost_class*(positive-negative)+self.cost_bbox*l1-self.cost_giou*giou
        return operation('assignment',cost,targets['valid'])


class SetCriterion(nn.Module):
    def __init__(self,num_classes,weight_dict=None,alpha=0.75,gamma=2.0,classification='vfl',matcher=None):
        if classification not in ('vfl','focal','bce'): raise ValueError("classification must be vfl, focal or bce")
        self.num_classes,self.alpha,self.gamma,self.classification=num_classes,alpha,gamma,classification
        self.weight_dict=weight_dict or {'loss_'+classification:1.0,'loss_bbox':5.0,'loss_giou':2.0}
        self.matcher=matcher or HungarianMatcher()

    def _loss(self,outputs,targets,assignment,normalizer):
        logits,boxes=outputs['pred_logits'],outputs['pred_boxes']
        b,q,c=logits.shape;slots=targets['labels'].shape[1]
        valid=targets['valid']*(assignment>=0)
        matched=boxes.gather(1,assignment.unsqueeze(-1).expand(b,slots,4))
        iou,giou=box_overlap(matched,targets['boxes'])
        mapping=as_tensor(np.arange(q,dtype=np.float32).reshape(1,q,1)).eq(assignment.unsqueeze(1))*valid.unsqueeze(1)
        labels=one_hot(targets['labels'],c)
        target=mapping@labels
        if self.classification=='vfl':
            target_score=mapping@(labels*iou.detach().unsqueeze(-1))
            weight=self.alpha*(logits.sigmoid().detach()**self.gamma)*(1-target)+target_score
            classification=((logits.softplus()-logits*target_score)*weight).sum()/normalizer
        elif self.classification=='focal':
            probability=logits.sigmoid()
            pt=probability*target+(1-probability)*(1-target)
            alpha_t=self.alpha*target+(1-self.alpha)*(1-target)
            classification=((logits.softplus()-logits*target)*alpha_t*((1-pt)**self.gamma)).sum()/normalizer
        else: classification=(logits.softplus()-logits*target).sum()/normalizer
        return {'loss_'+self.classification:classification,
                'loss_bbox':((matched-targets['boxes']).abs()*valid.unsqueeze(-1)).sum()/normalizer,
                'loss_giou':((1-giou)*valid).sum()/normalizer}

    def forward(self,outputs,targets,num_boxes=None):
        # num_boxes may be supplied explicitly for globally normalized accumulation.
        normalizer=targets['valid'].sum().clamp(min=1) if num_boxes is None else as_tensor(num_boxes).clamp(min=1)
        losses={}
        def add(out,truth,indices,norm,suffix):
            for name,value in self._loss(out,truth,indices,norm).items():
                if name in self.weight_dict: losses[name+suffix]=value*self.weight_dict[name]
        add(outputs,targets,self.matcher(outputs,targets),normalizer,'')
        for i,out in enumerate(outputs.get('aux_outputs',[])):
            add(out,targets,self.matcher(out,targets),normalizer,f'_aux_{i}')
        if 'dn_aux_outputs' in outputs:
            meta=outputs['dn_meta'];groups=meta['dn_num_group'];slots=meta['max_gt'];b=targets['valid'].shape[0]
            if not 1<=slots<=targets['valid'].shape[1]: raise ValueError("denoising target bucket mismatch")
            truth={name:value[:,:slots] for name,value in targets.items()}
            truth={name:value.repeat(1,groups,*([1]*(value.ndim-2))) if groups>1 else value for name,value in truth.items()}
            idx=np.concatenate([np.arange(slots,dtype=np.float32)+g*2*slots for g in range(groups)])
            indices=as_tensor(idx.reshape(1,-1)).expand(b,-1)
            for i,out in enumerate(outputs['dn_aux_outputs']): add(out,truth,indices,normalizer*groups,f'_dn_{i}')
        return losses

    @staticmethod
    def total(losses): return sum(losses.values())
