"""Load unchanged upstream math without its application/config registry.

Registry imports and distributed-normalizer discovery are adapted for the
single-process harness. The unused torchvision import is omitted, and its
float32 box_area helper is expressed with the same four-coordinate formula.
No compiled torchvision operators are used by the selected VFL/box recipe.
Source checksums are verified before execution.
"""
import hashlib
import json
from pathlib import Path
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[3]
_loaded=None


def load_reference():
    global _loaded
    if _loaded is not None:return _loaded
    manifest=json.loads((ROOT/'examples/rtdetr/tests/rtdetr-reference.json').read_text())
    folder=ROOT/'artifacts/rtdetr-reference'
    package=types.ModuleType('_munet_upstream_rtdetr');package.__path__=[str(folder)];sys.modules[package.__name__]=package
    for name in ['box_ops','utils','common','presnet','denoising','hybrid_encoder','rtdetr_decoder','matcher','rtdetr_criterion']:
        filename=name+'.py';path=folder/filename
        if not path.is_file(): raise RuntimeError('Run python examples/rtdetr/fetch_reference.py before the required detector parity gate')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['files'][filename]['sha256']: raise RuntimeError('upstream source checksum mismatch')
        text=data.decode().replace('from src.core import register','register = lambda cls: cls')
        text=text.replace('from src.misc.dist import get_world_size, is_dist_available_and_initialized','get_world_size = lambda: 1\nis_dist_available_and_initialized = lambda: False')
        text=text.replace('import torchvision\n','')
        text=text.replace('from torchvision.ops.boxes import box_area','def box_area(boxes):\n    return (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])')
        module=types.ModuleType(package.__name__+'.'+name);module.__package__=package.__name__;module.__file__=str(path)
        sys.modules[module.__name__]=module;exec(compile(text,str(path),'exec'),module.__dict__);setattr(package,name,module)
    _loaded=package;return package


def model_for(config):
    ref=load_reference()
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=ref.presnet.PResNet(config['backbone_depth'],variant='d',return_idx=[1,2,3],freeze_at=config['freeze_at'],freeze_norm=config['freeze_norm'],pretrained=False)
            channels=[512,1024,2048] if config['backbone_depth']>=50 else [128,256,512]
            self.encoder=ref.hybrid_encoder.HybridEncoder(in_channels=channels,hidden_dim=config['hidden_dim'],nhead=config['nhead'],dim_feedforward=config['dim_feedforward'],expansion=config['expansion'],depth_mult=config['depth_mult'],eval_spatial_size=config['eval_spatial_size'])
            self.decoder=ref.rtdetr_decoder.RTDETRTransformer(num_classes=config['num_classes'],hidden_dim=config['hidden_dim'],num_queries=config['num_queries'],feat_channels=[config['hidden_dim']]*3,nhead=config['nhead'],num_decoder_layers=config['num_decoder_layers'],num_decoder_points=config['num_decoder_points'],dim_feedforward=config['dim_feedforward'],num_denoising=config['num_denoising'],eval_spatial_size=config['eval_spatial_size'])
        def forward(self,x,targets=None,denoising=None):
            if denoising is None:return self.decoder(self.encoder(self.backbone(x)),targets)
            original=ref.rtdetr_decoder.get_contrastive_denoising_training_group
            def prepared(*args,**kwargs):
                labels=torch.tensor(denoising['labels'],dtype=torch.long)
                boxes=torch.tensor(denoising['boxes_unact'])
                mask=torch.tensor(denoising['attn_mask'],dtype=torch.bool)
                slots,groups=denoising['max_gt'],denoising['num_group']
                indices=[torch.cat([torch.arange(len(t['labels']))+g*2*slots for g in range(groups)]) for t in targets]
                meta={'dn_num_group':groups,'dn_num_split':[labels.shape[1],config['num_queries']],'dn_positive_idx':indices}
                return self.decoder.denoising_class_embed(labels),boxes,mask,meta
            # Common random inputs isolate model/loss parity from different PRNG sequences.
            ref.rtdetr_decoder.get_contrastive_denoising_training_group=prepared
            try:return self.decoder(self.encoder(self.backbone(x)),targets)
            finally:ref.rtdetr_decoder.get_contrastive_denoising_training_group=original
    return Detector()


def criterion_for(classes):
    r=load_reference()
    matcher=r.matcher.HungarianMatcher({'cost_class':2,'cost_bbox':5,'cost_giou':2},use_focal_loss=True,alpha=0.25,gamma=2)
    return r.rtdetr_criterion.SetCriterion(matcher,{'loss_vfl':1,'loss_bbox':5,'loss_giou':2},['vfl','boxes'],alpha=0.75,gamma=2,num_classes=classes)
