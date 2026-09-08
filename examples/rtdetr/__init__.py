"""RT-DETR v1: ResNet-vd, hybrid encoder, and iterative deformable decoder."""
import numpy as np
from munet import nn
from .backbone import PResNet
from .encoder import HybridEncoder
from .decoder import RTDETRTransformer


class RTDETR(nn.Module):
    def __init__(self,num_classes=80,*,backbone_depth=50,hidden_dim=256,nhead=8,
                 num_queries=300,num_decoder_layers=6,num_decoder_points=4,
                 dim_feedforward=1024,num_denoising=100,expansion=1.0,depth_mult=1.0,
                 freeze_at=0,freeze_norm=True,eval_spatial_size=None,seed=0):
        if num_classes<=0 or num_queries<=0 or num_decoder_layers<=0: raise ValueError("invalid detector dimensions")
        if hidden_dim%4 or hidden_dim%nhead: raise ValueError("hidden dimension must be divisible by four and the head count")
        self.config=dict(num_classes=num_classes,backbone_depth=backbone_depth,hidden_dim=hidden_dim,nhead=nhead,
                         num_queries=num_queries,num_decoder_layers=num_decoder_layers,num_decoder_points=num_decoder_points,
                         dim_feedforward=dim_feedforward,num_denoising=num_denoising,expansion=expansion,depth_mult=depth_mult,
                         freeze_at=freeze_at,freeze_norm=freeze_norm,eval_spatial_size=None if eval_spatial_size is None else list(eval_spatial_size),seed=seed)
        self.num_classes,self.num_queries=num_classes,num_queries
        token=nn._initialization_rng.set(np.random.default_rng(seed))
        try:
            self.backbone=PResNet(backbone_depth,variant='d',return_idx=[1,2,3],freeze_at=freeze_at,freeze_norm=freeze_norm)
            self.encoder=HybridEncoder(in_channels=self.backbone.out_channels,hidden_dim=hidden_dim,nhead=nhead,
                                       dim_feedforward=dim_feedforward,expansion=expansion,depth_mult=depth_mult,
                                       eval_spatial_size=eval_spatial_size)
            self.decoder=RTDETRTransformer(num_classes=num_classes,hidden_dim=hidden_dim,num_queries=num_queries,
                                          feat_channels=[hidden_dim]*3,nhead=nhead,num_decoder_layers=num_decoder_layers,
                                          num_decoder_points=num_decoder_points,dim_feedforward=dim_feedforward,
                                          num_denoising=num_denoising,eval_spatial_size=eval_spatial_size)
        finally: nn._initialization_rng.reset(token)

    def forward(self,images,denoising=None):
        if images.ndim!=4 or images.shape[1]!=3 or any(s%32 for s in images.shape[-2:]):
            raise ValueError("RT-DETR expects NCHW RGB images with dimensions divisible by 32")
        if self.config['eval_spatial_size'] is not None and not self.training and tuple(images.shape[-2:])!=tuple(self.config['eval_spatial_size']):
            raise ValueError("image size does not match cached evaluation geometry")
        features=self.encoder(self.backbone(images))
        if sum(x.shape[2]*x.shape[3] for x in features)<self.num_queries: raise ValueError("feature map has fewer positions than requested queries")
        return self.decoder(features,denoising=denoising)


def rtdetr_r50vd(num_classes=80,**kwargs): return RTDETR(num_classes,backbone_depth=50,**kwargs)
def rtdetr_r18vd(num_classes=80,**kwargs): return RTDETR(num_classes,backbone_depth=18,**kwargs)

from .loss import SetCriterion, HungarianMatcher
from .denoising import pad_targets, prepare_denoising
from .training import DetectorTrainer, rtdetr_adamw, load_torch_checkpoint
from .data import PostProcessor, Predictor, CocoDetection, preprocess, coco_results, evaluate_coco
