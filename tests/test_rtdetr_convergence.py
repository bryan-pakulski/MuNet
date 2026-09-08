"""Small-data convergence gate, separate from the reference numerical gate."""
import numpy as np
import munet as mu
from munet.models.rtdetr import RTDETR,DetectorTrainer


def test_tiny_detector_head_overfits_a_fixed_batch():
    model=RTDETR(2,backbone_depth=18,hidden_dim=16,nhead=4,num_queries=4,num_decoder_layers=2,
                 num_decoder_points=2,dim_feedforward=32,num_denoising=0,expansion=.5,depth_mult=.34,seed=6)
    model.backbone.requires_grad_(False);model.encoder.requires_grad_(False)
    optimizer=mu.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.005,weight_decay=0)
    trainer=DetectorTrainer(model,device='cpu',optimizer=optimizer,max_norm=10,ema=False)
    images=np.random.default_rng(4).uniform(size=(2,3,32,64)).astype(np.float32)
    targets=[{'labels':[0],'boxes':[[.223,.319,.137,.217]]},{'labels':[1],'boxes':[[.713,.617,.193,.113]]}]
    losses=[trainer.step(images,targets)['loss'] for _ in range(100)]
    assert np.isfinite(losses).all()
    assert losses[-1]<losses[0]*.65,(losses[0],losses[-1])
