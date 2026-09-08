"""Full default R50-vd training smoke (300 queries, six layers, 100 DN queries).

Use --size 640 for deployment resolution. The default 160 has enough encoder
positions for all 300 queries and is suitable for a CPU/software-Vulkan gate.
This validates execution and state advancement, not COCO convergence or speed.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from examples.rtdetr import rtdetr_r50vd,DetectorTrainer,pad_targets,prepare_denoising


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='vulkan');p.add_argument('--size',type=int,default=160)
    p.add_argument('--output',default='artifacts/validation/full-r50-training.json');p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    model=rtdetr_r50vd(seed=7);trainer=DetectorTrainer(model,device=args.device,target_slots=2,seed=17)
    x=np.random.default_rng(5).uniform(size=(2,3,args.size,args.size)).astype(np.float32)
    targets=[{'labels':[1,3],'boxes':[[.223,.319,.137,.217],[.713,.617,.193,.113]]},{'labels':[],'boxes':np.empty((0,4),np.float32)}]
    if args.plan_only:
        padded=pad_targets(targets,model.num_classes,2)
        dn=prepare_denoising(padded,model.num_classes,model.num_queries,num_denoising=100,rng=trainer.rng)
        arrays=[x,padded['labels'],padded['boxes'],padded['valid'],dn['labels'],dn['boxes_unact'],dn['attn_mask']]
        program=trainer._program(dn);program.device='cpu';program._capture(arrays)
        result={'config':model.config,'input_shape':list(x.shape),'stats':program.stats(),'execution':'planning only; tensor arena allocation deferred'}
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    probe=model.decoder.dec_score_head[0].weight;before=probe.numpy();start=time.perf_counter()
    metrics=trainer.step(x,targets);elapsed=time.perf_counter()-start
    if not np.isfinite(list(metrics.values())).all() or np.array_equal(before,probe.numpy()): raise AssertionError('training did not advance finite model state')
    stats=next(iter(trainer.programs.values())).stats()
    result={'model':'RT-DETR v1 R50-vd','config':model.config,'input_shape':list(x.shape),'metrics':metrics,'stats':stats,
            'first_step_seconds':elapsed,'steps':trainer.steps,'ema_steps':trainer.ema.steps.numpy().item(),
            'optimizer_parameters':len(trainer.optimizer.state),'claim':'one finite full-architecture training step; no COCO AP/performance claim'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__': main()
