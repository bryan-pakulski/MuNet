"""Train RT-DETR on COCO-format data; run from an installed munet-nn package."""
import argparse
import json
from pathlib import Path
from munet.models.rtdetr import rtdetr_r50vd, DetectorTrainer, CocoDetection, load_torch_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--output',default='runs/rtdetr');p.add_argument('--device',default='vulkan')
    p.add_argument('--epochs',type=int,default=72);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--size',type=int,default=640);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume');p.add_argument('--torch-weights');p.add_argument('--target-slots',type=int)
    p.add_argument('--export-onnx',action='store_true')
    args=p.parse_args()
    if args.batch_size<2: p.error('use batch-size >= 2 for training BatchNorm')
    if args.resume and args.torch_weights: p.error('select either resume or initial torch weights')
    data=CocoDetection(args.images,args.annotations,size=(args.size,args.size),horizontal_flip=.5)
    if len(data)<args.batch_size: p.error('dataset must contain at least one full batch')
    model=rtdetr_r50vd(len(data.classes),seed=args.seed)
    if args.torch_weights: load_torch_checkpoint(model,args.torch_weights)
    trainer=DetectorTrainer(model,device=args.device,target_slots=args.target_slots,seed=args.seed)
    if args.resume: trainer.load(args.resume)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'categories.json').write_text(json.dumps({'ids':data.category_ids,'names':data.classes}))
    for epoch in range(trainer.epoch,args.epochs):
        for images,targets,_,_ in data.batches(args.batch_size,rng=trainer.rng):
            metrics=trainer.step(images,targets)
            print(json.dumps({'epoch':epoch+1,'step':trainer.steps,**metrics}),flush=True)
        trainer.finish_epoch();trainer.save(out/'last.mnet')
    image,_,_,_=data.get(0)
    trainer.export(out/'inference.mnet',image[None],onnx_path=out/'inference.onnx' if args.export_onnx else None)


if __name__=='__main__': main()
