"""Compute COCO bbox AP from an RT-DETR training checkpoint (optional pycocotools)."""
import argparse
from munet import compile
from munet.checkpoint import load_state
from munet.models.rtdetr import RTDETR,PostProcessor,CocoDetection,coco_results,evaluate_coco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--images',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--device',default='vulkan');p.add_argument('--size',type=int,default=640)
    args=p.parse_args();state=load_state(args.checkpoint);model=RTDETR(**state['config']).eval()
    model.load_state_dict(state['ema']['shadow'] if state.get('ema') else state['model'])
    data=CocoDetection(args.images,args.annotations,size=(args.size,args.size));post=PostProcessor()
    program=compile(lambda x,s:post(model(x),s),device=args.device);records=[]
    for x,_,ids,sizes in data.batches(1,shuffle=False,drop_last=False):
        output={k:v.numpy() for k,v in program(x,sizes).items()}
        records.extend(coco_results([{k:v[0] for k,v in output.items()}],ids,data.category_ids))
    evaluate_coco(args.annotations,records)


if __name__=='__main__': main()
