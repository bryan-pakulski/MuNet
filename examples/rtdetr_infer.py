"""Run a training checkpoint on images and write boxes, labels and scores as JSON."""
import argparse
import json
from pathlib import Path
from PIL import Image
from munet.checkpoint import load_state
from munet.models.rtdetr import RTDETR, Predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--images',nargs='+',required=True)
    p.add_argument('--device',default='vulkan');p.add_argument('--size',type=int,default=640)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--output',default='detections.json')
    args=p.parse_args();state=load_state(args.checkpoint)
    model=RTDETR(**state['config']);model.load_state_dict(state['ema']['shadow'] if state.get('ema') else state['model'])
    predictor=Predictor(model,size=(args.size,args.size),device=args.device)
    records=[]
    for name in args.images:
        with Image.open(name) as image: pred=predictor([image],args.threshold)[0]
        records.append({'image':name,**{k:v.tolist() for k,v in pred.items()}})
    Path(args.output).write_text(json.dumps(records,indent=2))


if __name__=='__main__': main()
