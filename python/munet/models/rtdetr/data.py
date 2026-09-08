"""COCO data I/O and RT-DETR RGB preprocessing. Pillow is an optional dependency."""
import json
from pathlib import Path
import numpy as np
from ... import nn,compile
from .loss import cxcywh_to_xyxy


class PostProcessor(nn.Module):
    """Top scoring query/class pairs, no NMS; original_sizes is float32 [B,(W,H)]."""
    def __init__(self,num_top_queries=300): self.num_top_queries=num_top_queries
    def forward(self,outputs,original_sizes):
        logits,boxes=outputs['pred_logits'],outputs['pred_boxes'];b,q,c=logits.shape
        if original_sizes.shape!=(b,2): raise ValueError('original_sizes must contain width,height per image')
        scores,indices=logits.sigmoid().flatten(1).topk(min(self.num_top_queries,q*c),1)
        queries=(indices/c).floor();labels=indices-queries*c
        boxes=cxcywh_to_xyxy(boxes).gather(1,queries.unsqueeze(-1).expand(b,queries.shape[1],4))
        scale=original_sizes.repeat(1,2).unsqueeze(1)
        return {'labels':labels,'scores':scores,'boxes':boxes*scale}


def preprocess(images,size=(640,640)):
    """PIL/uint8 RGB -> NCHW float32 in [0,1], with original width,height."""
    from PIL import Image
    if len(size)!=2 or any(s<=0 or s%32 for s in size): raise ValueError('image size must be positive and divisible by 32')
    arrays=[];original=[]
    for image in images:
        if not isinstance(image,Image.Image): image=Image.fromarray(np.asarray(image,dtype=np.uint8))
        image=image.convert('RGB');original.append(image.size)
        image=image.resize((size[1],size[0]),Image.Resampling.BILINEAR)
        arrays.append(np.asarray(image,np.float32).transpose(2,0,1)/255.)
    if not arrays: raise ValueError('at least one image is required')
    return np.ascontiguousarray(np.stack(arrays)),np.asarray(original,np.float32)


class Predictor:
    def __init__(self,model,*,size=(640,640),device='vulkan',num_top_queries=300):
        self.model=model.eval();self.size=size;self.postprocess=PostProcessor(num_top_queries)
        self.program=compile(lambda x,s:self.postprocess(self.model(x),s),device=device)
    def __call__(self,images,score_threshold=.5):
        x,sizes=preprocess(images,self.size);out={k:v.numpy() for k,v in self.program(x,sizes).items()}
        result=[]
        for b in range(len(x)):
            keep=out['scores'][b]>=score_threshold
            result.append({'labels':out['labels'][b,keep].astype(np.int64),'scores':out['scores'][b,keep],'boxes':out['boxes'][b,keep]})
        return result


class CocoDetection:
    """COCO boxes/category mapping; resize and optional random horizontal flip.

    This deliberately exposes the transform boundary: stronger crop/photometric
    augmentation may be supplied by a data pipeline without changing the model.
    Crowd annotations and degenerate boxes are excluded from the training targets.
    """
    def __init__(self,images,annotations,*,size=(640,640),horizontal_flip=0.):
        self.images=Path(images);self.annotations=Path(annotations);self.size=tuple(size);self.horizontal_flip=horizontal_flip
        if not 0<=horizontal_flip<=1: raise ValueError('invalid flip probability')
        data=json.loads(self.annotations.read_text());self.records=sorted(data['images'],key=lambda x:x['id'])
        categories=sorted(data['categories'],key=lambda x:x['id']);self.category_ids=[c['id'] for c in categories];self.classes=[c['name'] for c in categories]
        mapping={v:i for i,v in enumerate(self.category_ids)};self.targets={r['id']:[] for r in self.records}
        for a in data['annotations']:
            if a.get('iscrowd',0) or a['category_id'] not in mapping: continue
            if a['image_id'] in self.targets: self.targets[a['image_id']].append((mapping[a['category_id']],a['bbox']))
    def __len__(self): return len(self.records)
    def get(self,index,*,rng=None,size=None):
        from PIL import Image
        record=self.records[index]
        with Image.open(self.images/record['file_name']) as source: image=source.convert('RGB')
        w,h=image.size;labels=[];boxes=[]
        for label,(x,y,bw,bh) in self.targets[record['id']]:
            x1,y1=np.clip([x,y],0,[w,h]);x2,y2=np.clip([x+bw,y+bh],0,[w,h])
            if x2-x1<1 or y2-y1<1: continue
            labels.append(label);boxes.append([(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h])
        boxes=np.asarray(boxes,np.float32).reshape(-1,4)
        if self.horizontal_flip and (rng or np.random.default_rng()).random()<self.horizontal_flip:
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);boxes[:,0]=1-boxes[:,0]
        x,original=preprocess([image],size or self.size)
        return x[0],{'labels':np.asarray(labels,np.int64),'boxes':boxes},record['id'],original[0]
    def batches(self,batch_size,*,rng=None,shuffle=True,drop_last=True,multiscale=None):
        if batch_size<=0: raise ValueError('batch_size must be positive')
        rng=rng or np.random.default_rng();order=rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            if len(indices)<batch_size and drop_last: break
            size=self.size
            if multiscale: size=(int(rng.choice(multiscale)),)*2
            records=[self.get(int(i),rng=rng,size=size) for i in indices]
            yield np.stack([r[0] for r in records]),[r[1] for r in records],[r[2] for r in records],np.stack([r[3] for r in records])


def coco_results(predictions,image_ids,category_ids):
    """Map contiguous labels back to dataset IDs and xyxy boxes to COCO xywh."""
    result=[]
    for pred,image_id in zip(predictions,image_ids):
        for label,score,box in zip(pred['labels'],pred['scores'],pred['boxes']):
            label=int(label)
            if not 0<=label<len(category_ids): raise ValueError('predicted label outside category mapping')
            x1,y1,x2,y2=map(float,box)
            result.append({'image_id':int(image_id),'category_id':int(category_ids[label]),'score':float(score),'bbox':[x1,y1,x2-x1,y2-y1]})
    return result


def evaluate_coco(annotations,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth=COCO(str(annotations))
    if not predictions:
        result=COCO();result.dataset={'images':truth.dataset['images'],'categories':truth.dataset['categories'],'annotations':[]};result.createIndex()
    else: result=truth.loadRes(predictions)
    evaluation=COCOeval(truth,result,'bbox');evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
    return evaluation.stats.copy()
