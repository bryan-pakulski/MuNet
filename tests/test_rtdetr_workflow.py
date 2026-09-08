"""Executable training -> checkpoint/resume -> native/ONNX -> detections workflow."""
import json
import numpy as np
import pytest
import munet as mu
from munet.models.rtdetr import RTDETR,DetectorTrainer,PostProcessor,CocoDetection,coco_results,preprocess
from munet.interop import to_onnx,from_onnx,from_torch
from rtdetr_reference import model_for


def small_model():
    return RTDETR(3,backbone_depth=18,hidden_dim=8,nhead=2,num_queries=4,num_decoder_layers=2,
                  num_decoder_points=2,dim_feedforward=16,num_denoising=4,expansion=.5,depth_mult=.34)


def test_detector_checkpoint_resumes_adamw_rng_bn_ema_and_scheduler(device,tmp_path):
    model=small_model()
    # The full-backbone derivative gate is in test_rtdetr.py. This workflow gate
    # selects a head fine-tune, reducing optimizer-state allocation in CI.
    model.backbone.requires_grad_(False);model.encoder.requires_grad_(False)
    trainer=DetectorTrainer(model,device=device,target_slots=2,milestones=(1,),seed=17)
    image=np.random.default_rng(5).uniform(size=(2,3,64,64)).astype(np.float32)
    raw=[{'labels':[1],'boxes':[[.32,.48,.17,.21]]},{'labels':[],'boxes':np.zeros((0,4))}]
    first=trainer.step(image,raw);assert np.isfinite(list(first.values())).all()
    trainer.finish_epoch();trainer.save(tmp_path/'state.mnet')
    other=small_model();other.backbone.requires_grad_(False);other.encoder.requires_grad_(False)
    resumed=DetectorTrainer(other,device=device,target_slots=2,milestones=(1,),seed=999).load(tmp_path/'state.mnet')
    assert resumed.steps==1 and resumed.epoch==1
    for _ in range(2):
        expected=trainer.step(image,raw);actual=resumed.step(image,raw)
        np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=3e-4,atol=3e-5)
    for name,p in model.named_parameters(): np.testing.assert_allclose(p.numpy(),dict(other.named_parameters())[name].numpy(),rtol=2e-4,atol=3e-5)
    for name,p in trainer.ema.shadow.items(): np.testing.assert_allclose(p.numpy(),resumed.ema.shadow[name].numpy(),rtol=2e-4,atol=3e-5)
    # One GT / four groups and two GT / two groups have identical array shapes.
    # They need different static DN metadata, then must synchronize state when
    # returning to the first cached program.
    two=[{'labels':[1,0],'boxes':[[.32,.48,.17,.21],[.71,.61,.19,.11]]},raw[1]]
    expected=trainer.step(image,two);actual=resumed.step(image,two)
    np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=3e-4,atol=3e-5)
    assert len(trainer.programs)==2
    expected=trainer.step(image,raw);actual=resumed.step(image,raw)
    np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=3e-4,atol=3e-5)
    inference=trainer.export(tmp_path/'detector.mnet',image[:1])
    loaded=mu.load(tmp_path/'detector.mnet',device=device)
    expected=inference(image[:1]);actual=loaded(image[:1])
    for key in expected: np.testing.assert_allclose(actual[key].numpy(),expected[key].numpy(),rtol=2e-5,atol=2e-5)


def test_detector_native_onnx_roundtrip_and_reference_evaluator(tmp_path):
    # Export the full detector path with a compact backbone/head; the R50 graph
    # uses the same standard ops and is separately checked against upstream.
    import onnx
    from onnx.reference import ReferenceEvaluator
    model=small_model().eval();x=np.random.default_rng(8).uniform(size=(1,3,32,64)).astype(np.float32)
    program=mu.compile(model,device='cpu');expected={k:v.numpy() for k,v in program(x).items()}
    path=tmp_path/'detector.onnx';to_onnx(program,path)
    graph=onnx.load(path)
    # ONNX 1.22's evaluator uses the opset-20 GridSample mode spelling even for
    # opset 18. Its standard version converter supplies that spelling correctly.
    ref=ReferenceEvaluator(onnx.version_converter.convert_version(graph,20))
    actual=ref.run(None,{graph.graph.input[0].name:x})
    for value,want in zip(actual,expected.values()): np.testing.assert_allclose(value,want,rtol=3e-4,atol=3e-5)
    import os
    if os.environ.get('CI')=='true':
        import onnxruntime as ort
        ort.disable_telemetry_events()
        runtime=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        for value,want in zip(runtime.run(None,{runtime.get_inputs()[0].name:x}),expected.values()):
            np.testing.assert_allclose(value,want,rtol=3e-4,atol=3e-5)
    imported=from_onnx(path,device='cpu');out=imported(x)
    for key,value in out.items(): np.testing.assert_allclose(value.numpy(),expected[key],rtol=3e-4,atol=3e-5)
    mu.save(imported,tmp_path/'imported.mnet')
    saved=mu.load(tmp_path/'imported.mnet',device='cpu')(x)
    for key,value in saved.items(): np.testing.assert_allclose(value.numpy(),expected[key],rtol=3e-4,atol=3e-5)


def test_official_torch_detector_import(tmp_path):
    import torch
    torch.manual_seed(10);model=small_model();reference=model_for(model.config).eval()
    x=torch.tensor(np.random.default_rng(2).uniform(size=(1,3,32,64)).astype(np.float32))
    imported=from_torch(reference,(x,),device='cpu')
    with torch.no_grad(): want=reference(x)
    actual=imported(x.numpy())
    for result,expected in zip(actual,want.values()): np.testing.assert_allclose(result.numpy(),expected.numpy(),rtol=5e-4,atol=6e-5)


def test_postprocess_category_mapping_and_coco_data(device,tmp_path):
    from PIL import Image
    logits=np.asarray([[[3.,1.],[-2.,2.]]],np.float32);boxes=np.asarray([[[.5,.5,.2,.4],[.3,.4,.2,.2]]],np.float32)
    p=PostProcessor(3);f=mu.compile(lambda l,b,s:p({'pred_logits':l,'pred_boxes':b},s),device=device)
    result={k:v.numpy()[0] for k,v in f(logits,boxes,np.array([[100,200]],np.float32)).items()}
    np.testing.assert_array_equal(result['labels'],[0,1,1])
    np.testing.assert_allclose(result['boxes'],[[40,60,60,140],[20,60,40,100],[40,60,60,140]],atol=2e-5)
    records=coco_results([result],[17],[3,9]);assert [r['category_id'] for r in records]==[3,9,9]
    np.testing.assert_allclose(records[0]['bbox'],[40,60,20,80],atol=2e-5)
    Image.new('RGB',(100,200),'red').save(tmp_path/'im.png')
    annotation={'images':[{'id':17,'file_name':'im.png'}],'categories':[{'id':9,'name':'b'},{'id':3,'name':'a'}],
                'annotations':[{'id':1,'image_id':17,'category_id':9,'bbox':[10,20,30,40],'iscrowd':0}]}
    (tmp_path/'coco.json').write_text(json.dumps(annotation));data=CocoDetection(tmp_path,tmp_path/'coco.json',size=(32,64),horizontal_flip=1.)
    image,target,ident,size=data.get(0,rng=np.random.default_rng(1))
    assert image.shape==(3,32,64) and image.dtype==np.float32 and ident==17
    np.testing.assert_array_equal(size,[100,200]);np.testing.assert_array_equal(target['labels'],[1])
    np.testing.assert_allclose(target['boxes'],[[.75,.2,.3,.2]],atol=1e-7)
