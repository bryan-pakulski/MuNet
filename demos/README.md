# Demo Catalog

Demos are organized by high-level feature areas so you can quickly map library
scope to practical examples.

## Visual

- `visual/object_detection/` — object-detection oriented demos (scaffold).
- `visual/semantic_segmentation/` — segmentation training/inference demos:
  - `mnist.py`
  - `unet.py`
- `visual/instance_segmentation/` — instance-segmentation oriented demos (scaffold).

## Operators

- `operators/transformer_ops_showcase.py` — focused operator usage walkthrough.

## Serialization

- `serialization/munet/serialization_roundtrip_demo.py` — MuNet save/load roundtrip.
- `serialization/framework_interop/` — framework interoperability examples:
  - `pytorch_export_demo.py`
  - `pytorch_import_demo.py`
  - `test_torch.py`

## Inference

- `inference/foundation/` — concrete deploy-style baseline demos:
  - `batch_forward_demo.py`
  - `e2e_train_save_load_infer.py`
- `inference/quantization/` — quantization demo slot (scaffold for upcoming work).

## MultiGPU

- `multigpu/multi_gpu_allreduce_training_demo.py`
- `multigpu/model_offload_manual_demo.py`
- `multigpu/auto_offload_strategy_compare_demo.py`
- `multigpu/gpt3_multi_gpu_demo.py`

## Transformers

- `transformers/tiny_llm.py`
- `transformers/decoder_block_demo.py`
