# MuNet Dtype-Sensitive Code Path Inventory

This inventory captures the major places where dtype behavior, scalar representation, and float-specialized logic currently affect correctness, extensibility, or mixed-precision readiness.

Its purpose is to make Phase 0 concrete: future phases can point to this document instead of rediscovering the same hotspots.

## 1. Core dtype and scalar primitives

### `src/types.hpp`

Current responsibilities:

- defines `DataType`
- defines dtype helpers (`dtype_name`, `is_floating`, `is_integral`, `is_low_precision`, `dtype_size`)
- defines promotion / accumulation helpers (`promote_types`, `accumulation_type`)
- defines `TensorOptions` and `ScalarValue`
- defines float32 <-> float16 bit conversion helpers

Why this is dtype-sensitive:

- every later phase should treat this file as the canonical dtype policy layer
- any new dtype (bf16, fp8, uint8, etc.) will start here first
- promotion and accumulation behavior should be extended here rather than in backends or individual ops

## 2. Tensor construction, conversion, and scalar access

### `src/tensor.hpp` / `src/tensor.cpp`

Current dtype-sensitive areas:

- `TensorOptions`-based construction
- `Tensor::to(Device)` / `Tensor::to(DataType)` / `Tensor::to(const TensorOptions &)`
- CPU-mediated dtype conversion helpers
- `item_value()` and legacy `item()` behavior
- `backward()` root gradient seed creation
- `uniform_()` floating-only validation
- `clone()` / `contiguous()` preserving dtype and options

Why this is dtype-sensitive:

- tensor creation and conversion semantics define how the rest of the library should reason about dtype
- any ambiguity here multiplies across ops, backends, optimizers, and bindings
- mixed precision should extend tensor-level policy rather than bypass it

## 3. Backend interface contracts

### `src/core/backend.hpp`

Current dtype-sensitive areas:

- backend contracts operate on `Storage` objects rather than typed kernel signatures
- `fill_uniform` and optimizer kernels expose `float` hyperparameter assumptions directly
- capability signatures do not yet advertise dtype support/fallback behavior

Why this is dtype-sensitive:

- backend support for new dtypes is currently implicit rather than declared
- later phases should separate capability detection from raw storage-level entry points

## 4. CPU backend implementation hotspots

### `src/backend/cpu_backend.hpp`

Current dtype-sensitive areas:

- elementwise ops cast storage to `float *` / `const float *`
- matmul, losses, convolution, pooling, normalization, reductions, optimizer kernels, and random fill are overwhelmingly float-specialized
- dtype conversions are not delegated to typed kernel families yet

Why this is dtype-sensitive:

- CPU is the current reference backend, so its float-specialization is the clearest signal of where later kernel splitting is required
- future dtype work should avoid scattering additional `if (dtype == ...)` logic through this file without a dispatch design in place

## 5. CUDA and Vulkan backend surfaces

### `src/backend/cuda_backend.hpp` / `src/backend/cuda_backend.cu`
### `src/backend/vulkan_backend.hpp` / `src/backend/vulkan_backend.cpp`

Current dtype-sensitive areas:

- backend signatures mirror the same float-oriented kernel assumptions as CPU
- CUDA kernels are primarily written against `float` buffers
- Vulkan/CUDA interfaces will need a capability model before partial dtype coverage becomes manageable

Why this is dtype-sensitive:

- backend parity for new dtypes will be difficult unless dispatch/capability work is centralized first
- mixed precision should not require editing every backend entry point blindly without support metadata

## 6. Op glue and autograd coupling

### `src/ops.hpp`
### `src/autograd/engine.hpp`

Current dtype-sensitive areas:

- some op implementations and backward helpers still create or inspect float-oriented CPU buffers directly
- autograd accumulation relies on backend add/sum behavior being valid for the tensor dtype in question
- the monolithic op layer makes it easy for dtype policy to leak into unrelated ops

Why this is dtype-sensitive:

- Phase 2 should break this surface apart so dtype behavior can be localized per op
- autograd should eventually depend on explicit dtype guarantees rather than incidental backend behavior

## 7. Module and layer initialization

### `src/nn/linear.hpp`
### `src/nn/normalization.hpp`
### `src/nn/activations.hpp`
### `src/nn/attention.hpp`
### `src/nn/pooling.hpp`

Current dtype-sensitive areas:

- module parameters and buffers are still commonly initialized as `DataType::Float32`
- helper constants and masks often assume float-valued scalar construction
- training/inference modules still need a more explicit dtype policy

Current transition status:

- core NN modules now accept `TensorOptions` for parameter construction
- normalization running-stat buffers use normalization accumulation dtype rather than blindly matching low-precision parameter dtype
- serialization preserves model tensor dtype fidelity and reconstructs supported built-ins from saved dtype metadata
- optimizer state follows `optimizer_state_type(parameter_dtype)`

Why this is dtype-sensitive:

- module constructors are where dtype defaults become user-visible behavior
- Phase 5 should make parameter/buffer dtype choices explicit rather than implicit

## 8. Optimizers and training state

### `src/optim.hpp`

Current dtype-sensitive areas:

- optimizer hyperparameters are `float`
- optimizer state tensors are created without a broader policy for model dtype vs state dtype vs master weights
- future grad-scaling/master-weight behavior has no dedicated abstraction yet

Current transition status:

- backends now expose a coarse capability surface (`BackendFeature` x `DataType`) instead of requiring callers to infer support from implementation details
- optimizer state now follows `optimizer_state_type(parameter_dtype)`
- serialization keeps tensor dtype fidelity instead of forcing float32 on save/load
- mixed parameter/state backend kernels still need broader typed-kernel coverage
- accepted policy: fp32 master weights and grad-scaling metadata belong to optimizer/trainer checkpoints, not module parameter checkpoints

Why this is dtype-sensitive:

- mixed precision training needs optimizer-state rules, not just dtype conversion APIs
- this is a later-phase concern, but it should stay visible from Phase 0 onward

## 9. Bindings and external API assumptions

### `src/bindings.cpp`

Current dtype-sensitive areas:

- Python bindings expose `TensorOptions` and dtype conversion APIs
- dtype-aware NumPy ingress and tensor factories now preserve `float16` / `int32` where supported
- helper code outside the core bindings can still diverge from the C++ dtype model unless it shares the same supported-dtype checks

Why this is dtype-sensitive:

- Python user expectations will diverge quickly if binding behavior does not track core dtype semantics
- later phases should align numpy interop and factory paths with the evolving dtype model

## 10. Existing regression coverage and remaining gaps

### Coverage that already exists

- `tests/test_tensor.cpp` now covers dtype helpers, `TensorOptions`, dtype conversion round-trips, and `item_value()` behavior
- backend registration and autograd behavior already have dedicated tests elsewhere in the suite

### Gaps still visible in Phase 0

- no architecture decision log existed before this phase
- no single dtype inventory existed before this phase
- most backend/op/module tests still assume float-centric execution
- no capability/fallback tests exist yet for partial dtype support
- backend capability reporting is still implicit rather than queryable

## Recommended use in later phases

- **Phase 1:** extend the type/tensor sections rather than adding ad hoc dtype rules elsewhere
- **Phase 2:** use this inventory to split dtype-sensitive op glue into per-op modules
- **Phase 3:** use the backend sections as the checklist for capability-based dispatch work
- **Phase 5:** revisit the module/optimizer sections when mixed-precision training semantics are introduced
