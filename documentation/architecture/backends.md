# Backend Architecture

## Contract

All backends implement `Backend` in `src/backend.hpp`.

## Registration

Use `BackendManager::register_backend(DeviceType, factory)` to install or
override backend constructors without touching `tensor.cpp` dispatch code.

## Debug/Profiling Wrapper

`DebugBackend` can wrap any backend instance and is enabled when debug/profile
flags are active.

## Current Backends

- CPU
- CUDA (optional)
- Vulkan (optional)

## Notes

Device selection is index-aware for CUDA and Vulkan through `Device.index`.
