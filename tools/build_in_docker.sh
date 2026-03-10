#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="munet-builder:latest"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[builder] Building Docker image: ${IMAGE_NAME}"
docker build -f "${ROOT_DIR}/docker/Dockerfile.builder" -t "${IMAGE_NAME}" "${ROOT_DIR}"

echo "[builder] Running CMake build inside container"
docker run --rm -it \
  -v "${ROOT_DIR}:/workspace/MuNet" \
  -w /workspace/MuNet \
  "${IMAGE_NAME}" \
  bash -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"'
