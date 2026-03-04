#!/bin/bash

set -e

cd build
make munet_tests -j $(nproc)

./munet_tests

python ../tests/test_python.py
