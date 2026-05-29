#!/usr/bin/env bash
cd src && cmake --preset linux-default
cmake --build build -j
CC=/usr/bin/clang
CXX=/usr/lib64/rocm/llvm/bin/clang++
