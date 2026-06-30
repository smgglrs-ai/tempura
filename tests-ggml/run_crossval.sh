#!/bin/bash
# Generate ggml test vectors and run Rust cross-validation tests.
set -euo pipefail
LLAMA_DIR=~/Code/github.com/ggml-org/llama.cpp
cd "$(dirname "$0")"

echo "=== Building C test vector generator ==="
gcc -O2 -I$LLAMA_DIR/ggml/include -I$LLAMA_DIR/ggml/src \
    -o gen_test_vectors gen_test_vectors.c \
    -L$LLAMA_DIR/build/bin -lggml-base -lm

echo "=== Generating test vectors ==="
LD_LIBRARY_PATH=$LLAMA_DIR/build/bin ./gen_test_vectors

echo "=== Running Rust cross-validation ==="
cd ..
cargo test --package tempura-kernels -- ggml_crossval --nocapture
