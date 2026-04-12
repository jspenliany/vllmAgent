#!/bin/bash

MODEL_PATH="/home/smart6learn/models/huggingface/gemma4/gemma-4-gguf/gemma-4-31B-it-Q4_K_M.gguf"


/home/smart6learn/program/llama-bin/llama-src/llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    --ctx-size 16384 \
    --n-gpu-layers -1 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --flash-attn on \
    --host 0.0.0.0 \
    --port 8000
