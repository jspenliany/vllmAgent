#!/bin/bash

MODEL_PATH="/home/smart/models/huggingface/gemma3/gemma-3-27b-it"

# These fix the "Broken Pipe" and Networking issues
export NCCL_P2P_DISABLE=1
export VLLM_CONFIGURE_LOGGING=1
export CUDA_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization compressed-tensors \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000 \
    --served-model-name gemma-3-27b-it \
    --enable-auto-tool-choice \
    --tool-call-parser functiongemma \
    --distributed-executor-backend mp

