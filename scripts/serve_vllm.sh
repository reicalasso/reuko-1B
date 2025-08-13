#!/bin/bash

# VLLM ile Qwen2.5-mini modeli servis et
MODEL_PATH=${1:-"Qwen/Qwen2.5-0.5B"}
PORT=${2:-8000}

echo "🚀 Starting vLLM server for model: $MODEL_PATH on port $PORT"

vllm --model $MODEL_PATH \
     --num-gpus 1 \
     --port $PORT \
     --max-seq-len 1024 \
     --log-level info
echo "✅ vLLM server is running at http://localhost:$PORT"