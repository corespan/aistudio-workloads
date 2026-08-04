#!/bin/bash
# Entrypoint when the container is used as a standalone vLLM server.
# aistudio-server overrides this CMD with its own `vllm serve` arguments.
#
# Environment variables:
#   MODEL                 — model path or HuggingFace repo (required)
#   TENSOR_PARALLEL_SIZE  — number of GPUs (default: 1)
#   MAX_MODEL_LEN         — max context length (default: 4096)
#   PORT                  — port to listen on (default: 8000)
#
set -e

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL env var is required."
    echo "  Set it to a model path or HuggingFace repo ID, e.g.:"
    echo "  -e MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    exit 1
fi

TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-8000}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code
