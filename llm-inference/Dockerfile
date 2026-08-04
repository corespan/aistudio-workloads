# LLM Inference Workload — open-source edition
#
# Base image already contains vLLM v0.14.1, PyTorch, CUDA runtime, and
# attention kernels pre-compiled. No nvcc or separate build stage needed.
#
# Build:
#   docker build -t llminference:1.0.0-nvidia .
#
# Run (standalone server):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models \
#     -e MODEL=/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#     llminference:1.0.0-nvidia
#
# Run benchmark:
#   docker run --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v /path/to/results:/results \
#     llminference:1.0.0-nvidia \
#     python3 benchmark.py \
#       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --output_data_path /results/run-001 \
#       --run_name tinyllama-tp1 \
#       --concurrencies 32,64,128
FROM vllm/vllm-openai:v0.14.1

ENV PYTHONUNBUFFERED=1

# Install extra packages (vLLM, PyTorch, CUDA already in base)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

WORKDIR /llm-inference
COPY . /llm-inference/

CMD ["bash", "-c", "sh ./script.sh"]
