# LLM Inference Workload — open-source edition
#
# Base image already contains vLLM v0.14.1, PyTorch, CUDA runtime, and
# attention kernels pre-compiled. No nvcc or separate build stage needed.
#
# Build:
#   docker build -t llminference:4.0.0-nvidia .
#
# Run (standalone server):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models \
#     -e MODEL=/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#     llminference:4.0.0-nvidia
#
# Run benchmark:
#   docker run --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v /path/to/results:/results \
#     llminference:4.0.0-nvidia \
#     python3 benchmark.py \
#       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --output_data_path /results/run-001 \
#       --run_name tinyllama-tp1 \
#       --concurrencies 32,64,128 \
#       --dataset-name custom --dataset-path /datasets/prompts.jsonl
#
# The image also carries NVIDIA nvbandwidth for --interconnect_check, built in
# its own stage so no compiler reaches the final image. The build has no GPU to
# detect, so the CUDA architectures are listed: Volta to Blackwell as machine
# code, plus PTX of the newest for GPUs released later.
ARG NVBANDWIDTH_VERSION=v0.8
ARG NVBANDWIDTH_CUDA_ARCHS="70-real;75-real;80-real;86-real;89-real;90-real;100-real;120"

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS nvbandwidth
ARG NVBANDWIDTH_VERSION
ARG NVBANDWIDTH_CUDA_ARCHS
RUN apt-get update \
 && apt-get install -y --no-install-recommends git cmake build-essential libboost-program-options-dev \
 && rm -rf /var/lib/apt/lists/*
# Boost is linked statically by nvbandwidth's own CMakeLists; libcuda and NVML
# come from the driver at run time.
RUN git clone --depth 1 --branch "${NVBANDWIDTH_VERSION}" https://github.com/NVIDIA/nvbandwidth /src \
 && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=${NVBANDWIDTH_CUDA_ARCHS}" \
 && cmake --build /build -j"$(nproc)"

FROM vllm/vllm-openai:v0.14.1

ENV PYTHONUNBUFFERED=1

COPY --from=nvbandwidth /build/nvbandwidth /usr/local/bin/nvbandwidth

# Install extra packages (vLLM, PyTorch, CUDA already in base)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

WORKDIR /llm-inference
COPY . /llm-inference/

# Clear the vllm/vllm-openai base image ENTRYPOINT so docker run
# commands can specify python3 directly without vllm intercepting them.
ENTRYPOINT []
CMD ["bash", "-c", "sh ./script.sh"]
