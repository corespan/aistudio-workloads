# AIStudio Workloads

Standalone GPU workloads for LLM benchmarking and interactive Jupyter environments. Each workload runs as an isolated Docker container on any NVIDIA GPU node and can be used independently or managed through [AIStudio Server](https://github.com/corespan/aistudio-server).

## Workloads

### LLM Inference (`llm-inference`)
vLLM-based throughput and latency benchmarking. Starts its own vLLM server, sweeps concurrency levels, and outputs structured benchmark results (tokens/sec, TTFT, TPOT, cost estimates).

**Supported models:** Any HuggingFace-compatible model. Gated models (Llama, etc.) require an HF token — see [MODEL-LICENSES.md](https://github.com/corespan/aistudio-server/blob/master/MODEL-LICENSES.md).

### Jupyter Notebook (`jupyter-notebook`)
JupyterLab environment for remote GPU nodes: GPU microbenchmarks (compute and bandwidth, warmed up and timed with CUDA events), a vLLM benchmark that serves a model from the node's models share the way an AIStudio benchmark run does, MLPerf-style LLM and vision notebooks, and in the terminal `nvidia-smi`, `check-p2p` / `enable-p2p` and `ai-studio-cli nvbandwidth run` (host and GPU-to-GPU copy bandwidth, P2P included). The GPU notebooks need compute capability 7.0 or newer (T4 and later).

## Running Standalone

Pull the image and run directly on any NVIDIA GPU node (no AIStudio Server required):

```bash
# LLM Inference
docker run --gpus all --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/results:/results \
  -v /path/to/dataset.json:/path/to/dataset.json \
  --entrypoint python3 \
  us-docker.pkg.dev/aimlworkbench/aistudio/llminference:4.0.0-nvidia \
  /llm-inference/benchmark.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset-path /path/to/dataset.json \
    --concurrencies 4 \
    --output_data_path /results/run1 \
    --run_name run1

# Jupyter Notebook
docker run --gpus all -d \
  -p 8888:7008 \
  -v /path/to/data:/data \
  -v /opt/amw/models:/models:ro \
  -e workload_id=my-session \
  -e workload_port=7008 \
  -e JUPYTER_TOKEN="$(openssl rand -hex 24)" \
  us-docker.pkg.dev/aimlworkbench/aistudio/jupyternotebook:4.0.0-nvidia
```

The notebooks read model weights from `/models` (mount the node's models share there, read-only). The notebook asks for its token: open `http://<node>:8888/tree?token=<JUPYTER_TOKEN>`. Without `JUPYTER_TOKEN`, Jupyter picks a random one and prints it in `docker logs`.

**Dataset:** Any dataset type `vllm bench serve` supports, chosen with `--dataset-name` (default `sharegpt`, a JSON list of conversations). Common choices:

| `--dataset-name` | `--dataset-path` | Notes |
|---|---|---|
| `sharegpt` (default) | JSON list of conversations | |
| `custom` | JSONL, one `{"prompt": "..."}` per line | output length via `--extra_bench_args "--custom-output-len 128"` |
| `random` | none | lengths via `--extra_bench_args "--random-input-len 512 --random-output-len 128"` |
| `hf` | a Hugging Face dataset id | needs internet |

No dataset is bundled or auto-downloaded. You can use [OpenOrca (MIT)](https://huggingface.co/datasets/Open-Orca/OpenOrca) or [Dolly (CC-BY-SA-3.0)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) as clean alternatives.

**Model-specific settings:** `--extra_serve_args` passes flags straight to `vllm serve` (for example `"--gpu-memory-utilization 0.95 --kv-cache-dtype fp8"`), and loading has no fixed time limit: the script checks the vLLM server every 2 s, fails at once on a fatal vLLM error (out of memory, engine failed to start), and otherwise waits as long as vLLM keeps printing progress. `--server_stall_timeout` (default 1800 s) fails a load that prints nothing for that long; `--server_timeout` adds a total limit if you want one (default 0, none). The model can be a Hugging Face id or a local path, such as weights on an NFS mount.

**GPU interconnect:** `--interconnect_check` measures, before vLLM starts, how the GPUs given to the container are connected, and writes it to `interconnect.json` (also printed as one `INTERCONNECT_RESULT:` line):

- PCIe bus id and link generation/width (current and max) per GPU, from `nvidia-smi`;
- whether each pair of GPUs can reach the other peer to peer (P2P);
- [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth) copy bandwidth host↔GPU for every GPU, and GPU↔GPU plus latency with two or more GPUs.

GPUs without P2P are logged ("No P2P detected between GPU 0 and GPU 1") and the run continues. `--interconnect_timeout` (default 900 s) bounds the bandwidth test. Off unless the flag is given.

## Integration with AIStudio Server

When used with [AIStudio Server](https://github.com/corespan/aistudio-server), these workloads are orchestrated automatically — the server handles node selection, SSH execution, result collection, and the benchmark dashboard.

To use a specific workload image version, set the `image_tag` in the server's `catalog.json`:

```json
{
  "workload_types": [
    { "name": "LLMInference", "image_tag": "4.0.0-nvidia" },
    { "name": "JupyterNotebook", "image_tag": "4.0.0-nvidia" }
  ]
}
```

The server passes `dataset_path` (set in the benchmark config UI) as a bind mount and CLI argument into the container automatically.

## Building

```bash
# LLM Inference
cd llm-inference
docker build -t llminference:local .

# Jupyter Notebook
cd jupyter-notebook
docker build -t jupyternotebook:local .
```

## License

Apache-2.0 — see [LICENSE](LICENSE). This licence covers CoreSpan source code only, not model weights or datasets. See [NOTICE](NOTICE) for third-party dependency terms.
