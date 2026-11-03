# AIStudio Workloads

Standalone GPU workloads for LLM benchmarking and interactive Jupyter environments. Each workload runs as an isolated Docker container on any NVIDIA GPU node and can be used independently or managed through [AIStudio Server](https://github.com/corespan/aistudio-server).

## Workloads

### LLM Inference (`llm-inference`)
vLLM-based throughput and latency benchmarking. Starts its own vLLM server, sweeps concurrency levels, and outputs structured benchmark results (tokens/sec, TTFT, TPOT, cost estimates).

**Supported models:** Any HuggingFace-compatible model. Gated models (Llama, etc.) require an HF token — see [MODEL-LICENSES.md](https://github.com/corespan/aistudio-server/blob/master/MODEL-LICENSES.md).

### Jupyter Notebook (`jupyter-notebook`)
JupyterLab environment with GPU profiling utilities and MLPerf microbenchmarks, pre-configured for remote GPU nodes.

## Running Standalone

Pull the image and run directly on any NVIDIA GPU node (no AIStudio Server required):

```bash
# LLM Inference
docker run --gpus all --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/results:/results \
  -v /path/to/dataset.json:/path/to/dataset.json \
  --entrypoint python3 \
  us-docker.pkg.dev/aimlworkbench/aistudio/llminference:1.1.0-nvidia \
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
  -e workload_id=my-session \
  -e workload_port=7008 \
  us-docker.pkg.dev/aimlworkbench/aistudio/jupyternotebook:1.0.0-nvidia
```

**Dataset:** Any dataset type `vllm bench serve` supports, chosen with `--dataset-name` (default `sharegpt`, a JSON list of conversations). Common choices:

| `--dataset-name` | `--dataset-path` | Notes |
|---|---|---|
| `sharegpt` (default) | JSON list of conversations | |
| `custom` | JSONL, one `{"prompt": "..."}` per line | output length via `--extra_bench_args "--custom-output-len 128"` |
| `random` | none | lengths via `--extra_bench_args "--random-input-len 512 --random-output-len 128"` |
| `hf` | a Hugging Face dataset id | needs internet |

No dataset is bundled or auto-downloaded. You can use [OpenOrca (MIT)](https://huggingface.co/datasets/Open-Orca/OpenOrca) or [Dolly (CC-BY-SA-3.0)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) as clean alternatives.

**Model-specific settings:** `--extra_serve_args` passes flags straight to `vllm serve` (for example `"--gpu-memory-utilization 0.95 --kv-cache-dtype fp8"`), and loading has no fixed time limit: the script checks the vLLM server every 2 s, fails at once on a fatal vLLM error (out of memory, engine failed to start), and otherwise waits as long as vLLM keeps printing progress. `--server_stall_timeout` (default 1800 s) fails a load that prints nothing for that long; `--server_timeout` adds a total limit if you want one (default 0, none). The model can be a Hugging Face id or a local path, such as weights on an NFS mount.

## Integration with AIStudio Server

When used with [AIStudio Server](https://github.com/corespan/aistudio-server), these workloads are orchestrated automatically — the server handles node selection, SSH execution, result collection, and the benchmark dashboard.

To use a specific workload image version, set the `image_tag` in the server's `catalog.json`:

```json
{
  "workload_types": [
    { "name": "LLMInference", "image_tag": "1.0.0-nvidia" },
    { "name": "JupyterNotebook", "image_tag": "1.0.0-nvidia" }
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
