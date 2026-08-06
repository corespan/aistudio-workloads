# Contributing to AIStudio Workloads

Thanks for your interest in contributing. This document covers how to get started, what areas are open for extension, and what to verify before opening a pull request.

## Getting Started

1. Fork the repo and create a branch from `main`.
2. Make your changes inside the relevant workload directory (`llm-inference/` or `jupyter-notebook/`).
3. Build and test your image locally before opening a PR.
4. Open a pull request against `main` with a clear description of what and why.

For new workload types or significant changes, open an issue first to discuss the approach.

## Areas Open for Extension

| Area | Examples |
|------|---------|
| **New workload types** | Training jobs, embedding benchmarks, diffusion model inference, ASR |
| **AMD ROCm support** | ROCm-compatible Dockerfiles alongside the existing NVIDIA images |
| **Benchmark metrics** | Extend `benchmark.py` with additional vLLM metrics or cost models |
| **Dataset formats** | Support for dataset formats beyond ShareGPT (e.g. OASST, Dolly) |
| **Jupyter notebooks** | New GPU profiling or MLPerf benchmark notebooks |
| **Hardware pricing** | Extend `hardware_pricing.json` with additional GPU types and cloud regions |

## Best Practices

- **One workload per directory.** Each workload is self-contained with its own `Dockerfile`, `requirements.txt`, and entrypoint.
- **No bundled datasets.** Workloads must not download or embed datasets automatically. Use `--dataset-path` and require the operator to supply the file.
- **Permissive licences only.** New dependencies must be Apache-2.0, MIT, or BSD. No GPL, AGPL, or packages with contested provenance without prior discussion. Update the NOTICE file if you add a package with notice obligations.
- **Keep images lean.** Do not add packages to `requirements.txt` that are already in the base image (`vllm/vllm-openai` for llm-inference).

## Before Raising a PR

```bash
# 1. Build the image locally
cd llm-inference   # or jupyter-notebook
docker build -t llminference:dev .

# 2. Run a smoke test (requires a GPU node with Docker)
docker run --gpus all --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/results:/results \
  -v /path/to/dataset.json:/path/to/dataset.json \
  --entrypoint python3 llminference:dev \
  /llm-inference/benchmark.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset-path /path/to/dataset.json \
    --concurrencies 1 \
    --output_data_path /results/smoke

# 3. Confirm BENCH_RESULT lines are printed and results written to /tmp/results
```

CI runs a lint and image build check on every PR. PRs that fail to build will not be merged.

## Reporting Bugs

Open a GitHub issue with: steps to reproduce, expected vs actual behaviour, relevant log output, GPU type, and image tag used.

## Security Issues

Do not open a public issue for security vulnerabilities. Email [muraharirao@corespan.ai](mailto:muraharirao@corespan.ai) directly.
