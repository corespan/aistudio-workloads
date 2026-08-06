#!/usr/bin/env python3
"""
benchmark.py — LLM inference benchmarking for AIStudio (open-source edition).

Starts a vLLM server, sweeps through concurrency levels using `vllm bench serve`,
and writes results to <output_data_path>/.

The dataset must be supplied by the caller via --dataset-path. It must be a
ShareGPT-format JSON file (a list of conversation objects). The file is not
downloaded or fetched at runtime — the caller is responsible for placing it on
the GPU node before the benchmark runs.

Usage (standalone):
    python3 benchmark.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output_data_path /results/run-001 \
        --run_name tinyllama-tp1 \
        --tp 1 \
        --concurrencies 32,64,128 \
        --dataset-path /datasets/sharegpt.json

Results are written to:
    <output_data_path>/benchmark_result.json  — raw per-concurrency rows (jsonl)
    <output_data_path>/summary.json           — aggregated metrics + cost estimates
    <output_data_path>/logs/status.json       — current run status
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests
import urllib3

from costEngine import CostEngine

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Logging — structured, timestamped, to stdout so Docker captures it
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM LLM inference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or local path (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    p.add_argument("--output_data_path", required=True,
                   help="Directory where benchmark results are written")
    p.add_argument("--run_name", required=True,
                   help="Label for this run (used in log messages)")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor-parallel size (number of GPUs)")
    p.add_argument("--concurrencies", default="32,64,128",
                   help="Comma-separated concurrency levels to sweep")

    # Server tuning
    p.add_argument("--enable_optimizations", action="store_true",
                   help="Enable CUDA graph (FULL) + -O3 compiler flags. "
                        "Requires sm80+ (Ampere/Hopper). Do NOT set for T4/V100.")
    p.add_argument("--max_model_len", type=int, default=None,
                   help="Override vLLM max model context length. "
                        "If omitted, vLLM reads max_position_embeddings from model config.")

    # Optional: profiler
    p.add_argument("--enable_profiler", action="store_true",
                   help="Run a 2-request profiler warmup and save Torch traces")

    # Dataset
    p.add_argument("--dataset-path", required=True,
                   help="Path to a ShareGPT-format JSON dataset file on the local filesystem. "
                        "Must be provided — no automatic download is performed.")

    # Optional: GPU power monitoring via Prometheus
    p.add_argument("--enable_power_metrics", action="store_true",
                   help="Collect GPU power and utilisation via Prometheus")
    p.add_argument("--prometheus_url", default="http://localhost:9090",
                   help="Prometheus server URL (used only with --enable_power_metrics)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def write_json(path: Path, data: dict) -> None:
    """Atomically write *data* as JSON to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_status(log_dir: Path, status: str, reason: str = "") -> None:
    payload = {"status": status}
    if reason:
        payload["reason"] = reason
    write_json(log_dir / "status.json", payload)
    log.info("Status → %s%s", status, f" ({reason})" if reason else "")


# ---------------------------------------------------------------------------
# System helpers
# ---------------------------------------------------------------------------
def free_port(port: int) -> None:
    """Kill any process currently listening on *port*."""
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            pid = conn.pid
            if pid:
                log.info("Freeing port %d — terminating PID %d", port, pid)
                proc = psutil.Process(pid)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()


def get_gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        ).strip().split("\n")[0].strip()
    except Exception:
        return "DEFAULT"


def get_gpu_uuids(count: int) -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return [l.strip() for l in out.strip().split("\n") if l.strip()][:count]
    except Exception as exc:
        log.warning("Could not query GPU UUIDs: %s", exc)
        return []


def wait_server_ready(server: subprocess.Popen, port: int, timeout: int = 600) -> None:
    """Poll /v1/models until the vLLM server responds 200 or *timeout* seconds elapse."""
    url = f"http://127.0.0.1:{port}/v1/models"
    for elapsed in range(timeout):
        if server.poll() not in (None, 0):
            stderr = server.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"vLLM server exited prematurely:\n{stderr.strip()}")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                log.info("vLLM server ready after %ds", elapsed)
                return
        except Exception:
            pass
        if elapsed > 0 and elapsed % 30 == 0:
            log.info("Waiting for vLLM server on port %d … (%ds / %ds)", port, elapsed, timeout)
        time.sleep(1)
    raise RuntimeError(f"vLLM server did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# GPU power monitor
# ---------------------------------------------------------------------------
class GPUMonitor:
    """
    Polls Prometheus for per-GPU power and utilisation at *interval* seconds.
    Call start() before a benchmark run and stop() after to collect samples.
    """

    def __init__(self, prometheus_url: str, gpu_uuids: list[str], interval: float = 1.0):
        self.prometheus_url = prometheus_url
        self.gpu_uuids = gpu_uuids
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple] = []  # (timestamp, avg_power_w, avg_util_pct)

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def metrics(self) -> tuple[float, float, list]:
        if not self.samples:
            return 0.0, 0.0, []
        avg_p = round(sum(s[1] for s in self.samples) / len(self.samples), 3)
        avg_u = round(sum(s[2] for s in self.samples) / len(self.samples), 3)
        return avg_p, avg_u, self.samples

    def _query_prometheus(self, query: str, fallback: str | None = None) -> float | None:
        url = f"{self.prometheus_url}/api/v1/query"
        for q in ([query] + ([fallback] if fallback else [])):
            try:
                r = requests.get(url, params={"query": q}, timeout=2, verify=False)
                if r.status_code == 200:
                    result = r.json().get("data", {}).get("result", [])
                    if result:
                        return float(result[0]["value"][1])
            except Exception:
                pass
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            total_p, total_u, n = 0.0, 0.0, 0
            for uuid in self.gpu_uuids:
                clean = uuid.replace("GPU-", "")
                p = self._query_prometheus(
                    f'nvidia_smi_power_draw_watts{{uuid="{clean}"}}',
                    "nvidia_smi_power_draw_watts",
                )
                u = self._query_prometheus(
                    f'nvidia_smi_utilization_gpu_ratio{{uuid="{clean}"}}',
                    "nvidia_smi_utilization_gpu_ratio",
                )
                if p is not None and u is not None:
                    total_p += p
                    total_u += u * 100.0
                    n += 1
            if n:
                self.samples.append((time.time(), total_p / n, total_u / n))
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------
def aggregate_results(
    args: argparse.Namespace,
    output_dir: Path,
    power_map: dict[int, tuple],
    gpu_name: str,
) -> None:
    """
    Read benchmark_result.json (jsonl), attach power + cost metrics,
    write summary.json, and print a human-readable table.
    """
    benchmark_file = output_dir / "benchmark_result.json"
    if not benchmark_file.exists():
        log.warning("benchmark_result.json not found — skipping aggregation")
        return

    required_keys = [
        "duration", "total_input_tokens", "total_output_tokens",
        "output_throughput", "total_token_throughput", "request_throughput",
        "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "mean_e2el_ms",
        "p50_e2el_ms", "p99_e2el_ms",
        # vllm bench serve calls these num_prompts / completed
        "num_prompts", "completed",
    ]

    engine = CostEngine()

    summary: dict = {}
    with benchmark_file.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conc = row.get("concurrency")
            if conc is None:
                continue

            m: dict = {k: round(v, 2) for k in required_keys
                       if isinstance((v := row.get(k)), (int, float))}

            conc_int = int(conc)
            if args.enable_power_metrics and conc_int in power_map:
                avg_p, avg_u, _ = power_map[conc_int]
                m["avg_gpu_power_watts"] = round(avg_p, 3)
                m["avg_gpu_utilization_percent"] = round(avg_u, 3)
                throughput = m.get("total_token_throughput", 0)
                m["gpu_power_efficiency_score"] = (
                    round((throughput / avg_p) * (avg_u / 100.0), 4) if avg_p > 0 else 0.0
                )

            if m.get("duration") and m.get("total_input_tokens"):
                m["cost_metrics"] = engine.calculate_llm(
                    gpu_name=gpu_name,
                    num_gpus=args.tp,
                    duration_seconds=m["duration"],
                    avg_power_watts=m.get("avg_gpu_power_watts", 300.0),
                    total_input_tokens=m["total_input_tokens"],
                    total_output_tokens=m["total_output_tokens"],
                )

            summary[f"#gpus-{args.tp} concurrency-{conc}"] = m

    write_json(output_dir / "summary.json", summary)

    # Human-readable table
    print("\n" + "─" * 70)
    print(f"  Benchmark Summary — {args.run_name}")
    print("─" * 70)
    for label, m in summary.items():
        print(f"\n  {label}")
        print(f"    throughput : {m.get('total_token_throughput', 'n/a')} tok/s")
        print(f"    ttft p50   : {m.get('mean_ttft_ms', 'n/a')} ms")
        print(f"    tpot p50   : {m.get('mean_tpot_ms', 'n/a')} ms")
        print(f"    e2el p50   : {m.get('mean_e2el_ms', 'n/a')} ms")
    print("\n" + "─" * 70 + "\n")

    log.info("Summary written → %s/summary.json", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_data_path)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    set_status(log_dir, "Starting")

    # ── Validate dataset path ────────────────────────────────────────────────
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        log.error("Dataset file not found: %s", dataset_path)
        set_status(log_dir, "Failed", "dataset file not found: %s" % dataset_path)
        sys.exit(1)
    log.info("Dataset: %s", dataset_path)

    # ── Validate concurrencies ───────────────────────────────────────────────
    try:
        concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
        assert all(c > 0 for c in concurrencies), "all values must be positive"
    except (ValueError, AssertionError) as exc:
        log.error("Invalid --concurrencies '%s': %s", args.concurrencies, exc)
        sys.exit(1)

    # ── Profiler setup ───────────────────────────────────────────────────────
    prof_dir: Path | None = None
    if args.enable_profiler:
        prof_dir = output_dir / "VLLM_TORCH_PROFILER_DIR"
        prof_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(prof_dir)
        log.info("Profiler enabled — traces → %s", prof_dir)

    # ── Build serve command ──────────────────────────────────────────────────
    port = 9123
    free_port(port)

    serve_cmd = [
        "vllm", "serve", args.model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(args.tp),
        "--disable-log-requests",
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]
    if args.max_model_len:
        serve_cmd += ["--max-model-len", str(args.max_model_len)]
    if args.enable_optimizations:
        # sm80+ only (Ampere / Hopper / Blackwell). Skip for T4 / V100.
        serve_cmd += ["--compilation-config", '{"cudagraph_mode": "FULL"}', "-O3"]
        log.info("Compiler optimizations enabled (CUDA graph FULL + -O3)")
    else:
        log.info("Running without compiler optimizations (safe for T4 / V100)")

    # ── Start vLLM server ────────────────────────────────────────────────────
    log.info("Starting vLLM server — tp=%d, port=%d", args.tp, port)
    server = subprocess.Popen(serve_cmd, stderr=subprocess.PIPE, start_new_session=False)

    monitor: GPUMonitor | None = None
    _shutdown_done = False

    def shutdown(signum=None, frame=None, exit_code: int = 0) -> None:
        nonlocal _shutdown_done
        if _shutdown_done:
            return
        _shutdown_done = True
        log.info("Shutting down vLLM server …")
        if monitor:
            monitor.stop()
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        os._exit(exit_code)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        wait_server_ready(server, port)
    except RuntimeError as exc:
        log.error("Server failed to start: %s", exc)
        set_status(log_dir, "Failed", "vLLM server did not start")
        shutdown(exit_code=1)

    set_status(log_dir, "Running")

    # ── Profiler warmup (optional) ───────────────────────────────────────────
    if args.enable_profiler and prof_dir:
        log.info("Running profiler warmup (2 requests) …")
        warmup_cmd = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--host", "127.0.0.1", "--port", str(port),
            "--endpoint", "/v1/completions",
            "--model", args.model,
            "--dataset-name", "sharegpt",
            "--dataset-path", str(dataset_path),
            "--num-prompts", "2", "--max-concurrency", "1",
            "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--profile", "--trust-remote-code",
        ]
        subprocess.run(warmup_cmd, check=True)
        log.info("Profiler warmup done — traces saved to %s", prof_dir)

    # ── GPU power monitor setup ──────────────────────────────────────────────
    power_map: dict[int, tuple] = {}
    if args.enable_power_metrics:
        uuids = get_gpu_uuids(args.tp)
        if uuids:
            monitor = GPUMonitor(args.prometheus_url, uuids)
            log.info("GPU power monitoring active — %d GPU(s)", len(uuids))
        else:
            log.warning("No GPU UUIDs found — power monitoring disabled")

    gpu_name = get_gpu_name()
    log.info("GPU detected: %s", gpu_name)

    # ── Concurrency sweep ────────────────────────────────────────────────────
    for conc in concurrencies:
        log.info("─" * 50)
        log.info("Concurrency = %d  (prompts = %d)", conc, 10 * conc)

        bench_cmd = [
            "vllm", "bench", "serve",
            "--backend", "vllm",
            "--host", "127.0.0.1", "--port", str(port),
            "--endpoint", "/v1/completions",
            "--model", args.model,
            "--dataset-name", "sharegpt",
            "--dataset-path", str(dataset_path),
            "--num-prompts", str(10 * conc),
            "--max-concurrency", str(conc),
            "--ignore-eos",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result", "--append-result",
            "--result-dir", str(output_dir),
            "--result-filename", "benchmark_result.json",
            "--metadata", f"concurrency={conc}",
            "--trust-remote-code",
        ]

        if monitor:
            monitor.start()

        try:
            subprocess.run(bench_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            log.error("Benchmark failed at concurrency=%d: %s", conc, exc)
            set_status(log_dir, "Failed", f"benchmark error at concurrency={conc}")
            shutdown(exit_code=1)

        if monitor:
            monitor.stop()
            avg_p, avg_u, samples = monitor.metrics()
            power_map[conc] = (avg_p, avg_u, samples)
            log.info("Power — avg=%.1fW  util=%.1f%%  samples=%d", avg_p, avg_u, len(samples))

    # ── Aggregate results ────────────────────────────────────────────────────
    aggregate_results(args, output_dir, power_map, gpu_name)
    set_status(log_dir, "Completed")
    log.info("Run '%s' complete.", args.run_name)

    # ── Emit BENCH_RESULT: lines for aistudio-server ─────────────────────────
    # The worker reads these lines from SSH stdout to create BenchmarkResult rows.
    # One line per concurrency level; worker creates one DB row per line.
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            for label, m in summary.items():
                # Extract concurrency from label "#gpus-N concurrency-M"
                conc_val = None
                for part in label.split():
                    if part.startswith("concurrency-"):
                        try:
                            conc_val = int(part.split("-", 1)[1])
                        except ValueError:
                            pass
                result = {
                    "concurrency":              conc_val,
                    "total_token_throughput":   m.get("total_token_throughput"),
                    "output_token_throughput":  m.get("output_throughput"),
                    "mean_ttft_ms":             m.get("mean_ttft_ms"),
                    "p99_ttft_ms":              m.get("p99_ttft_ms"),
                    "mean_tpot_ms":             m.get("mean_tpot_ms"),
                    "mean_e2el_ms":             m.get("mean_e2el_ms"),
                    "p50_e2el_ms":              m.get("p50_e2el_ms"),
                    "p99_e2el_ms":              m.get("p99_e2el_ms"),
                    "duration_s":               m.get("duration"),
                    "total_requests":           m.get("num_prompts"),
                    "successful_requests":      m.get("completed"),
                    "total_input_tokens":       m.get("total_input_tokens"),
                    "total_output_tokens":      m.get("total_output_tokens"),
                    "avg_gpu_power_watts":      m.get("avg_gpu_power_watts"),
                    "avg_gpu_utilization_pct":  m.get("avg_gpu_utilization_percent"),
                    "gpu_power_efficiency":     m.get("gpu_power_efficiency_score"),
                    "cost_metrics":             m.get("cost_metrics"),
                }
                print("BENCH_RESULT:" + json.dumps(result), flush=True)
        except Exception as exc:
            log.warning("Could not emit BENCH_RESULT lines: %s", exc)

    shutdown()


if __name__ == "__main__":
    main()
