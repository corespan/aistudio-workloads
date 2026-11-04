"""
interconnect.py — how the GPUs given to this run are connected, measured before vLLM starts.

Three views, all from inside the container, so they cover exactly the GPUs the
caller handed to `docker run --gpus`:

  gpus       nvidia-smi per GPU: index, UUID, PCIe bus id, PCIe link generation
             and width (current and max). An idle GPU may report a lower current
             generation than it trains to under load; the bandwidth test is the
             measured figure.
  p2p        CUDA peer access between every pair, as torch reports it.
  bandwidth  NVIDIA nvbandwidth (https://github.com/NVIDIA/nvbandwidth) copy-engine
             tests: host<->GPU for every GPU, GPU<->GPU when there are two or more.

The checks run in child processes so this process never holds a CUDA context,
which would keep GPU memory busy for the whole benchmark. CUDA_DEVICE_ORDER is
set to PCI_BUS_ID so CUDA's device numbers match nvidia-smi's.

Nothing here is model-specific: it reports on whatever GPUs the run was given.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

NVBANDWIDTH = os.environ.get("NVBANDWIDTH_BIN", "nvbandwidth")

HOST_TESTS = ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"]
PEER_TESTS = [
    "device_to_device_memcpy_read_ce",
    "device_to_device_bidirectional_memcpy_read_ce",
    "device_to_device_latency_sm",
]

_GPU_FIELDS = [
    "index", "uuid", "name", "pci.bus_id",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
]

_P2P_SCRIPT = (
    "import json, torch\n"
    "n = torch.cuda.device_count()\n"
    "print(json.dumps([[None if i == j else bool(torch.cuda.can_device_access_peer(i, j))"
    " for j in range(n)] for i in range(n)]))\n"
)


def _child_env() -> dict:
    return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def _number(value):
    """nvidia-smi and nvbandwidth print numbers as text, and 'N/A' or '[N/A]' when absent."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def query_gpus(timeout: float = 30) -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + ",".join(_GPU_FIELDS), "--format=csv,noheader,nounits"],
        encoding="utf-8", timeout=timeout,
    )
    gpus = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(_GPU_FIELDS):
            continue
        row = dict(zip(_GPU_FIELDS, cells))
        gpus.append({
            "index": _number(row["index"]),
            "uuid": row["uuid"],
            "name": row["name"],
            "pci_bus_id": row["pci.bus_id"],
            "pcie_gen_current": _number(row["pcie.link.gen.current"]),
            "pcie_gen_max": _number(row["pcie.link.gen.max"]),
            "pcie_width_current": _number(row["pcie.link.width.current"]),
            "pcie_width_max": _number(row["pcie.link.width.max"]),
        })
    return gpus


def query_p2p(timeout: float = 120) -> list[list]:
    """matrix[i][j]: whether GPU i can access GPU j's memory directly; None on the diagonal."""
    out = subprocess.check_output(
        ["python3", "-c", _P2P_SCRIPT], encoding="utf-8", timeout=timeout, env=_child_env(),
    )
    return json.loads(out.strip().splitlines()[-1])


def missing_p2p_pairs(matrix: list[list]) -> list[tuple[int, int]]:
    return [(i, j) for i, row in enumerate(matrix) for j, ok in enumerate(row) if ok is False]


def parse_nvbandwidth(output: str) -> dict:
    """The parts of `nvbandwidth -j` output a report needs; matrices become numbers (None for N/A)."""
    root = json.loads(output).get("nvbandwidth", {})
    tests = []
    for case in root.get("testcases", []):
        tests.append({
            "name": case.get("name"),
            "status": case.get("status"),
            "description": case.get("bandwidth_description"),
            "matrix": [[_number(v) for v in row] for row in case.get("bandwidth_matrix", [])],
            "sum": _number(case.get("sum")),
        })
    return {
        "version": root.get("git_version") or root.get("version"),
        "driver": root.get("Driver Version"),
        "cuda_runtime": root.get("CUDA Runtime Version"),
        "error": root.get("error"),
        "tests": tests,
    }


def run_nvbandwidth(gpu_count: int, timeout: float) -> dict:
    tests = HOST_TESTS + (PEER_TESTS if gpu_count >= 2 else [])
    proc = subprocess.run(
        [NVBANDWIDTH, "-j", "-t", *tests],
        capture_output=True, text=True, timeout=timeout, env=_child_env(),
    )
    # nvbandwidth prints its JSON even when a test is waived; a non-zero exit with
    # no JSON is a real failure.
    try:
        return parse_nvbandwidth(proc.stdout)
    except ValueError:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError("nvbandwidth exited %d: %s"
                           % (proc.returncode, detail[-1] if detail else "no output"))


def check(output_dir: Path, timeout: float) -> dict:
    """
    Collect all three views. Nothing here stops the run: missing P2P is reported
    in p2p_missing, and a view that cannot be collected is recorded with its error.
    """
    result: dict = {"errors": {}}

    try:
        result["gpus"] = query_gpus()
    except Exception as exc:
        result["errors"]["gpus"] = str(exc)
        result["gpus"] = []

    try:
        result["p2p"] = query_p2p()
        result["p2p_missing"] = [list(p) for p in missing_p2p_pairs(result["p2p"])]
    except Exception as exc:
        result["errors"]["p2p"] = str(exc)
        result["p2p"] = None
        result["p2p_missing"] = None

    gpu_count = len(result["gpus"]) or len(result["p2p"] or [])
    if shutil.which(NVBANDWIDTH) is None:
        result["errors"]["bandwidth"] = "%s is not installed in this image" % NVBANDWIDTH
        result["bandwidth"] = None
    else:
        try:
            result["bandwidth"] = run_nvbandwidth(gpu_count, timeout)
        except Exception as exc:
            result["errors"]["bandwidth"] = str(exc)
            result["bandwidth"] = None

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "interconnect.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
