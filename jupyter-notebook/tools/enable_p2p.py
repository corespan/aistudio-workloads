#!/usr/bin/env python3
"""
enable_p2p.py — Try enabling CUDA peer-to-peer access between every GPU pair.

CUDA peer access is switched on per process: it lasts only while the process
that asked for it runs. So this does not leave P2P "on" for your notebook or
for vLLM. It proves which pairs CAN be enabled (PyTorch, NCCL and vLLM enable
it themselves when they need it). Pairs the hardware cannot connect are skipped.

Usage (Jupyter terminal):
    python enable_p2p.py
    enable-p2p          # if symlinked via Dockerfile
"""

import ctypes
import sys

CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704

def load_cudart():
    for lib in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    print("ERROR: libcudart.so not found. Is the container running with --gpus all?")
    sys.exit(1)

def main():
    cuda = load_cudart()

    count = ctypes.c_int(0)
    cuda.cudaGetDeviceCount(ctypes.byref(count))
    n = count.value

    if n == 0:
        print("No CUDA GPUs found.")
        sys.exit(0)

    if n == 1:
        print("Only 1 GPU detected — P2P not applicable.")
        sys.exit(0)

    print(f"GPUs detected: {n}")
    print()

    enabled  = []
    skipped  = []
    already  = []
    failed   = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            can = ctypes.c_int(0)
            cuda.cudaDeviceCanAccessPeer(ctypes.byref(can), i, j)

            if not can.value:
                skipped.append((i, j))
                continue

            cuda.cudaSetDevice(i)
            ret = cuda.cudaDeviceEnablePeerAccess(j, ctypes.c_uint(0))

            if ret == 0:
                enabled.append((i, j))
            elif ret == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                already.append((i, j))
            else:
                failed.append((i, j, ret))

    if enabled:
        print("Can be enabled (enabled for this check only):")
        for i, j in enabled:
            print(f"  GPU{i} → GPU{j}")

    if already:
        print("Already enabled:")
        for i, j in already:
            print(f"  GPU{i} → GPU{j}")

    if skipped:
        print("No P2P support (hardware limitation — skipped):")
        for i, j in skipped:
            print(f"  GPU{i} → GPU{j}")

    if failed:
        print("Failed:")
        for i, j, code in failed:
            print(f"  GPU{i} → GPU{j}  (CUDA error {code})")

    print()
    if not failed and not skipped:
        print("✓ Every GPU pair can use P2P. Programs enable it for themselves when they run.")
    elif not failed and skipped:
        print("✓ P2P works where the hardware allows it.")
        print("  Skipped pairs have no direct path (no NVLink/NVSwitch or PCIe peer route).")
    else:
        print("✗ Some pairs failed. Check CUDA error codes above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
