#!/usr/bin/env python3
"""
check_p2p.py — Check CUDA peer-to-peer access between all GPU pairs.

Usage (Jupyter terminal):
    python check_p2p.py
    check-p2p          # if symlinked via Dockerfile
"""

import ctypes
import sys

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

    print(f"GPUs detected: {n}")
    print()

    # Print device names
    for i in range(n):
        cuda.cudaSetDevice(i)
        # cudaGetDeviceProperties is complex; use nvidia-smi for names
    try:
        import subprocess
        names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
    except Exception:
        names = [f"GPU{i}" for i in range(n)]

    for i, name in enumerate(names):
        print(f"  GPU {i}: {name.strip()}")
    print()

    # P2P matrix
    header = "        " + "".join(f"  GPU{j}  " for j in range(n))
    print(header)
    print("       " + "-" * (8 * n))

    all_ok = True
    for i in range(n):
        row = f"  GPU{i}  |"
        for j in range(n):
            if i == j:
                row += "   --  "
            else:
                can = ctypes.c_int(0)
                cuda.cudaDeviceCanAccessPeer(ctypes.byref(can), i, j)
                if can.value:
                    row += "  YES  "
                else:
                    row += "   NO  "
                    all_ok = False
        print(row)

    print()
    if n == 1:
        print("Single GPU — P2P not applicable.")
    elif all_ok:
        print("✓ All GPU pairs support P2P access.")
    else:
        print("✗ Some pairs lack P2P support.")
        print("  Run enable_p2p.py to attempt enabling it.")

if __name__ == "__main__":
    main()
