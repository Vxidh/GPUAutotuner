"""Proves python/design_space.is_valid agrees with the kernel's compile-time
static_asserts on EVERY candidate in the full grid: each candidate is compiled
(device code, no linking) and compile success must equal is_valid exactly.
Writes results/feasibility_audit.csv."""
import csv
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design_space import enumerate_space, FACTORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NVCC = "/usr/local/cuda-13.0/bin/nvcc"
TU = os.path.join(tempfile.gettempdir(), "feas_tu.cu")
with open(TU, "w") as f:
    f.write('#include "gemm_kernel.cuh"\n')


def compiles(cfg):
    defs = [f"-D{k}={cfg[k]}" for k in FACTORS]
    r = subprocess.run([NVCC, "-std=c++17", "-arch=sm_90", "-c", "-o", "/dev/null",
                        "-I", os.path.join(ROOT, "cuda"), TU] + defs,
                       capture_output=True, text=True)
    msg = ""
    if r.returncode:
        for line in r.stderr.splitlines():
            if "static assertion failed" in line:
                msg = line.split('with "')[-1].rstrip('"')
                break
        msg = msg or r.stderr.strip().splitlines()[-1][:120]
    return r.returncode == 0, msg


def main():
    space = enumerate_space()
    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(lambda t: compiles(t[0]), space))
    mismatches = 0
    rows = []
    for (cfg, ok, reason), (comp_ok, msg) in zip(space, results):
        if ok != comp_ok:
            mismatches += 1
            print("MISMATCH", cfg, "python:", ok, reason, "| nvcc:", comp_ok, msg)
        rows.append({**cfg, "python_valid": ok, "python_reason": reason,
                     "nvcc_compiles": comp_ok, "nvcc_message": msg})
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "feasibility_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_valid = sum(r["python_valid"] for r in rows)
    print(f"{len(rows)} candidates, {n_valid} feasible, {mismatches} mismatches between python filter and nvcc")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
