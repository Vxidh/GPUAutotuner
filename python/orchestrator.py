"""Measurement engine for the final protocol. Runs ON the H200 host.

- compiles each configuration once per source hash (cache key includes a hash
  of bench.cu + gemm_kernel.cuh + nvcc flags, so stale binaries are impossible)
- runs one measurement = one bench process = one block of timed launches
- detects other tenants' GPU activity during the measurement window via NVML
  per-process accounting, waits for a quiet GPU before measuring, and retries
  contended measurements (contract: timing.contention_policy)
- verifies every configuration once per shape (contract: correctness)
- logs every launch (raw_launches.csv) and every measurement block
  (measurements.csv) append-only, with run id, order index and environment
"""
import atexit
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pynvml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_DIR = os.path.join(ROOT, "cuda")
NVCC = "/usr/local/cuda-13.0/bin/nvcc"
NVCC_FLAGS = ["-O3", "-std=c++17", "-arch=sm_90", "-lcublas"]
FACTORS = ["BM", "BN", "BK", "TM", "TN", "UNROLL"]
CONTRACT = json.load(open(os.path.join(ROOT, "contract.json")))
CONTRACT_SHA = hashlib.sha256(open(os.path.join(ROOT, "contract.json"), "rb").read()).hexdigest()[:12]

MAX_ATTEMPTS = 6
QUIET_WINDOW_S = 0.4
QUIET_MAX_WAIT_S = 30.0


def source_hash():
    h = hashlib.sha256()
    for f in ["bench.cu", "gemm_kernel.cuh"]:
        h.update(open(os.path.join(CUDA_DIR, f), "rb").read())
    h.update(" ".join(NVCC_FLAGS).encode())
    return h.hexdigest()[:12]


SRC_HASH = source_hash()
BIN_DIR = os.path.join(ROOT, "bin", SRC_HASH)
os.makedirs(BIN_DIR, exist_ok=True)


def cfg_key(cfg):
    return "_".join(f"{k}{cfg[k]}" for k in FACTORS)


# ---------------------------------------------------------------- compile ---

def compile_one(cfg):
    out = os.path.join(BIN_DIR, f"bench_{cfg_key(cfg)}")
    if os.path.exists(out):
        return cfg_key(cfg), out, None, 0.0
    t0 = time.time()
    r = subprocess.run([NVCC] + NVCC_FLAGS + ["-o", out + ".tmp", os.path.join(CUDA_DIR, "bench.cu")]
                       + [f"-D{k}={cfg[k]}" for k in FACTORS], capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode:
        return cfg_key(cfg), None, r.stderr.strip()[-400:], dt
    os.replace(out + ".tmp", out)
    return cfg_key(cfg), out, None, dt


def compile_many(cfgs, run, workers=20):
    uniq = {cfg_key(c): c for c in cfgs}
    todo = [c for k, c in uniq.items() if not os.path.exists(os.path.join(BIN_DIR, f"bench_{k}"))]
    res = {}
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for key, path, err, dt in ex.map(compile_one, todo):
                res[key] = (path, err)
                run.log_compile(key, path is not None, err, dt)
    for k in uniq:
        if k not in res:
            res[k] = (os.path.join(BIN_DIR, f"bench_{k}"), None)
    return res


# ------------------------------------------------------------------- NVML ---

class Gpu:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def state(self):
        h = self.h
        return {
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1),
        }

    def other_activity_since(self, t0_us, own_pids):
        """Max SM util of any process other than own_pids since t0 (us)."""
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(self.h, int(t0_us))
        except pynvml.NVMLError:
            return 0, []
        seen = {}
        for s in samples:
            if s.pid in own_pids or s.timeStamp < t0_us:
                continue
            if s.smUtil > 0:
                seen[s.pid] = max(seen.get(s.pid, 0), s.smUtil)
        return (max(seen.values()) if seen else 0), sorted(seen)

    def wait_quiet(self, own_pids):
        """Block until no other process used the SMs for QUIET_WINDOW_S."""
        t_start = time.time()
        while time.time() - t_start < QUIET_MAX_WAIT_S:
            busy, _ = self.other_activity_since((time.time() - QUIET_WINDOW_S) * 1e6, own_pids)
            if not busy:
                return time.time() - t_start
            time.sleep(0.1)
        return time.time() - t_start


# ------------------------------------------------------------ clock lock ---

_locked = False


def unlock_clocks():
    global _locked
    if _locked:
        subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)
        subprocess.run(["sudo", "nvidia-smi", "-rmc"], capture_output=True)
        _locked = False


def lock_clocks():
    global _locked
    subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], capture_output=True)
    subprocess.run(["sudo", "nvidia-smi", "-lgc", "1980,1980"], capture_output=True)
    subprocess.run(["sudo", "nvidia-smi", "-lmc", "3201,3201"], capture_output=True)
    _locked = True


atexit.register(unlock_clocks)
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, lambda *a: (unlock_clocks(), sys.exit(130)))


# -------------------------------------------------------------------- run ---

class Run:
    """One immutable run directory; all files append-only."""

    def __init__(self, run_id=None):
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = os.path.join(ROOT, "results", self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.gpu = Gpu()
        self.own_pids = {os.getpid()}
        self.order = self._load_counter()
        self.verified = self._load_json("verified.json", {})
        self.tol = CONTRACT["correctness"]["tolerance"]
        env_path = os.path.join(self.dir, "environment.json")
        if not os.path.exists(env_path):
            json.dump(self.environment(), open(env_path, "w"), indent=2)

    def _load_json(self, name, default):
        p = os.path.join(self.dir, name)
        return json.load(open(p)) if os.path.exists(p) else default

    def _load_counter(self):
        p = os.path.join(self.dir, "measurements.csv")
        if not os.path.exists(p):
            return 0
        with open(p) as f:
            return sum(1 for _ in f) - 1

    def environment(self):
        nv = subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
        return {
            "run_id": self.run_id, "contract_sha256_12": CONTRACT_SHA, "source_hash": SRC_HASH,
            "gpu": pynvml.nvmlDeviceGetName(self.gpu.h),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "cuda_driver_version": pynvml.nvmlSystemGetCudaDriverVersion(),
            "nvcc": nv, "nvcc_flags": NVCC_FLAGS,
            "kernel": os.uname().release, "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

    def _append(self, name, row):
        p = os.path.join(self.dir, name)
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def _append_many(self, name, rows):
        if not rows:
            return
        p = os.path.join(self.dir, name)
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    def log_compile(self, key, ok, err, dt):
        self._append("compile_log.csv", {"run_id": self.run_id, "cfg_key": key, "compiled": ok,
                                          "compile_s": round(dt, 3), "error": (err or "")[:300]})

    # -- one measurement block, with contention detection and retry --------
    def measure(self, binary, cfg, shape, phase, block, point_id, compare=None):
        M, N, K = shape
        tp = CONTRACT["timing"]["per_shape"][str(M)]
        skey = f"{M}x{N}x{K}"
        vkey = f"{skey}|{cfg_key(cfg)}"
        need_verify = vkey not in self.verified
        args = [binary, str(M), str(N), str(K), str(tp["iters"]), str(tp["warmup"])]
        if need_verify:
            args.append("--verify")
        if compare:
            args += ["--compare", str(compare[0]), compare[1]]
        env = dict(os.environ, NVIDIA_TF32_OVERRIDE="0")

        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            waited = self.gpu.wait_quiet(self.own_pids)
            st0 = self.gpu.state()
            t0 = time.time() * 1e6
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            self.own_pids.add(p.pid)
            try:
                out, err = p.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = "", "timeout"
            st1 = self.gpu.state()
            busy, pids = self.gpu.other_activity_since(t0, self.own_pids)
            attempts.append((busy, pids))
            if not busy:
                break
        contended = bool(busy)

        try:
            d = json.loads(out.strip().splitlines()[-1])
        except Exception:
            d = {"status": "launch_invalid:no_json:" + (err or out)[-200:]}

        status = d.get("status", "launch_invalid")
        if need_verify and status == "OK":
            rel = d.get("rel_fro_err")
            ok = rel is not None and np.isfinite(rel) and rel <= self.tol
            self.verified[vkey] = {"rel_fro_err": rel, "max_abs_err": d.get("max_abs_err"), "ok": bool(ok)}
            json.dump(self.verified, open(os.path.join(self.dir, "verified.json"), "w"), indent=1)
        if status == "OK" and vkey in self.verified and not self.verified[vkey]["ok"]:
            status = "numerically_invalid:tolerance"
        validity = "valid" if status == "OK" else status.split(":")[0]

        tc = d.get("t_custom_ms", []) or []
        tb = d.get("t_cublas_ms", []) or []
        self.order += 1
        row = {
            "run_id": self.run_id, "order": self.order, "phase": phase, "shape": skey,
            "M": M, "N": N, "K": K, "block": block, "point_id": point_id,
            **{f: cfg[f] for f in FACTORS}, "cfg_key": cfg_key(cfg),
            "validity": validity, "status": status,
            "attempts": len(attempts), "contended": contended,
            "other_pids": ";".join(map(str, pids)), "other_max_smutil": busy,
            "quiet_wait_s": round(waited, 2),
            "sm_clock_start": st0["sm_clock_mhz"], "sm_clock_end": st1["sm_clock_mhz"],
            "mem_clock_start": st0["mem_clock_mhz"], "temp_c": st1["temp_c"], "power_w": st1["power_w"],
            "threads": d.get("threads"), "smem_bytes": d.get("smem_bytes"),
            "regs_per_thread": d.get("regs_per_thread"), "local_bytes_per_thread": d.get("local_bytes_per_thread"),
            "blocks_per_sm": d.get("blocks_per_sm"), "occupancy": d.get("occupancy"),
            "verified_this_run": need_verify,
            "rel_fro_err": self.verified.get(vkey, {}).get("rel_fro_err"),
            "n_launches": len(tc),
            "median_ms": statistics.median(tc) if tc else None,
            "p25_ms": float(np.percentile(tc, 25)) if tc else None,
            "p75_ms": float(np.percentile(tc, 75)) if tc else None,
            "mean_ms": statistics.fmean(tc) if tc else None,
            "min_ms": min(tc) if tc else None,
            "max_ms": max(tc) if tc else None,
            "tflops": (2.0 * M * N * K / (statistics.median(tc) / 1e3) / 1e12) if tc else None,
            "cublas_mode": d.get("cublas_math_mode") if compare else None,
            "cublas_median_ms": statistics.median(tb) if tb else None,
            "cublas_tflops": (2.0 * M * N * K / (statistics.median(tb) / 1e3) / 1e12) if tb else None,
            "wall_time": time.strftime("%H:%M:%S"),
        }
        self._append("measurements.csv", row)
        raw = [{"run_id": self.run_id, "order": self.order, "phase": phase, "shape": skey,
                "cfg_key": cfg_key(cfg), "block": block, "impl": "custom", "launch": i, "latency_ms": t}
               for i, t in enumerate(tc)]
        raw += [{"run_id": self.run_id, "order": self.order, "phase": phase, "shape": skey,
                 "cfg_key": cfg_key(cfg), "block": block, "impl": "cublas", "launch": i, "latency_ms": t}
                for i, t in enumerate(tb)]
        self._append_many("raw_launches.csv", raw)
        return row

    # -- R randomised replicate blocks over a list of design points --------
    def run_blocks(self, points, shape, phase, n_blocks, seed, compare=None, progress_every=25):
        """points: list of (point_id, cfg). Each block measures every point once,
        in an independently shuffled order."""
        bins = compile_many([c for _, c in points], self)
        rng = np.random.default_rng(seed)
        rows = []
        total = n_blocks * len(points)
        done = 0
        for b in range(1, n_blocks + 1):
            for idx in rng.permutation(len(points)):
                pid, cfg = points[idx]
                path, err = bins[cfg_key(cfg)]
                if path is None:
                    self.order += 1
                    row = {"run_id": self.run_id, "order": self.order, "phase": phase,
                           "shape": "x".join(map(str, shape)), "block": b, "point_id": pid,
                           **{f: cfg[f] for f in FACTORS}, "cfg_key": cfg_key(cfg),
                           "validity": "compile_invalid", "status": "compile_invalid:" + (err or "")[:120]}
                    self._append("compile_failures.csv", row)
                    rows.append(row)
                else:
                    rows.append(self.measure(path, cfg, shape, phase, b, pid, compare))
                done += 1
                if done % progress_every == 0 or done == total:
                    c = sum(1 for r in rows if r.get("contended"))
                    retried = sum(1 for r in rows if (r.get("attempts") or 1) > 1)
                    print(f"[{phase} {shape[0]}^3] {done}/{total} measured | retried {retried} | "
                          f"still contended {c} | last clock {rows[-1].get('sm_clock_end')} MHz",
                          flush=True)
        return rows
