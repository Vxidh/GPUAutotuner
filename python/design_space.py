"""Shared definition of the GPU kernel configuration space and its hardware
constraints (Objective 1: constrained mathematical design matrix).

Mirrors the static_asserts in cuda/gemm_kernel.cuh exactly; the kernel will
not compile for any configuration rejected here, and vice versa.
"""
from itertools import product

SMEM_LIMIT_BYTES = 49152  # static __shared__ limit per block (48 KiB)
MAX_THREADS = 1024
FACTORS = ["BM", "BN", "BK", "TM", "TN", "UNROLL"]

# Full exploratory factor levels (exhaustive ground-truth grid)
FULL_LEVELS = {
    "BM": [32, 64, 128, 256],
    "BN": [32, 64, 128, 256],
    "BK": [8, 16, 32, 64],
    "TM": [2, 4, 8],
    "TN": [2, 4, 8],
    "UNROLL": [1, 2, 4, 8],
}

# 2-level screening factors for the fractional factorial design
SCREEN_LEVELS = {
    "BM": [64, 128],
    "BN": [64, 128],
    "BK": [8, 32],
    "TM": [4, 8],
    "TN": [4, 8],
    "UNROLL": [1, 8],
}


def is_valid(cfg, M=4096, N=4096, K=4096):
    """Returns (ok, reason). Constraint order matches gemm_kernel.cuh."""
    bm, bn, bk, tm, tn = cfg["BM"], cfg["BN"], cfg["BK"], cfg["TM"], cfg["TN"]
    if bm % tm or bn % tn:
        return False, "thread_tile_not_dividing_block_tile"
    if (bm * bn) % (tm * tn):
        return False, "threads_not_integer"
    threads = (bm * bn) // (tm * tn)
    if threads > MAX_THREADS:
        return False, "threads_per_block_gt_1024"
    if (bm * bk + bk * bn) * 4 > SMEM_LIMIT_BYTES:
        return False, "shared_mem_gt_48KiB"
    if threads % bk:
        return False, "A_load_stride_not_integer"
    if threads % bn:
        return False, "B_load_stride_not_integer"
    if threads > bm * bk:
        return False, "A_tile_load_overflow"
    if threads > bk * bn:
        return False, "B_tile_load_overflow"
    if bm % (threads // bk):
        return False, "A_tile_rows_not_covered"
    if bk % (threads // bn):
        return False, "B_tile_rows_not_covered"
    if M % bm or N % bn or K % bk:
        return False, "shape_not_divisible_by_tile"
    return True, "ok"


def enumerate_space(levels=FULL_LEVELS, M=4096, N=4096, K=4096):
    """All candidate configurations with feasibility status and reason."""
    out = []
    for combo in product(*[levels[f] for f in FACTORS]):
        cfg = dict(zip(FACTORS, combo))
        ok, reason = is_valid(cfg, M, N, K)
        out.append((cfg, ok, reason))
    return out


def enumerate_valid(levels=FULL_LEVELS, M=4096, N=4096, K=4096):
    space = enumerate_space(levels, M, N, K)
    return [c for c, ok, _ in space if ok], [(c, r) for c, ok, r in space if not ok]


if __name__ == "__main__":
    import collections
    space = enumerate_space()
    valid = [c for c, ok, _ in space if ok]
    print(f"Full grid: {len(space)} candidates, {len(valid)} feasible")
    print(collections.Counter(r for _, ok, r in space if not ok))
    sv, si = enumerate_valid(SCREEN_LEVELS)
    print(f"Screening 2-level grid: {len(sv)} feasible, {len(si)} infeasible")
