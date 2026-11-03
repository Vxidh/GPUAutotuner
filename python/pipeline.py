"""Phase runner for the final protocol. Each phase is a separate invocation so
results can be inspected (and reported) between phases. Decisions that later
phases need (retained factors, validation candidates, top-20 list) are read
from results/<run_id>/decisions/, written by analysis.py.

  python pipeline.py <run_id> stability
  python pipeline.py <run_id> screening <M>
  python pipeline.py <run_id> ccd
  python pipeline.py <run_id> validate
  python pipeline.py <run_id> exhaustive
  python pipeline.py <run_id> recheck
  python pipeline.py <run_id> compare
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import Run, CONTRACT, lock_clocks, unlock_clocks, FACTORS
from design_space import enumerate_valid, FULL_LEVELS
import designs

SEED = CONTRACT["randomisation"]["seed"]
BLOCKS = CONTRACT["randomisation"]["replicate_blocks"]
PRIMARY = (4096, 4096, 4096)
DEFAULT_CFG = {"BM": 64, "BN": 64, "BK": 8, "TM": 4, "TN": 4, "UNROLL": 1}


def decision(run, name):
    return json.load(open(os.path.join(run.dir, "decisions", name)))


def main():
    run_id, phase = sys.argv[1], sys.argv[2]
    run = Run(run_id)
    lock_clocks()
    try:
        if phase == "stability":
            alt = {"BM": 128, "BN": 128, "BK": 32, "TM": 8, "TN": 8, "UNROLL": 8}
            rows = run.run_blocks([("default", DEFAULT_CFG), ("alt", alt)], PRIMARY,
                                  "stability", 10, SEED + 1, progress_every=10)
            df = pd.DataFrame(rows)
            out = {}
            for pid, g in df.groupby("point_id"):
                m = g["median_ms"].astype(float)
                out[pid] = {"block_medians_ms": m.round(5).tolist(),
                            "cv_percent": float(100 * m.std(ddof=1) / m.mean()),
                            "contended_blocks": int(g["contended"].sum()),
                            "sm_clocks_seen": sorted(set(g["sm_clock_end"].astype(int)))}
            out["gate_cv_max_percent"] = 1.0
            out["gate_passed"] = all(v["cv_percent"] <= 1.0 for k, v in out.items() if isinstance(v, dict))
            json.dump(out, open(os.path.join(run.dir, "stability.json"), "w"), indent=2)
            print(json.dumps(out, indent=2))

        elif phase == "screening":
            M = int(sys.argv[3])
            shape = (M, M, M)
            run.run_blocks(designs.screening_points(), shape, "screening",
                           BLOCKS["screening"], SEED + M)

        elif phase == "ccd":
            d = decision(run, "retention.json")
            pts, alpha = designs.ccd_points(d["retained"], d["fixed"])
            json.dump([{"point_id": p, "kind": k, "nominal": n, **c} for p, c, n, k in pts],
                      open(os.path.join(run.dir, "decisions", "ccd_design.json"), "w"), indent=1)
            print(f"CCD: {len(pts)} points on {d['retained']}, fixed {d['fixed']}, alpha={alpha:.3f}")
            run.run_blocks([(p, c) for p, c, _, _ in pts], PRIMARY, "ccd", BLOCKS["ccd"], SEED + 7)

        elif phase == "validate":
            cands = decision(run, "validation_candidates.json")["candidates"]
            run.run_blocks([(c["candidate_id"], {f: c[f] for f in FACTORS}) for c in cands],
                           PRIMARY, "validation", BLOCKS["validation"], SEED + 11)

        elif phase == "exhaustive":
            valid, _ = enumerate_valid(FULL_LEVELS)
            run.run_blocks([(f"G{i:04d}", c) for i, c in enumerate(valid)], PRIMARY,
                           "exhaustive", BLOCKS["exhaustive"], SEED + 13, progress_every=100)

        elif phase == "recheck":
            top = decision(run, "exhaustive_top20.json")["top"]
            run.run_blocks([(t["cfg_key"], {f: t[f] for f in FACTORS}) for t in top], PRIMARY,
                           "exhaustive_recheck", BLOCKS["exhaustive_top_recheck"], SEED + 17)

        elif phase == "compare":
            sel = decision(run, "final_selection.json")
            pts = [("default", DEFAULT_CFG), ("rsm_selected", sel["rsm_selected"]),
                   ("exhaustive_best", sel["exhaustive_best"])]
            for mode in ("default", "pedantic"):
                run.run_blocks(pts, PRIMARY, f"compare_{mode}", 3, SEED + 19,
                               compare=(10, mode), progress_every=3)
        else:
            raise SystemExit(f"unknown phase {phase}")
    finally:
        unlock_clocks()


if __name__ == "__main__":
    main()
