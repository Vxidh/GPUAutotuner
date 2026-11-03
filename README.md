# DOE–RSM autotuning of a CUDA SGEMM kernel

Minor project, Dept. of Mathematics, MIT Manipal. Design of experiments and response surface methodology used to tune six tiling parameters (BM, BN, BK, TM, TN, UNROLL) of a CUDA FP32 GEMM kernel on an H200, compared against exhaustive search, random search and cuBLAS.

DOE–RSM found (128,128,32,8,8,8) at 34.34 TFLOPS with 59 evaluations: #2 of 1,396 feasible configs, 0.17% from the exhaustive optimum (34.40 TFLOPS). Equal-budget random search had a median gap of 7.26%. The tuned kernel reaches 72% of cuBLAS FP32 and 2.61× the untuned default.

| Path | Contents |
|---|---|
| `cuda/` | Tunable SGEMM kernel and benchmark harness |
| `python/` | Design space, designs, run pipeline, analysis, figures |
| `contract.json` | Pre-registered analysis contract (sha in `contract.sha`) |
| `run_phase1.sh`, `run_phase2.sh`, `phase*.log` | Final-run drivers and their logs |
| `results/final_20261028/` | Raw measurements, decisions, analysis outputs |
| `deliverables/` | Report and deck builders, figures, final PDFs and PPTX |
