#!/bin/bash
# Phase 1 of the final run: stability gate, then screening at 3 shapes.
cd ~/autotuner && source venv/bin/activate && cd python
RUN=final_20261028
python3 pipeline.py $RUN stability || exit 1
python3 -c "import json,sys; s=json.load(open(\"../results/$RUN/stability.json\")); print(\"STABILITY GATE PASSED\" if s[\"gate_passed\"] else \"STABILITY GATE FAILED\"); sys.exit(0 if s[\"gate_passed\"] else 3)" || exit 3
for M in 4096 2048 1024; do python3 pipeline.py $RUN screening $M || exit 1; done
echo PHASE1_DONE
