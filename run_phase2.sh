#!/bin/bash
# Remaining phases, run back-to-back once the exhaustive sweep finishes.
cd ~/autotuner/python && source ../venv/bin/activate
RUN=final_20261028
LOG=../results/$RUN
while ps -eo args | grep -q "[p]ipeline.py $RUN exhaustive"; do sleep 10; done
echo "=== MARK exhaustive_done $(date +%T)"
python3 analysis.py $RUN exhaustive > $LOG/analysis_exhaustive.log 2>&1 || { echo FAIL analysis_exhaustive; exit 1; }
echo "=== MARK exhaustive_analysed $(date +%T)"
python3 pipeline.py $RUN validate || { echo FAIL validate; exit 1; }
python3 analysis.py $RUN validation > $LOG/analysis_validation.log 2>&1 || { echo FAIL analysis_validation; exit 1; }
echo "=== MARK validation_done $(date +%T)"
python3 pipeline.py $RUN recheck || { echo FAIL recheck; exit 1; }
python3 analysis.py $RUN recheck > $LOG/analysis_recheck.log 2>&1 || { echo FAIL analysis_recheck; exit 1; }
echo "=== MARK recheck_done $(date +%T)"
python3 pipeline.py $RUN compare || { echo FAIL compare; exit 1; }
echo "=== MARK compare_done $(date +%T)"
python3 analysis.py $RUN final > $LOG/analysis_final.log 2>&1 || { echo FAIL analysis_final; exit 1; }
echo "=== MARK ALL_DONE $(date +%T)"
