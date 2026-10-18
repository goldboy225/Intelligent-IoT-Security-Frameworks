#!/bin/bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/workspace/edgeguard/matplotlib XDG_CACHE_HOME=/workspace/edgeguard/cache
cd "$experiment_dir"
/workspace/edgeguard/venv/bin/python -m unittest -v test_protocol
experiment_cpu="$(/workspace/edgeguard/venv/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')"
experiment_output="${EDGEGUARD_OUTPUT_DIR:-/workspace/edgeguard/results/hybrid}"
mkdir -p "$experiment_output"
echo "Running experiment; progress is in $experiment_output/run.log"
exec taskset -c "$experiment_cpu" /workspace/edgeguard/venv/bin/python -c 'import run_experiment; run_experiment.main()' \
  --data /workspace/edgeguard/data/UNSW_NB15_testing-set.csv \
  --output "$experiment_output" --extended "$@" > "$experiment_output/run.log" 2>&1
