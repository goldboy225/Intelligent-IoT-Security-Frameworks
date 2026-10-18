#!/bin/bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/workspace/edgeguard/matplotlib XDG_CACHE_HOME=/workspace/edgeguard/cache
cd "$experiment_dir"
/workspace/edgeguard/venv/bin/python fetch_additional_data.py
experiment_cpu="$(/workspace/edgeguard/venv/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')"
mkdir -p /workspace/edgeguard/results/edgeiiot /workspace/edgeguard/results/botiot_alternative
taskset -c "$experiment_cpu" /workspace/edgeguard/venv/bin/python -c 'import run_experiment;run_experiment.main()' \
  --data /workspace/edgeguard/data/ML-EdgeIIoT-dataset.csv \
  --output /workspace/edgeguard/results/edgeiiot --dataset edgeiiot --models EdgeGuard > /workspace/edgeguard/results/edgeiiot/run.log 2>&1
/workspace/edgeguard/venv/bin/python botiot_data.py
taskset -c "$experiment_cpu" /workspace/edgeguard/venv/bin/python -c 'import run_experiment;run_experiment.main()' \
  --data /workspace/edgeguard/data/Bot-IoT-5percent-independent-sample.csv \
  --output /workspace/edgeguard/results/botiot_alternative --dataset botiot_alternative --models EdgeGuard > /workspace/edgeguard/results/botiot_alternative/run.log 2>&1
taskset -c "$experiment_cpu" /workspace/edgeguard/venv/bin/python botiot_transfer.py > /workspace/edgeguard/results/botiot_transfer.log 2>&1
