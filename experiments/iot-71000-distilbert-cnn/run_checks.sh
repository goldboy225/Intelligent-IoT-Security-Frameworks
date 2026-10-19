#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
export MPLCONFIGDIR=/workspace/sherlock-review/matplotlib XDG_CACHE_HOME=/workspace/sherlock-review/cache
export HF_HOME=/workspace/sherlock-review/huggingface IPYTHONDIR=/workspace/sherlock-review/ipython
export JUPYTER_RUNTIME_DIR=/workspace/sherlock-review/jupyter-runtime OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p results
/workspace/sherlock-review/venv/bin/python -m unittest -v test_protocol > results/protocol_tests.log 2>&1
/workspace/sherlock-review/venv/bin/python audit_paper.py > results/arithmetic_audit.log 2>&1
/workspace/sherlock-review/venv/bin/python inspect_architecture.py > results/architecture_checks.log 2>&1
/workspace/sherlock-review/venv/bin/python preflight.py > results/preflight.log 2>&1
/workspace/sherlock-review/venv/bin/python -m ipykernel install --prefix /workspace/sherlock-review/venv --name sherlock-review
/workspace/sherlock-review/venv/bin/python build_report.py
echo 'Checks complete. Read REVIEW_REPORT.md for the unresolved real-data training status.'
