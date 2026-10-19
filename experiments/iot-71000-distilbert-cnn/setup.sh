#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR=/workspace/.onboarding/uv-cache
if [ ! -x /workspace/sherlock-review/venv/bin/python ]; then
  uv venv --python python3.12 /workspace/sherlock-review/venv
fi
uv pip install --python /workspace/sherlock-review/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu -r "$script_dir/requirements.lock.txt"
uv pip check --python /workspace/sherlock-review/venv/bin/python
