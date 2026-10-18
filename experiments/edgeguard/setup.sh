#!/bin/bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR=/workspace/.onboarding/uv-cache
mkdir -p /workspace/edgeguard/data
if [ ! -x /workspace/edgeguard/venv/bin/python ]; then
  uv venv --python python3.12 /workspace/edgeguard/venv
fi
uv pip install --python /workspace/edgeguard/venv/bin/python -r "$experiment_dir/requirements.lock.txt"
uv pip check --python /workspace/edgeguard/venv/bin/python
/workspace/edgeguard/venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
import requests
path=Path('/workspace/edgeguard/data/UNSW_NB15_testing-set.csv')
expected='7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c'
if not path.exists():
    url='https://raw.githubusercontent.com/tylerhoward15/sklearn-RandomForestsClassifier-UNSW-NB15/a011b64e3d1efad4a4bdfbf8ecc4d5e7da21736e/UNSW_NB15_testing-set.csv'
    response=requests.get(url,timeout=120)
    response.raise_for_status()
    payload=response.content
    assert hashlib.sha256(payload).hexdigest()==expected,'Downloaded data checksum mismatch'
    path.write_bytes(payload)
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Existing data checksum mismatch'
print('Dataset SHA-256 verified')
PY
