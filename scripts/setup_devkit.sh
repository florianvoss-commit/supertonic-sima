#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${SUPERTONIC_APP_ROOT:-/media/nvme/llima/supertonic-tts}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SUPERTONIC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VENV="${APP_ROOT}/.venv"
PYNEAT_WHEEL="${PYNEAT_WHEEL:-/media/nvme/neat/pyneat-0.4.0-cp311-cp311-linux_aarch64.whl}"

export HF_HOME="${APP_ROOT}/hf-cache"
export PIP_CACHE_DIR="${APP_ROOT}/pip-cache"
export TMPDIR="${APP_ROOT}/tmp"

mkdir -p \
  "${APP_ROOT}" \
  "${APP_ROOT}/output" \
  "${APP_ROOT}/models" \
  "${HF_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"

if [[ ! -f "${PYNEAT_WHEEL}" ]]; then
  echo "Missing pyneat wheel: ${PYNEAT_WHEEL}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/requirements-devkit.txt" ]]; then
  echo "Missing requirements file under ${REPO_ROOT}" >&2
  exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
sed -i \
  's/^include-system-site-packages = true$/include-system-site-packages = false/' \
  "${VENV}/pyvenv.cfg"

"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${REPO_ROOT}/requirements-devkit.txt"
"${VENV}/bin/python" -m pip install --no-deps "${PYNEAT_WHEEL}"

SUPERTONIC_APP_ROOT="${APP_ROOT}" \
  SUPERTONIC_REPO_ROOT="${REPO_ROOT}" \
  bash "${REPO_ROOT}/scripts/download_models.sh"

"${VENV}/bin/python" - <<'PY'
import numpy
import onnxruntime
import pyneat

print(f"numpy={numpy.__version__}")
print(f"onnxruntime={onnxruntime.__version__}")
print(f"pyneat={getattr(pyneat, '__version__', '0.4.0')}")
PY

echo "DevKit environment ready: ${VENV}"
echo "Model assets ready: ${APP_ROOT}/models"
