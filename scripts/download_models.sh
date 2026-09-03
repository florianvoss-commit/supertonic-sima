#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SUPERTONIC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
APP_ROOT="${SUPERTONIC_APP_ROOT:-/media/nvme/supertonic-tts}"
VENV="${APP_ROOT}/.venv"
MODEL_ROOT="${APP_ROOT}/models"
UPSTREAM_DIR="${MODEL_ROOT}/supertonic-3"
COMPILED_DIR="${MODEL_ROOT}/supertonic-3-sima"

export HF_HOME="${APP_ROOT}/hf-cache"
export TMPDIR="${APP_ROOT}/tmp"
mkdir -p "${MODEL_ROOT}" "${HF_HOME}" "${TMPDIR}"

if [[ ! -x "${VENV}/bin/hf" ]]; then
  echo "Missing Hugging Face CLI in ${VENV}; run ${REPO_ROOT}/scripts/setup_devkit.sh" >&2
  exit 2
fi

"${VENV}/bin/hf" download florianvoss/supertonic-3-sima \
  supertonic_vector_field_sima_mpk.tar.gz \
  supertonic_vocoder_sima_bf16_mpk.tar.gz \
  supertonic_runtime_data.npz \
  artifact_manifest.json \
  vocoder_bf16_manifest.json \
  --local-dir "${COMPILED_DIR}"

"${VENV}/bin/hf" download Supertone/supertonic-3 \
  onnx/duration_predictor.onnx \
  onnx/text_encoder.onnx \
  onnx/vocoder.onnx \
  onnx/tts.json \
  onnx/unicode_indexer.json \
  voice_styles/F1.json voice_styles/F2.json voice_styles/F3.json \
  voice_styles/F4.json voice_styles/F5.json \
  voice_styles/M1.json voice_styles/M2.json voice_styles/M3.json \
  voice_styles/M4.json voice_styles/M5.json \
  --revision 724fb5abbf5502583fb520898d45929e62f02c0b \
  --local-dir "${UPSTREAM_DIR}"

(
  cd "${COMPILED_DIR}"
  printf '%s  %s\n' \
    '2f6b8c918e0c402453e48bd2686dbea429e6ce1dd98151c940d88229980e8dd2' \
    'supertonic_vector_field_sima_mpk.tar.gz' \
    '90b2d6a089c8527826dd1d0cb5b557316ac703045f422d6e1332deeabd84e0cb' \
    'supertonic_vocoder_sima_bf16_mpk.tar.gz' \
    '81fc7a131dd0eafe6fa7062b23f49e0dc9e3fb178f5473bf8d404ab555bcf5aa' \
    'supertonic_runtime_data.npz' \
    '1659e891f4da0cc80c6dd7a5fbb3ca87f3a12ad7f0f2bce0bfb0963433709628' \
    'vocoder_bf16_manifest.json' \
    | sha256sum --check
)

echo "Model assets ready: ${MODEL_ROOT}"
