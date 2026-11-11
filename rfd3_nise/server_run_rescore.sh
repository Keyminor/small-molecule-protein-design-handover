#!/usr/bin/env bash
# LASErMPNN (3 sequences) then Boltz-2x structure rescore. Does not start NISE
# and does not run the affinity head. p(bind) is not an admission score.
set -euo pipefail

export PATH="/root/autodl-tmp/.local:/root/autodl-tmp/cuda-12.8/bin:/root/autodl-tmp/foundry-venv/bin:/root/autodl-tmp/NISE/.venv/bin:/usr/bin:/bin"
export CUDA_HOME=/root/autodl-tmp/cuda-12.8
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export UV_CACHE_DIR=/root/autodl-tmp/.cache/uv
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export BOLTZ_CACHE=/root/autodl-tmp/.cache/boltz
export TMPDIR=/root/autodl-tmp/tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NISE=/root/autodl-tmp/NISE
if [ ! -f "${NISE}/run_nise_boltz2x.py" ]; then
  NISE="C:/Users/EDY/Documents/GitHub/NISE"
fi
export PYTHONPATH="${NISE}${PYTHONPATH:+:${PYTHONPATH}}"
NISE_PY="${NISE}/.venv/bin/python"
WORK="${SCRIPT_DIR}"

mkdir -p "${XDG_CACHE_HOME}" "${BOLTZ_CACHE}" "${TMPDIR}"
if [ ! -e /root/.boltz ]; then
  ln -sfn "${BOLTZ_CACHE}" /root/.boltz
fi
cd "${NISE}"

STAGE="${1:-all}"
echo "===== $(date -Is) rescore stage ${STAGE} kernels=on ====="
"${NISE_PY}" "${WORK}/rescore_maturation.py" "${STAGE}" \
  --scaffolds "${WORK}/rfd3_outputs" \
  --work "${WORK}"
echo "===== $(date -Is) rescore done ====="
df -h /root/autodl-tmp
