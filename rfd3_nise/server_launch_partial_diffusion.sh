#!/usr/bin/env bash
# Partial diffusion of the three oxaliplatin parents. 18 jobs x 800 scaffolds.
# Does not start NISE. Sequence redesign and Boltz run after these backbones exist.
set -euo pipefail

export PATH="/root/autodl-tmp/.local:/root/autodl-tmp/foundry-venv/bin:/root/autodl-tmp/NISE/.venv/bin:/usr/bin:/bin"
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export UV_CACHE_DIR=/root/autodl-tmp/.cache/uv
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export BOLTZ_CACHE=/root/autodl-tmp/.cache/boltz
export FOUNDRY_CHECKPOINT_DIRS=/root/autodl-tmp/.foundry/checkpoints
export TMPDIR=/root/autodl-tmp/tmp
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NISE=/root/autodl-tmp/NISE
if [ ! -f "${NISE}/run_nise_boltz2x.py" ]; then
  NISE="C:/Users/EDY/Documents/GitHub/NISE"
fi
NISE_PY="${NISE}/.venv/bin/python"
FOUNDRY_PY=/root/autodl-tmp/foundry-venv/bin/python
RFD3=/root/autodl-tmp/foundry-venv/bin/rfd3
YAML="${SCRIPT_DIR}/partial_diffusion_jobs.yaml"
OUT="${SCRIPT_DIR}/rfd3_outputs"

mkdir -p "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${HF_HOME}" "${BOLTZ_CACHE}" "${TMPDIR}" "${OUT}"
cd "${NISE}"

echo "===== $(date -Is) convert Boltz parents to RFD3 ====="
"${NISE_PY}" "${SCRIPT_DIR}/prepare_parents.py"
echo "===== $(date -Is) validate partial_t and ligand ====="
"${FOUNDRY_PY}" "${SCRIPT_DIR}/validate_spec.py"
echo "===== $(date -Is) RFD3 partial diffusion dry-run ====="
"${NISE_PY}" "${SCRIPT_DIR}/run_rfdiffusion3.py" \
  --rfd3-executable "${RFD3}" \
  --inputs-yaml "${YAML}" \
  --output-dir "${OUT}" \
  --scale pocket \
  --n-designs 800 \
  --diffusion-batch-size 4 \
  --dry-run
echo "===== $(date -Is) RFD3 partial diffusion design ====="
"${NISE_PY}" "${SCRIPT_DIR}/run_rfdiffusion3.py" \
  --rfd3-executable "${RFD3}" \
  --inputs-yaml "${YAML}" \
  --output-dir "${OUT}" \
  --scale pocket \
  --n-designs 800 \
  --diffusion-batch-size 4 \
  -- \
  skip_existing=True
echo "===== $(date -Is) RFD3 partial diffusion done ====="
find "${OUT}" -type f \( -name '*.cif.gz' -o -name '*.cif' -o -name '*.pdb' \) | wc -l
