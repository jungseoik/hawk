#!/usr/bin/env bash
# Stage 3 — evaluation / demo (anomaly description + QA).
# Requires FULL env + a finetuned checkpoint + test split (added LAST).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${ENV_NAME:-cerberus}"
CFG="${CFG:-configs/eval_configs/eval.yaml}"
GPU_ID="${GPU_ID:-0}"

: "${LLAMA_MODEL:?set LLAMA_MODEL}"
: "${EVAL_CKPT:?set EVAL_CKPT (finetuned checkpoint; also set in cfg ckpt:)}"

echo "[run_eval] env=${ENV_NAME} cfg=${CFG} gpu=${GPU_ID}"
conda run -n "${ENV_NAME}" python app.py \
  --cfg-path "${CFG}" --model_type llama_v2 --gpu-id "${GPU_ID}"
