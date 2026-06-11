#!/usr/bin/env bash
# Stage 2 — finetune on the anomaly-understanding datasets (3-stream concat,
# unified language loss; L_sim/L_dis keep streams separated).
# Requires FULL env + Stage-1 checkpoint + anomaly datasets (added LAST).
#
#   LLAMA_MODEL  : llama-2-7b-chat-hf path
#   STAGE1_CKPT  : checkpoint from Stage 1 (set in cfg `ckpt:`)
#   DATA_ROOT    : anomaly datasets root
#   ABLATION     : optional name from experiments/configs/ablation.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${ENV_NAME:-cerberus}"
CFG="${CFG:-configs/train_configs/stage2_finetune.yaml}"
NPROC="${NPROC:-2}"
GPUS="${GPUS:-0,1}"
ABLATION="${ABLATION:-cerberus_full}"

echo "[run_stage2] env=${ENV_NAME} cfg=${CFG} ablation=${ABLATION} gpus=${GPUS}"
echo "[run_stage2] llama_model + prompt_path already wired in ${CFG} (local)."
echo "[run_stage2] REMAINING: set ckpt: (stage-1 output) and the anomaly dataset paths in ${CFG}."
echo "[run_stage2] NOTE: ablation switches (background_branch/L_BL/L_dis) live in"
echo "             experiments/configs/ablation.yaml — wire them into the cfg/model."

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPUS}" \
  conda run -n "${ENV_NAME}" torchrun --nproc_per_node="${NPROC}" --master_port=12001 \
  train.py --cfg-path "${CFG}"
