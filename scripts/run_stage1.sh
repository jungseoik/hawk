#!/usr/bin/env bash
# Stage 1 — pretrain on WebVid (3-stream, all 5 losses active).
# Requires the FULL env tier and the model weights + WebVid data (added LAST).
#
# Fill these before running (configs/train_configs/stage1_pretrain.yaml has the
# original author's absolute paths that do NOT exist on this machine):
#   LLAMA_MODEL  : path to llama-2-7b-chat-hf
#   DATA_ROOT    : WebVid videos + annotations root
#   NPROC        : #GPUs (default 2 on this Blackwell box)
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${ENV_NAME:-cerberus}"
CFG="${CFG:-configs/train_configs/stage1_pretrain.yaml}"
NPROC="${NPROC:-2}"
GPUS="${GPUS:-0,1}"

echo "[run_stage1] env=${ENV_NAME} cfg=${CFG} gpus=${GPUS} nproc=${NPROC}"
echo "[run_stage1] llama_model is already wired in ${CFG} (local weights)."
echo "[run_stage1] REMAINING: set WebVid paths in ${CFG} ->"
echo "             datasets.webvid.build_info.{anno_dir,videos_dir}"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPUS}" \
  conda run -n "${ENV_NAME}" torchrun --nproc_per_node="${NPROC}" --master_port=10000 \
  train.py --cfg-path "${CFG}"
