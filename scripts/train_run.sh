#!/usr/bin/env bash
# =============================================================================
# Chunked-training launcher with provenance + auto-resume.
#
# Designed for stop-and-continue training (interrupt anytime for other work,
# relaunch to pick up where you left off). One STABLE run dir per logical
# experiment; checkpoints + TensorBoard accumulate there across all chunks, and
# each relaunch auto-resumes from the latest checkpoint. TB curves stay a single
# continuous line (global step = epoch*iters + i).
#
# Usage:
#   bash scripts/train_run.sh <cfg> <run_name> [gpus] [nproc]
#   bash scripts/train_run.sh configs/train_configs/stage1_main.yaml core 0,1 2
#   # ...train a while, Ctrl-C / kill for other work, then LATER just rerun the
#   #    SAME command -> it auto-resumes from the latest checkpoint.
#
# Env: RUNS_ROOT (default /data/pia/runs), ENV_NAME (default cerberus),
#      MASTER_PORT (default 10000), FRESH=1 to ignore existing checkpoints.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: train_run.sh <cfg> <run_name> [gpus] [nproc]}"
NAME="${2:?provide a STABLE run_name (reused across chunks), e.g. 'core'}"
GPUS="${3:-0,1}"
NPROC="${4:-$(echo "$GPUS" | tr ',' '\n' | grep -c .)}"
ENV_NAME="${ENV_NAME:-cerberus}"
RUNS_ROOT="${RUNS_ROOT:-/data/pia/runs}"

RUN_DIR="${RUNS_ROOT}/${NAME}"
JOB_ID="main"                       # fixed => checkpoints/TB land in RUN_DIR/main across chunks
OUT_JOB="${RUN_DIR}/${JOB_ID}"
mkdir -p "$OUT_JOB"

# ---- auto-resume: newest checkpoint in this run dir ----
RESUME=""
if [ "${FRESH:-0}" != "1" ]; then
  LATEST="$(ls -1 "${OUT_JOB}"/checkpoint_*.pth 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$LATEST" ] && RESUME="$LATEST"
fi

# ---- provenance (appended per chunk) ----
TS="$(date +%Y%m%d-%H%M%S)"
{
  echo "==== chunk @ $TS ===="
  echo "config     : $CFG"
  echo "gpus       : $GPUS  nproc: $NPROC"
  echo "git_commit : $(git rev-parse HEAD)   dirty: $(git status --porcelain | wc -l)"
  echo "resume_from: ${RESUME:-<fresh start>}"
} >> "$RUN_DIR/run_info.txt"
cp "$CFG" "$RUN_DIR/config.yaml"
git diff > "$RUN_DIR/git_diff_${TS}.patch" 2>/dev/null || true

echo "[train_run] RUN_DIR=$RUN_DIR  (job=$JOB_ID)"
echo "[train_run] resume: ${RESUME:-FRESH}"
echo "[train_run] log -> $RUN_DIR/train.log   | TB -> $OUT_JOB/tensorboard"

OPTS=(run.output_dir="$RUN_DIR")
[ -n "$RESUME" ] && OPTS+=(run.resume_ckpt_path="$RESUME")

# --no-capture-output: stream stdout/stderr live (plain `conda run` buffers the pipe,
# leaving train.log empty until exit).
HAWK_JOB_ID="$JOB_ID" NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n "$ENV_NAME" torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-10000}" \
  train.py --cfg-path "$CFG" --options "${OPTS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
