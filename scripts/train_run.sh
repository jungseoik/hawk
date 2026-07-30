#!/usr/bin/env bash
# =============================================================================
# Paper-grade training launcher — captures full provenance so any loss/metric
# can be traced back later (for ablation tables & the paper).
#
# Creates a per-run directory and records: git commit + dirty diff, the exact
# config, GPU/env info, and the full training stdout/stderr log. TensorBoard and
# checkpoints go INSIDE this run dir (via --options run.output_dir=...), so each
# run/ablation is fully self-contained and comparable.
#
# Usage:
#   bash scripts/train_run.sh <cfg> [run_name] [gpus] [nproc]
#   bash scripts/train_run.sh configs/train_configs/stage1_main.yaml stage1_main 0,1 2
# Resume: pass a config whose run.resume_ckpt_path points at a checkpoint, OR
#   append --options at call sites; the launcher forwards none by default.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

CFG="${1:?usage: train_run.sh <cfg> [run_name] [gpus] [nproc]}"
NAME="${2:-$(basename "${CFG%.yaml}")}"
GPUS="${3:-0,1}"
NPROC="${4:-$(echo "$GPUS" | tr ',' '\n' | grep -c .)}"
ENV_NAME="${ENV_NAME:-cerberus}"
RUNS_ROOT="${RUNS_ROOT:-/data/pia/runs}"

TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUNS_ROOT}/${NAME}-${TS}"
mkdir -p "$RUN_DIR"

# ---- provenance ----
{
  echo "run_name   : $NAME"
  echo "timestamp  : $TS"
  echo "config     : $CFG"
  echo "gpus       : $GPUS   nproc: $NPROC"
  echo "git_commit : $(git rev-parse HEAD)"
  echo "git_branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git_dirty  : $(git status --porcelain | wc -l) changed files"
} > "$RUN_DIR/run_info.txt"
git status --porcelain > "$RUN_DIR/git_status.txt" || true
git diff > "$RUN_DIR/git_diff.patch" || true          # exact uncommitted state
cp "$CFG" "$RUN_DIR/config.yaml"
conda run -n "$ENV_NAME" python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" >> "$RUN_DIR/run_info.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv >> "$RUN_DIR/run_info.txt" 2>/dev/null || true

echo "[train_run] RUN_DIR=$RUN_DIR"
echo "[train_run] logging to $RUN_DIR/train.log  (tail -f to watch)"
echo "[train_run] TensorBoard/checkpoints -> $RUN_DIR/<job_id>/"

# ---- launch (checkpoints + TB land under RUN_DIR via output_dir override) ----
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 \
  conda run -n "$ENV_NAME" torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-10000}" \
  train.py --cfg-path "$CFG" --options run.output_dir="$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/train.log"
