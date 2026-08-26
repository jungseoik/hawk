#!/usr/bin/env bash
# 2차 평가 — dual-branch 대조군과 예산 정합 비교.
#
#   no_static  : 우리가 직접 돌린 진짜 dual-branch (같은 stage-1, 40 epoch) → 인과 귀속
#   hawk_official : HAWK 공개 체크포인트 (stage-2 11 epoch)               → published baseline
#   flow_ep10  : 우리 flow 를 epoch 10 에서 끊은 것 — HAWK 공개본과 **학습량 동일**
#
# 3-브랜치/2-브랜치는 체크포인트 가중치에서 자동 판정한다(`detect_use_background`).
set -u
cd "$(dirname "$0")/.." || exit 1
R="${CERBERUS_ROOT:-/home/work/seoik}"
PY="${CERBERUS_PY:-$R/miniconda3/envs/cerberus/bin/python}"
ANNO="$R/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json"
mkdir -p experiments/out logs

eval_one() {
  local tag="$1" ckpt="$2" gpu="$3"
  local out="experiments/out/eval_${tag}.json"
  [ -s "$out" ] && { echo "[eval2] $tag 이미 완료 — 건너뜀"; return 0; }
  echo "[eval2] $tag (GPU $gpu) 시작 $(date '+%F %T')"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" scripts/evaluate.py \
    --ckpt "$ckpt" --anno "$ANNO" --gpu-id "$gpu" --out "$out" \
    > "logs/eval_${tag}.log" 2>&1
  echo "[eval2] $tag 종료 rc=$? $(date '+%F %T')"
}

{ eval_one abl2_no_static "$R/runs/abl2_no_static/main/checkpoint_39.pth" 0
  eval_one abl2_flow_ep10 "$R/runs/abl2_flow/main/checkpoint_10.pth"      0; } &
P0=$!
{ eval_one hawk_official  "$R/hawk_official/finetuned.pth"                1; } &
P1=$!
wait $P0 $P1
echo "[eval2] 전부 종료 $(date '+%F %T')"
ls -la experiments/out/eval_abl2_no_static.json experiments/out/eval_abl2_flow_ep10.json experiments/out/eval_hawk_official.json 2>/dev/null
