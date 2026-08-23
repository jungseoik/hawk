#!/usr/bin/env bash
# dual-branch 대조군(`no_static`) 학습 — GPU 2장, 40 epoch.
#
# 다른 arm 과 effective batch 를 맞춘다: 2 GPU x batch 2 = 4 (다른 arm 은 1 GPU x batch 4).
# 표현 손실 가중치는 전 arm 과 동일하게 0.
#
#   bash scripts/run_no_static.sh --check
#   bash scripts/run_no_static.sh
set -u
cd "$(dirname "$0")/.." || exit 1
R="${CERBERUS_ROOT:-/home/work/seoik}"
ARM=no_static
TARGET="${NS_TARGET:-40}"
MAX_RETRY="${NS_MAX_RETRY:-12}"
LOG="$R/runs/abl2_${ARM}/train.log"

done_epochs() { [ -f "$LOG" ] && grep -c 'Averaged stats' "$LOG" || echo 0; }

if [ "${1:-}" = "--check" ]; then
  printf "  %-12s %2d/%d\n" "$ARM" "$(done_epochs)" "$TARGET"; exit 0
fi

RUNNING=$(pgrep -f 'train[.]py --cfg-path' | wc -l)
if [ "$RUNNING" -gt 0 ] && [ "${NS_FORCE:-0}" != "1" ]; then
  echo "[no_static] ⚠ 학습이 이미 $RUNNING 개 실행 중입니다. NS_FORCE=1 로 무시 가능."; exit 1
fi

export CERBERUS_REPR_LOSS_WEIGHT=0
mkdir -p "$R/runs/abl2_${ARM}"
attempt=0
while :; do
  n="$(done_epochs)"
  [ "$n" -ge "$TARGET" ] && { echo "[no_static] 완료 ($n/$TARGET)"; break; }
  [ "$attempt" -ge "$MAX_RETRY" ] && { echo "[no_static] 재시도 소진 ($n/$TARGET)"; exit 1; }
  attempt=$((attempt + 1))
  echo "[no_static] 시도 $attempt  현재 $n/$TARGET  $(date '+%F %T')"
  bash scripts/train_run.sh configs/train_configs/ablation/stage2_no_static.yaml \
    "abl2_${ARM}" 0,1 2 >> "$R/runs/abl2_${ARM}/runner.out" 2>&1
done
echo "[no_static] 종료 $(date '+%F %T')"
