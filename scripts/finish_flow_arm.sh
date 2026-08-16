#!/usr/bin/env bash
# abl2_flow 를 40 epoch 까지 마저 채운다 (v2 러너가 재시도 상한 3 을 소진해 15 에서 멈춤).
# 현재 zero arm 이 도는 중이므로 그것이 끝난 뒤 실행되도록 대기한다.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "[finish-flow] zero arm 종료 대기  $(date '+%F %T')"
while pgrep -f "train[.]py --cfg-path.*stage2_zero" > /dev/null; do sleep 300; done
echo "[finish-flow] 시작  $(date '+%F %T')"
ABL_ARMS="flow" ABL_MAX_RETRY=12 exec bash scripts/run_ablation_v2.sh
