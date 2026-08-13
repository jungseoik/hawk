#!/usr/bin/env bash
# 평가가 끝나면 자동으로 v2 절제 재실행을 시작한다.
# 사람이 지켜보지 않아도 GPU 가 노는 시간이 없도록 잇는 용도.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "[chain] 평가 종료 대기  $(date '+%F %T')"
while [ "$(pgrep -f 'scripts/evaluate[.]py' | wc -l)" -gt 0 ]; do sleep 60; done
echo "[chain] 평가 종료 확인. 60초 후 v2 시작  $(date '+%F %T')"
sleep 60
exec bash scripts/run_ablation_v2.sh
