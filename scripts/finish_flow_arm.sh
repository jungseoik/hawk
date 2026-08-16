#!/usr/bin/env bash
# abl2_flow 를 40 epoch 까지 마저 채운다.
#
# 왜 별도로 도는가: v2 러너가 재시도 상한 3 을 소진해 flow 를 15/40 에서 두고 다음 arm 으로
# 넘어갔다(원인은 컨테이너 240GB 제한에 의한 DataLoader worker OOM kill — 재개는 정상 작동).
# 러너의 arm 목록은 이미 flow 를 지났으므로 돌아오지 않는다.
#
# **모든 학습이 끝난 뒤**에 시작한다. 동시에 돌리면 이미 병목인 메모리를 더 다투게 되고,
# v2 러너의 가드가 정당하게 거부한다.
set -u
cd "$(dirname "$0")/.." || exit 1
echo "[finish-flow] 전체 학습 종료 대기  $(date '+%F %T')"
while pgrep -f "train[.]py --cfg-path" > /dev/null || pgrep -f "run_ablation_v[2]" > /dev/null; do
  sleep 600
done
echo "[finish-flow] 시작  $(date '+%F %T')"
ABL_ARMS="flow" ABL_MAX_RETRY=12 exec bash scripts/run_ablation_v2.sh
