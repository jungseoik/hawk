#!/usr/bin/env bash
# 시드 재현 실행 — 핵심 세 arm 을 seed 43, 12 epoch 으로 다시 돌린다.
#
# 왜 12 epoch 인가: 잰 모든 arm·모든 지표에서 ep10 이 ep40 보다 좋았다(recall -0.091 등).
# 40 을 돌 이유가 없고, 그래서 시드 재현 비용이 예전의 1/3 이다.
# 왜 이 세 arm 인가: 논문의 핵심 대비가 flow vs duplicate(입력 선택)와
# 세 브랜치 vs 두 브랜치(no_static)이기 때문이다.
#
# 세 arm 모두 1 GPU x batch 4 = effective 4 로 통일한다(새 세 arm 끼리의 통제).
set -u
cd "$(dirname "$0")/.." || exit 1
R="${CERBERUS_ROOT:-/home/work/seoik}"; T="${S1_TARGET:-12}"; MR="${S1_MAX_RETRY:-8}"
ARMS="flow duplicate no_static"
done_ep(){ local l="$R/runs/s1_$1/train.log"; { [ -f "$l" ] && grep -c 'Averaged stats' "$l"; } | head -1; }
[ "${1:-}" = "--check" ] && { for a in $ARMS; do printf "  s1_%-10s %2d/%d\n" "$a" "$(done_ep $a)" "$T"; done; exit 0; }
RUN=$(pgrep -f 'train[.]py --cfg-path' | wc -l)
[ "$RUN" -gt 0 ] && [ "${S1_FORCE:-0}" != "1" ] && { echo "[s1] 학습 $RUN 개 실행 중"; exit 1; }
export CERBERUS_REPR_LOSS_WEIGHT=0
one(){ local a="$1" g="$2" t=0
  mkdir -p "$R/runs/s1_$a"
  while :; do
    n="$(done_ep $a)"; [ "${n:-0}" -ge "$T" ] && { echo "[s1] $a 완료 ($n)"; return 0; }
    [ "$t" -ge "$MR" ] && { echo "[s1] $a 재시도 소진"; return 1; }
    t=$((t+1)); echo "[s1] $a (GPU $g) 시도 $t  현재 ${n:-0}/$T  $(date '+%F %T')"
    MASTER_PORT=$((14000+g*53)) bash scripts/train_run.sh \
      "configs/train_configs/ablation/stage2_${a}_s1.yaml" "s1_${a}" "$g" 1 \
      >> "$R/runs/s1_${a}/runner.out" 2>&1
  done; }
one flow 0 & P0=$!
one duplicate 1 & P1=$!
wait $P0 $P1
one no_static 0 & P2=$!
wait $P2
echo "[s1] 종료 $(date '+%F %T')"
