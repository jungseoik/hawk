#!/usr/bin/env bash
# L_BL 복원 실험 — 배경 분기에 전용 감독을 주면 상수 붕괴가 풀리는가.
# arm 2개(flow/zero)를 GPU 1장씩, 12 epoch. flow > zero 가 나오면 CVD 가 살아난다.
set -u
cd "$(dirname "$0")/.." || exit 1
R="${CERBERUS_ROOT:-/home/work/seoik}"; T="${BL_TARGET:-12}"; MR="${BL_MAX_RETRY:-8}"
done_ep(){ local l="$R/runs/bl_$1/train.log"; { [ -f "$l" ] && grep -c 'Averaged stats' "$l"; } | head -1; }
[ "${1:-}" = "--check" ] && { for a in flow zero; do printf "  bl_%-6s %2d/%d\n" "$a" "$(done_ep $a)" "$T"; done; exit 0; }
RUN=$(pgrep -f 'train[.]py --cfg-path' | wc -l)
[ "$RUN" -gt 0 ] && [ "${BL_FORCE:-0}" != "1" ] && { echo "[bl] 학습 $RUN 개 실행 중"; exit 1; }
export CERBERUS_REPR_LOSS_WEIGHT=0
one(){ local a="$1" g="$2" t=0
  mkdir -p "$R/runs/bl_$a"
  while :; do
    n="$(done_ep $a)"; [ "${n:-0}" -ge "$T" ] && { echo "[bl] $a 완료 ($n)"; return 0; }
    [ "$t" -ge "$MR" ] && { echo "[bl] $a 재시도 소진"; return 1; }
    t=$((t+1)); echo "[bl] $a (GPU $g) 시도 $t  현재 ${n:-0}/$T  $(date '+%F %T')"
    MASTER_PORT=$((13000+g*41)) bash scripts/train_run.sh \
      "configs/train_configs/ablation/stage2_${a}_bl.yaml" "bl_${a}" "$g" 1 \
      >> "$R/runs/bl_${a}/runner.out" 2>&1
  done; }
one flow 0 & P0=$!
one zero 1 & P1=$!
wait $P0 $P1; echo "[bl] 종료 $(date '+%F %T')"
