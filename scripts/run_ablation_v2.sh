#!/usr/bin/env bash
# 절제 실험 재실행 (v2) — 표현 손실을 끄고 다섯 arm 을 처음부터.
#
# 왜 재실행인가
# -------------
# v1 은 다음 세 가지로 실패했다.
#   1. `random_mask` 가 epoch 5 에서 DataLoader worker OOM kill (러너가 감지 못함)
#   2. `duplicate` 가 epoch 16 에서 발산 — 파라미터 193/231 이 비유한, 회복 불가
#   3. `flow` 는 완주했으나 epoch 28 에 표현 손실이 정류점을 이탈해 마지막 12 epoch 을
#      다른 arm 과 **다른 실효 목적함수**로 학습
#
# 셋 다 원인이 같다. `L_sim`·`L_dis` 가 정류점에 갇혀 있다가 장기 학습에서 이탈하고,
# 그 시점이 arm 마다 다르며, 어떤 arm 에서는 학습을 죽인다. 논문은 두 항을 기여로
# 주장하지 않으므로(Appendix A) 0 으로 두는 데 잃는 것이 없다.
#
#   CERBERUS_REPR_LOSS_WEIGHT=0  → 두 항이 기울기를 주지 않는다
#
# v1 결과는 지우지 않는다. `runs/abl_*` 는 그대로 두고 v2 는 `runs/abl2_*` 에 쓴다 —
# 이탈 현상 자체가 Appendix A.1 의 근거이므로 원자료가 남아야 한다.
set -u
cd "$(dirname "$0")/.." || exit 1

ABL_CHECK_ONLY=0
[ "${1:-}" = "--check" ] && ABL_CHECK_ONLY=1

ABL_GPUS="${ABL_GPUS:-0,1}"
ABL_NPROC="${ABL_NPROC:-2}"
ABL_ARMS="${ABL_ARMS-flow zero random_mask duplicate flow_reinit}"
ABL_TARGET_EPOCH="${ABL_TARGET_EPOCH:-40}"
ABL_MAX_RETRY="${ABL_MAX_RETRY:-3}"
RUNS_ROOT="${CERBERUS_ROOT:-/home/work/seoik}/runs"
export CERBERUS_REPR_LOSS_WEIGHT="${CERBERUS_REPR_LOSS_WEIGHT:-0}"

completed_epochs() {
  local log="${RUNS_ROOT}/abl2_$1/train.log"
  [ -f "$log" ] && grep -c 'Averaged stats' "$log" || echo 0
}

# 발산 감지 — 최근 출력 iteration 중 NaN 비율이 임계를 넘으면 재시도해도 소용없다.
diverged() {
  local log="${RUNS_ROOT}/abl2_$1/train.log"
  [ -f "$log" ] || return 1
  local tail_lines nan
  tail_lines=$(grep 'Train: data epoch' "$log" | tail -100)
  [ -z "$tail_lines" ] && return 1
  nan=$(echo "$tail_lines" | grep -c 'totalloss: nan')
  [ "$nan" -ge 50 ]
}

# GPU 를 쓰는 다른 작업이 있으면 시작하지 않는다. 학습뿐 아니라 **평가**도 본다 —
# 평가가 GPU 를 점유한 상태에서 학습을 띄우면 둘 다 OOM 으로 죽을 수 있다.
#
# `pgrep -c` 는 매치가 없을 때 "0" 을 출력하면서 exit 1 도 반환하므로 `|| echo 0` 을
# 붙이면 값이 "0\n0" 이 되어 정수 비교가 깨진다. `wc -l` 로 세는 편이 안전하다.
RUNNING=$(( $(pgrep -f 'train[.]py --cfg-path' | wc -l) + $(pgrep -f 'scripts/evaluate[.]py' | wc -l) ))
if [ "$RUNNING" -gt 0 ] && [ "$ABL_CHECK_ONLY" -eq 0 ] && [ "${ABL_FORCE:-0}" != "1" ]; then
  echo "[v2] ⚠ GPU 를 쓰는 작업이 이미 $RUNNING 개 실행 중입니다 (학습 또는 평가)."
  echo "     동시 실행하면 양쪽이 OOM 으로 죽을 수 있습니다. ABL_FORCE=1 로 무시 가능."
  exit 1
fi

echo "=============================================================="
echo "[v2] arms: ${ABL_ARMS:-<없음>}"
echo "[v2] CERBERUS_REPR_LOSS_WEIGHT = $CERBERUS_REPR_LOSS_WEIGHT"
echo "[v2] 목표 $ABL_TARGET_EPOCH epoch · 재시도 상한 $ABL_MAX_RETRY · GPU $ABL_GPUS"
echo "=============================================================="

if [ "$ABL_CHECK_ONLY" -eq 1 ]; then
  for arm in flow zero random_mask duplicate flow_reinit; do
    n="$(completed_epochs "$arm")"; mark="  "
    [ "$n" -ge "$ABL_TARGET_EPOCH" ] && mark="✅"
    d=""; diverged "$arm" && d="  ⚠발산"
    printf "  %s %-12s %2d/%d%s\n" "$mark" "$arm" "$n" "$ABL_TARGET_EPOCH" "$d"
  done
  exit 0
fi

for arm in $ABL_ARMS; do
  cfg="configs/train_configs/ablation/stage2_${arm}.yaml"
  [ -f "$cfg" ] || { echo "[v2] config 없음: $cfg — 건너뜁니다"; continue; }

  attempt=0
  while :; do
    done_ep="$(completed_epochs "$arm")"
    [ "$done_ep" -ge "$ABL_TARGET_EPOCH" ] && { echo "[v2] $arm 완료 ($done_ep) — 건너뜀"; break; }
    if diverged "$arm"; then
      echo "[v2] ⚠ $arm 발산 감지 ($done_ep epoch). 재시도해도 회복되지 않으므로 중단합니다."
      echo "     로그: ${RUNS_ROOT}/abl2_${arm}/train.log"
      break
    fi
    [ "$attempt" -ge "$ABL_MAX_RETRY" ] && { echo "[v2] $arm 재시도 소진 ($done_ep/$ABL_TARGET_EPOCH)"; break; }

    attempt=$((attempt + 1))
    echo "--------------------------------------------------------------"
    echo "[v2] $arm 시도 $attempt/$ABL_MAX_RETRY (현재 $done_ep/$ABL_TARGET_EPOCH)  $(date '+%F %T')"
    MASTER_PORT=$((11300 + $(echo "$arm" | cksum | cut -d' ' -f1) % 500)) \
      bash scripts/train_run.sh "$cfg" "abl2_${arm}" "$ABL_GPUS" "$ABL_NPROC"
    echo "[v2] $arm 종료 (exit $?)  epoch $done_ep → $(completed_epochs "$arm")"
  done
done

echo "=============================================================="
echo "[v2] 최종  $(date '+%F %T')"
for arm in flow zero random_mask duplicate flow_reinit; do
  n="$(completed_epochs "$arm")"; mark="  "
  [ "$n" -ge "$ABL_TARGET_EPOCH" ] && mark="✅"
  d=""; diverged "$arm" && d="  ⚠발산"
  printf "  %s %-12s %2d/%d%s\n" "$mark" "$arm" "$n" "$ABL_TARGET_EPOCH" "$d"
done
