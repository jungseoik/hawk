#!/usr/bin/env bash
# 절제 실험 후속 러너 — 중단된 arm 재개 + 남은 arm 실행.
#
# 왜 별도 스크립트인가
# --------------------
# `run_ablation_arms.sh` 는 arm 을 순차 실행하지만 **실패를 감지하지 않는다.**
# random_mask arm 이 epoch 5 에서 DataLoader worker 가 SIGKILL 로 죽어 exit 1 로
# 끝났는데, 러너는 그대로 다음 arm 으로 넘어갔다. 그 결과 40 epoch 예산으로 비교해야
# 할 arm 하나가 5 epoch 짜리로 남았다.
#
# 그리고 실행 중인 bash 스크립트는 편집하면 안 된다 — bash 는 스크립트를 바이트
# 오프셋으로 읽어 나가므로, 진행 중인 파일을 고치면 남은 부분을 엉뚱하게 해석한다.
# 그래서 원본을 두고 이 스크립트를 따로 둔다.
#
# 이 스크립트가 다르게 하는 것
# ----------------------------
#  1. arm 이 끝날 때마다 **완료 epoch 수를 확인**하고, 목표에 못 미치면 재시도한다.
#  2. 재시도는 `train_run.sh` 의 자동 재개를 쓴다 — 최신 체크포인트에서 이어가므로
#     처음부터 다시 돌지 않는다(체크포인트에 optimizer·scaler·epoch 이 모두 있다).
#  3. 재시도 상한을 두어 같은 지점에서 무한히 죽는 경우를 막는다.
#
# 사용
# ----
#   bash scripts/run_ablation_followup.sh --check            # 상태만 출력 (학습 안 함)
#   bash scripts/run_ablation_followup.sh                     # 기본: random_mask flow_reinit
#   ABL_ARMS="zero" bash scripts/run_ablation_followup.sh      # 특정 arm 만
#
# ⚠ `--check` 없이 실행하면 **즉시 학습이 시작된다.** 다른 arm 이 GPU 를 쓰고 있는지
#   먼저 확인하십시오 — 동시 실행하면 서로 메모리를 다투다 죽는다. 실제로 검증하려다
#   ABL_ARMS="" 로 실행해 학습을 띄운 적이 있다(빈 문자열은 `:-` 로 기본값이 된다).
#
# 변수 이름에 반드시 접두사를 붙인다. 이 컨테이너에는 `NPROC=42` 가 미리 심어져 있어
# 흔한 이름을 쓰면 torchrun 이 GPU 42 장을 요구하고 즉사한다.
set -u

cd "$(dirname "$0")/.." || exit 1

# --check 는 상태만 보고 종료한다. 검증 목적으로 실행할 때 학습이 시작되지 않게 하는
# 유일하게 확실한 방법이다.
ABL_CHECK_ONLY=0
[ "${1:-}" = "--check" ] && ABL_CHECK_ONLY=1

ABL_GPUS="${ABL_GPUS:-0,1}"
ABL_NPROC="${ABL_NPROC:-2}"
# `:-` 가 아니라 `-` 를 쓴다. `:-` 는 **빈 문자열도** 기본값으로 치환하므로,
# ABL_ARMS="" 로 "아무 arm 도 돌리지 말라"고 지시해도 기본 arm 이 실행된다.
ABL_ARMS="${ABL_ARMS-random_mask flow_reinit}"
ABL_TARGET_EPOCH="${ABL_TARGET_EPOCH:-40}"
ABL_MAX_RETRY="${ABL_MAX_RETRY:-3}"
RUNS_ROOT="${CERBERUS_ROOT:-/home/work/seoik}/runs"

completed_epochs() {   # $1 = arm 이름
  local log="${RUNS_ROOT}/abl_$1/train.log"
  [ -f "$log" ] && grep -c 'Averaged stats' "$log" || echo 0
}

# 다른 학습이 이미 GPU 를 쓰고 있으면 동시 실행은 둘 다 위험하게 만든다.
RUNNING="$(pgrep -fc 'train.py --cfg-path.*ablation' 2>/dev/null || echo 0)"
if [ "$RUNNING" -gt 0 ] && [ "$ABL_CHECK_ONLY" -eq 0 ]; then
  echo "[followup] ⚠ 절제 학습 프로세스가 이미 $RUNNING 개 실행 중입니다."
  echo "[followup]   동시 실행하면 GPU 메모리를 다투다 양쪽이 죽을 수 있습니다."
  echo "[followup]   진행하려면 ABL_FORCE=1 을 지정하십시오."
  [ "${ABL_FORCE:-0}" != "1" ] && exit 1
fi

echo "=============================================================="
echo "[followup] arms: ${ABL_ARMS:-<없음>}"
echo "[followup] 목표 $ABL_TARGET_EPOCH epoch · 재시도 상한 $ABL_MAX_RETRY 회"
echo "[followup] GPU $ABL_GPUS (nproc $ABL_NPROC)"
echo "=============================================================="

if [ "$ABL_CHECK_ONLY" -eq 1 ]; then
  echo "[followup] --check: 상태만 출력하고 종료합니다 (학습 없음)"
  for arm in flow random_mask duplicate zero flow_reinit; do
    n="$(completed_epochs "$arm")"
    mark="  "; [ "$n" -ge "$ABL_TARGET_EPOCH" ] && mark="✅"
    printf "  %s %-12s %2d/%d\n" "$mark" "$arm" "$n" "$ABL_TARGET_EPOCH"
  done
  exit 0
fi

for arm in $ABL_ARMS; do
  cfg="configs/train_configs/ablation/stage2_${arm}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "[followup] config 없음: $cfg — 건너뜁니다"
    continue
  fi

  attempt=0
  while :; do
    done_ep="$(completed_epochs "$arm")"
    if [ "$done_ep" -ge "$ABL_TARGET_EPOCH" ]; then
      echo "[followup] arm=$arm 이미 $done_ep/$ABL_TARGET_EPOCH — 건너뜁니다"
      break
    fi
    if [ "$attempt" -ge "$ABL_MAX_RETRY" ]; then
      echo "[followup] arm=$arm 재시도 $ABL_MAX_RETRY 회 소진 ($done_ep/$ABL_TARGET_EPOCH)."
      echo "[followup] 같은 지점에서 반복 실패하는 중일 수 있습니다 — 로그를 확인하십시오:"
      echo "           ${RUNS_ROOT}/abl_${arm}/train.log"
      break
    fi

    attempt=$((attempt + 1))
    echo "--------------------------------------------------------------"
    echo "[followup] arm=$arm 시도 $attempt/$ABL_MAX_RETRY  (현재 $done_ep/$ABL_TARGET_EPOCH)"
    echo "[followup] $(date '+%F %T')"
    echo "--------------------------------------------------------------"

    # master port 를 arm 이름에서 결정론적으로 뽑아 동시 실행 충돌을 피한다.
    MASTER_PORT=$((10700 + $(echo "$arm" | cksum | cut -d' ' -f1) % 500)) \
      bash scripts/train_run.sh "$cfg" "abl_${arm}" "$ABL_GPUS" "$ABL_NPROC"
    rc=$?

    after="$(completed_epochs "$arm")"
    echo "[followup] arm=$arm 종료 (exit $rc)  epoch $done_ep → $after"

    # 진전이 전혀 없으면 재시도해도 같은 자리에서 죽을 가능성이 높다.
    if [ "$rc" -ne 0 ] && [ "$after" -le "$done_ep" ]; then
      echo "[followup] ⚠ 진전 없이 실패했습니다. 재개가 같은 지점에서 막히는지 확인이 필요합니다."
    fi
  done
done

echo "=============================================================="
echo "[followup] 최종 상태  $(date '+%F %T')"
for arm in flow random_mask duplicate zero flow_reinit; do
  n="$(completed_epochs "$arm")"
  mark="  "; [ "$n" -ge "$ABL_TARGET_EPOCH" ] && mark="✅"
  printf "  %s %-12s %2d/%d\n" "$mark" "$arm" "$n" "$ABL_TARGET_EPOCH"
done
echo "=============================================================="
