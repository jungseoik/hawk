#!/usr/bin/env bash
# GPU 3장 환경 — 남은 arm 을 **GPU 1장씩 병렬**로 돌린다.
#
# 왜 이 방식인가
# --------------
# GPU 를 한 작업에 더 붙이면 epoch 당 샘플 수만 늘고 (`iters_per_epoch` 이 2500 으로
# 고정이므로) epoch 시간은 그대로다. 반면 arm 을 나눠 병렬로 돌리면 벽시계 시간이
# arm 개수만큼 줄어든다.
#
#   2 GPU × batch 2 = effective 4,  2500 iter × 4 = 10,000 샘플/epoch
#   1 GPU × batch 4 = effective 4,  2500 iter × 4 = 10,000 샘플/epoch   ← 동일
#
# 조건이 같으므로 **이미 완주한 arm 을 다시 돌릴 필요가 없다.**
#
#   [A] 2 GPU 순차 유지        남은 124 epoch × 1.30h = 6.7일
#   [B] 3 GPU 한 작업          arm 불일치로 전부 재실행 = 10.8일
#   [C] 이 스크립트            가장 긴 arm 40 × 1.88h  = 3.1일   ← 채택
#
# 주의
# ----
# 병목은 GPU 가 아니라 **컨테이너 메모리**다(cgroup `memory.max`). arm 3 개를 동시에
# 돌리면 DataLoader 워커가 3 배가 되므로 `*_1gpu.yaml` 은 `num_workers` 를 4 로 낮춰
# 두었다. 학습은 compute-bound 이므로(`data:` 지연 실측 0.0000) 속도 영향은 없다.
#
# OOM kill 이 나면 rank 가 죽고 NCCL watchdog 가 SIGABRT 를 던진다. 재개는 체크포인트에서
# 정상 동작하므로 진행분은 잃지 않는다 — 그래서 각 arm 을 재시도 루프로 감쌌다.
#
# 사용
# ----
#   bash scripts/run_arms_parallel.sh --check     # 상태만
#   bash scripts/run_arms_parallel.sh             # 실행 (남은 arm 자동 선택)
#   ARM_GPUS="0 1 2" bash scripts/run_arms_parallel.sh
set -u
cd "$(dirname "$0")/.." || exit 1

ARM_GPUS="${ARM_GPUS:-0 1 2}"
ARM_TARGET="${ARM_TARGET:-40}"
ARM_MAX_RETRY="${ARM_MAX_RETRY:-12}"
RUNS_ROOT="${CERBERUS_ROOT:-/home/work/seoik}/runs"
ALL_ARMS="flow zero random_mask duplicate flow_reinit"

completed() { local l="${RUNS_ROOT}/abl2_$1/train.log"; [ -f "$l" ] && grep -c 'Averaged stats' "$l" || echo 0; }

status() {
  for a in $ALL_ARMS; do
    n="$(completed "$a")"; m="  "; [ "$n" -ge "$ARM_TARGET" ] && m="✅"
    printf "  %s %-12s %2d/%d\n" "$m" "$a" "$n" "$ARM_TARGET"
  done
}

if [ "${1:-}" = "--check" ]; then status; exit 0; fi

RUNNING=$(pgrep -f 'train[.]py --cfg-path' | wc -l)
if [ "$RUNNING" -gt 0 ] && [ "${ARM_FORCE:-0}" != "1" ]; then
  echo "[parallel] ⚠ 학습이 이미 $RUNNING 개 실행 중입니다. ARM_FORCE=1 로 무시 가능."
  exit 1
fi

# 남은 arm 을 **남은 epoch 이 많은 순**으로 정렬한다. 긴 것을 먼저 붙여야 전체가 빨리 끝난다.
TODO=""
for a in $ALL_ARMS; do
  n="$(completed "$a")"
  [ "$n" -lt "$ARM_TARGET" ] && TODO="$TODO $((ARM_TARGET - n)):$a"
done
TODO=$(echo "$TODO" | tr ' ' '\n' | grep -v '^$' | sort -rn -t: | cut -d: -f2 | tr '\n' ' ')
[ -z "$TODO" ] && { echo "[parallel] 남은 arm 없음"; status; exit 0; }

echo "=============================================================="
echo "[parallel] 남은 arm: $TODO"
echo "[parallel] GPU: $ARM_GPUS · 목표 $ARM_TARGET epoch · 재시도 $ARM_MAX_RETRY"
echo "[parallel] CERBERUS_REPR_LOSS_WEIGHT=${CERBERUS_REPR_LOSS_WEIGHT:-0}"
echo "=============================================================="

export CERBERUS_REPR_LOSS_WEIGHT="${CERBERUS_REPR_LOSS_WEIGHT:-0}"

# arm 하나를 GPU 하나에 붙여 목표 epoch 까지 재시도하며 돌린다.
run_arm() {
  local arm="$1" gpu="$2" attempt=0
  local cfg="configs/train_configs/ablation/stage2_${arm}_1gpu.yaml"
  [ -f "$cfg" ] || { echo "[parallel] config 없음: $cfg"; return 1; }
  while :; do
    local n; n="$(completed "$arm")"
    [ "$n" -ge "$ARM_TARGET" ] && { echo "[parallel] $arm 완료 ($n)"; return 0; }
    [ "$attempt" -ge "$ARM_MAX_RETRY" ] && { echo "[parallel] $arm 재시도 소진 ($n/$ARM_TARGET)"; return 1; }
    attempt=$((attempt + 1))
    echo "[parallel] $arm (GPU $gpu) 시도 $attempt  현재 $n/$ARM_TARGET  $(date '+%F %T')"
    MASTER_PORT=$((12000 + gpu * 37)) \
      bash scripts/train_run.sh "$cfg" "abl2_${arm}" "$gpu" 1 > "${RUNS_ROOT}/abl2_${arm}/parallel_${gpu}.out" 2>&1
  done
}

i=0
PIDS=""
for arm in $TODO; do
  gpu=$(echo "$ARM_GPUS" | cut -d' ' -f$((i + 1)))
  [ -z "$gpu" ] && { echo "[parallel] GPU 부족 — $arm 은 이번 웨이브에서 제외"; continue; }
  mkdir -p "${RUNS_ROOT}/abl2_${arm}"
  run_arm "$arm" "$gpu" &
  PIDS="$PIDS $!"
  i=$((i + 1))
done

echo "[parallel] ${i}개 arm 병렬 시작. 완료 대기…"
for p in $PIDS; do wait "$p"; done

echo "=============================================================="
echo "[parallel] 웨이브 종료  $(date '+%F %T')"
status
echo "남은 arm 이 있으면 같은 명령을 다시 실행하십시오 (완료된 것은 건너뜁니다)."
echo "=============================================================="
