#!/usr/bin/env bash
# =============================================================================
# 정적 스트림 통제군 4-arm 순차 실행
#
# 네 arm 은 아키텍처·파라미터 수·시각 토큰 수가 모두 같고, 정적 스트림에 들어가는
# *내용* 만 다르다. 따라서 성능 차이는 용량이 아니라 내용에 귀속된다.
#
#   flow        정적 = (1 − M) ⊙ x        ← 제안 방식
#   random_mask M 을 블록 단위로 뒤섞음      ← 면적은 같고 위치는 무의미
#   duplicate   정적 = 원본 프레임 전체      ← 분해 없이 스트림만 추가
#   zero        정적 = 0                   ← 용량만 있고 내용 없음
#
# 실행 순서가 중요하다: flow 와 random_mask 가 논문의 생사를 가르는 비교이므로
# 먼저 돌린다. 중간에 끊겨도 그 둘의 결과는 확보된다.
#
# 각 arm 은 max_epoch 40 (= 400k 샘플, 약 1.6일). 전체 약 6.4일.
# train_run.sh 가 arm 별로 자동 재개하므로, 끊긴 뒤 같은 명령을 다시 실행하면 이어진다.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
ARMS="${ARMS:-flow random_mask duplicate zero}"

for arm in $ARMS; do
  cfg="configs/train_configs/ablation/stage2_${arm}.yaml"
  [ -f "$cfg" ] || { echo "[ablation] config 없음: $cfg"; exit 1; }

  echo "=============================================================="
  echo "[ablation] arm=${arm} 시작  $(date '+%F %T')"
  echo "=============================================================="

  MASTER_PORT=$((10200 + $(echo "$arm" | cksum | cut -d' ' -f1) % 500)) \
    bash scripts/train_run.sh "$cfg" "abl_${arm}" "$GPUS" "$NPROC"

  echo "[ablation] arm=${arm} 종료 (exit $?)  $(date '+%F %T')"
done

echo "[ablation] 전체 완료 $(date '+%F %T')"
