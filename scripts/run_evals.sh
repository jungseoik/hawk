#!/usr/bin/env bash
# 5개 arm 을 GPU 2장에 나눠 평가한다. arm 별 정적 입력 모드는 run 의 config.yaml 에서 자동 검출.
#
# ⚠ CUBLAS_WORKSPACE_CONFIG=:4096:8 은 필수다 — 없으면 같은 체크포인트가 실행마다 다른
#   문장을 낸다(실측 3/3 불일치, 하나는 `4 4 4 4 …` 로 퇴화).
set -u
cd "$(dirname "$0")/.." || exit 1
R="${CERBERUS_ROOT:-/home/work/seoik}"
PY="${CERBERUS_PY:-$R/miniconda3/envs/cerberus/bin/python}"
ANNO="$R/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json"
mkdir -p experiments/out logs

eval_one() {
  local arm="$1" gpu="$2"
  local out="experiments/out/eval_abl2_${arm}.json"
  [ -s "$out" ] && { echo "[eval] $arm 이미 완료 — 건너뜀"; return 0; }
  echo "[eval] $arm (GPU $gpu) 시작 $(date '+%F %T')"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" scripts/evaluate.py \
    --ckpt "$R/runs/abl2_${arm}/main/checkpoint_39.pth" \
    --anno "$ANNO" --gpu-id "$gpu" --out "$out" \
    > "logs/eval_abl2_${arm}.log" 2>&1
  echo "[eval] $arm 종료 rc=$? $(date '+%F %T')"
}

# GPU 0 과 1 에 번갈아 배정하고, 각 GPU 안에서는 순차 실행한다.
{ for a in flow random_mask flow_reinit; do eval_one "$a" 0; done; } &
P0=$!
{ for a in zero duplicate;              do eval_one "$a" 1; done; } &
P1=$!
wait $P0 $P1
echo "[eval] 전부 종료 $(date '+%F %T')"
ls -la experiments/out/eval_abl2_*.json
