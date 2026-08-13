#!/usr/bin/env bash
# flow arm 의 이탈 전/후 체크포인트를 **순차** 평가한 뒤 v2 절제 재실행으로 넘어간다.
# 순차인 이유: 두 평가를 동시에 돌리면 디코딩 중 호스트 RAM 을 함께 잡아 위험하다.
set -u
cd "$(dirname "$0")/.." || exit 1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="${CERBERUS_PY:-/home/work/seoik/miniconda3/envs/cerberus/bin/python}"
ANNO="${CERBERUS_ROOT:-/home/work/seoik}/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json"
for e in 27 39; do
  out="experiments/out/eval_flow_ep${e}.json"
  if [ -f "$out" ]; then echo "[evals] ep$e 이미 있음 — 건너뜀"; continue; fi
  echo "[evals] flow ckpt_$e 평가 시작  $(date '+%F %T')"
  "$PY" scripts/evaluate.py \
    --ckpt "${CERBERUS_ROOT:-/home/work/seoik}/runs/abl_flow/main/checkpoint_${e}.pth" \
    --anno "$ANNO" --out "$out" --gpu-id 0
  echo "[evals] ckpt_$e 종료 (exit $?)  $(date '+%F %T')"
done
echo "[evals] 평가 종료. v2 절제 재실행 시작  $(date '+%F %T')"
exec bash scripts/run_ablation_v2.sh
