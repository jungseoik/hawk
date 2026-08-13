# 현재 상태 — 세션이 끊겨도 여기부터 읽으면 이어갈 수 있다

**마지막 갱신 2026-08-13.** 긴 서술은 다른 문서에 있다. 여기는 *지금 무엇이 돌고 있고
다음이 무엇인가*만 둔다.

## 지금 돌고 있는 것

| 작업 | 위치 | 예상 |
|---|---|---|
| `flow` ckpt_27 평가 (이탈 전) | `experiments/out/eval_flow_ep27.json` | 2~3시간 |
| `flow` ckpt_39 평가 (이탈 후) | `experiments/out/eval_flow_ep39.json` | 2~3시간 |
| 자동 연결 | `scripts/chain_eval_then_v2.sh` | 평가 끝나면 v2 자동 시작 |

확인: `bash scripts/run_ablation_v2.sh --check`

## 다음 (자동)

`scripts/run_ablation_v2.sh` — 5 arm 재실행, **`CERBERUS_REPR_LOSS_WEIGHT=0`**,
출력 `runs/abl2_*`. arm 당 약 2.1일, 전체 약 10.5일.

순서: `flow → zero → random_mask → duplicate → flow_reinit`

## 왜 재실행하는가 (v1 실패 요약)

| arm | v1 결과 |
|---|---|
| flow | 40/40 완주. 단 **epoch 28 에 표현 손실이 정류점 이탈**, 마지막 12 epoch 이 다른 arm 과 다른 실효 목적함수 |
| random_mask | epoch 5 에서 DataLoader worker OOM kill. 러너가 감지 못하고 다음 arm 으로 진행 |
| duplicate | epoch 16 에서 **발산** — 파라미터 193/231 비유한, 회복 불가. 마지막 정상은 ckpt_15 |
| zero | 미시작 |
| flow_reinit | 미시작 |

셋 다 원인이 같다 — `L_sim`·`L_dis` 가 `cos = ±1` 정류점에 갇혀 있다가 장기 학습에서
이탈하고, 이탈 시점이 arm 마다 다르며, 어떤 arm 에서는 학습을 죽인다. 논문은 두 항을
기여로 주장하지 않으므로(Appendix A) 0 으로 두는 데 잃는 것이 없다.

**v1 결과는 지우지 않는다** — 이탈 현상 자체가 Appendix A.1 의 근거다.

## 끝나면 할 일

```bash
# arm 별 평가 (정적 입력 모드는 run 의 config.yaml 에서 자동 검출)
for a in flow zero random_mask duplicate flow_reinit; do
  CUBLAS_WORKSPACE_CONFIG=:4096:8 $CERBERUS_PY scripts/evaluate.py \
    --ckpt $CERBERUS_ROOT/runs/abl2_$a/main/checkpoint_39.pth \
    --anno $CERBERUS_ROOT/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json \
    --out experiments/out/eval_abl2_$a.json
done
# 통계 비교 (쌍 부트스트랩 + Holm + 도메인별)
$CERBERUS_PY scripts/compare_arms.py $(for a in flow zero random_mask duplicate flow_reinit; do
  echo --eval experiments/out/eval_abl2_$a.json; done) --out experiments/out/arm_comparison.json
```

## 결과가 나오면 판정하는 것

`experiment-roadmap.md` §0 의 S1/S2/S3 사다리와 §6 의 결과별 대응을 그대로 따른다.
주 종점은 **도메인별** 값이다(풀링은 DoTA 62.8% 편중).

## 미해결 (Loop 3 must-fix 중 남은 것)

| | 내용 |
|---|---|
| W4 | "용량·정규화로 만들 수 없다" 문구 — 분모 정정으로 상당 부분 해소, 잔여 문구 정리 필요 |
| W8 | §4.2 의 5범주 스키마와 Appendix H.1 의 4범주 공존 — 4범주로 통일 |
| W11 | Scene-word Recall 표제어화·동의어 미처리 — **결과 나오기 전에 동결할 것** |
| — | Stage-1 재학습 여부 (정렬 결함 근본 해결 + C3 증거 확보를 한 번에) — GPU 예산 결정 |
| — | 사람 주석자 2인째 (κ 를 사람–사람으로 만들려면 필요) |

## 최근에 고친 것 (되돌리지 말 것)

- **생성 비결정론** — `do_sample=False` 만으로 부족했다. 같은 체크포인트·같은 클립에서
  3/3 다른 출력, 하나는 `4 4 4 4 …` 퇴화. `enforce_determinism()` 추가 후 3/3 동일.
  평가는 반드시 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 과 함께 실행할 것.
- **추론 경로가 arm 설정을 무시** — `load_streams_aligned` 가 항상 `flow` 로 고정돼 있었다.
  이제 run 의 `config.yaml` 에서 자동 검출한다.
- **정답 설명 오염** — UCF-Crime 10.8% 가 LLM 거부 응답. 평가에서만 제외(학습은 통제 유지).
- **Stage-1 정렬 결함 공개** — `checkpoint_106` 은 정렬 수정 커밋보다 10시간 앞선다.
  §4.1 에 공개했고 Appendix A 에 전제로 달았다.
