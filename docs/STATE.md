# 현재 상태 — 세션이 끊겨도 여기부터 읽으면 이어갈 수 있다

**마지막 갱신 2026-08-18.** 긴 서술은 다른 문서에 있다. 여기는 *지금 무엇이 돌고 있고
다음이 무엇인가*만 둔다.

## ▶ 현재 — 5개 arm 중 4개 완주, `flow` 만 남음 (2026-08-23)

```
  ✅ zero         40/40
  ✅ duplicate    40/40   oriloss 1.2402 → 1.0682
  ✅ flow_reinit  40/40   oriloss 1.2567 → 0.7255
  ✅ random_mask  40/40   oriloss 1.2810 → 0.9230
  🔄 flow         36/40   (GPU 0)
```

완주한 arm 은 전부 체크포인트가 건강하다(비유한 파라미터 0/231, 손실 단조 하강).

**주의 — 아직 해석하지 말 것.** arm 별 최종 학습 손실은 하류 성능이 아니다. 판정은 held-out
평가와 `compare_arms.py` 의 클립 단위 쌍 부트스트랩으로만 한다
(`experiment-roadmap.md` §0 의 S1/S2/S3).

남은 4 epoch × 1.81h ≈ **7시간**, 완료 예상 **8월 23일 저녁**.
그다음은 아래 "끝나면 할 일" 의 평가 + 통계 비교로 바로 넘어간다.

재개 명령(중단됐을 경우):
```bash
cd $CERBERUS_ROOT/hawk && ARM_GPUS="0" bash scripts/run_arms_parallel.sh
```

### 실측 처리량 — 추정에 이것만 쓸 것

**1.81 h/epoch** (전 구간 실측: 웨이브 1 두 arm 38 epoch 이 68.6시간). 로그의 순간
`s/it` 값은 1.8~4.7 사이로 크게 진동하므로 **완료 예상 계산에 쓰지 말 것** — 실제로 이
값으로 추정했다가 완료일을 이틀 앞당겨 잘못 보고한 적이 있다. 6시간당 +3 epoch 이
관측된 실제 속도다.

## 🔁 대화 기록이 이제 영속된다 (2026-08-18)

`~/.claude` 는 컨테이너와 함께 사라지므로 `claude --resume` 목록도 사라졌다. 정본을
`seoik/claude_state/` 에 두고 `scripts/claude_state_sync.sh` 로 세 겹 보존한다 — 부팅 시
`--restore`, 5분 주기 데몬 `--save`, 정상 종료 시 `SessionEnd` 훅. 부트스트랩 1.5 단계가
자동 처리하므로 새 컨테이너에서 `claude --continue` 로 바로 이어갈 수 있다.
**부트스트랩을 먼저 돌린 다음 claude 를 띄워야 한다** (목록은 실행 시점에 읽힌다).
과거 세션 5개가 이관돼 실제 재개까지 검증됐다.

## ⚠ 컨테이너 메모리 제한이 모든 미스터리 사망의 원인이었다

```
/sys/fs/cgroup/memory.max      240 GB   (구 컨테이너)  →  739 GB  (2026-08-18 신 컨테이너)
/sys/fs/cgroup/memory.events   oom_kill 6              →  0
```

`free` 는 호스트 2TB 를 보여준다 — 속지 말고 항상 cgroup 값을 볼 것.
구 컨테이너에서는 DataLoader worker 가 OOM kill → rank 사망 → NCCL watchdog SIGABRT 가
약 5 epoch(6.6시간)마다 발생했다. 신 컨테이너는 여유가 3배라 재발 가능성이 낮지만,
`num_workers` 4 와 재시도 상한 12 는 그대로 둔다(마진만 커진다).
**재개는 정상 작동하므로 진행분을 잃지 않는다.** random_mask v1 중단, 평가 2건 실패,
flow v2 의 주기적 사망이 전부 이것이었다.

확인: `cat /sys/fs/cgroup/memory.events`

확인: `bash scripts/run_ablation_v2.sh --check`
가중치 확인: `grep -m1 REPR_LOSS_WEIGHT $CERBERUS_ROOT/runs/abl2_flow/train.log`

## 이미 나온 실측 (첫 결과)

`flow` v1 arm 의 이탈 전/후 비교 — 표현 손실이 활성화되면 성능이 떨어진다.

| | ep27 (이탈 전) | ep39 (이탈 후) |
|---|---|---|
| Scene-word Recall | **0.4085** | 0.3208 |
| BLEU-1 | 0.3259 | 0.2866 |

Δ = +0.0920, 95% CI [+0.0564, +0.1276], n = 424 — 구간이 0 을 포함하지 않는다.
원고 반영: Appendix A.2. **이것이 `t = 0` 재실행의 실증 근거다.**
결과: `experiments/out/eval_flow_ep{27,39}.json`

## 설계 결정 — 원본과의 예산 일치는 필수가 아니다 (2026-08-17)

원본 HAWK 의 effective batch·샘플 예산에 맞추는 것은 **요구사항이 아니다.** 필요한 것은
**arm 끼리 서로 같은 조건**뿐이다. GPU 구성 변경을 검토할 때 "원본과 달라진다"를 반대
근거로 쓰지 말 것 (루트 `CLAUDE.md` §5 완화 항목 참조).

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

| — | Stage-1 재학습 여부 (정렬 결함 근본 해결 + C3 증거 확보를 한 번에) — GPU 예산 결정 |
| — | C3 증거 경로: Appendix F(손실 항 누적 절제)를 실행할지, C3 를 기여에서 내릴지 |

## Loop 3 이후 완료한 것

- **가설 형식 제거** — H1~H7 → 발견 서술. 심사자가 찾은 모순 3건 동반 해소
- **벤치마크 "재사용 가능" 주장 철회** — 라벨은 LLM 생성 + 사람 1인 80건 검증. 그 규모로
  독립 벤치마크를 주장하면 정작 튼튼한 것들까지 신뢰를 잃는다. 두 번째 주석자도 불필요해짐
- **Scene-word Recall 편향 3종 보정** (결과 전 동결) — 표제어화, 동의어 15군, 속성 상충
  무효화. 합성 6케이스 검증
- **라벨 스키마 4범주로 통일**, `context_critical` → `causal`
- **부록 A.1·A.2 위치 정정** (Appendix I 뒤에 있었음), 수식 (A.1) → (B.1)
- **[54]/[55]/[57] 기술 정정** — Barlow Twins ≠ VICReg, RESOUND 은 예측 불가능성이 아니라 최소화

## 최근에 고친 것 (되돌리지 말 것)

- **생성 비결정론** — `do_sample=False` 만으로 부족했다. 같은 체크포인트·같은 클립에서
  3/3 다른 출력, 하나는 `4 4 4 4 …` 퇴화. `enforce_determinism()` 추가 후 3/3 동일.
  평가는 반드시 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 과 함께 실행할 것.
- **추론 경로가 arm 설정을 무시** — `load_streams_aligned` 가 항상 `flow` 로 고정돼 있었다.
  이제 run 의 `config.yaml` 에서 자동 검출한다.
- **정답 설명 오염** — UCF-Crime 10.8% 가 LLM 거부 응답. 평가에서만 제외(학습은 통제 유지).
- **Stage-1 정렬 결함 공개** — `checkpoint_106` 은 정렬 수정 커밋보다 10시간 앞선다.
  §4.1 에 공개했고 Appendix A 에 전제로 달았다.
