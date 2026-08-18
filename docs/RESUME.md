# 새 세션에서 이어가기 — 여기부터 읽으세요

`/home/work/seoik` 만 영구입니다. 그 밖(`$HOME` 포함)은 컨테이너와 함께 사라집니다.
아래 순서대로 하면 중단 지점에서 그대로 이어집니다.

---

## 1. 부트스트랩 (제일 먼저, 반드시)

```bash
bash /home/work/seoik/bootstrap_cerberus.sh   # 멱등 — 여러 번 돌려도 안전
source ~/.bashrc
```

복구되는 것: `~/.claude` 심링크(에이전트·스킬), 지난 대화 기록,
`CERBERUS_PY`·`CERBERUS_ROOT`·`TORCH_HOME`·`HF_HOME`·`HF_TOKEN`·`GEMINI_API_KEY`·
`CUBLAS_WORKSPACE_CONFIG`, git 사용자 설정. 데이터·모델·체크포인트 존재도 검증합니다.

**건너뛰면 `$CERBERUS_PY` 가 비어 모든 학습·평가 명령이 조용히 실패합니다.**

지난 대화를 이어받으려면 부트스트랩 **후에** `claude --continue` 또는 `claude --resume`
(목록은 claude 실행 시점에 읽히므로 순서가 중요합니다).

## 2. 상황 파악 — 이 순서로 읽으세요

| 문서 | 내용 |
|---|---|
| **`docs/STATE.md`** | 지금 무엇이 어디까지 됐고 다음이 무엇인가 — **가장 먼저** |
| `docs/MIGRATION-3GPU.md` | 컨테이너 이전·GPU 구성 변경 절차 |
| `docs/experiment-roadmap.md` | 실험 계획, 결과별 대응, 기각된 실험과 그 이유 |
| `docs/training-log.md` | 학습 이력, 실패와 원인, 설정 변경 근거 |
| `docs/evidence-index.md` | 각 주장의 근거와 재현 명령 |
| `docs/review-log.md` | 심사 지적 → 조치 → 커밋, **하지 않은 것과 그 이유** |
| 루트 `CLAUDE.md` | 세션 운영·서버 함정. **§5 완화 항목** 필독 |

## 3. 현재 상태 확인

```bash
cd $CERBERUS_ROOT/hawk
bash scripts/run_arms_parallel.sh --check      # arm 진행률
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cat /sys/fs/cgroup/memory.max                  # 컨테이너 메모리 한계 (free 는 호스트를 보여줌)
git status --porcelain | wc -l                 # 0 이어야 정상
```

## 4. 학습 재개

```bash
# GPU 2장이면
ARM_GPUS="0 1" bash scripts/run_arms_parallel.sh
# GPU 3장이면
bash scripts/run_arms_parallel.sh
```

남은 arm 을 **GPU 1장씩 병렬**로 돌립니다. 완료된 arm 은 건너뛰고, 웨이브가 끝난 뒤 남은
arm 이 있으면 같은 명령을 다시 실행하면 됩니다.

**GPU 를 한 작업에 묶지 마세요.** `iters_per_epoch: 2500` 이 고정이라 GPU 를 더 붙여도
epoch 시간이 줄지 않고 epoch 당 샘플 수만 늘어납니다. arm 을 쪼개야 벽시계 시간이 줍니다.

## 5. 반드시 지킬 것 세 가지

1. **평가는 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 과 함께.** 없으면 생성이 비결정론적이라 같은
   체크포인트가 실행마다 다른 문장을 냅니다(실측 3/3 불일치, 하나는 `4 4 4 4 …` 퇴화).
2. **`CERBERUS_REPR_LOSS_WEIGHT=0`.** 러너가 export 하지만, 직접 학습을 띄울 때는 확인하세요.
   이 항이 켜지면 arm 마다 다른 시점에 활성화되어 통제가 깨지고, 한 arm 은 발산했습니다.
3. **실행 중인 bash 스크립트를 편집하지 마세요.** bash 가 바이트 오프셋으로 읽어 나가므로
   남은 부분을 잘못 해석합니다. 사본을 만들어 고치세요.

## 6. 학습이 끝나면

```bash
for a in flow zero random_mask duplicate flow_reinit; do
  CUBLAS_WORKSPACE_CONFIG=:4096:8 $CERBERUS_PY scripts/evaluate.py \
    --ckpt $CERBERUS_ROOT/runs/abl2_$a/main/checkpoint_39.pth \
    --anno $CERBERUS_ROOT/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json \
    --out experiments/out/eval_abl2_$a.json
done
$CERBERUS_PY scripts/compare_arms.py \
  $(for a in flow zero random_mask duplicate flow_reinit; do echo --eval experiments/out/eval_abl2_$a.json; done) \
  --out experiments/out/arm_comparison.json
```

평가는 **순차로** 돌리세요. 동시에 돌리면 디코딩 중 호스트 메모리를 함께 잡습니다.

## 7. 백업 위치

| | |
|---|---|
| 코드·논문·문서 | GitHub `jungseoik/hawk` |
| 스킬·에이전트 | GitHub `jungseoik/seoik_skills` (독립 저장소) |
| 체크포인트·로그·결과 | HF `backseollgi/Cerberus` — `ablation_v2/`·`docs/`·`results/`·`benchmark/` |
| 대화 기록 | `seoik/claude_state/` (정본) · `seoik/session_archive/` (스냅샷) |

체크포인트 전량은 `seoik/runs/` 에만 있습니다(arm 당 77GB). HF 에는 완주한 arm 의 최종본만
올립니다.
