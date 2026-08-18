# 컨테이너 이전 — GPU 3장 환경에서 이어가기

이 문서 하나로 재개 가능하도록 쓴다. 순서대로 따라가면 된다.

---

## 0. 먼저 알아야 할 것

**`/home/work/seoik` 만 영구다.** `$HOME`(=`/home/work`) 의 나머지는 컨테이너를 내리면
사라진다 — `~/.claude` 포함.

영구 볼륨에 이미 다 있다.

| 자산 | 크기 | 상태 |
|---|---|---|
| `seoik/hawk` (코드·논문) | 13G | git 동기화 완료 |
| `seoik/runs` (체크포인트) | 369G | v1·v2 전부 보존 |
| `seoik/miniconda3` (conda 환경) | 11G | **재설치 불필요** |
| `seoik/hawk_anomaly` (Stage-2 데이터) | 121G | 그대로 |
| `seoik/session_archive` | 22M | 대화 기록 |

---

## 1. 컨테이너 내리기 **전**에 할 일

```bash
cd $CERBERUS_ROOT/hawk

# (a) 미커밋 확인 — 0 이어야 한다
git status --porcelain | wc -l

# (b) 원격과 동기화 확인 — 0 이어야 한다
git fetch -q "https://$(cat ../.github_token)@github.com/jungseoik/hawk.git" main
git rev-list --count HEAD ^FETCH_HEAD

# (c) 세션 기록 아카이브 (컨테이너와 함께 사라진다)
D=$CERBERUS_ROOT/session_archive/$(date +%Y%m%d)
mkdir -p "$D"
cp -r ~/.claude/projects/-home-work-seoik "$D"/ 2>/dev/null
cp ~/.claude/history.jsonl "$D"/ 2>/dev/null

# (d) 학습 정지 — 체크포인트는 epoch 마다 저장되므로 진행분을 잃지 않는다
pkill -f "run_ablation_v[2]"; pkill -f "train[.]py --cfg-path"
```

체크포인트가 epoch 단위로 저장되므로 **아무 때나 끊어도 된다.** 진행 중이던 epoch 만
다시 돈다.

---

## 2. 새 컨테이너에서 복구

```bash
bash /home/work/seoik/bootstrap_cerberus.sh   # 멱등 — 여러 번 돌려도 안전
source ~/.bashrc
```

이것이 복구하는 것: `~/.claude` 심링크(에이전트·스킬), 셸 환경
(`CERBERUS_PY`·`TORCH_HOME`·`HF_HOME`·`HF_TOKEN`·`GEMINI_API_KEY`), git 사용자 설정.
그리고 데이터·모델·체크포인트가 살아있는지 검증한다.

**이걸 건너뛰면 `$CERBERUS_PY` 가 비어 모든 학습·평가 명령이 조용히 실패한다.**

확인:

```bash
echo $CERBERUS_PY                       # 비어 있으면 안 된다
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader   # 3장 확인
cat /sys/fs/cgroup/memory.max           # 컨테이너 메모리 제한 — 병목이다
bash $CERBERUS_ROOT/hawk/scripts/run_arms_parallel.sh --check     # arm 진행률
```

---

## 3. 이어서 학습 — GPU 3장 병렬

```bash
cd $CERBERUS_ROOT/hawk
bash scripts/run_arms_parallel.sh
```

남은 arm 을 자동으로 골라 **GPU 1장씩 병렬**로 돌린다. 완료된 arm 은 건너뛴다.
웨이브가 끝나고 남은 arm 이 있으면 같은 명령을 다시 실행하면 된다.

### 왜 GPU 를 한 작업에 몰지 않는가

`iters_per_epoch` 이 2500 으로 **고정**이라, GPU 를 더 붙이면 epoch 당 샘플 수만 늘고
epoch 시간은 그대로다. arm 을 나눠 병렬로 돌려야 벽시계 시간이 줄어든다.

```
2 GPU × batch 2 = effective 4,  2500 iter × 4 = 10,000 샘플/epoch
1 GPU × batch 4 = effective 4,  2500 iter × 4 = 10,000 샘플/epoch   ← 동일
```

조건이 같으므로 **이미 완주한 arm(`zero`)을 다시 돌릴 필요가 없다.**

| 방식 | 소요 |
|---|---|
| 2 GPU 순차 유지 | 6.7일 |
| 3 GPU 한 작업 (arm 불일치로 전부 재실행) | 10.8일 |
| **arm 3개 병렬 (`run_arms_parallel.sh`)** | **3.1일** |

### 주의 — 병목은 GPU 가 아니라 메모리다

```
/sys/fs/cgroup/memory.max      240 GB   (기존 컨테이너 기준)
/sys/fs/cgroup/memory.events   oom_kill 6
```

`free` 는 호스트 값을 보여주므로 속지 말 것. DataLoader worker 가 OOM kill 되면 rank 가
죽고 NCCL watchdog 가 SIGABRT 를 던진다. **재개는 정상 동작하므로 진행분을 잃지 않으며**,
러너가 재시도로 감싸 두었다(상한 12).

`*_1gpu.yaml` 은 `num_workers` 를 4 로 낮춰 두었다 — arm 3 개면 워커가 3 배가 되기 때문이다.
학습은 compute-bound 이므로(`data:` 지연 실측 0.0000) 속도 영향은 없다.

새 컨테이너의 메모리 제한이 다르면 `num_workers` 를 조정할 것.

---

## 4. 학습이 끝나면

```bash
cd $CERBERUS_ROOT/hawk

# arm 별 평가 — 순차로 돌린다. 동시에 돌리면 디코딩 중 메모리를 함께 잡는다.
for a in flow zero random_mask duplicate flow_reinit; do
  CUBLAS_WORKSPACE_CONFIG=:4096:8 $CERBERUS_PY scripts/evaluate.py \
    --ckpt $CERBERUS_ROOT/runs/abl2_$a/main/checkpoint_39.pth \
    --anno $CERBERUS_ROOT/hawk_anomaly/Annotation/All_Mix/all_videos_heldout.local.json \
    --out experiments/out/eval_abl2_$a.json
done

# 통계 비교 — 클립 단위 쌍 부트스트랩 + Holm + 도메인별
$CERBERUS_PY scripts/compare_arms.py \
  $(for a in flow zero random_mask duplicate flow_reinit; do echo --eval experiments/out/eval_abl2_$a.json; done) \
  --out experiments/out/arm_comparison.json
```

⚠ **평가는 반드시 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 과 함께 실행한다.** 없으면 생성이
비결정론적이어서 같은 체크포인트가 실행마다 다른 문장을 낸다(실측 3/3 불일치, 하나는
`4 4 4 4 …` 로 퇴화).

---

## 5. 결과를 어떻게 읽는가

`docs/experiment-roadmap.md` §0 의 S1/S2/S3 사다리와 §6 의 결과별 대응.
주 종점은 **도메인별** 값이다(풀링은 DoTA 62.8% 편중).

---

## 6. 이어서 읽을 문서

| 문서 | 내용 |
|---|---|
| `docs/STATE.md` | 지금 무엇이 돌고 있고 다음이 무엇인가 — **가장 먼저** |
| `docs/experiment-roadmap.md` | 실험 계획·결과별 대응·기각된 실험과 그 이유 |
| `docs/training-log.md` | 학습 이력·실패와 원인·설정 변경 근거 |
| `docs/evidence-index.md` | 각 주장의 근거와 재현 명령 |
| `docs/review-log.md` | 심사 지적 → 조치 → 커밋, 그리고 **하지 않은 것과 그 이유** |
| 루트 `CLAUDE.md` | 세션 운영·서버 함정. **§5 완화 항목**(원본과의 예산 일치는 필수가 아님)을 반드시 볼 것 |

## 7. 백업 상태

- **GitHub** `jungseoik/hawk` — 코드·논문·문서 전부. 미커밋 0
- **HF** `backseollgi/Cerberus` — `ablation_v2/`(완주 arm 체크포인트·로그·설정),
  `docs/`, `results/`, `benchmark/`

체크포인트는 arm 당 77GB 라 **완주한 arm 의 최종본만** 올린다. 중간 체크포인트가 필요하면
`seoik/runs` 에서 직접 가져올 것.
