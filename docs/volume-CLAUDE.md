# CERBERUS 작업 볼륨 — Claude 세션 길잡이

이 파일은 `/home/work/seoik`에서 세션을 열면 자동으로 읽힙니다. **다른 무엇보다 먼저 여기부터.**

## 0. 새 세션이라면 — **`hawk/docs/RESUME.md` 부터 읽으세요**

부트스트랩 → 상황 파악 → 학습 재개까지 한 장으로 정리돼 있습니다. 아래는 그 요약입니다.

## 0.1 첫 명령

```bash
bash /home/work/seoik/bootstrap_cerberus.sh   # 멱등 — 여러 번 돌려도 안전
source ~/.bashrc
```

`~/.claude`(에이전트·스킬 심링크), 셸 환경(`CERBERUS_PY`·`TORCH_HOME`·`HF_HOME`·`HF_TOKEN`),
git 사용자 설정을 복구하고 데이터·모델·체크포인트가 살아있는지 검증합니다.
**이걸 건너뛰면 `$CERBERUS_PY`가 비어 있어 모든 학습/평가 명령이 조용히 실패합니다.**

---

## 1. 이 컨테이너의 유일한 규칙: `seoik`만 영구

| 경로 | 운명 |
|---|---|
| `/home/work/seoik` (NFS 9.1T) | **영구** — 데이터·모델·conda·캐시·체크포인트 전부 여기 |
| `$HOME` = `/home/work` 나머지 | 세션 종료 시 **통째로 초기화** (`~/.claude` 포함) |

- **어떤 산출물도 `$HOME`에 두지 마세요.** 캐시(`TORCH_HOME`/`HF_HOME`)도 seoik로 고정돼 있습니다.
- **대화 기록은 이제 자동으로 영속화됩니다** (2026-08-18). 정본은 `seoik/claude_state/`,
  동기화는 `hawk/scripts/claude_state_sync.sh` 가 담당합니다 — 부트스트랩이 `--restore` 로
  되돌리고 5분 주기 데몬이 `--save` 로 밀어 넣으며, 세션 정상 종료 시 `SessionEnd` 훅이 한 번 더 저장합니다
  (훅은 `seoik/.claude/settings.local.json`). 그래서 **새 컨테이너에서 `claude --continue` /
  `claude --resume` 으로 지난 대화를 그대로 이어갈 수 있습니다.**
  `seoik/session_archive/` 는 이전 방식(수동 스냅샷)의 보관소로 남겨 둡니다.
- `~/.claude/projects/-home-work-seoik/memory/`는 세션마다 날아갑니다. **영속 지식은 여기 CLAUDE.md와
  `hawk/docs/`에 쓰세요.** (에이전트 축적 메모리는 예외 — `hawk/.claude/agent-memory/`, git 추적됨)

---

## 2. 문서 역할 분담 (중복 금지)

| 문서 | 다루는 것 |
|---|---|
| **이 파일** (`seoik/CLAUDE.md`) | 세션 운영·서버 함정·현재 진행 상황 |
| **`hawk/docs/experiment-roadmap.md`** | ⭐ **무엇을 왜 어떤 순서로 — 실험 계획의 단일 출처.** 다른 서버의 에이전트는 이것부터 읽으면 된다 |
| `hawk/docs/review-2026-08-10-methodology.md` | 심사 지적과 대응, 코드로 재확인한 사실들 |
| `seoik/README.md` | 자산 지도(어디에 뭐가 있는지)·백업 상태 — 사람용 요약 |
| `hawk/CLAUDE.md` | **연구 내용**: HAWK(베이스) vs CERBERUS(신규) 구분, 논문 집필 규칙, 코드 분류 |
| `hawk/docs/training-log.md` | 학습 청크별 이력·중단 사유·설정 변경 근거 |
| `hawk/docs/session-recovery.md` | README와 동일 내용의 레포 내 사본 |

연구 얘기(Tri-Branch 구조, 논문 포맷, 에이전트 사용법)는 **`hawk/CLAUDE.md`로 가세요.** 여기서 반복하지 않습니다.

---

## 3. 이 서버 특유의 함정 (전부 실측으로 확인된 것)

1. **`conda run -n cerberus`는 실패합니다.** 컨테이너가 자체 conda(`/home/work/miniconda3`)를 base로
   활성화하고 `envs_dirs`를 자기 것으로 고정해서, 이름으로 찾으면 엉뚱한 경로로 풀립니다.
   → **항상 절대 prefix**: `$CERBERUS_PY` 직접 실행, 또는
   `$CERBERUS_ROOT/miniconda3/bin/conda run -p $CERBERUS_ROOT/miniconda3/envs/cerberus ...`
   (`train_run.sh`·`setup_env.sh`는 이미 수정됨)
2. **NFS가 유일한 병목입니다.** CPU 84코어·cgroup 메모리 739GB는 남아돕니다(2026-08-18 실측).
   소파일 쓰기가 **병렬도와 무관하게 ~50 files/s에서 포화**(파일당 35ms 왕복 지연)하므로
   워커를 늘려도 빨라지지 않습니다. 대량 추출 작업에 워커 증설을 제안하지 마세요.
   단 학습 중 `data:` 지연은 실측 0.0000 — **학습은 compute-bound라 NFS 영향 없습니다.**
3. **Claude 에이전트/스킬은 세션 시작 시점에만 스캔됩니다.** 중간에 추가하면 한동안 안 잡히니
   부트스트랩을 먼저 돌리세요.
4. **컨테이너 `eth0` 카운터에 NFS 트래픽이 안 잡힙니다**(호스트 커널이 처리). 네트워크가
   한가해 보여도 병목이 아니라는 근거가 되지 못합니다.
5. **환경에 `NPROC`이 미리 심어져 있습니다**(CPU 코어 수 — 2026-08-18 컨테이너는 `84`, 이전은 42). 셸 스크립트에서
   `NPROC="${NPROC:-2}"` 같은 흔한 관용구를 쓰면 이 값을 물려받아 `torchrun`이 GPU 42장을
   요구하고 `CUDA error: invalid device ordinal`로 즉사합니다. 스크립트 변수는 반드시
   네임스페이스를 붙이세요(`ABL_NPROC` 등). 다른 일반 이름도 먼저 `echo`로 확인할 것.
6. **이 이미지는 `/usr/share/doc/*` 를 지웁니다** (`dpkg.cfg.d` 의 `path-exclude`). apt 로 설치해도
   문서에 딸린 셸 스크립트가 안 깔립니다 — 실측으로 `fzf` 의 `key-bindings.bash` 가 없어서
   **Ctrl-R 퍼지 검색이 조용히 죽어 있었습니다.** 사본을
   `seoik_skills/skills/server-init/config/fzf-key-bindings.bash` 로 벤더링해 해결했습니다.
   비슷한 도구(zsh/vim 플러그인 등)를 붙일 때 같은 함정을 예상하세요.
7. **tmux·ble.sh·fzf 같은 서버 환경도 매 컨테이너 재설치 대상입니다** — apt 패키지와
   `~/.local` 이 전부 이미지와 함께 되돌아갑니다. 부트스트랩 1.7 단계가 자동 처리하며
   (`seoik_skills/skills/server-init`), 이미 갖춰져 있으면 건너뜁니다. 적용은 `exec bash`.


---

## 3.5. 지금 무엇이 돌고 있나

> **`hawk/docs/STATE.md` 를 먼저 읽으세요.** 현재 실행 중인 작업·다음 단계·미해결 항목이
> 거기 한 장으로 정리돼 있고, 이 파일보다 자주 갱신됩니다.

빠른 확인:
```bash
bash $CERBERUS_ROOT/hawk/scripts/run_ablation_v2.sh --check   # arm 진행률
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

⚠ **평가는 반드시 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 과 함께 실행하세요.** 생성이
비결정론적이어서 같은 체크포인트가 실행마다 다른 문장을 냅니다(실측 3/3 불일치).

## 4. 진행 상황 (2026-08-10 기준)

| 단계 | 상태 |
|---|---|
| WebVid 다운로드 + mp4 추출 | ✅ 1330/1330 shard, **263.8만 clip** |
| **Stage-1 사전학습** | ✅ **완주** — epoch 106, LR→min_lr(1e-6), loss 3.815→2.8506 |
| Stage-2 이상행동 데이터 | ✅ 7,899개 / 121GB (HAWK 7종), `hawk_anomaly/` |
| 논문 도구 | ✅ 자체 에이전트 4 + vendor 스킬 4 |
| **E1/E2 실측** | ⏳ **미실행** (합성 데이터 자체검증만 완료) |
| **Stage-2 finetune** | ⏳ **미시작** (데이터·config 준비 완료) |
| 논문 원고 | ◐ 5개 장 작성, 결과·conclusion·references 미작성 |

**Stage-1 핵심 결과**: 분리 손실 두 항이 **0.0000 수렴**.
최종 체크포인트: `runs/core/main/checkpoint_106.pth` (2.0GB, 총 54개 보존)

> ⚠️ **로그의 `middleloss`/`middleloss_bg`는 코사인 값이 아니라 손실값입니다** (`base_task.py:264-270`):
> `middleloss = 1 − cos(z_a,z_m)`, `middleloss_bg = (1 + cos(z_m,z_b))/2`.
> 따라서 둘 다 0이라는 건 **`cos(z_a,z_m) = +1`(정렬), `cos(z_m,z_b) = −1`(반상관)** 이라는 뜻이고,
> 이게 방향성 비대칭 목적함수의 설계 의도대로 달성된 결과입니다(논문 `02_related_work.md:23`과 일치).
> `runs/core/STOPPED.md:26-27`·`README.md`·`hawk/docs/session-recovery.md`는 이 값을 "코사인이 0으로
> 수렴"이라고 **잘못 표기**하고 있습니다 — 인용하지 말고 고쳐야 합니다. 논문 원고는 정확합니다.

> **stage1의 loss 값 자체는 논문 성능 지표가 아닙니다.** Table-1의 VAD 수치는 stage2 이후에 나옵니다.
> stage1은 "구조가 작동하는가" 검증 단계이고, 그 증거(상보성 수렴)는 이미 확보됐습니다.

---

## 5. 확정된 설계 결정과 그 근거 (되돌리지 말 것)

> ### ⚠ 2026-08-17 완화 — 원본과의 예산·배치 일치는 **필수가 아니다**
>
> 아래 항목들이 "원본 HAWK와 동일한 샘플 예산·effective batch"를 강하게 규정하고 있어,
> GPU 구성을 바꿀 때마다 그 제약이 걸림돌로 작용했다. 사용자 판단으로 **완화한다.**
>
> **중요한 것과 아닌 것을 구분한다.**
>
> | | 중요도 |
> |---|---|
> | 원본 HAWK와 같은 batch·샘플 예산 | **중요하지 않다.** 깨져도 된다 — 대부분의 논문은 베이스라인의 배치 크기까지 맞추지 않는다. 우리가 보는 것은 "효과가 있느냐"이고, 수렴이 충분하면 된다 |
> | **arm 끼리** 같은 batch·샘플 예산 | **중요하다.** 이게 다르면 성능 차이가 배치 때문인지 배경 스트림 때문인지 가릴 수 없다 |
>
> 따라서 GPU 수·배치를 바꾸는 제안을 검토할 때 **원본과의 일치를 근거로 반대하지 말 것.**
> 확인할 것은 하나뿐이다 — 비교하는 arm 들이 서로 같은 조건인가.


이미 논의를 거쳐 확정된 사항입니다. 다시 제안하기 전에 근거를 먼저 읽으세요.

- **`max_epoch: 107`** — 이 코드에서 `max_epoch`은 학습량 상한이 아니라 **cosine LR 스케줄의 길이**입니다.
  원래 200은 "3일 돌리고 아무 때나 끊는다"는 전제의 넉넉한 값이라 LR이 제대로 annealing되지 않았습니다.
  107은 원본 HAWK stage1 계획과 동일한 160만 샘플 예산에 맞춘 값이었다 — 논문에서 "HAWK와 동등 학습량"을
  주장하기 위한 통제입니다. (`iters_per_epoch` 고정이라 GPU를 늘려도 epoch당 시간은 안 줄고,
  `max_epoch`을 함께 낮춰야 실제 시간이 단축됩니다.)
- **freeze 정책은 원본 그대로** — EVA-ViT·Q-Former·LLaMA-2-7B 전부 freeze, video Q-Former +
  projection + fusion head만 학습. 원본 HAWK도 stage1/stage2 모두 동일하며 백본을 푸는 단계가 없습니다.
  모델의 98.3%가 고정이라 **loss가 완만하게 내려가는 건 설계상 당연한 결과**입니다(플래토 아님).
- **Stage-2 데이터는 원본 HAWK 것을 그대로 사용** — 자체 데이터셋 구축은 하지 않습니다.
  "같은 데이터 + 같은 freeze + 같은 샘플 예산 → 차이는 오직 구조"가 이 논문의 통제 논리이고,
  리뷰어가 가장 먼저 찌를 지점입니다. 데이터를 바꾸면 `max_epoch`·freeze를 맞춘 노력이 무의미해집니다.
- **vendor 스킬은 라이선스 표기 후 레포 커밋** — `vendor/academic-research-skills` (CC BY-NC 4.0),
  표기는 `NOTICE.md`. 우리 자체 에이전트에 없는 기능만 흡수하는 방침.

---

## 6. 다음 작업

> ⚠️ **GPU 장수가 바뀌었습니다.** stage1 때는 H100 3장이었으나 **현재 세션은 2장**입니다.
> 아래 명령의 GPU 인자는 실제 `nvidia-smi` 결과로 확인 후 조정하세요.

**① E1/E2 실측** (짧음, GPU 1장이면 충분)
```bash
cd /home/work/seoik/hawk
$CERBERUS_PY scripts/extract_representations.py --cfg configs/eval_configs/eval.yaml \
    --ckpt /home/work/seoik/runs/core/main/checkpoint_106.pth --out experiments/out/reps.npz
$CERBERUS_PY experiments/disentanglement.py --reps experiments/out/reps.npz   # CDS
```

**② Stage-2 finetune** (장시간)
`configs/train_configs/stage2_finetune.yaml`에서 경로 3개를 먼저 교체:
- `ckpt:` → `/home/work/seoik/runs/core/main/checkpoint_106.pth`
- `anno_dir:` → `/home/work/seoik/hawk_anomaly/Annotation/All_Mix/all_videos_train.json`
- `videos_dir:` → `/home/work/seoik/hawk_anomaly/Videos/`

```bash
# 인자: <cfg> <tag> <gpu-ids> <nproc>  — GPU 2장 기준
bash scripts/train_run.sh configs/train_configs/stage2_finetune.yaml stage2 0,1 2
```

---

## 7. 작업 규칙

- **장시간 GPU 작업은 시작 전 승인을 받으세요.** 학습 재개·finetune 착수는 항상 확인 후 진행.
- **주기적으로 보고**하세요 — 사용자가 모니터링합니다. 진행률·ETA·이상 징후를 먼저 알립니다.
- **백업은 두 갈래**: 코드·문서·figure는 GitHub `jungseoik/hawk`, 체크포인트·로그는
  HF `backseollgi/Cerberus/stage1_core/`. 토큰은 `seoik/.github_token`, `seoik/.hf_token`.
  큰 변경 후에는 양쪽 다 밀어두세요 — 다른 서버에서 재현 가능해야 합니다
  (`hawk/docs/reproduce.md`, `hawk/docs/MIGRATION.md`).
- **CERBERUS 관련 논문 작업은 자체 에이전트 우선** (`hawk/.claude/agents/` 4종 — 고유 맥락이
  `agent-memory/`에 축적돼 있음). vendor 스킬은 인용 검증·PRISMA·학회 포맷 같은 범용 기능에만.
  상세 판단 기준은 `hawk/CLAUDE.md`.
- **`webvid_10m*` 1.1TB는 추출이 끝나 삭제 가능**합니다. 재다운로드는
  `hawk/scripts/resilient_hf_download.py`로 약 7시간.

---

> 이 파일은 `/home/work/seoik/CLAUDE.md` 의 사본입니다. 원본은 볼륨 루트에 있으며
> hawk 저장소 밖이라 git 이 추적하지 않습니다. 저장소만 clone 한 경우를 위해 여기 둡니다.
> **원본을 고치면 이 사본도 갱신하세요.**
