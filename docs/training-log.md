# CERBERUS Stage-1 — Training Log & Decisions

Chunk-by-chunk history of the Stage-1 core pretrain (`runs/core`), and **why** the schedule
was changed on 2026-08-09. Companion to [`training_plan.md`](training_plan.md) (the plan)
and [`MIGRATION.md`](MIGRATION.md) (how to move servers).

> Every relaunch appends provenance to `runs/core/run_info.txt` (git commit, dirty count,
> GPUs, resume source) and a `git_diff_<ts>.patch`. This file is the human-readable summary.

---

## Chunk history

| # | Window | Server | GPUs | eff. batch | Epochs | Ended by |
|---|---|---|---|---|---|---|
| 1 | 2026-07-30 15:51 → 07-31 08:33 | `/data/pia` (local disk) | 2 (`0,1`) | 8 | 0 → 25 | host reboot |
| 2 | 2026-08-03 12:02 → 08-04 07:16 | `/data/pia` (local disk) | 2 (`0,1`) | 8 | 24 → 54* | manual stop |
| 3 | 2026-08-09 ~02:00 → | Backend.AI container, `/home/work/seoik` (NFS) | **3** (`0,1,2`) | **12** | 53 → 107 | (running) |

\* epoch 54 reached iter 1264/1500 and was discarded — resume granularity is per-epoch, so
**`checkpoint_53` is the real continuation point**.

### State at the end of chunk 2 (the resume point)
- Total loss: 8.0 at start → **~2.9 plateau from ~epoch 10 onward** (ep30 2.94, ep50 2.92, ep54 2.90 smoothed)
- Complementarity LOSS terms `middleloss` (= 1 − cos(z_a,z_m)) and `middleloss_bg` (= (1 + cos(z_m,z_b))/2)
  → **≈ 0** (converged, as designed). NOTE: these are losses, not cosines — inverting them gives
  **cos(z_a,z_m) → +1** (alignment) and **cos(z_m,z_b) → −1** (anti-correlation). See `base_task.py:264-270`.
- LR: 1e-5 → **8.53e-6** — i.e. only 15% decayed after 53 epochs
- Checkpoints: local `0..53`; HF backup `backseollgi/Cerberus/stage1_core/` `0..45` (+50)
- Samples consumed: 79,500 step × 8 = **636,000**

---

## Decision (2026-08-09): `max_epoch` 200 → **107**, GPUs 2 → **3**

### Why the plateau was not a convergence problem

`hawk/common/optims.py:99` anneals cosine over **`max_epoch × iters_per_epoch` total steps**:

```python
lr = (init_lr - min_lr) * 0.5 * (1 + cos(pi * total_step / (max_epoch * iters_per_epoch))) + min_lr
```

So `max_epoch` is the **schedule length**, not a harmless cap. It was set to 200 as a
"generous cap — stop anytime after ~3 days" (commits `e11860c`, `95eda28`), which stretched
the cosine over 300,000 steps. After 53 epochs we were only **26.5% into the schedule**, LR
was still 8.53e-6, and the late-cosine decay that actually drives convergence had not begun.
Continuing under `max_epoch: 200` meant **95 more hours** at a near-constant high LR.

### Why 107 specifically — match the sample budget, not the epoch count

Epoch numbers are **not comparable** between HAWK and CERBERUS; the units differ:

| | original HAWK stage-1 | CERBERUS (chunk 3) |
|---|---|---|
| `iters_per_epoch` | 2,500 | 1,500 |
| batch (per GPU) | 1 | 4 |
| GPUs | 4 | 3 |
| **samples per "epoch"** | **10,000** | **18,000** |

Copying `max_epoch: 160` from HAWK would therefore train 20% *longer*, not equally.
Matching **samples** instead:

```
HAWK stage-1 plan  : 160 ep × 2500 it × 4  = 1.60M samples   (ran to checkpoint_127 = 1.27M)
CERBERUS target    : 107 ep × 1500 it × 12 = 1.60M samples   ← equivalent budget
```

Consequences at resume:

| | under `max_epoch: 200` | under `max_epoch: 107` |
|---|---|---|
| position of `checkpoint_53` | 26.5% of schedule | **49.5%** |
| LR on resume | 8.53e-6 | **5.57e-6** |
| remaining wall-clock | 95.5 h (2 GPU) | **34.6 h** (3 GPU) |
| LR at finish | ~5e-6 (never annealed) | **1e-6 = `min_lr`** |

### Why 3 GPUs actually saves time here (and why it isn't automatic)

`iters_per_epoch` is **fixed at 1500**, so an "epoch" is a 39-minute checkpoint interval, not
a pass over the data (one real pass over 2.63M samples would be ~329k iters). Adding a GPU
therefore does **not** shrink epoch wall-clock — it raises samples/step from 8 to 12.

The speedup only materialises because we lowered `max_epoch` to match: reaching the same
1.60M-sample budget needs 133,333 steps at 12/step instead of 200,000 at 8/step.

| | 2 GPU | 3 GPU |
|---|---|---|
| steps to 1.60M samples | 200,000 | **133,333** |
| required `max_epoch` | 133 | **107** |
| remaining wall-clock | 52.2 h | **34.6 h** |

### Known discontinuities at epoch 54 (expected, recorded here on purpose)

1. **LR steps down** 8.53e-6 → 5.57e-6. Downward, so it does not fight the restored Adam
   moments (raising LR mid-run is the dangerous direction) and it helps leave the plateau.
2. **Effective batch 8 → 12.** Gradients get less noisy from epoch 54 on. Report Stage-1 as
   "epochs 0–53 at effective batch 8, 54–107 at 12" rather than a single number.
3. Both land at the same point on the curve, so a small step in the loss/TensorBoard trace at
   epoch 54 is expected and is **not** a bug.

`init_lr`, `min_lr`, `warmup_steps`, `iters_per_epoch`, `batch_size_train` and `seed` were
**left untouched**, so the only intended changes are the two above.

---

## Environment notes for chunk 3 (new server)

Moved from `/data/pia` (local disk) to a Backend.AI container with `/home/work/seoik` on NFS.
Nothing else about the run changed. Gotchas hit during setup, recorded so they are not
re-discovered:

- **`/home/work/seoik` is the only persistent volume.** `$HOME` (`/home/work`) is container
  scratch and disappears with the session. Conda, caches and data must live under `seoik/`.
  `train_run.sh` now pins `TORCH_HOME` / `HF_HOME` there.
- **The container ships its own conda** at `/home/work/miniconda3`, active as `base`, and its
  `envs_dirs` does not include ours — so `conda run -n cerberus` silently resolves to the
  wrong prefix and fails with `No module named 'torch'`. `train_run.sh` now addresses the env
  by **absolute prefix** (`conda run -p /home/work/seoik/miniconda3/envs/cerberus`).
- **NFS is the throughput bottleneck**, not CPU/RAM (83 cores, 676 GB idle). Measured: 25.8 MB/s
  sequential, 35 ms per small-file write, and **saturated at ~50 files/s regardless of
  parallelism** (8 / 32 / 64 workers all measured the same). Extracting 2.66M mp4 files took
  ~3 h; raising worker count does not help.
- `scripts/extract_all_webvid.py` (new) replaces `build_webvid_split.py` here: one 9.1T volume
  means no big/small split or union symlinks are needed. It is resumable per shard via
  `<out>/.done/<page_dir>` markers.
- **TensorBoard is NOT a complete record of this run.** chunk 1-2's event files stayed on the
  old server, so `runs/core/main/tensorboard/` starts at step 81,005. `train.log` is the only
  source covering epoch 0 onward (it is appended across chunks and travelled via HF). Use
  `scripts/plot_training_curves.py` to rebuild the full curves + a CSV from the log; it handles
  the re-run epochs at the resume points (25, 54) by letting the later record win.
- Model build verified before training with
  `python scripts/smoke_test.py --cfg configs/train_configs/stage1_main.yaml` → all three
  streams (appearance / motion / background) forward OK on H100 (sm_90), torch 2.11+cu128,
  transformers 4.28.

---

# Stage-2 finetune (anomaly instruction tuning) — 이 서버, 2026-08-10 착수

## 설정과 그 근거

| 항목 | 값 | 근거 |
|---|---|---|
| `ckpt` | `runs/core/main/checkpoint_106.pth` | stage1 최종 (LR이 `min_lr`까지 annealing 완료) |
| `max_epoch` | **160** (원본 그대로) | 아래 예산 계산 참조 |
| `batch_size_train` | **2** (원본 1) | GPU가 4장→2장이라 effective batch를 맞추기 위해 |
| `iters_per_epoch` | 2500 (원본 그대로) | |
| `num_workers` | 8 (원본 16) | 실측 `data:` 지연 0.0000 — 8로 충분, NFS 부하 절감 |
| GPU | H100 2장 (stage1은 3장) | 세션 재할당으로 변경됨 |

**샘플 예산 — 원본과 동일하게 통제:**

```
원본 HAWK stage2 : 160 ep x 2500 it x batch 1 x 4 GPU = 1.60M 샘플 (effective batch 4)
이 서버           : 160 ep x 2500 it x batch 2 x 2 GPU = 1.60M 샘플 (effective batch 4)
```

GPU 장수가 절반이 된 만큼 batch를 2배로 올려, **샘플 예산과 effective batch를 둘 다** 원본과
일치시켰다. 그 결과 `max_epoch`은 원본 값 160을 그대로 쓸 수 있다 (stage1에서 107로 재계산해야
했던 것과 다른 점 — stage1은 `iters_per_epoch`도 1500으로 바꿨기 때문).

**batch를 더 키우지 않은 이유 (2026-08-10 실측, 150 iter 공정 비교, H100 2장):**

| batch | 정상상태 s/it | max mem | 처리량 | 1.6M 샘플 소요 |
|---|---|---|---|---|
| 2 | 1.369 | 45.7 GB | **2.92 samples/s** | 6.3일 |
| 4 | 2.773 | 56.6 GB | 2.89 samples/s | 6.4일 |
| 6 | 4.31 | 69.0 GB | 2.78 samples/s | 6.7일 |
| 8 | — | OOM (75.9 GB 할당 후 실패) | — | — |

batch에 비례해 iter 시간이 같이 늘어 **처리량이 평평하다** = 32프레임 x 3스트림 조합이 이미
샘플당 연산 포화. VRAM을 더 채워도 이득이 없고 batch 6부터는 오히려 저하되므로, 여유 34GB를
남겨 다일 학습의 OOM 위험을 줄이는 batch 2를 택했다. `data:` 지연은 모든 설정에서 0.0000이라
NFS는 병목이 아니다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`도 batch 6의 저하를
해소하지 못했다(4.72 s/it) — 단편화가 아니라 연산 한계라는 근거.

## 착수 전 처리한 이관 문제 (다른 서버에서 재현할 때도 필요)

1. **annotation의 비디오 경로가 원저자 서버 절대경로였다.** 배포본 JSON의 `video` 필드가
   `/remote-home/share/jiaqitang/Data/...`인데, 로더는 `os.path.join(vis_root, sample['video'])`
   (`video_instruct_dataset.py:128`)를 쓴다. **두 번째 인자가 절대경로면 `vis_root`가 통째로
   무시되므로** config의 `videos_dir`를 고쳐도 소용이 없다. 게다가 데이터셋 이름이 실제 배포본
   디렉토리명과 다르다(`UCF-Crime`↔`UCF_Crime`, `avenue/avenue/`↔`CUHK_Avenue/`,
   `ped1/ped1/`↔`Ped1/`, `ShanghaiTechDataset`↔`ShanghaiTech`).
   → `scripts/localize_anomaly_annotations.py`가 `videos_dir` 기준 상대경로로 변환해
   `*.local.json`을 만든다(원본 보존). train 7066 / test 786 / all 7852 전부 매칭 확인.
2. **config에 남아 있던 외부 경로 6곳**(`llama_model`, `tokenizer_name`, `prompt_path`, `ckpt`,
   `anno_dir`, `videos_dir`)을 이 서버 기준으로 교체. `output_dir`도 레포 안(`output/`)을 가리키던
   것을 `runs/` 절대경로로 옮겼다(`train_run.sh` 사용 시에는 run별로 덮어써짐).

## stage2의 손실 구조는 stage1과 다르다 (논문 서술 시 주의)

instruct 데이터는 `conv_type='multi'` 경로(`video_llama.py:738-806`)를 타는데, 여기서는 세 스트림의
임베딩을 **토큰으로 concat해 LLaMA를 한 번만** 통과시키고 `loss_motion`·`loss_background`에
같은 `loss` 값을 복사해 반환한다. 따라서 stage2 총손실은 실질적으로

```
total = 1.2 x LM손실 + 0.1 x middleloss + 0.1 x middleloss_bg
```

이며, stage1처럼 스트림별 독립 LM 손실이 존재하지 않는다. 이는 원본 HAWK 설계 그대로이고 버그가
아니다. 로그의 `motionloss`/`backgroundloss`가 `oriloss`와 동일하게 찍히는 것은 정상이다.

## 논문 작성에 필요한 산출물 — 학습 완료 시 아래가 전부 수집되어야 한다

`train_run.sh`가 `runs/stage2/`에 자동으로 남기는 것:

- `main/checkpoint_{0..159}.pth` — epoch별 체크포인트 (약 2GB x 160 = 320GB)
- `main/tensorboard/` — 스칼라 7종: `Loss/total`, `Loss/ori`, `Loss/motion`, `Loss/background`,
  `Loss/middle`, `Loss/middle_bg`, `Learning Rate` (global step = epoch x 2500 + i)
- `train.log` — 전 구간 append. **TB보다 이쪽이 권위 있는 기록**이다(chunk 재개 시 TB는 끊길 수 있음)
- `run_info.txt` — chunk별 provenance: config 경로, GPU, git commit + dirty 여부, resume 지점
- `config.yaml` — 실행 시점 config 사본
- `git_diff_<ts>.patch` — 실행 시점 워킹트리 diff

학습 완료 후 추가로 만들어야 하는 것:

- `scripts/plot_training_curves.py`로 stage2 곡선 + CSV 생성 → `figs/stage2_curves.{png,pdf,csv}`
- HF `backseollgi/Cerberus/stage2_finetune/`에 체크포인트(마일스톤+최종)·`train.log`·`config.yaml`·
  곡선 백업 (stage1과 동일한 방식)
- E1/E2 실측: `scripts/extract_representations.py` + `experiments/disentanglement.py`로 CDS 산출.
  **stage1 체크포인트와 stage2 체크포인트 양쪽에서 뽑아야** 분리도가 finetune으로 어떻게 변했는지
  비교할 수 있다.
- Table-1용 VAD 정량 수치 (`scripts/run_eval.sh`, test split은 `all_videos_test.local.json`)

---

## NaN epoch 평균의 정체 (2026-08-12, 실측)

`abl_flow` arm의 로그에서 `Averaged stats`의 `totalloss`가 28 epoch 중 **9회 `nan`** 으로
찍혔고, 26–28에서 3연속으로 나타났다. 학습 실패로 읽힐 수 있는 형태이므로 원인을 확정했다.

**실측.** 출력된 iteration 7,032개 중 NaN은 **1개**(0.014%)다. epoch 평균은 2,500 iteration의
평균이므로 **NaN 하나가 섞이면 평균 전체가 NaN**이 된다. 즉 로그의 `nan`은 "이 epoch이
실패했다"가 아니라 "이 epoch에 NaN iteration이 **적어도 하나** 있었다"는 뜻이다.

실제 발생률은 epoch당 약 0.4회다. 그러면 `P(epoch에 NaN ≥ 1) = 9/28 = 0.32`이고,
3연속이 나타날 확률은 28 epoch 구간에서 약 60%다. **3연속은 악화의 증거가 아니라 이
발생률에서 예상되는 사건이다.**

**가중치 확인.** `checkpoint_27.pth`의 학습 파라미터 231개 전부 NaN/Inf 없음. AMP GradScaler가
NaN 손실에서 옵티마이저 스텝을 건너뛰므로 가중치가 오염되지 않는다. 정상 epoch의 loss는
단조 하강한다(1.537 → 0.983).

**감시 기준 교체.** "연속 3 epoch NaN"은 이 발생률에서 우연과 악화를 구별하지 못하므로 폐기한다.

| 폐기 | 대체 |
|---|---|
| 연속 3 epoch NaN | **출력 iteration의 NaN 비율 > 1%** (현재 0.014%) |
| | **체크포인트에 NaN/Inf 파라미터** (현재 없음) |
| | **정상 epoch loss가 3회 연속 상승** (현재 단조 하강) |

**논문 표기.** arm별 NaN epoch 수를 보고할 때 이 각주가 함께 가야 한다. 수치만 제시하면
심사자가 9/28을 학습 불안정으로 읽는다. 보고 형식은 "NaN epoch 9/28 (iteration 수준
NaN 비율 0.014%; epoch 평균은 단일 NaN에 의해 오염된다)"로 한다.

재현:
```bash
$CERBERUS_PY - <<'EOF'
import re, collections
tot = collections.Counter(); nan = collections.Counter()
for l in open('/home/work/seoik/runs/abl_flow/train.log', errors='ignore'):
    m = re.search(r'Train: data epoch: \[(\d+)\].*?totalloss: (nan|[0-9.]+)', l)
    if m:
        e = int(m.group(1)); tot[e] += 1
        if m.group(2) == 'nan': nan[e] += 1
print(f'출력 {sum(tot.values())}개 중 NaN {sum(nan.values())}개')
EOF
```

---

## random_mask arm 중단과 러너의 실패 미감지 (2026-08-13)

`abl_random_mask` 가 **epoch 5, iteration 320 부근**에서 다음과 같이 죽었다.

```
RuntimeError: DataLoader worker (pid 3751766) is killed by signal: Killed.
[ablation] arm=random_mask 종료 (exit 1)
[ablation] arm=duplicate 시작              ← 러너가 실패를 무시하고 진행
```

**자원 문제가 아니다.** 사후 확인: RAM 2,015GB 중 1,898GB 여유, `/dev/shm` 64GB(사용
276KB), GPU 46.8/81.6GB. 같은 데이터로 `flow` 는 40 epoch 완주했고 `duplicate` 는 문제없이
진행 중이다. 일시적 메모리 스파이크로 보이며 재현되지 않았다.

**복구 가능.** `checkpoint_4.pth` 에 `model`·`optimizer`·`scaler`·`epoch` 이 모두 있어
epoch 5 부터 이어갈 수 있다(`train_run.sh` 가 최신 체크포인트를 자동으로 찾아
`resume_ckpt_path` 로 넘긴다).

### 러너의 결함과 대응

`run_ablation_arms.sh` 는 `train_run.sh` 의 종료 코드를 확인하지 않는다. arm 하나가 5 epoch
에서 죽어도 다음 arm 으로 넘어가므로, **동일 예산 통제가 조용히 깨진다.**

실행 중인 스크립트는 편집하지 않았다 — bash 는 스크립트를 바이트 오프셋으로 읽으므로
진행 중 파일을 고치면 남은 부분을 잘못 해석한다. 대신 `run_ablation_followup.sh` 를 새로
두고, 완료 epoch 수를 확인해 목표에 못 미치면 재시도하도록 했다.

### 검증하려다 실제 학습을 띄운 사고

`ABL_ARMS="" bash scripts/run_ablation_followup.sh` 로 "arm 없이 상태만 보자"고 실행했는데
학습이 시작되어 `duplicate` 와 GPU 를 다투었다. 원인은 `${ABL_ARMS:-기본값}` 의 `:-` 가
**빈 문자열도 기본값으로 치환**한다는 점이다. 즉시 종료하여 피해는 없었다(`duplicate`
정상 진행, `random_mask` 체크포인트·로그 불변).

세 가지 안전장치를 넣었다.

| 장치 | 효과 |
|---|---|
| `--check` 플래그 | 상태만 출력하고 종료. 검증 시 학습이 시작될 수 없다 |
| `${ABL_ARMS-...}` (`:` 제거) | 빈 문자열을 "arm 없음"으로 존중 |
| 실행 중 프로세스 감지 | 다른 절제 학습이 있으면 거부(`ABL_FORCE=1` 로만 무시) |

### 남은 일정

```
flow          40/40  완주
random_mask    5/40  재개 필요
duplicate     12/40  진행 중
zero           0/40  대기
flow_reinit    0/40  대기
```

`zero` 종료 후 `bash scripts/run_ablation_followup.sh` 로 `random_mask` 재개 +
`flow_reinit` 실행. 약 144 epoch × 75분 ≈ 7.5일.

