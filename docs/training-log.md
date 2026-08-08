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
- Complementarity terms `cos(z_a,z_m)` and `cos(z_m,z_b)` → **≈ 0** (converged, as designed)
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
- Model build verified before training with
  `python scripts/smoke_test.py --cfg configs/train_configs/stage1_main.yaml` → all three
  streams (appearance / motion / background) forward OK on H100 (sm_90), torch 2.11+cu128,
  transformers 4.28.
