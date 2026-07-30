# CERBERUS Training Plan (paper-aligned, chunked-friendly)

> Overall plan for training CERBERUS and producing every number the paper needs.
> Designed for **chunked training** — stop anytime for other work, relaunch to
> auto-resume (validated: same run dir + latest checkpoint + continuous curves).
> Maps to contributions **C1–C5** and experiments **E1–E4** in
> [`cerberus-research-plan.md`](cerberus-research-plan.md) / `paper_translation/improved/04_experiments.md`.

## Chunked training — the workflow (this is fully fine)
Interrupted/stop-and-continue training gives the **same result** as a continuous run:
resume restores model + optimizer + AMP scaler + LR schedule + epoch, and TensorBoard
stays one continuous line (global step = `epoch*iters + i`).
```bash
# start (or continue) — SAME command each time; auto-resumes from latest checkpoint
bash scripts/train_run.sh configs/train_configs/stage1_main.yaml core 0,1 2
# stop: Ctrl-C or kill the process anytime. Rerun the same line later to continue.
```
- One stable run dir: `/data/pia/runs/<name>/` (job `main`). Checkpoints, `tensorboard/`,
  `train.log`, and provenance (`run_info.txt`, `git_diff_*.patch`, `config.yaml`) accumulate there.
- **Caveat**: resume granularity = per-epoch, so a mid-epoch stop loses only that epoch's
  partial progress. Keep `iters_per_epoch` moderate (checkpoint ~every 1 h) so little is lost.
- Monitor: `tail -f /data/pia/runs/core/train.log` ; `tensorboard --logdir /data/pia/runs/core/main/tensorboard`.

## Phase 0 — Core model (Stage 1 pretrain)  ← DO THIS FIRST
- Config: `stage1_main.yaml` (batch 2, 2×GPU). Run **as long as feasible** (the more
  converged, the more convincingly the tri-branch / CVD advantage shows). Chunk freely.
- Produces: the **best core checkpoint** + loss & complementarity curves
  (`Loss/middle` = cos(z_a,z_m), `Loss/middle_bg` = cos(z_m,z_b)) — direct E1/E3 evidence.
- Throughput ~3 clips/s on 2× Blackwell (compute-bound). Reference: ~6 days ≈ HAWK-equiv
  1.6M samples; ~10 days ≈ one pass over 2.63M.

## Phase 1 — Stage 2 finetune (the headline VAD numbers)  ← needs anomaly data
- ⚠️ The paper's Table-1 (BLEU / GPT-guided) numbers come from **Stage 2** on the HAWK
  7-dataset anomaly set — **not yet downloaded**. Stage 1 alone validates the architecture,
  not the headline metrics.
- Config: `stage2_finetune.yaml`; set `ckpt:` to the Phase-0 core checkpoint.

## Phase 2 — Ablations (for the paper's ablation TABLE)  ← rigorous, not forked
**Do NOT fork ablations from the core checkpoint** — that confounds the result (you'd measure
late-removal, not the component's training contribution). Instead run each ablation as an
**independent run from the same init at an equal, reduced budget** (fair comparison), including
a *full-model-at-reduced-budget* as the ablation baseline:
| run | change vs full | paper |
|---|---|---|
| full (reduced budget) | — (ablation baseline) | Ablation A |
| w/o Background branch | drop static stream (→ dual-branch) | C1 |
| w/o `L_BL` | no scene-language supervision | C3 |
| `L_dis` = `1−cos` (buggy) | wrong loss direction | E3 (causal) |
| w/o `L_dis` | no dissimilarity objective | C2 |
| τ ∈ {0.05…0.5} | flow-threshold sweep | Analysis C |

Each = its own `train_run.sh ... <name>` run dir → directly comparable. (Ablation configs to be
added under `configs/train_configs/ablations/`.)

## Phase 3 — Diagnostics & analysis (post-hoc on checkpoints)
- **E1 (CDS) / representation viz**: `scripts/extract_representations.py --ckpt <core> --out reps.npz`
  → `experiments/disentanglement.py --reps reps.npz` + `representation_viz.py`.
- **E3 (loss-direction)**: compare the buggy vs fixed `L_dis` runs' cos(z_m,z_b) trajectories
  (from `Loss/middle_bg` curves) + representation separation.
- **E2 (BSI / bg-critical)**, **E4 (generality)**: need the curated benchmark / 2nd backbone (later).

## What's recorded automatically (paper-traceable)
Per run dir: per-iter CLI losses (`train.log`), TensorBoard scalars (all 5 losses + LR),
per-epoch checkpoints (weights+optimizer+scaler+**config**), and git-commit provenance.
→ Any reported number is traceable to exact code + config + data.

## Disk
~2 GB/checkpoint. Bound the count via `iters_per_epoch` (fewer, spaced checkpoints) or prune
old ones; `/data/pia` has ~2.3 TB free. Extracted dataset ~1.18 TB already resident.
