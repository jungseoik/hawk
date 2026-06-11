# CERBERUS — Experiment Code

> Code for the paper's experiment section. Design rationale & metric definitions:
> [`../docs/cerberus-research-plan.md`](../docs/cerberus-research-plan.md) §3.
> Datasets are integrated **last**; everything here is verifiable today on
> synthetic inputs so the pipeline is proven before data arrives.

## Mapping: paper experiment → code

| Paper | What it proves | Code | Runnable now? |
|---|---|---|---|
| **E1** Disentanglement measurement | dynamic/static streams are non-redundant yet jointly cover full-frame info (CDS metric) | `disentanglement.py` | ✅ synthetic self-test |
| **E1/E3** Representation viz | motion/background clusters separate under correct loss | `representation_viz.py` | ✅ synthetic regimes |
| **E2** Background-critical benchmark | model uses static scene context causally (BSI metric) | `bg_critical_benchmark/` | ◐ schema + compositor + BSI self-test (data later) |
| **E3** Loss-direction causal check | `(1+cos)/2` disentangles, `1-cos` collapses | `loss_direction.py` | ✅ synthetic convergence demo |
| **E4** Complementarity generality | CVD helps across backbones/tasks (not HAWK-specific) | `configs/plugin_ablation.yaml` | ✗ needs training |
| Ablation A/C | per-component contribution | `configs/ablation.yaml` | ✗ needs training |
| Analysis C | flow-threshold sensitivity | `configs/threshold_sweep.yaml` | ✗ needs training |
| Analysis D / §P4 | unified flow ≈ 2x faster | `efficiency_benchmark.py` | ✅ synthetic frames (cv2) |

## Quick start (no dataset needed)

```bash
conda activate cerberus
python experiments/disentanglement.py --self-test     # CDS separates regimes
python experiments/loss_direction.py                  # buggy vs fixed convergence
python experiments/representation_viz.py --regime separated --method pca
python experiments/representation_viz.py --regime collapsed --method pca
python experiments/bg_critical_benchmark/background_swap.py --self-test
python experiments/efficiency_benchmark.py            # needs --full tier (cv2)
```

## On a trained model

```bash
# 1) dump middle representations z_a, z_m, z_b over a test split
python scripts/extract_representations.py --cfg configs/eval_configs/eval.yaml \
    --ckpt <ckpt.pth> --out experiments/out/reps.npz
# 2) measure
python experiments/disentanglement.py --reps experiments/out/reps.npz
python experiments/representation_viz.py --reps experiments/out/reps.npz --method tsne
```
