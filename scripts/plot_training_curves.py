"""
Reconstruct the FULL Stage-1 training curves from train.log and render paper figures.

Why parse the log instead of TensorBoard: `train.log` is appended across every
chunk (and survived the server migration via HF), so it holds the complete run from
epoch 0. The TensorBoard event files do NOT — chunk 1-2's events were left on the old
server, so TB alone starts at step 81,005. The log is the only complete source.

Handles the things that make this log non-trivial:
  - chunks overlap (a crashed epoch is re-run on resume) -> later record wins per epoch
  - `Averaged stats` prints lr rounded to 0.0000 -> LR is taken from the per-iter lines
  - effective batch changed at epoch 54 (2 GPU -> 3 GPU), annotated on the figure

Usage:
    python scripts/plot_training_curves.py                       # defaults below
    python scripts/plot_training_curves.py --log <path> --out figs/stage1_curves
"""
import argparse, os, re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated palette (light surface).
# Validator: lightness/chroma/CVD/normal-vision all PASS; aqua warns on contrast,
# so every series is ALSO direct-labeled — identity never rests on color alone.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

RE_EPOCH_START = re.compile(r"Start training epoch (\d+)")
RE_RESUME = re.compile(r"Resume checkpoint from .*checkpoint_(\d+)\.pth")
RE_AVG = re.compile(
    r"Averaged stats:.*?totalloss: ([\d.]+).*?oriloss: ([\d.]+).*?middleloss: ([\d.]+)"
    r".*?motionloss: ([\d.]+).*?backgroundloss: ([\d.]+).*?middleloss_bg: ([\d.]+)")
RE_ITER = re.compile(
    r"Train: data epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?lr: ([\d.]+).*?totalloss: ([\d.]+)")


def parse(path):
    """-> (epoch_stats, iter_points, chunk_starts). Later chunks overwrite re-run epochs."""
    epochs, iters, chunks = OrderedDict(), [], []
    cur = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = RE_RESUME.search(line)
            if m:
                chunks.append(int(m.group(1)) + 1)  # resumes at the NEXT epoch
                continue
            m = RE_EPOCH_START.search(line)
            if m:
                cur = int(m.group(1))
                continue
            m = RE_ITER.search(line)
            if m:
                ep, it, ipe, lr, tot = int(m.group(1)), int(m.group(2)), int(m.group(3)), \
                    float(m.group(4)), float(m.group(5))
                iters.append((ep + it / ipe, lr, tot))
                continue
            m = RE_AVG.search(line)
            if m and cur is not None:
                epochs[cur] = dict(zip(
                    ("total", "ori", "middle", "motion", "background", "middle_bg"),
                    (float(x) for x in m.groups())))
    # a re-run epoch appears twice; keep the last write, then sort by epoch
    return OrderedDict(sorted(epochs.items())), iters, sorted(set(chunks))


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)


def mark_batch_change(ax, at, label=True):
    """Epoch 54: resumed on 3 GPUs, effective batch 8 -> 12 (see docs/training-log.md)."""
    ax.axvline(at, color=INK_MUTED, linewidth=1.0, linestyle="--", alpha=0.7)
    if label:
        ax.annotate("resume: eff. batch 8→12,\nLR schedule 200→107 ep",
                    xy=(at, ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=7.5, color=INK_MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/home/work/seoik/runs/core/train.log")
    ap.add_argument("--out", default="figs/stage1_curves")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--batch-change", type=int, default=54)
    args = ap.parse_args()

    epochs, iters, chunks = parse(args.log)
    if not epochs:
        raise SystemExit(f"no epoch stats parsed from {args.log}")
    ep = list(epochs.keys())
    get = lambda k: [epochs[e][k] for e in ep]
    print(f"[plot] epochs {ep[0]}..{ep[-1]} ({len(ep)} records), "
          f"{len(iters)} iter points, resume points at epochs {chunks}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="white")
    fig.suptitle("CERBERUS Stage-1 pretraining on WebVid (2.64M clips)",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.985)

    # A — total loss, the headline curve. Single series: no legend box, title names it.
    ax = axes[0][0]
    ax.plot(ep, get("total"), color=C_BLUE, linewidth=2)
    style(ax, "epoch", "loss", "A. Total loss")
    mark_batch_change(ax, args.batch_change)
    ax.annotate(f"{get('total')[-1]:.3f}", xy=(ep[-1], get("total")[-1]),
                xytext=(-4, 8), textcoords="offset points", ha="right",
                fontsize=8.5, color=INK)

    # B — components. Three series: legend AND direct labels (aqua fails contrast alone).
    ax = axes[0][1]
    for key, col, lab in ((("ori"), C_BLUE, "language (ori)"),
                          ("motion", C_ORANGE, "motion"),
                          ("background", C_AQUA, "background")):
        ax.plot(ep, get(key), color=col, linewidth=2, label=lab)
        ax.annotate(lab, xy=(ep[-1], get(key)[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    style(ax, "epoch", "loss", "B. Loss components")
    mark_batch_change(ax, args.batch_change, label=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")
    ax.set_xlim(right=ep[-1] + (ep[-1] - ep[0]) * 0.22)

    # C — LR from the per-iter lines (Averaged stats rounds lr to 0.0000).
    ax = axes[1][0]
    if iters:
        xs = [p[0] for p in iters][::20]
        ys = [p[1] for p in iters][::20]
        ax.plot(xs, ys, color=C_BLUE, linewidth=2)
    style(ax, "epoch", "learning rate", "C. Learning rate (cosine)")
    mark_batch_change(ax, args.batch_change, label=False)
    ax.annotate("schedule horizon shortened\n→ annealing actually reaches min_lr",
                xy=(args.batch_change, ax.get_ylim()[1]), xytext=(6, -6),
                textcoords="offset points", va="top", fontsize=7.5, color=INK_MUTED)

    # D — complementarity: the CVD claim. Both should sit at ~0.
    ax = axes[1][1]
    for key, col, lab in (("middle", C_BLUE, "cos(z_a, z_m)"),
                          ("middle_bg", C_ORANGE, "cos(z_m, z_b)")):
        ax.plot(ep, get(key), color=col, linewidth=2, label=lab)
    style(ax, "epoch", "cosine similarity", "D. Stream complementarity")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper right")
    mark_batch_change(ax, args.batch_change, label=False)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):   # pdf = vector, for camera-ready
        p = f"{args.out}.{ext}"
        fig.savefig(p, dpi=args.dpi, facecolor="white", bbox_inches="tight")
        print(f"[plot] wrote {p}")

    # CSV so the numbers behind the figure are inspectable (and table-view accessible)
    csv_path = f"{args.out}.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,total,ori,motion,background,middle,middle_bg\n")
        for e in ep:
            d = epochs[e]
            f.write(f"{e},{d['total']},{d['ori']},{d['motion']},{d['background']},"
                    f"{d['middle']},{d['middle_bg']}\n")
    print(f"[plot] wrote {csv_path}")


if __name__ == "__main__":
    main()
