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
import argparse, math, os, re
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
# `nan` 도 매칭한다. AMP 학습에서는 2500 iteration 중 한둘이 fp16 오버플로로 NaN 손실을
# 내는 일이 있고(GradScaler 가 해당 스텝을 건너뛰므로 학습 자체는 진행된다), 그 한 번이
# epoch 전체 평균을 NaN 으로 만든다. 숫자만 매칭하면 해당 epoch 이 곡선에서 **조용히
# 사라져** 구멍이 있다는 사실조차 드러나지 않는다. 매칭한 뒤 아래에서 반복 로그로부터
# 강건하게 재계산하고, 몇 개를 제외했는지 보고한다.
_NUM = r"(-?[\d.]+|nan)"
RE_AVG = re.compile(
    rf"Averaged stats:.*?totalloss: {_NUM}.*?oriloss: {_NUM}.*?middleloss: {_NUM}"
    rf".*?motionloss: {_NUM}.*?backgroundloss: {_NUM}.*?middleloss_bg: {_NUM}")
RE_ITER = re.compile(
    rf"Train: data epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?lr: ([\d.]+).*?totalloss: {_NUM}"
    rf".*?oriloss: {_NUM}.*?middleloss: {_NUM}.*?motionloss: {_NUM}"
    rf".*?backgroundloss: {_NUM}.*?middleloss_bg: {_NUM}")

# Averaged stats 와 반복 로그의 항목 순서가 다르다 (Averaged: total,ori,middle,motion,bg,middle_bg
# / 반복 로그도 동일 순서). 재계산 시 이 키 순서를 공유한다.
LOSS_KEYS = ("total", "ori", "middle", "motion", "background", "middle_bg")


def parse(path):
    """-> (epoch_stats, iter_points, chunk_starts). Later chunks overwrite re-run epochs.

    NaN 처리: 로그의 `Averaged stats` 가 NaN 이면 그 epoch 의 값을 **반복 로그로부터
    재계산**한다(유한한 iteration 만 평균). 재계산에 쓸 수 있는 iteration 이 없으면 그
    epoch 은 NaN 으로 남기되, 어느 epoch 이 그랬는지 호출자에게 알린다 — 곡선에서 조용히
    빠지는 것보다 명시적으로 결측인 편이 낫다.
    """
    epochs, iters, chunks = OrderedDict(), [], []
    per_epoch_iters = {}
    nan_epochs = []
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
                ep, it, ipe = int(m.group(1)), int(m.group(2)), int(m.group(3))
                lr = float(m.group(4))
                losses = [float(m.group(i)) for i in range(5, 11)]
                iters.append((ep + it / ipe, lr, losses[0]))
                per_epoch_iters.setdefault(ep, []).append(losses)
                continue
            m = RE_AVG.search(line)
            if m and cur is not None:
                vals = [float(x) for x in m.groups()]
                if not any(math.isfinite(v) for v in vals) or not math.isfinite(vals[0]):
                    rows = per_epoch_iters.get(cur, [])
                    n_used = 0
                    for j in range(len(vals)):
                        finite = [r[j] for r in rows if math.isfinite(r[j])]
                        if not math.isfinite(vals[j]):
                            vals[j] = sum(finite) / len(finite) if finite else float("nan")
                        n_used = max(n_used, len(finite))
                    nan_epochs.append((cur, n_used))
                epochs[cur] = dict(zip(LOSS_KEYS, vals))

    if nan_epochs:
        print(f"[curves] Averaged stats 가 NaN 인 epoch {len(nan_epochs)}개 — "
              "반복 로그의 유한값으로 재계산했습니다 "
              "(AMP 오버플로로 일부 iteration 이 NaN; GradScaler 가 해당 스텝을 건너뜁니다).")
        for ep, n in nan_epochs[:5]:
            print(f"          epoch {ep}: 유한 iteration {n}개로 평균")

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
