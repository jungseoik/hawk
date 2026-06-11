"""
E1/E3 — Representation-space Visualization
==========================================
Projects the three middle representations (z_a appearance, z_m motion,
z_b background) to 2-D with t-SNE or UMAP and renders a scatter plot.

Used to visualize:
  - Dual-Branch (HAWK)              : only appearance + motion clusters
  - Tri-Branch + buggy L_dis        : motion and background OVERLAP (collapse)
  - Tri-Branch + fixed L_dis        : motion and background SEPARATE

Input: an .npz with z_a, z_m, z_b (from scripts/extract_representations.py).
With no input it fabricates the 'collapsed' vs 'separated' regimes so the
figure pipeline is verifiable today.

Dependencies: numpy, matplotlib (required); scikit-learn or umap-learn (for 2-D).
"""
from __future__ import annotations

import argparse
import numpy as np


def _embed(X: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", random_state=seed,
                    perplexity=min(30, max(5, X.shape[0] // 10))).fit_transform(X)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(method)


def plot(z_a, z_m, z_b, out: str, method: str = "tsne", title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(z_a), len(z_m), len(z_b))
    X = np.concatenate([z_a[:n], z_m[:n], z_b[:n]], 0)
    emb = _embed(X, method)
    ea, em, eb = emb[:n], emb[n:2 * n], emb[2 * n:3 * n]

    plt.figure(figsize=(6, 6))
    plt.scatter(*ea.T, s=12, alpha=0.6, label="appearance $z_a$", marker="o")
    plt.scatter(*em.T, s=12, alpha=0.6, label="motion $z_m$", marker="^")
    plt.scatter(*eb.T, s=12, alpha=0.6, label="background $z_b$", marker="s")
    plt.legend()
    plt.title(title or f"Middle-representation space ({method.upper()})")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved -> {out}")


def _synthetic(regime: str, n=200, d=64, seed=0):
    rng = np.random.default_rng(seed)
    fa = rng.standard_normal((n, d)) + np.array([5.0] + [0] * (d - 1))
    fm = rng.standard_normal((n, d)) + np.array([0, 5.0] + [0] * (d - 2))
    if regime == "separated":
        fb = rng.standard_normal((n, d)) + np.array([0, 0, 5.0] + [0] * (d - 3))
    else:  # collapsed: background sits on top of motion
        fb = fm + 0.2 * rng.standard_normal((n, d))
    return fa, fm, fb


def main():
    ap = argparse.ArgumentParser(description="E1/E3 representation visualization")
    ap.add_argument("--reps", type=str, default=None, help="npz with z_a, z_m, z_b")
    ap.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    ap.add_argument("--out", type=str, default="experiments/out/repviz.png")
    ap.add_argument("--regime", choices=["separated", "collapsed"], default="separated",
                    help="synthetic regime when --reps is not given")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.reps:
        d = np.load(args.reps)
        plot(d["z_a"], d["z_m"], d["z_b"], args.out, args.method)
    else:
        za, zm, zb = _synthetic(args.regime)
        plot(za, zm, zb, args.out, args.method,
             title=f"Synthetic ({args.regime}) — {args.method.upper()}")


if __name__ == "__main__":
    main()
